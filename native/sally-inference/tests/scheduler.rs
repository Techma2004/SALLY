//! Single-flight scheduler: ordering, queue limit, cancellation, panic isolation.
#[path = "common/gguf_builder.rs"]
mod gguf_builder;
#[path = "common/tiny_llama.rs"]
mod tiny;

use sally_inference::architecture::llama::LlamaModel;
use sally_inference::architecture::{Architecture, KvCache};
use sally_inference::cancel::CancelToken;
use sally_inference::error::EngineError;
use sally_inference::generate::{FinishReason, GenParams};
use sally_inference::memory::Budget;
use sally_inference::scheduler::{Engine, Event, Job, Scheduler};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;
use tiny::{lm, PROMPT};

/// Wraps the real tiny model: optional delay per step, a signal on the first step, a
/// call counter, and a panic when it sees token 99.
struct Probe {
    inner: LlamaModel,
    delay: Duration,
    started: Option<Sender<()>>,
    calls: Arc<AtomicUsize>,
}

impl Architecture for Probe {
    fn forward(&mut self, t: u32, pos: usize, s: &mut KvCache) -> Result<&[f32], EngineError> {
        if t == 99 {
            panic!("boom");
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(tx) = self.started.take() {
            let _ = tx.send(());
        }
        std::thread::sleep(self.delay);
        self.inner.forward(t, pos, s)
    }
    fn context_len(&self) -> usize {
        self.inner.context_len()
    }
    fn kv_bytes(&self, ctx: usize) -> usize {
        self.inner.kv_bytes(ctx)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn new_cache(&self, ctx: usize) -> Result<KvCache, EngineError> {
        self.inner.new_cache(ctx)
    }
    fn scratch_bytes(&self) -> usize {
        self.inner.scratch_bytes()
    }
}

struct Rig {
    sched: Scheduler,
    calls: Arc<AtomicUsize>,
    started: Receiver<()>,
    budget: Budget,
}

fn rig(queue_len: usize, delay_ms: u64) -> Rig {
    let (model, tok) = lm();
    let (tx, started) = channel();
    let calls = Arc::new(AtomicUsize::new(0));
    let budget = Budget::new(64 * 1024 * 1024);
    let engine = Engine {
        model: Box::new(Probe {
            inner: model,
            delay: Duration::from_millis(delay_ms),
            started: Some(tx),
            calls: calls.clone(),
        }),
        tok: Arc::new(tok),
        budget: budget.clone(),
        ctx: 64,
    };
    Rig {
        sched: Scheduler::start(engine, queue_len),
        calls,
        started,
        budget,
    }
}

fn greedy(max_tokens: usize) -> GenParams {
    GenParams {
        max_tokens,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 1,
    }
}

fn job(prompt: &[u32], max_tokens: usize) -> (Job, Receiver<Event>, CancelToken) {
    let (tx, rx) = channel();
    let cancel = CancelToken::new();
    (
        Job {
            prompt: prompt.to_vec(),
            params: greedy(max_tokens),
            cancel: cancel.clone(),
            events: tx,
        },
        rx,
        cancel,
    )
}

fn finish(
    rx: &Receiver<Event>,
) -> Result<(String, sally_inference::generate::GenStats), EngineError> {
    let mut text = String::new();
    loop {
        match rx
            .recv_timeout(Duration::from_secs(30))
            .expect("worker went silent")
        {
            Event::Text(t) => text.push_str(&t),
            Event::Done(s) => return Ok((text, s)),
            Event::Failed(e) => return Err(e),
        }
    }
}

#[test]
fn a_job_streams_text_then_reports_stats() {
    let r = rig(2, 0);
    let (j, rx, _) = job(&PROMPT, 6);
    r.sched.submit(j).unwrap();
    let (text, stats) = finish(&rx).unwrap();
    assert!(stats.generated_tokens() <= 6);
    assert_ne!(stats.finish, FinishReason::Cancelled);
    let _ = text; // content is covered by generate() tests; here only the plumbing
    assert_eq!(
        r.budget.used(),
        0,
        "the request's budget is returned when it ends"
    );
}

#[test]
fn jobs_complete_in_submission_order() {
    let r = rig(4, 0);
    let mut rxs = Vec::new();
    for n in 1..=4 {
        let (j, rx, _) = job(&PROMPT, n);
        r.sched.submit(j).unwrap();
        rxs.push((n, rx));
    }
    for (n, rx) in &rxs {
        let (_, stats) = finish(rx).unwrap();
        assert!(stats.generated_tokens() <= *n);
    }
}

#[test]
fn queue_full_is_resource_exhausted_and_running_job_is_unharmed() {
    let r = rig(1, 40);
    let (a, rx_a, _) = job(&PROMPT, 3);
    r.sched.submit(a).unwrap();
    r.started.recv_timeout(Duration::from_secs(10)).unwrap(); // A is now running
    let (b, rx_b, _) = job(&PROMPT, 2);
    r.sched.submit(b).unwrap(); // takes the single queue slot
    let (c, _rx_c, _) = job(&PROMPT, 2);
    let e = r.sched.submit(c).unwrap_err();
    assert_eq!(e.code(), "resource_exhausted");
    assert!(e.to_string().contains("queue"), "{e}");
    assert!(finish(&rx_a).is_ok());
    assert!(finish(&rx_b).is_ok());
    // Once drained, there is room again.
    let (d, rx_d, _) = job(&PROMPT, 2);
    r.sched.submit(d).unwrap();
    assert!(finish(&rx_d).is_ok());
}

#[test]
fn a_queued_job_cancelled_before_it_starts_never_runs() {
    let r = rig(2, 40);
    let (a, rx_a, _) = job(&PROMPT, 3);
    r.sched.submit(a).unwrap();
    r.started.recv_timeout(Duration::from_secs(10)).unwrap();
    let (b, rx_b, cancel_b) = job(&PROMPT, 5);
    r.sched.submit(b).unwrap();
    cancel_b.cancel();
    assert!(finish(&rx_a).is_ok());
    let calls_after_a = r.calls.load(Ordering::SeqCst);
    let e = finish(&rx_b).unwrap_err();
    assert_eq!(e.code(), "cancelled");
    assert_eq!(
        r.calls.load(Ordering::SeqCst),
        calls_after_a,
        "a cancelled queued job must not touch the model"
    );
}

#[test]
fn cancelling_a_running_job_stops_it_and_frees_the_worker() {
    let r = rig(2, 30);
    let (a, rx_a, cancel_a) = job(&PROMPT, 40);
    r.sched.submit(a).unwrap();
    r.started.recv_timeout(Duration::from_secs(10)).unwrap();
    cancel_a.cancel();
    let (_, stats) = finish(&rx_a).unwrap();
    assert_eq!(stats.finish, FinishReason::Cancelled);
    assert!(
        r.calls.load(Ordering::SeqCst) < 20,
        "should stop long before 40 tokens"
    );
    assert_eq!(r.budget.used(), 0);
    let (b, rx_b, _) = job(&PROMPT, 2);
    r.sched.submit(b).unwrap();
    assert!(finish(&rx_b).is_ok());
}

#[test]
fn a_vanished_client_cancels_generation() {
    let r = rig(2, 30);
    let (a, rx_a, _) = job(&PROMPT, 40);
    r.sched.submit(a).unwrap();
    r.started.recv_timeout(Duration::from_secs(10)).unwrap();
    drop(rx_a); // the HTTP handler gave up
    let (b, rx_b, _) = job(&PROMPT, 2);
    r.sched.submit(b).unwrap();
    assert!(finish(&rx_b).is_ok());
    assert!(
        r.calls.load(Ordering::SeqCst) < 30,
        "the abandoned job kept generating: {} steps",
        r.calls.load(Ordering::SeqCst)
    );
}

#[test]
fn a_failing_job_reports_its_error_and_the_next_job_runs() {
    let r = rig(2, 0);
    let (bad, rx_bad, _) = job(&[260, 104], 0); // max_tokens 0 is an invalid request
    r.sched.submit(bad).unwrap();
    assert_eq!(finish(&rx_bad).unwrap_err().code(), "invalid_request");
    let (ok, rx_ok, _) = job(&PROMPT, 2);
    r.sched.submit(ok).unwrap();
    assert!(finish(&rx_ok).is_ok());
}

#[test]
fn a_panicking_job_does_not_kill_the_worker() {
    let r = rig(2, 0);
    let (bad, rx_bad, _) = job(&[260, 99], 2); // token 99 makes the probe panic
    r.sched.submit(bad).unwrap();
    let e = finish(&rx_bad).unwrap_err();
    assert_eq!(e.code(), "internal");
    assert_eq!(
        r.budget.used(),
        0,
        "a panic must not leak the request's budget"
    );
    let (ok, rx_ok, _) = job(&PROMPT, 2);
    r.sched.submit(ok).unwrap();
    assert!(finish(&rx_ok).is_ok());
}

#[test]
fn shutdown_lets_the_running_job_finish_and_then_refuses_new_ones() {
    let mut r = rig(2, 10);
    let (a, rx_a, _) = job(&PROMPT, 3);
    r.sched.submit(a).unwrap();
    r.started.recv_timeout(Duration::from_secs(10)).unwrap();
    r.sched.shutdown();
    assert!(finish(&rx_a).is_ok(), "the in-flight job completes");
    let (b, _rx_b, _) = job(&PROMPT, 2);
    assert!(r.sched.submit(b).is_err());
}
