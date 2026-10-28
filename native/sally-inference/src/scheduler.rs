//! One worker thread owns the model and runs one generation at a time. Requests wait in a
//! bounded queue; when it is full, `submit` fails with `resource_exhausted` instead of
//! letting work pile up (spec sections 5 and 7).

use crate::architecture::Architecture;
use crate::cancel::CancelToken;
use crate::error::EngineError;
use crate::generate::{generate, GenParams, GenStats};
use crate::memory::{open_cache, Budget};
use crate::tokenizer::Tokenizer;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{sync_channel, Sender, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Everything the worker needs to serve requests.
pub struct Engine {
    pub model: Box<dyn Architecture + Send>,
    pub tok: Arc<Tokenizer>,
    pub budget: Budget,
    /// KV cache size (positions) allocated for each request.
    pub ctx: usize,
}

pub struct Job {
    pub prompt: Vec<u32>,
    pub params: GenParams,
    pub cancel: CancelToken,
    pub events: Sender<Event>,
}

/// What the worker reports back for one job. A job always ends with exactly one of
/// `Done` or `Failed`.
#[derive(Debug)]
pub enum Event {
    Text(String),
    Done(GenStats),
    Failed(EngineError),
}

pub struct Scheduler {
    tx: Option<SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
}

impl Scheduler {
    /// Starts the worker. Up to `queue_len` jobs may wait while one runs.
    pub fn start(engine: Engine, queue_len: usize) -> Scheduler {
        let (tx, rx) = sync_channel::<Job>(queue_len);
        let worker = std::thread::Builder::new()
            .name("sally-worker".into())
            .spawn(move || {
                let mut engine = engine;
                for job in rx {
                    run_job(&mut engine, job);
                }
            })
            .expect("failed to spawn the worker thread");
        Scheduler {
            tx: Some(tx),
            worker: Some(worker),
        }
    }

    /// Queues a job, or fails immediately with `resource_exhausted` if the queue is full.
    pub fn submit(&self, job: Job) -> Result<(), EngineError> {
        let Some(tx) = &self.tx else {
            return Err(EngineError::Internal("the scheduler is shut down".into()));
        };
        tx.try_send(job).map_err(|e| match e {
            TrySendError::Full(_) => {
                EngineError::ResourceExhausted("the request queue is full; retry shortly".into())
            }
            TrySendError::Disconnected(_) => {
                EngineError::Internal("the worker thread has stopped".into())
            }
        })
    }

    /// Stops accepting jobs, lets queued and running jobs finish, then joins the worker.
    pub fn shutdown(&mut self) {
        self.tx = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_job(engine: &mut Engine, job: Job) {
    if job.cancel.is_cancelled() {
        let _ = job.events.send(Event::Failed(EngineError::Cancelled));
        return;
    }
    let events = job.events.clone();
    // A panic inside one request must not take the whole service down.
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<GenStats, EngineError> {
        let (mut cache, _reservation) = open_cache(&*engine.model, engine.ctx, &engine.budget)?;
        generate(
            &mut *engine.model,
            &engine.tok,
            &mut cache,
            &job.prompt,
            &job.params,
            &job.cancel,
            &mut |text| {
                // A closed channel means the client is gone: stop generating for it.
                if events.send(Event::Text(text.to_string())).is_err() {
                    job.cancel.cancel();
                }
            },
        )
    }));
    let event = match result {
        Ok(Ok(stats)) => Event::Done(stats),
        Ok(Err(e)) => Event::Failed(e),
        Err(_) => Event::Failed(EngineError::Internal(
            "the request failed unexpectedly (internal panic)".into(),
        )),
    };
    let _ = job.events.send(event);
}
