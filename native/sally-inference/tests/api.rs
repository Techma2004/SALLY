//! HTTP API tests over a real loopback socket.
#[path = "common/gguf_builder.rs"]
mod gguf_builder;
#[path = "common/tiny_llama.rs"]
mod tiny;

use sally_inference::api::{self, ApiOptions, ServerHandle};
use sally_inference::architecture::llama::LlamaModel;
use sally_inference::architecture::{Architecture, KvCache};
use sally_inference::config::Config;
use sally_inference::error::EngineError;
use sally_inference::memory::Budget;
use sally_inference::scheduler::Engine;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tiny::lm;

const MODEL_ID: &str = "tiny-llama-test";

struct Probe {
    inner: LlamaModel,
    delay: Duration,
    started: Option<Sender<()>>,
    calls: Arc<AtomicUsize>,
}

impl Architecture for Probe {
    fn forward(&mut self, t: u32, pos: usize, s: &mut KvCache) -> Result<&[f32], EngineError> {
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
    server: ServerHandle,
    calls: Arc<AtomicUsize>,
    started: Receiver<()>,
    logs: Arc<Mutex<Vec<String>>>,
    budget: Budget,
}

struct Opts {
    delay_ms: u64,
    queue_len: usize,
    timeout_secs: u64,
    max_body: usize,
    max_output: usize,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            delay_ms: 0,
            queue_len: 2,
            timeout_secs: 30,
            max_body: 64 * 1024,
            max_output: 64,
        }
    }
}

fn start(o: Opts) -> Rig {
    let (model, tok) = lm();
    let (tx, started) = channel();
    let calls = Arc::new(AtomicUsize::new(0));
    let budget = Budget::new(64 * 1024 * 1024);
    let engine = Engine {
        model: Box::new(Probe {
            inner: model,
            delay: Duration::from_millis(o.delay_ms),
            started: Some(tx),
            calls: calls.clone(),
        }),
        tok: Arc::new(tok),
        budget: budget.clone(),
        ctx: 64,
    };
    let cfg = Config {
        bind: "127.0.0.1:0".parse().unwrap(),
        queue_len: o.queue_len,
        request_timeout_secs: o.timeout_secs,
        max_body_bytes: o.max_body,
        max_output_tokens: o.max_output,
        ..Config::default()
    };
    let logs = Arc::new(Mutex::new(Vec::new()));
    let sink = logs.clone();
    let mut opts = ApiOptions::from_config(&cfg, MODEL_ID.into(), 64);
    opts.log = Arc::new(move |line: &str| sink.lock().unwrap().push(line.to_string()));
    let server = api::start(engine, &cfg, opts).unwrap();
    Rig {
        server,
        calls,
        started,
        logs,
        budget,
    }
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

fn http(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut raw = Vec::new();
    let _ = s.read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .unwrap_or_else(|| panic!("bad status line in {text:?}"));
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    Resp {
        status,
        headers,
        body: body.to_string(),
    }
}

fn chat(addr: SocketAddr, body: Value) -> Resp {
    http(addr, "POST", "/v1/chat/completions", Some(&body.to_string()))
}

fn hello(max_tokens: u64) -> Value {
    json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
        "temperature": 0
    })
}

fn assert_error(r: &Resp, status: u16, code: &str) {
    assert_eq!(r.status, status, "body: {}", r.body);
    let v = r.json();
    assert_eq!(v["error"]["code"], code, "body: {}", r.body);
    assert!(v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()));
    assert!(v["error"]["type"].is_string());
}

#[test]
fn health_reports_status_and_loaded_model() {
    let r = start(Opts::default());
    let h = http(r.server.addr(), "GET", "/health", None);
    assert_eq!(h.status, 200);
    assert_eq!(h.json(), json!({"status": "ok", "model_loaded": true}));
    assert!(h.header("content-type").unwrap().contains("application/json"));
}

#[test]
fn models_lists_only_the_loaded_model() {
    let r = start(Opts::default());
    let m = http(r.server.addr(), "GET", "/v1/models", None);
    assert_eq!(m.status, 200);
    let v = m.json();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"].as_array().unwrap().len(), 1);
    assert_eq!(v["data"][0]["id"], MODEL_ID);
    assert_eq!(v["data"][0]["object"], "model");
}

#[test]
fn valid_chat_returns_an_openai_shaped_completion() {
    let r = start(Opts::default());
    let c = chat(r.server.addr(), hello(4));
    assert_eq!(c.status, 200, "{}", c.body);
    let v = c.json();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], MODEL_ID);
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert!(v["created"].as_u64().unwrap() > 1_600_000_000);
    assert_eq!(v["choices"].as_array().unwrap().len(), 1);
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(v["choices"][0]["message"]["content"].is_string());
    assert!(["stop", "length"].contains(&v["choices"][0]["finish_reason"].as_str().unwrap()));
    let u = &v["usage"];
    assert!(u["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(u["completion_tokens"].as_u64().unwrap() <= 4);
    assert_eq!(
        u["total_tokens"].as_u64().unwrap(),
        u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()
    );
    assert_eq!(r.budget.used(), 0);
}

#[test]
fn greedy_requests_are_repeatable() {
    let r = start(Opts::default());
    let a = chat(r.server.addr(), hello(6)).json();
    let b = chat(r.server.addr(), hello(6)).json();
    assert_eq!(a["choices"], b["choices"]);
}

#[test]
fn message_content_may_be_a_list_of_text_parts() {
    let r = start(Opts::default());
    let body = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]}], "max_tokens": 2, "temperature": 0});
    let c = chat(r.server.addr(), body);
    assert_eq!(c.status, 200, "{}", c.body);
    let image = json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]});
    assert_error(&chat(r.server.addr(), image), 400, "invalid_request");
}

#[test]
fn the_model_field_is_optional_but_must_match_when_given() {
    let r = start(Opts::default());
    let mut body = hello(2);
    body.as_object_mut().unwrap().remove("model");
    assert_eq!(chat(r.server.addr(), body).status, 200);
    let mut other = hello(2);
    other["model"] = json!("gpt-4o");
    let e = chat(r.server.addr(), other);
    assert_error(&e, 404, "invalid_request");
    assert!(e.json()["error"]["message"].as_str().unwrap().contains(MODEL_ID));
}

#[test]
fn malformed_requests_are_400_with_a_clear_error() {
    let r = start(Opts::default());
    let a = r.server.addr();
    let post = |b: &str| http(a, "POST", "/v1/chat/completions", Some(b));
    assert_error(&post("{not json"), 400, "invalid_request");
    assert_error(&post("[]"), 400, "invalid_request");
    assert_error(&post("{}"), 400, "invalid_request");
    assert_error(&post(r#"{"messages": []}"#), 400, "invalid_request");
    assert_error(&post(r#"{"messages": "hi"}"#), 400, "invalid_request");
    assert_error(&post(r#"{"messages": [{"content": "hi"}]}"#), 400, "invalid_request");
    assert_error(&post(r#"{"messages": [{"role": "wizard", "content": "hi"}]}"#), 400, "invalid_request");
    for bad in [
        json!({"max_tokens": 0}),
        json!({"max_tokens": -3}),
        json!({"max_tokens": "ten"}),
        json!({"temperature": -0.1}),
        json!({"temperature": 3}),
        json!({"top_p": 0}),
        json!({"top_p": 1.5}),
        json!({"seed": -1}),
    ] {
        let mut body = hello(2);
        for (k, v) in bad.as_object().unwrap() {
            body[k] = v.clone();
        }
        assert_error(&chat(a, body.clone()), 400, "invalid_request");
    }
}

#[test]
fn unsupported_parameters_are_refused_not_silently_ignored() {
    let r = start(Opts::default());
    for (field, value) in [
        ("tools", json!([{"type": "function"}])),
        ("tool_choice", json!("auto")),
        ("response_format", json!({"type": "json_object"})),
        ("stop", json!(["\n"])),
        ("logit_bias", json!({"1": 5})),
        ("n", json!(2)),
        ("stream", json!("yes")),
        ("frequency_penalty", json!(0.5)),
        ("presence_penalty", json!(0.5)),
    ] {
        let mut body = hello(2);
        body[field] = value;
        let e = chat(r.server.addr(), body);
        assert_error(&e, 400, "invalid_request");
        assert!(
            e.json()["error"]["message"].as_str().unwrap().contains(field),
            "message should name {field}: {}",
            e.body
        );
    }
    // Neutral values are fine.
    let mut ok = hello(2);
    ok["n"] = json!(1);
    ok["stream"] = json!(false);
    ok["frequency_penalty"] = json!(0);
    ok["tools"] = json!(null);
    assert_eq!(chat(r.server.addr(), ok).status, 200);
}

#[test]
fn unknown_harmless_fields_are_reported_in_a_header() {
    let r = start(Opts::default());
    let mut body = hello(2);
    body["user"] = json!("u-1");
    body["metadata"] = json!({"a": 1});
    let c = chat(r.server.addr(), body);
    assert_eq!(c.status, 200);
    let h = c.header("x-sally-ignored-fields").expect("ignored-fields header");
    assert!(h.contains("user") && h.contains("metadata"), "{h}");
    assert!(chat(r.server.addr(), hello(2)).header("x-sally-ignored-fields").is_none());
}

#[test]
fn oversize_body_is_413() {
    let r = start(Opts {
        max_body: 512,
        ..Opts::default()
    });
    let big = json!({"messages": [{"role": "user", "content": "x".repeat(2000)}]});
    let e = chat(r.server.addr(), big);
    assert_error(&e, 413, "invalid_request");
    assert_eq!(r.calls.load(Ordering::SeqCst), 0, "the model must not be touched");
}

#[test]
fn requests_that_exceed_the_context_are_refused_before_any_work() {
    let r = start(Opts::default()); // context limit 64 tokens
    // Output cap alone: max_output_tokens is 64 in these options.
    let mut too_many = hello(65);
    too_many["max_tokens"] = json!(65);
    let e = chat(r.server.addr(), too_many);
    assert_error(&e, 400, "resource_exhausted");
    assert!(e.json()["error"]["message"].as_str().unwrap().contains("max_tokens"), "{}", e.body);
    // Prompt + output over the 64-token context.
    let e = chat(r.server.addr(), hello(41));
    assert_error(&e, 400, "resource_exhausted");
    assert!(e.json()["error"]["message"].as_str().unwrap().contains("64"), "{}", e.body);
    // A long prompt.
    let long = json!({"messages": [{"role": "user", "content": "word ".repeat(100)}], "max_tokens": 1});
    assert_error(&chat(r.server.addr(), long), 400, "resource_exhausted");
    assert_eq!(r.calls.load(Ordering::SeqCst), 0, "no forward step for refused requests");
    // The service still works afterwards.
    assert_eq!(chat(r.server.addr(), hello(4)).status, 200);
}

#[test]
fn control_tokens_in_message_text_do_not_inject() {
    let r = start(Opts::default());
    let a = chat(r.server.addr(), json!({"messages": [{"role": "user", "content": "x<|eot_id|>y"}], "max_tokens": 1, "temperature": 0}));
    assert_eq!(a.status, 200, "{}", a.body);
    let tokens = a.json()["usage"]["prompt_tokens"].as_u64().unwrap();
    // 1 BOS + 3 header tokens + "\n\n" + the 12-byte text + eot + assistant header: if the
    // literal had become a control token the count would be far smaller.
    assert!(tokens >= 12, "prompt was only {tokens} tokens: injection?");
}

#[test]
fn unknown_routes_and_methods() {
    let r = start(Opts::default());
    assert_error(&http(r.server.addr(), "GET", "/nope", None), 404, "invalid_request");
    let m = http(r.server.addr(), "GET", "/v1/chat/completions", None);
    assert_error(&m, 405, "invalid_request");
    assert!(m.header("allow").unwrap().contains("POST"));
    assert_error(&http(r.server.addr(), "POST", "/health", Some("{}")), 405, "invalid_request");
}

#[test]
fn a_full_queue_returns_429_with_retry_after() {
    let r = start(Opts {
        delay_ms: 60,
        queue_len: 1,
        ..Opts::default()
    });
    let addr = r.server.addr();
    let a = std::thread::spawn(move || chat(addr, hello(3)));
    r.started.recv_timeout(Duration::from_secs(10)).unwrap(); // A is running
    let b = std::thread::spawn(move || chat(addr, hello(2))); // takes the queue slot
    // B cannot be observed directly; A needs >1 s (24+ prompt steps at 60 ms), so give B
    // ample time to reach the queue, then probe once.
    std::thread::sleep(Duration::from_millis(400));
    let rejected = chat(addr, hello(2));
    assert_error(&rejected, 429, "resource_exhausted");
    assert!(rejected.header("retry-after").is_some());
    assert_eq!(a.join().unwrap().status, 200);
    assert_eq!(b.join().unwrap().status, 200);
}

#[test]
fn health_answers_while_a_generation_is_running() {
    let r = start(Opts {
        delay_ms: 80,
        ..Opts::default()
    });
    let addr = r.server.addr();
    let gen = std::thread::spawn(move || chat(addr, hello(5)));
    r.started.recv_timeout(Duration::from_secs(10)).unwrap();
    let t = Instant::now();
    let h = http(addr, "GET", "/health", None);
    assert_eq!(h.status, 200);
    assert!(t.elapsed() < Duration::from_millis(300), "health blocked for {:?}", t.elapsed());
    assert_eq!(gen.join().unwrap().status, 200);
}

#[test]
fn a_request_that_runs_too_long_times_out_and_is_cancelled() {
    let r = start(Opts {
        delay_ms: 300,
        timeout_secs: 1,
        ..Opts::default()
    });
    let t = Instant::now();
    let e = chat(r.server.addr(), hello(20));
    assert!(t.elapsed() < Duration::from_secs(4), "took {:?}", t.elapsed());
    assert_error(&e, 504, "cancelled");
    assert!(e.json()["error"]["message"].as_str().unwrap().contains("timeout"));
    // The abandoned generation stops instead of running all 20 steps.
    std::thread::sleep(Duration::from_millis(1500));
    assert!(r.calls.load(Ordering::SeqCst) < 15, "kept running: {}", r.calls.load(Ordering::SeqCst));
    assert_eq!(r.budget.used(), 0);
}

#[test]
fn logs_are_structured_and_never_contain_the_prompt() {
    let r = start(Opts::default());
    let secret = "quartz-zebra-secret-7731";
    let body = json!({"messages": [{"role": "user", "content": secret}], "max_tokens": 2, "temperature": 0});
    assert_eq!(chat(r.server.addr(), body).status, 200);
    http(r.server.addr(), "GET", "/health", None);
    let logs = r.logs.lock().unwrap().clone();
    assert!(logs.len() >= 2, "{logs:?}");
    for line in &logs {
        assert!(!line.contains(secret), "prompt leaked into log: {line}");
        let v: Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["event"], "request");
        assert!(v["status"].is_u64() && v["path"].is_string() && v["elapsed_ms"].is_number());
    }
    let chat_line: Value = serde_json::from_str(logs.iter().find(|l| l.contains("/v1/chat")).unwrap()).unwrap();
    assert!(chat_line["prompt_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn the_server_only_binds_loopback_by_default() {
    assert!(Config::default().bind.ip().is_loopback());
    let r = start(Opts::default());
    assert!(r.server.addr().ip().is_loopback());
}

#[test]
fn a_non_loopback_bind_is_refused() {
    let (model, tok) = lm();
    let engine = Engine {
        model: Box::new(model),
        tok: Arc::new(tok),
        budget: Budget::new(1 << 26),
        ctx: 64,
    };
    let cfg = Config {
        bind: "0.0.0.0:0".parse().unwrap(),
        ..Config::default()
    };
    let e = api::start(engine, &cfg, ApiOptions::from_config(&cfg, "m".into(), 64)).err().unwrap();
    assert!(e.to_string().contains("loopback"), "{e}");
}

#[test]
fn shutdown_finishes_the_running_request_then_stops_listening() {
    let r = start(Opts {
        delay_ms: 50,
        ..Opts::default()
    });
    let addr = r.server.addr();
    let gen = std::thread::spawn(move || chat(addr, hello(4)));
    r.started.recv_timeout(Duration::from_secs(10)).unwrap();
    r.server.shutdown();
    assert_eq!(gen.join().unwrap().status, 200, "in-flight request must complete");
    assert!(TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_err());
}

// ---- streaming ----

/// Reads the whole response, recording when each read returned.
fn read_timed(s: &mut TcpStream) -> (Vec<(Instant, usize)>, String) {
    let mut raw = Vec::new();
    let mut marks = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match s.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                marks.push((Instant::now(), raw.len()));
                if raw.ends_with(b"0\r\n\r\n") {
                    break; // the terminating chunk
                }
            }
        }
    }
    (marks, String::from_utf8_lossy(&raw).into_owned())
}

fn open_stream(addr: SocketAddr, max_tokens: u64) -> TcpStream {
    let body = json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": max_tokens, "temperature": 0, "stream": true}).to_string();
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    s
}

fn sse_events(text: &str) -> Vec<String> {
    let mut rest = text.split_once("\r\n\r\n").map(|x| x.1).unwrap_or("");
    let mut body = String::new();
    // De-chunk: "<hex size>\r\n<data>\r\n" ... "0\r\n\r\n".
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let n = usize::from_str_radix(size, 16).expect("chunk size");
        if n == 0 {
            break;
        }
        body.push_str(&tail[..n]);
        rest = &tail[n + 2..];
    }
    body.split("\n\n")
        .filter_map(|e| e.strip_prefix("data: ").map(str::to_string))
        .collect()
}

#[test]
fn stream_emits_incremental_chunks_then_done() {
    let r = start(Opts {
        delay_ms: 25,
        ..Opts::default()
    });
    let t0 = Instant::now();
    let mut s = open_stream(r.server.addr(), 6);
    let (marks, text) = read_timed(&mut s);
    assert!(text.starts_with("HTTP/1.1 200"), "{text}");
    assert!(text.to_lowercase().contains("content-type: text/event-stream"), "{text}");
    assert!(text.to_lowercase().contains("transfer-encoding: chunked"), "{text}");
    let events = sse_events(&text);
    assert_eq!(events.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = events[..events.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    let last = chunks.last().unwrap();
    assert!(["stop", "length"].contains(&last["choices"][0]["finish_reason"].as_str().unwrap()));
    let streamed: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    // Same text as the non-streaming answer for the same greedy request.
    let plain = chat(r.server.addr(), hello(6)).json();
    assert_eq!(streamed, plain["choices"][0]["message"]["content"].as_str().unwrap());
    // Incremental: data arrived in several reads, spread over the generation time.
    assert!(marks.len() >= 3, "only {} reads", marks.len());
    let first_read = marks[0].0.duration_since(t0);
    let end = marks.last().unwrap().0.duration_since(t0);
    assert!(end - first_read > Duration::from_millis(60), "all data arrived at once: {first_read:?}..{end:?}");
}

#[test]
fn stream_errors_before_generation_are_plain_json() {
    let r = start(Opts::default());
    let body = json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 500, "stream": true});
    assert_error(&chat(r.server.addr(), body), 400, "resource_exhausted");
}

#[test]
fn disconnect_cancels_and_frees_kv_and_the_next_request_works() {
    let r = start(Opts {
        delay_ms: 40,
        ..Opts::default()
    });
    let addr = r.server.addr();
    let mut s = open_stream(addr, 30); // prompt 30 + 30 fits the 64-token context
    // Read until the first content chunk, then vanish.
    let mut seen = Vec::new();
    let mut buf = [0u8; 1024];
    while !String::from_utf8_lossy(&seen).contains("\"content\"") {
        let n = s.read(&mut buf).unwrap();
        assert!(n > 0, "stream ended early");
        seen.extend_from_slice(&buf[..n]);
    }
    let calls_at_disconnect = r.calls.load(Ordering::SeqCst);
    let t = Instant::now();
    drop(s);
    while r.budget.used() != 0 {
        assert!(t.elapsed() < Duration::from_secs(3), "KV cache not freed after disconnect");
        std::thread::sleep(Duration::from_millis(10));
    }
    eprintln!("cancellation latency after disconnect: {:?}", t.elapsed());
    assert!(
        r.calls.load(Ordering::SeqCst) < calls_at_disconnect + 10,
        "generation kept running: {} -> {}",
        calls_at_disconnect,
        r.calls.load(Ordering::SeqCst)
    );
    assert_eq!(chat(addr, hello(4)).status, 200);
}
