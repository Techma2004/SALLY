//! Loopback-only, OpenAI-compatible HTTP API (spec section 7).
//!
//! One handler thread per connection (capped); generation itself is serialised by the
//! scheduler's single worker, so handlers only parse, validate, wait and format.

pub mod types;

use crate::cancel::CancelToken;
use crate::config::Config;
use crate::error::EngineError;
use crate::generate::GenParams;
use crate::scheduler::{Engine, Event, Job, Scheduler};
use crate::tokenizer::Tokenizer;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server};
use types::{ApiError, Completion};

#[derive(Clone)]
pub struct ApiOptions {
    pub model_id: String,
    pub max_output_tokens: usize,
    pub max_body_bytes: usize,
    pub request_timeout: Duration,
    /// Total positions (prompt + output) a request may use.
    pub context_limit: usize,
    /// Simultaneous open requests (running, queued or being parsed); beyond this, 503.
    pub max_connections: usize,
    /// Receives one JSON line per request. Never contains prompt text.
    pub log: Arc<dyn Fn(&str) + Send + Sync>,
}

impl ApiOptions {
    pub fn from_config(cfg: &Config, model_id: String, context_limit: usize) -> ApiOptions {
        ApiOptions {
            model_id,
            max_output_tokens: cfg.max_output_tokens,
            max_body_bytes: cfg.max_body_bytes,
            request_timeout: Duration::from_secs(cfg.request_timeout_secs),
            context_limit,
            max_connections: 16,
            log: Arc::new(|line: &str| eprintln!("{line}")),
        }
    }
}

struct Shared {
    scheduler: Scheduler,
    tok: Arc<Tokenizer>,
    opts: ApiOptions,
    next_id: AtomicU64,
    in_flight: Mutex<usize>,
    idle: Condvar,
    open: AtomicUsize,
}

pub struct ServerHandle {
    addr: SocketAddr,
    server: Option<Arc<Server>>,
    shared: Arc<Shared>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    accept: Option<std::thread::JoinHandle<()>>,
}

impl ServerHandle {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stops accepting connections, lets in-flight requests finish, stops the worker.
    pub fn shutdown(mut self) {
        self.stop_all();
    }

    fn stop_all(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(s) = &self.server {
            s.unblock();
        }
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
        let mut n = self.shared.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        while *n > 0 {
            n = self.shared.idle.wait(n).unwrap_or_else(|e| e.into_inner());
        }
        drop(n);
        self.shared.scheduler.shutdown();
        // tiny_http closes its listening socket from its own thread once the server is
        // dropped; wait (bounded) so that "shut down" means "no longer listening".
        if self.server.take().is_some() {
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline
                && std::net::TcpStream::connect_timeout(&self.addr, Duration::from_millis(50)).is_ok()
            {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop_all();
    }
}

pub fn start(engine: Engine, cfg: &Config, opts: ApiOptions) -> Result<ServerHandle, EngineError> {
    if !cfg.bind.ip().is_loopback() {
        return Err(EngineError::InvalidRequest(format!(
            "refusing to bind {}: only loopback addresses are allowed (no authentication or TLS yet)",
            cfg.bind
        )));
    }
    let server = Server::http(cfg.bind)
        .map_err(|e| EngineError::Internal(format!("cannot listen on {}: {e}", cfg.bind)))?;
    let addr = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| EngineError::Internal("the listener has no IP address".into()))?;
    let server = Arc::new(server);
    let shared = Arc::new(Shared {
        tok: engine.tok.clone(),
        scheduler: Scheduler::start(engine, cfg.queue_len),
        opts,
        next_id: AtomicU64::new(1),
        in_flight: Mutex::new(0),
        idle: Condvar::new(),
        open: AtomicUsize::new(0),
    });
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let accept = {
        let (server, shared, stop) = (server.clone(), shared.clone(), stop.clone());
        std::thread::Builder::new()
            .name("sally-accept".into())
            .spawn(move || accept_loop(&server, &shared, &stop))
            .map_err(|e| EngineError::Internal(format!("cannot start the accept thread: {e}")))?
    };
    Ok(ServerHandle {
        addr,
        server: Some(server),
        shared,
        stop,
        accept: Some(accept),
    })
}

fn accept_loop(server: &Server, shared: &Arc<Shared>, stop: &std::sync::atomic::AtomicBool) {
    loop {
        let request = match server.recv() {
            Ok(r) => r,
            Err(_) if stop.load(Ordering::SeqCst) => return,
            Err(e) => {
                (shared.opts.log)(&format!(
                    "{{\"event\":\"accept_error\",\"error\":{}}}",
                    serde_json::Value::String(e.to_string())
                ));
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        if stop.load(Ordering::SeqCst) {
            return;
        }
        if shared.open.fetch_add(1, Ordering::SeqCst) >= shared.opts.max_connections {
            shared.open.fetch_sub(1, Ordering::SeqCst);
            let e = ApiError::busy("too many open requests").with_status(503);
            let _ = request.respond(error_response(&e));
            continue;
        }
        *shared.in_flight.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        let sh = shared.clone();
        let spawned = std::thread::Builder::new()
            .name("sally-request".into())
            .spawn(move || {
                handle(&sh, request);
                sh.open.fetch_sub(1, Ordering::SeqCst);
                let mut n = sh.in_flight.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                sh.idle.notify_all();
            });
        if spawned.is_err() {
            shared.open.fetch_sub(1, Ordering::SeqCst);
            *shared.in_flight.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        }
    }
}

type Reply = Response<std::io::Cursor<Vec<u8>>>;

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes())
        .unwrap_or_else(|_| Header::from_bytes(&b"X-Invalid"[..], &b"1"[..]).expect("static header"))
}

fn json_response(status: u16, body: String) -> Reply {
    Response::from_string(body)
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json"))
}

fn error_response(e: &ApiError) -> Reply {
    let mut r = json_response(e.status, e.body());
    for (k, v) in &e.headers {
        r = r.with_header(header(k, v));
    }
    r
}

/// A streaming response whose job is already queued; written by `stream_response`.
struct StreamPlan {
    rx: std::sync::mpsc::Receiver<Event>,
    cancel: CancelToken,
    id: String,
    created: u64,
    model: String,
    deadline: Instant,
    timeout_secs: u64,
}

enum Chat {
    Json(Reply, usize),
    Stream(StreamPlan),
}

struct Outcome {
    stream: Option<StreamPlan>,
    reply: Result<Reply, ApiError>,
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
}

impl Outcome {
    fn plain(reply: Result<Reply, ApiError>) -> Outcome {
        Outcome {
            stream: None,
            reply,
            prompt_tokens: None,
            completion_tokens: None,
        }
    }
}

fn handle(shared: &Shared, mut request: Request) {
    let started = Instant::now();
    let method = request.method().clone();
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();

    let mut out = route(shared, &mut request, &method, &path);
    if let Some(plan) = out.stream.take() {
        let (completion, outcome) = stream_response(request, plan);
        let line = serde_json::json!({
            "event": "request", "method": method.as_str(), "path": path, "status": 200,
            "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            "prompt_tokens": out.prompt_tokens, "completion_tokens": completion,
            "stream": outcome,
        });
        (shared.opts.log)(&line.to_string());
        return;
    }
    let (status, reply) = match out.reply {
        Ok(r) => (r.status_code().0, r),
        Err(e) => (e.status, error_response(&e)),
    };
    let mut line = serde_json::json!({
        "event": "request",
        "method": method.as_str(),
        "path": path,
        "status": status,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    if let Some(n) = out.prompt_tokens {
        line["prompt_tokens"] = n.into();
    }
    if let Some(n) = out.completion_tokens {
        line["completion_tokens"] = n.into();
    }
    // A client that has already gone away is not an error worth reporting.
    let _ = request.respond(reply);
    (shared.opts.log)(&line.to_string());
}

fn route(shared: &Shared, request: &mut Request, method: &Method, path: &str) -> Outcome {
    let allowed = match path {
        "/health" | "/v1/models" => "GET",
        "/v1/chat/completions" => "POST",
        _ => {
            return Outcome::plain(Err(ApiError::invalid(format!("no such route: {path}"))
                .with_status(404)))
        }
    };
    if method.as_str() != allowed {
        return Outcome::plain(Err(ApiError::invalid(format!(
            "method {} is not allowed on {path}",
            method.as_str()
        ))
        .with_status(405)
        .with_header("Allow", allowed)));
    }
    match path {
        "/health" => Outcome::plain(Ok(json_response(200, types::health_json()))),
        "/v1/models" => Outcome::plain(Ok(json_response(
            200,
            types::models_json(&shared.opts.model_id),
        ))),
        _ => chat(shared, request),
    }
}

fn read_body(shared: &Shared, request: &mut Request) -> Result<Vec<u8>, ApiError> {
    let max = shared.opts.max_body_bytes;
    let too_big = || {
        ApiError::invalid(format!("the request body exceeds the {max} byte limit")).with_status(413)
    };
    if request.body_length().is_some_and(|n| n > max) {
        return Err(too_big());
    }
    let mut body = Vec::new();
    request
        .as_reader()
        .take(max as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|e| ApiError::invalid(format!("could not read the request body: {e}")))?;
    if body.len() > max {
        return Err(too_big());
    }
    Ok(body)
}

fn chat(shared: &Shared, request: &mut Request) -> Outcome {
    let mut prompt_tokens = None;
    let result = chat_inner(shared, request, &mut prompt_tokens);
    match result {
        Ok(Chat::Json(reply, completion)) => Outcome {
            stream: None,
            reply: Ok(reply),
            prompt_tokens,
            completion_tokens: Some(completion),
        },
        Ok(Chat::Stream(plan)) => Outcome {
            stream: Some(plan),
            reply: Err(ApiError::invalid("internal: stream placeholder")),
            prompt_tokens,
            completion_tokens: None,
        },
        Err(e) => Outcome {
            stream: None,
            reply: Err(e),
            prompt_tokens,
            completion_tokens: None,
        },
    }
}

fn chat_inner(
    shared: &Shared,
    request: &mut Request,
    prompt_tokens: &mut Option<usize>,
) -> Result<Chat, ApiError> {
    let opts = &shared.opts;
    let body = read_body(shared, request)?;
    let req = types::parse_chat_request(&body)?;

    if let Some(m) = &req.model {
        if m != &opts.model_id {
            return Err(ApiError::invalid(format!(
                "model {m:?} is not loaded; this server serves {:?}",
                opts.model_id
            ))
            .with_status(404));
        }
    }
    if let Some(n) = req.max_tokens {
        if n > opts.max_output_tokens {
            return Err(ApiError::too_large(format!(
                "max_tokens {n} exceeds the server limit of {}",
                opts.max_output_tokens
            )));
        }
    }

    let prompt = shared
        .tok
        .encode_chat(&req.messages, true)
        .map_err(|e| ApiError::from_engine(&e))?;
    *prompt_tokens = Some(prompt.len());
    let room = opts.context_limit.saturating_sub(prompt.len());
    let max_tokens = req.max_tokens.unwrap_or(opts.max_output_tokens.min(room));
    if max_tokens == 0 || prompt.len() + max_tokens > opts.context_limit {
        return Err(ApiError::too_large(format!(
            "the prompt ({} tokens) plus max_tokens ({max_tokens}) exceeds the context limit of {} tokens",
            prompt.len(),
            opts.context_limit
        )));
    }

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let params = GenParams {
        max_tokens,
        temperature: req.temperature.unwrap_or(types::DEFAULT_TEMPERATURE),
        top_k: types::DEFAULT_TOP_K,
        top_p: req.top_p.unwrap_or(types::DEFAULT_TOP_P),
        seed: req.seed.unwrap_or(now.as_nanos() as u64),
    };
    let cancel = CancelToken::new();
    let (tx, rx) = channel();
    shared
        .scheduler
        .submit(Job {
            prompt: prompt.clone(),
            params,
            cancel: cancel.clone(),
            events: tx,
        })
        .map_err(|e| match e {
            EngineError::ResourceExhausted(m) => ApiError::busy(m),
            other => ApiError::from_engine(&other),
        })?;

    let deadline = Instant::now() + opts.request_timeout;
    if req.stream {
        let id = format!("chatcmpl-{}", shared.next_id.fetch_add(1, Ordering::SeqCst));
        return Ok(Chat::Stream(StreamPlan {
            rx,
            cancel,
            id,
            created: now.as_secs(),
            model: opts.model_id.clone(),
            deadline,
            timeout_secs: opts.request_timeout.as_secs(),
        }));
    }
    let mut text = String::new();
    let stats = loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok(Event::Text(t)) => text.push_str(&t),
            Ok(Event::Done(s)) => break s,
            Ok(Event::Failed(e)) => return Err(ApiError::from_engine(&e)),
            Err(RecvTimeoutError::Timeout) => {
                cancel.cancel();
                return Err(ApiError::timeout(opts.request_timeout.as_secs()));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(ApiError::from_engine(&EngineError::Internal(
                    "the worker dropped the request".into(),
                )))
            }
        }
    };

    let id = format!("chatcmpl-{}", shared.next_id.fetch_add(1, Ordering::SeqCst));
    let completion = stats.generated_tokens();
    let body = types::completion_json(&Completion {
        id: &id,
        created: now.as_secs(),
        model: &opts.model_id,
        content: &text,
        finish_reason: stats.finish.as_str(),
        prompt_tokens: prompt.len(),
        completion_tokens: completion,
    });
    let mut reply = json_response(200, body);
    if !req.ignored.is_empty() {
        let mut names = req.ignored.clone();
        names.sort();
        reply = reply.with_header(header("X-Sally-Ignored-Fields", &names.join(",")));
    }
    Ok(Chat::Json(reply, completion))
}

/// Writes the SSE stream with chunked transfer encoding (so the connection stays valid for
/// tiny_http's keep-alive handling); each event is its own chunk, flushed at once.
/// A write error means the client is gone: the job is cancelled so its KV cache is freed.
/// Returns (completion tokens, "done" | "client_disconnected" | "timeout" | "error").
fn stream_response(request: Request, plan: StreamPlan) -> (usize, &'static str) {
    let mut w = request.into_writer();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n";
    let send = |w: &mut dyn Write, payload: &str| -> std::io::Result<()> {
        let event = format!("data: {payload}\n\n");
        w.write_all(format!("{:x}\r\n{event}\r\n", event.len()).as_bytes())?;
        w.flush()
    };
    let chunk = |delta: serde_json::Value, finish: Option<&str>| {
        types::chunk_json(&plan.id, plan.created, &plan.model, delta, finish)
    };
    let first = w
        .write_all(head.as_bytes())
        .and_then(|_| send(&mut *w, &chunk(serde_json::json!({"role": "assistant"}), None)));
    if first.is_err() {
        plan.cancel.cancel();
        return (0, "client_disconnected");
    }
    let mut completion = 0usize;
    loop {
        let left = plan.deadline.saturating_duration_since(Instant::now());
        let (payload, last, outcome) = match plan.rx.recv_timeout(left) {
            Ok(Event::Text(t)) => {
                completion += 1;
                (chunk(serde_json::json!({"content": t}), None), false, "done")
            }
            Ok(Event::Done(s)) => {
                completion = s.generated_tokens();
                (chunk(serde_json::json!({}), Some(s.finish.as_str())), true, "done")
            }
            Ok(Event::Failed(e)) => (ApiError::from_engine(&e).body(), true, "error"),
            Err(RecvTimeoutError::Timeout) => {
                plan.cancel.cancel();
                (ApiError::timeout(plan.timeout_secs).body(), true, "timeout")
            }
            Err(RecvTimeoutError::Disconnected) => (
                ApiError::from_engine(&EngineError::Internal("the worker dropped the request".into())).body(),
                true,
                "error",
            ),
        };
        if send(&mut *w, &payload).is_err() {
            plan.cancel.cancel();
            return (completion, "client_disconnected");
        }
        if last {
            let _ = send(&mut *w, "[DONE]");
            let _ = w.write_all(b"0\r\n\r\n").and_then(|_| w.flush());
            return (completion, outcome);
        }
    }
}
