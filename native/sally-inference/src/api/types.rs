//! Request parsing, validation and response bodies for the OpenAI-compatible subset.
//!
//! Parameters that would change the output if honoured (tools, stop sequences, response
//! formats, streaming, penalties, n > 1) are refused with a clear error; they are never
//! silently dropped. Only fields with no bearing on the output are tolerated, and those
//! are reported back in the `x-sally-ignored-fields` response header.

use crate::error::EngineError;
use crate::tokenizer::ChatMessage;
use serde_json::{json, Map, Value};

pub const DEFAULT_TEMPERATURE: f32 = 0.8;
pub const DEFAULT_TOP_P: f32 = 0.95;
pub const DEFAULT_TOP_K: usize = 40;

/// Keys the server understands (accepted, or refused when they carry a non-neutral value).
const KNOWN: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "seed",
    "n",
    "stream",
    "logprobs",
    "top_logprobs",
    "frequency_penalty",
    "presence_penalty",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "logit_bias",
    "stop",
];

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub code: &'static str,
    pub etype: &'static str,
    pub message: String,
    pub headers: Vec<(&'static str, String)>,
}

impl ApiError {
    fn new(status: u16, code: &'static str, etype: &'static str, message: String) -> ApiError {
        ApiError {
            status,
            code,
            etype,
            message,
            headers: Vec::new(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> ApiError {
        ApiError::new(400, "invalid_request", "invalid_request_error", message.into())
    }

    pub fn with_status(mut self, status: u16) -> ApiError {
        self.status = status;
        self
    }

    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> ApiError {
        self.headers.push((name, value.into()));
        self
    }

    /// The request is too large for the configured limits. Not retryable as sent, so it is
    /// a 400 and not a 429 (OpenAI clients retry 429s automatically).
    pub fn too_large(message: impl Into<String>) -> ApiError {
        ApiError::new(400, "resource_exhausted", "invalid_request_error", message.into())
    }

    /// The server is busy; the same request may succeed shortly.
    pub fn busy(message: impl Into<String>) -> ApiError {
        ApiError::new(429, "resource_exhausted", "rate_limit_error", message.into())
            .with_header("Retry-After", "1")
    }

    pub fn timeout(secs: u64) -> ApiError {
        ApiError::new(
            504,
            "cancelled",
            "server_error",
            format!("generation exceeded the {secs} s request timeout and was cancelled"),
        )
    }

    pub fn from_engine(e: &EngineError) -> ApiError {
        let msg = e.to_string();
        match e {
            EngineError::InvalidRequest(_) => ApiError::invalid(msg),
            EngineError::ResourceExhausted(_) => ApiError::too_large(msg),
            EngineError::Cancelled => ApiError::new(503, "cancelled", "server_error", msg),
            EngineError::InvalidModel(_) => ApiError::new(500, "invalid_model", "server_error", msg),
            EngineError::UnsupportedArchitecture(_) => {
                ApiError::new(500, "unsupported_architecture", "server_error", msg)
            }
            EngineError::Internal(_) => ApiError::new(500, "internal", "server_error", msg),
        }
    }

    pub fn body(&self) -> String {
        json!({"error": {"message": self.message, "type": self.etype, "code": self.code}})
            .to_string()
    }
}

#[derive(Debug)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub stream: bool,
    /// Unknown fields that were tolerated, sorted.
    pub ignored: Vec<String>,
}

fn refuse(key: &str, why: &str) -> ApiError {
    ApiError::invalid(format!(
        "parameter {key:?} is not supported by this server ({why}); remove it from the request"
    ))
}

fn check_unsupported(obj: &Map<String, Value>) -> Result<(), ApiError> {
    for (k, v) in obj {
        if v.is_null() {
            continue;
        }
        match k.as_str() {
            "tools" | "tool_choice" | "functions" | "function_call" => {
                return Err(refuse(k, "tool calling is not implemented"))
            }
            "response_format" => {
                return Err(refuse(k, "constrained output is not available through the API yet"))
            }
            "logit_bias" | "top_logprobs" => return Err(refuse(k, "not implemented")),
            "stop" => return Err(refuse(k, "stop sequences are not implemented")),
            "n" if v.as_u64() != Some(1) => return Err(refuse(k, "only n = 1 is supported")),
            "stream" if v.as_bool().is_none() => {
                return Err(ApiError::invalid("\"stream\" must be a boolean"))
            }
            "logprobs" if v.as_bool() != Some(false) => return Err(refuse(k, "not implemented")),
            "frequency_penalty" | "presence_penalty" if v.as_f64() != Some(0.0) => {
                return Err(refuse(k, "penalties are not implemented"))
            }
            _ => {}
        }
    }
    Ok(())
}

fn content_text(content: &Value) -> Result<String, ApiError> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                let is_text = p.get("type").and_then(Value::as_str) == Some("text");
                match (is_text, p.get("text").and_then(Value::as_str)) {
                    (true, Some(t)) => out.push_str(t),
                    _ => {
                        return Err(ApiError::invalid(
                            "only text content parts are supported (no images or audio)",
                        ))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::invalid("message content must be a string or a list of text parts")),
    }
}

fn parse_messages(v: &Value) -> Result<Vec<ChatMessage>, ApiError> {
    let items = v
        .as_array()
        .ok_or_else(|| ApiError::invalid("\"messages\" must be an array"))?;
    if items.is_empty() {
        return Err(ApiError::invalid("\"messages\" must not be empty"));
    }
    items
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::invalid(format!("messages[{i}] needs a string \"role\"")))?;
            if !["system", "user", "assistant"].contains(&role) {
                return Err(ApiError::invalid(format!(
                    "messages[{i}]: role {role:?} is not supported (use system, user or assistant)"
                )));
            }
            let content = m
                .get("content")
                .ok_or_else(|| ApiError::invalid(format!("messages[{i}] needs \"content\"")))?;
            Ok(ChatMessage {
                role: role.to_string(),
                content: content_text(content).map_err(|e| {
                    ApiError::invalid(format!("messages[{i}]: {}", e.message))
                })?,
            })
        })
        .collect()
}

fn number(obj: &Map<String, Value>, key: &str, lo: f64, hi: f64, lo_open: bool) -> Result<Option<f32>, ApiError> {
    let Some(v) = obj.get(key).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let n = v
        .as_f64()
        .filter(|n| n.is_finite())
        .ok_or_else(|| ApiError::invalid(format!("{key:?} must be a number")))?;
    let ok = n <= hi && if lo_open { n > lo } else { n >= lo };
    if !ok {
        let bracket = if lo_open { '(' } else { '[' };
        return Err(ApiError::invalid(format!(
            "{key:?} must be in {bracket}{lo}, {hi}]"
        )));
    }
    Ok(Some(n as f32))
}

pub fn parse_chat_request(body: &[u8]) -> Result<ChatRequest, ApiError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid(format!("the request body is not valid JSON: {e}")))?;
    let obj = value
        .as_object()
        .ok_or_else(|| ApiError::invalid("the request body must be a JSON object"))?;
    check_unsupported(obj)?;

    let messages = parse_messages(
        obj.get("messages")
            .ok_or_else(|| ApiError::invalid("\"messages\" is required"))?,
    )?;

    let max_tokens = match ["max_tokens", "max_completion_tokens"]
        .iter()
        .find_map(|k| obj.get(*k).filter(|v| !v.is_null()).map(|v| (*k, v)))
    {
        None => None,
        Some((k, v)) => {
            let n = v
                .as_u64()
                .filter(|n| *n >= 1)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| ApiError::invalid(format!("{k:?} must be a positive integer")))?;
            Some(n)
        }
    };
    let seed = match obj.get("seed").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(
            v.as_u64()
                .ok_or_else(|| ApiError::invalid("\"seed\" must be a non-negative integer"))?,
        ),
    };
    let model = match obj.get("model").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(
            v.as_str()
                .ok_or_else(|| ApiError::invalid("\"model\" must be a string"))?
                .to_string(),
        ),
    };

    let ignored = obj
        .keys()
        .filter(|k| !KNOWN.contains(&k.as_str()))
        .cloned()
        .collect();

    Ok(ChatRequest {
        model,
        messages,
        max_tokens,
        temperature: number(obj, "temperature", 0.0, 2.0, false)?,
        top_p: number(obj, "top_p", 0.0, 1.0, true)?,
        seed,
        stream: obj.get("stream").and_then(Value::as_bool).unwrap_or(false),
        ignored,
    })
}

pub fn health_json() -> String {
    json!({"status": "ok", "model_loaded": true}).to_string()
}

pub fn models_json(id: &str) -> String {
    json!({
        "object": "list",
        "data": [{"id": id, "object": "model", "created": 0, "owned_by": "sally"}]
    })
    .to_string()
}

pub struct Completion<'a> {
    pub id: &'a str,
    pub created: u64,
    pub model: &'a str,
    pub content: &'a str,
    pub finish_reason: &'a str,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

pub fn completion_json(c: &Completion<'_>) -> String {
    json!({
        "id": c.id,
        "object": "chat.completion",
        "created": c.created,
        "model": c.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": c.content},
            "finish_reason": c.finish_reason
        }],
        "usage": {
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "total_tokens": c.prompt_tokens + c.completion_tokens
        }
    })
    .to_string()
}

/// One `chat.completion.chunk` SSE payload. `delta` is the JSON delta object.
pub fn chunk_json(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
    })
    .to_string()
}
