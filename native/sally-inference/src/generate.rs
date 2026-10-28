//! The generation loop: prefill, sample, stream text, stop on EOS / length / cancel.
//!
//! Requests that cannot fit in the context are refused up front with
//! `resource_exhausted`; nothing is truncated silently.

use crate::architecture::{Architecture, KvCache};
use crate::cancel::CancelToken;
use crate::error::EngineError;
use crate::sampling::Sampler;
use crate::tokenizer::{Tokenizer, Utf8Stream};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for GenParams {
    fn default() -> GenParams {
        GenParams {
            max_tokens: 128,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model produced an end-of-sequence token.
    Stop,
    /// `max_tokens` was reached.
    Length,
    Cancelled,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// Generated token ids, excluding the end-of-sequence token.
    pub tokens: Vec<u32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    /// Seconds from the start of the request (prefill included) to the first sampled token.
    pub first_token_secs: Option<f64>,
    pub finish: FinishReason,
}

impl GenStats {
    pub fn generated_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn prefill_tokens_per_sec(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_secs)
    }

    /// Decode speed over the tokens after the first (the first token's latency is
    /// dominated by prefill). `0.0` when fewer than two tokens were generated.
    pub fn decode_tokens_per_sec(&self) -> f64 {
        rate(self.tokens.len().saturating_sub(1), self.decode_secs)
    }
}

fn rate(n: usize, secs: f64) -> f64 {
    if n == 0 || secs <= 0.0 {
        0.0
    } else {
        n as f64 / secs
    }
}

/// Feeds `prompt` after whatever is already in `cache` (so a warm cache can be resumed),
/// then samples until EOS, `max_tokens` or cancellation. `on_text` receives each piece of
/// completed UTF-8 text as soon as it is known; it is never called with an empty string.
///
/// If `cancel` fires, generation stops before the next forward step and the cache keeps
/// exactly the positions that were fully processed.
pub fn generate(
    model: &mut dyn Architecture,
    tok: &Tokenizer,
    cache: &mut KvCache,
    prompt: &[u32],
    params: &GenParams,
    cancel: &CancelToken,
    on_text: &mut dyn FnMut(&str),
) -> Result<GenStats, EngineError> {
    if prompt.is_empty() {
        return Err(EngineError::InvalidRequest("the prompt is empty".into()));
    }
    if params.max_tokens == 0 {
        return Err(EngineError::InvalidRequest(
            "max_tokens must be at least 1".into(),
        ));
    }
    if tok.vocab_size() > model.vocab_size() {
        return Err(EngineError::InvalidModel(format!(
            "tokenizer has {} tokens but the model only {}",
            tok.vocab_size(),
            model.vocab_size()
        )));
    }
    let limit = cache.capacity().min(model.context_len());
    let need = cache
        .len()
        .checked_add(prompt.len())
        .and_then(|n| n.checked_add(params.max_tokens));
    match need {
        Some(n) if n <= limit => {}
        _ => {
            return Err(EngineError::ResourceExhausted(format!(
            "{} cached + {} prompt + {} output tokens does not fit in the {limit}-token context",
            cache.len(),
            prompt.len(),
            params.max_tokens
        )))
        }
    }

    let mut stats = GenStats {
        prompt_tokens: prompt.len(),
        tokens: Vec::new(),
        prefill_secs: 0.0,
        decode_secs: 0.0,
        first_token_secs: None,
        finish: FinishReason::Length,
    };
    let mut logits: Vec<f32> = Vec::new();
    let mut pos = cache.len();

    let started = Instant::now();
    for (i, &t) in prompt.iter().enumerate() {
        if cancel.is_cancelled() {
            stats.finish = FinishReason::Cancelled;
            stats.prefill_secs = started.elapsed().as_secs_f64();
            return Ok(stats);
        }
        let out = model.forward(t, pos, cache)?;
        pos += 1;
        if i + 1 == prompt.len() {
            logits.clear();
            logits.extend_from_slice(out);
        }
    }
    stats.prefill_secs = started.elapsed().as_secs_f64();

    let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p, params.seed);
    let mut utf8 = Utf8Stream::new();
    let mut first_done: Option<Instant> = None;
    loop {
        let next = sampler.sample(&mut logits);
        if tok.is_eos(next) {
            stats.finish = FinishReason::Stop;
            break;
        }
        stats.tokens.push(next);
        first_done.get_or_insert_with(Instant::now);
        stats
            .first_token_secs
            .get_or_insert_with(|| started.elapsed().as_secs_f64());
        let text = utf8.push(&tok.decode_piece(next));
        if !text.is_empty() {
            on_text(&text);
        }
        if stats.tokens.len() >= params.max_tokens {
            stats.finish = FinishReason::Length;
            break;
        }
        if cancel.is_cancelled() {
            stats.finish = FinishReason::Cancelled;
            break;
        }
        let out = model.forward(next, pos, cache)?;
        pos += 1;
        logits.clear();
        logits.extend_from_slice(out);
    }
    let tail = utf8.flush();
    if !tail.is_empty() {
        on_text(&tail);
    }
    // Time spent after the first token was sampled, i.e. pure decode steps.
    stats.decode_secs = first_done.map_or(0.0, |t| t.elapsed().as_secs_f64());
    Ok(stats)
}
