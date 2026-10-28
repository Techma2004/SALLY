//! Benchmark matrix (spec section 7): load, short prompt (first and warm run), long prompt,
//! repeated generation (memory stability) and cancellation latency. One JSON object per
//! row. Every row records the machine facts needed to interpret it; nothing is estimated.

use crate::architecture::llama::LlamaModel;
use crate::architecture::Architecture;
use crate::cancel::CancelToken;
use crate::config::Config;
use crate::error::EngineError;
use crate::generate::{generate, GenParams, GenStats};
use crate::gguf::GgufFile;
use crate::memory::{open_cache, Budget};
use crate::model_registry;
use crate::telemetry::{peak_rss_bytes, rss_bytes};
use crate::tensor_store::TensorStore;
use crate::tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;

fn row(name: &str, cfg: &Config, ctx: usize, fields: Value) -> String {
    let mut v = json!({
        "bench": name,
        "threads": cfg.threads,
        "context": ctx,
        "rss_bytes": rss_bytes(),
        "peak_rss_bytes": peak_rss_bytes(),
    });
    if let (Some(dst), Some(src)) = (v.as_object_mut(), fields.as_object()) {
        for (k, val) in src {
            dst.insert(k.clone(), val.clone());
        }
    }
    v.to_string()
}

fn gen_fields(s: &GenStats) -> Value {
    json!({
        "prompt_tokens": s.prompt_tokens,
        "generated_tokens": s.generated_tokens(),
        "first_token_ms": s.first_token_secs.map(|t| t * 1000.0),
        "prefill_tok_s": s.prefill_tokens_per_sec(),
        "decode_tok_s": s.decode_tokens_per_sec(),
        "finish": s.finish.as_str(),
    })
}

/// Runs the matrix and passes each JSON row to `emit` as soon as it is measured.
pub fn run(path: &Path, cfg: &Config, emit: &mut dyn FnMut(String)) -> Result<(), EngineError> {
    let started = Instant::now();
    let file = GgufFile::open(path)?;
    let arch = file.architecture().unwrap_or("").to_string();
    model_registry::lookup(&arch)?;
    let tok = Tokenizer::from_gguf(&file.metadata)?;
    let store = TensorStore::open(&file, path)?;
    let mut model = LlamaModel::load(store, &file.metadata, cfg.threads.max(1))?;
    let ctx = cfg.context_len.min(model.context_len());
    let budget = Budget::new(cfg.memory_budget_mb.saturating_mul(1024 * 1024));
    let _scratch = budget.reserve("scratch buffers", model.scratch_bytes())?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    emit(row("load", cfg, ctx, json!({"architecture": arch, "load_ms": load_ms})));

    let params = |max_tokens: usize| GenParams {
        max_tokens,
        temperature: 0.0,
        ..GenParams::default()
    };
    let short = tok.encode_chat(
        &[ChatMessage {
            role: "user".into(),
            content: "Say hello in one short sentence.".into(),
        }],
        true,
    )?;
    let one = |model: &mut LlamaModel, prompt: &[u32], max: usize| -> Result<GenStats, EngineError> {
        let (mut cache, _kv) = open_cache(&*model, ctx, &budget)?;
        generate(model, &tok, &mut cache, prompt, &params(max), &CancelToken::new(), &mut |_| {})
    };
    let max_short = 32usize.min(ctx.saturating_sub(short.len()));
    if max_short == 0 {
        return Err(EngineError::ResourceExhausted(
            "the context is too small to benchmark even a short prompt".into(),
        ));
    }

    // The first run after load pays for paging weights in; the second shows steady state.
    let first = one(&mut model, &short, max_short)?;
    emit(row("short_prompt_first_run", cfg, ctx, gen_fields(&first)));
    let warm = one(&mut model, &short, max_short)?;
    emit(row("short_prompt_warm", cfg, ctx, gen_fields(&warm)));

    let target = (ctx / 2).max(1);
    let text = "the quick brown fox jumps over the lazy dog ".repeat(target);
    let mut long = tok.encode(&text, true);
    long.truncate(target);
    let max_long = 16usize.min(ctx.saturating_sub(long.len()));
    if max_long > 0 {
        emit(row("long_prompt", cfg, ctx, gen_fields(&one(&mut model, &long, max_long)?)));
    }

    // Memory stability: RSS after each of several identical generations.
    let mut rss = Vec::new();
    for _ in 0..5 {
        one(&mut model, &short, max_short)?;
        rss.push(rss_bytes());
    }
    let first_rss = rss.first().copied().flatten();
    let last_rss = rss.last().copied().flatten();
    let growth = match (first_rss, last_rss) {
        (Some(a), Some(b)) if a > 0 => Some((b as f64 - a as f64) / a as f64 * 100.0),
        _ => None,
    };
    emit(row("repeated_generation", cfg, ctx, json!({
        "runs": rss.len(), "rss_after_each": rss, "rss_growth_pct_first_to_last": growth,
    })));

    // Cancellation: cancel after the third piece of text; time until `generate` returns.
    let (mut cache, _kv) = open_cache(&model, ctx, &budget)?;
    let cancel = CancelToken::new();
    let mut pieces = 0usize;
    let mut cancelled_at: Option<Instant> = None;
    let stats = generate(&mut model, &tok, &mut cache, &short, &params(max_short), &cancel, &mut |_| {
        pieces += 1;
        if pieces == 3 && cancelled_at.is_none() {
            cancelled_at = Some(Instant::now());
            cancel.cancel();
        }
    })?;
    emit(row("cancellation", cfg, ctx, json!({
        "cancel_to_return_ms": cancelled_at.map(|t| t.elapsed().as_secs_f64() * 1000.0),
        "pieces_before_cancel": pieces,
        "finish": stats.finish.as_str(),
    })));
    Ok(())
}
