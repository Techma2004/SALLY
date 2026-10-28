//! Repeated-request memory stability. This is deliberately the ONLY test in this binary:
//! RSS is process-wide, so any other test thread would add noise to the measurement.
#[path = "common/gguf_builder.rs"]
mod gguf_builder;
#[path = "common/tiny_llama.rs"]
mod tiny;

use sally_inference::architecture::Architecture;
use sally_inference::cancel::CancelToken;
use sally_inference::generate::{generate, GenParams};
use sally_inference::memory::{open_cache, Budget};
use sally_inference::telemetry::rss_bytes;
use tiny::{lm, PROMPT};

#[test]
fn twenty_repeated_requests_do_not_grow_memory() {
    let Some(_) = rss_bytes() else {
        eprintln!("skipped: RSS not readable on this platform");
        return;
    };
    let (mut model, tok) = lm();
    let budget = Budget::new(64 * 1024 * 1024);
    let params = GenParams {
        max_tokens: 20,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        seed: 3,
    };
    let mut once = |seed: u64| {
        let (mut cache, _reservation) = open_cache(&model, 64, &budget).unwrap();
        let p = GenParams {
            seed,
            ..params.clone()
        };
        let mut chars = 0;
        let stats = generate(
            &mut model,
            &tok,
            &mut cache,
            &PROMPT,
            &p,
            &CancelToken::new(),
            &mut |t| chars += t.len(),
        )
        .unwrap();
        assert!(stats.generated_tokens() > 0);
    };

    for seed in 0..5 {
        once(seed); // warm up allocator, thread stacks and page cache
    }
    let before = rss_bytes().unwrap();
    for seed in 5..25 {
        once(seed);
    }
    let after = rss_bytes().unwrap();
    assert_eq!(budget.used(), 0, "every request must give its budget back");

    let growth = after.saturating_sub(before);
    let allowed = before / 20; // 5%
    assert!(
        growth <= allowed,
        "RSS grew by {growth} bytes over 20 requests (before {before}, allowed {allowed})"
    );
    let _ = model.vocab_size();
}
