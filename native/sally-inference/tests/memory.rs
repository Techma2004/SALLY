//! Budget accounting, cache admission and telemetry tests (no model file needed).
use sally_inference::architecture::{Architecture, KvCache};
use sally_inference::error::EngineError;
use sally_inference::memory::{fmt_bytes, open_cache, Budget};
use sally_inference::telemetry::{mem_available_bytes, peak_rss_bytes, rss_bytes, Metrics};

/// Stand-in model: 1024 bytes of KV cache per position (1 layer, 1 kv head, 128 dims).
struct Stub;

impl Architecture for Stub {
    fn forward(&mut self, _: u32, _: usize, _: &mut KvCache) -> Result<&[f32], EngineError> {
        Err(EngineError::Internal("stub".into()))
    }
    fn context_len(&self) -> usize {
        4096
    }
    fn kv_bytes(&self, ctx: usize) -> usize {
        ctx.saturating_mul(1024)
    }
    fn vocab_size(&self) -> usize {
        1
    }
    fn new_cache(&self, ctx: usize) -> Result<KvCache, EngineError> {
        KvCache::try_new(1, 1, 128, ctx)
    }
    fn scratch_bytes(&self) -> usize {
        0
    }
}

#[test]
fn reserve_within_limit_tracks_usage() {
    let b = Budget::new(1000);
    let r1 = b.reserve("a", 300).unwrap();
    let r2 = b.reserve("b", 700).unwrap();
    assert_eq!((b.used(), b.remaining(), b.limit()), (1000, 0, 1000));
    assert_eq!((r1.bytes(), r1.what()), (300, "a"));
    drop(r2);
    assert_eq!(b.used(), 300);
}

#[test]
fn reserve_over_limit_fails_with_resource_exhausted() {
    let b = Budget::new(2048);
    let _held = b.reserve("scratch buffers", 1024).unwrap();
    let e = b.reserve("KV cache", 2048).unwrap_err();
    assert_eq!(e.code(), "resource_exhausted");
    let msg = e.to_string();
    for needle in ["KV cache", "2.0 KiB", "1.0 KiB"] {
        assert!(msg.contains(needle), "missing {needle:?} in {msg:?}");
    }
}

#[test]
fn failed_reserve_changes_nothing() {
    let b = Budget::new(100);
    let _r = b.reserve("a", 60).unwrap();
    assert!(b.reserve("b", 41).is_err());
    assert_eq!(b.used(), 60);
    assert!(b.reserve("c", 40).is_ok());
}

#[test]
fn drop_releases_reservation() {
    let b = Budget::new(100);
    {
        let _r = b.reserve("a", 100).unwrap();
        assert!(b.reserve("b", 1).is_err());
    }
    assert_eq!(b.used(), 0);
    assert!(b.reserve("b", 100).is_ok());
}

#[test]
fn reserve_overflow_is_refused_not_wrapped() {
    let b = Budget::new(usize::MAX);
    let _r = b.reserve("a", usize::MAX - 1).unwrap();
    let e = b.reserve("b", usize::MAX).unwrap_err();
    assert_eq!(e.code(), "resource_exhausted");
    assert_eq!(b.used(), usize::MAX - 1);
}

#[test]
fn zero_byte_reservation_is_allowed() {
    let b = Budget::new(0);
    assert!(b.reserve("nothing", 0).is_ok());
    assert!(b.reserve("something", 1).is_err());
}

#[test]
fn budget_is_shared_across_threads_without_overcommit() {
    let b = Budget::new(800);
    let handles: Vec<_> = (0..16)
        .map(|_| {
            let b = b.clone();
            std::thread::spawn(move || b.reserve("t", 100).ok())
        })
        .collect();
    let held: Vec<_> = handles
        .into_iter()
        .filter_map(|h| h.join().unwrap())
        .collect();
    assert_eq!(held.len(), 8, "exactly limit/size reservations may succeed");
    assert_eq!(b.used(), 800);
    drop(held);
    assert_eq!(b.used(), 0);
}

#[test]
fn kv_cache_reservation_rejects_oversized_context() {
    let budget = Budget::new(1024 * 1024); // room for 1024 positions
    let e = open_cache(&Stub, 2048, &budget).unwrap_err();
    assert_eq!(e.code(), "resource_exhausted");
    let msg = e.to_string();
    for needle in [
        "KV cache",
        "2048",
        "2.0 MiB",
        "1.0 MiB",
        "at most 1024 tokens",
    ] {
        assert!(msg.contains(needle), "missing {needle:?} in {msg:?}");
    }
    assert_eq!(budget.used(), 0, "a refused cache must not hold any budget");
}

#[test]
fn open_cache_charges_exactly_the_allocation() {
    let budget = Budget::new(4 * 1024 * 1024);
    let (cache, reservation) = open_cache(&Stub, 1024, &budget).unwrap();
    assert_eq!(cache.bytes(), 1024 * 1024);
    assert_eq!(reservation.bytes(), cache.bytes());
    assert_eq!(budget.used(), cache.bytes());
    drop(reservation);
    assert_eq!(budget.used(), 0);
}

#[test]
fn open_cache_with_huge_context_is_refused_not_overflowed() {
    let budget = Budget::new(usize::MAX);
    let e = open_cache(&Stub, usize::MAX, &budget).unwrap_err();
    assert_eq!(e.code(), "resource_exhausted");
}

#[test]
fn fmt_bytes_is_human_readable() {
    assert_eq!(fmt_bytes(0), "0 B");
    assert_eq!(fmt_bytes(1023), "1023 B");
    assert_eq!(fmt_bytes(1024), "1.0 KiB");
    assert_eq!(fmt_bytes(1536), "1.5 KiB");
    assert_eq!(fmt_bytes(3 * 1024 * 1024 + 512 * 1024), "3.5 MiB");
    assert_eq!(fmt_bytes(2 * 1024 * 1024 * 1024), "2.0 GiB");
}

#[cfg(target_os = "linux")]
#[test]
fn rss_reader_returns_nonzero() {
    let rss = rss_bytes().expect("VmRSS should be readable on Linux");
    assert!(rss > 100 * 1024, "implausibly small RSS {rss}");
    let peak = peak_rss_bytes().expect("VmHWM should be readable on Linux");
    assert!(peak >= rss);
    assert!(mem_available_bytes().expect("MemAvailable") > 0);
}

#[test]
fn rss_tracks_a_real_allocation() {
    let Some(before) = rss_bytes() else { return };
    let mut big = vec![0u8; 64 * 1024 * 1024];
    for i in (0..big.len()).step_by(4096) {
        big[i] = 1; // touch every page so it becomes resident
    }
    // Without black_box an optimised build removes the whole allocation as dead code.
    std::hint::black_box(&mut big);
    let after = rss_bytes().unwrap();
    std::hint::black_box(&big);
    assert!(
        after >= before + 48 * 1024 * 1024,
        "RSS did not reflect a 64 MiB allocation: {before} -> {after}"
    );
    drop(big);
}

#[test]
fn metrics_log_is_single_line_json_without_prompt_text() {
    let m = Metrics {
        load_ms: 12.5,
        prompt_tokens: 7,
        generated_tokens: 3,
        first_token_ms: Some(40.0),
        prefill_tok_s: 100.0,
        decode_tok_s: 5.5,
        peak_rss_bytes: Some(123456),
        kv_bytes: 4096,
        finish: "stop".into(),
        error: None,
    };
    let line = m.log_json();
    assert!(!line.contains('\n'));
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["event"], "generation");
    assert_eq!(v["prompt_tokens"], 7);
    assert_eq!(v["generated_tokens"], 3);
    assert_eq!(v["first_token_ms"], 40.0);
    assert_eq!(v["peak_rss_bytes"], 123456);
    assert_eq!(v["finish"], "stop");
    assert!(v["error"].is_null());
    let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
    assert!(
        keys.iter()
            .all(|k| !k.contains("prompt_text") && *k != "text" && *k != "content"),
        "log must not carry request text: {keys:?}"
    );
}

#[test]
fn metrics_for_a_failed_request_carry_the_error_code() {
    let m = Metrics::failed(&EngineError::ResourceExhausted("x".into()), 3.0);
    let v: serde_json::Value = serde_json::from_str(&m.log_json()).unwrap();
    assert_eq!(v["error"], "resource_exhausted");
    assert_eq!(v["finish"], "error");
    assert_eq!(v["generated_tokens"], 0);
}
