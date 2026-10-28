#[path = "common/gguf_builder.rs"]
mod gguf_builder;
#[path = "common/tiny_llama.rs"]
mod tiny;

use sally_inference::bench;
use sally_inference::config::Config;
use serde_json::Value;

#[test]
fn bench_emits_every_row_of_the_matrix_with_real_numbers() {
    let tmp = tiny::TempFile::new(&tiny::lm_bytes());
    let cfg = Config {
        context_len: 64,
        threads: 2,
        memory_budget_mb: 64,
        ..Config::default()
    };
    let mut rows = Vec::new();
    bench::run(&tmp.0, &cfg, &mut |l| rows.push(serde_json::from_str::<Value>(&l).unwrap())).unwrap();
    let names: Vec<&str> = rows.iter().map(|r| r["bench"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        ["load", "short_prompt_first_run", "short_prompt_warm", "long_prompt", "repeated_generation", "cancellation"]
    );
    assert!(rows[0]["load_ms"].as_f64().unwrap() > 0.0);
    for r in &rows[1..4] {
        assert!(r["generated_tokens"].as_u64().unwrap() >= 1, "{r}");
        assert!(r["prefill_tok_s"].as_f64().unwrap() > 0.0, "{r}");
        assert!(r["first_token_ms"].as_f64().unwrap() > 0.0, "{r}");
    }
    assert_eq!(rows[4]["runs"], 5);
    let c = &rows[5];
    assert_eq!(c["finish"], "cancelled");
    assert_eq!(c["pieces_before_cancel"], 3, "cancellation must stop on the next token: {c}");
    assert!(c["cancel_to_return_ms"].as_f64().unwrap() >= 0.0);
}
