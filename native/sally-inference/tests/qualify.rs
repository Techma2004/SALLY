#[path = "common/gguf_builder.rs"]
mod gguf_builder;

use gguf_builder::GgufBuilder;
use sally_inference::config::Config;
use sally_inference::gguf::GgmlType;
use sally_inference::qualify::qualify;
use std::path::PathBuf;
use std::process::Command;

fn write(tag: &str, bytes: Vec<u8>) -> PathBuf {
    let p = std::env::temp_dir().join(format!("sally-qualify-{}-{tag}.gguf", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn llama_meta(arch: &str) -> GgufBuilder {
    GgufBuilder::new()
        .kv_str("general.architecture", arch)
        .kv_u32(&format!("{arch}.context_length"), 4096)
        .kv_u32(&format!("{arch}.block_count"), 2)
        .kv_u32(&format!("{arch}.embedding_length"), 64)
        .kv_u32(&format!("{arch}.attention.head_count"), 4)
        .kv_u32(&format!("{arch}.attention.head_count_kv"), 2)
}

fn small(arch: &str) -> Vec<u8> {
    llama_meta(arch)
        .tensor("token_embd.weight", &[64, 8], GgmlType::F32, vec![0; 64 * 8 * 4])
        .build()
}

fn cfg(ctx: usize, mb: usize) -> Config {
    Config {
        context_len: ctx,
        memory_budget_mb: mb,
        ..Config::default()
    }
}

#[test]
fn unknown_architecture_reports_unsupported() {
    let p = write("unknown", small("mystery"));
    let r = qualify(&p, &cfg(128, 64)).unwrap();
    let _ = std::fs::remove_file(&p);
    assert!(!r.supported && !r.fits_in_budget);
    assert!(r.unsupported_ops.iter().any(|o| o.contains("mystery")), "{:?}", r.unsupported_ops);
    assert_eq!(r.kv_bytes_at_ctx, None);
    assert_eq!(r.exit_code(), 2);
    assert!(!r.verdict.to_lowercase().contains("fall back") && !r.verdict.to_lowercase().contains("use llama"));
}

#[test]
fn moe_expert_tensors_flagged_unsupported() {
    let bytes = llama_meta("llama")
        .tensor("token_embd.weight", &[64, 8], GgmlType::F32, vec![0; 64 * 8 * 4])
        .tensor("blk.0.ffn_gate_exps.weight", &[64, 8], GgmlType::F32, vec![0; 64 * 8 * 4])
        .build();
    let p = write("moe", bytes);
    let r = qualify(&p, &cfg(128, 64)).unwrap();
    let _ = std::fs::remove_file(&p);
    assert!(!r.supported);
    assert!(r.unsupported_ops.iter().any(|o| o.contains("mixture-of-experts")));
    assert_eq!(r.weight_bytes, 4096);
    assert_eq!(r.dense_bytes, Some(2048));
}

#[test]
fn mla_and_sparse_attention_are_named_not_approximated() {
    let bytes = llama_meta("llama")
        .tensor("blk.0.attn_kv_a_mqa.weight", &[8], GgmlType::F32, vec![0; 32])
        .tensor("blk.0.indexer.proj.weight", &[8], GgmlType::F32, vec![0; 32])
        .build();
    let p = write("mla", bytes);
    let r = qualify(&p, &cfg(128, 64)).unwrap();
    let _ = std::fs::remove_file(&p);
    let all = r.unsupported_ops.join(" | ");
    assert!(all.contains("latent attention") && all.contains("sparse attention"), "{all}");
}

#[test]
fn oversize_model_reports_does_not_fit() {
    let bytes = llama_meta("llama")
        .tensor("token_embd.weight", &[512 * 1024], GgmlType::F32, vec![0; 2 * 1024 * 1024])
        .build();
    let p = write("big", bytes);
    let r = qualify(&p, &cfg(128, 1)).unwrap(); // 1 MiB budget < 2 MiB of weights
    let _ = std::fs::remove_file(&p);
    assert!(r.supported, "{:?}", r.unsupported_ops);
    assert!(!r.fits_in_budget);
    assert_eq!(r.exit_code(), 1);
    assert!(r.verdict.contains("budget"));
}

#[test]
fn supported_small_model_passes() {
    let p = write("ok", small("llama"));
    let r = qualify(&p, &cfg(128, 64)).unwrap();
    assert!(r.supported && r.fits_in_budget, "{r:?}");
    // 2 * layers(2) * ctx(128) * kv_heads(2) * head_dim(16) * 4 bytes
    assert_eq!(r.kv_bytes_at_ctx, Some(2 * 2 * 128 * 2 * 16 * 4));
    assert_eq!(r.exit_code(), 0);
    assert_eq!(r.dense_bytes, None);
    let out = Command::new(env!("CARGO_BIN_EXE_sally-inference"))
        .args(["qualify", p.to_str().unwrap(), "--context", "128", "--memory-mb", "64"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("supported: true"));
    let u = write("unk2", small("mystery"));
    let out = Command::new(env!("CARGO_BIN_EXE_sally-inference")).args(["qualify", u.to_str().unwrap()]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(&u);
}

#[test]
fn context_is_capped_by_the_model_limit() {
    let p = write("cap", small("llama"));
    let r = qualify(&p, &cfg(1_000_000, 64)).unwrap();
    let _ = std::fs::remove_file(&p);
    assert_eq!(r.kv_bytes_at_ctx, Some(2 * 2 * 4096 * 2 * 16 * 4));
}
