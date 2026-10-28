#[path = "common/gguf_builder.rs"]
mod gguf_builder;

use gguf_builder::GgufBuilder;
use sally_inference::gguf::GgmlType;
use std::path::PathBuf;
use std::process::Command;

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sally-inspect-{}-{tag}.gguf", std::process::id()))
}

fn run(path: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_sally-inference"))
        .arg("inspect")
        .arg(path)
        .output()
        .unwrap()
}

#[test]
fn inspect_prints_report_for_valid_file() {
    let bytes = GgufBuilder::new()
        .kv_str("general.architecture", "llama")
        .kv_u32("llama.context_length", 4096)
        .kv_u32("llama.block_count", 16)
        .kv_u32("llama.embedding_length", 2048)
        .kv_str("tokenizer.ggml.model", "gpt2")
        .tensor("a", &[4, 2], GgmlType::F32, vec![0; 32])
        .tensor("b", &[64], GgmlType::Q4_0, vec![0; 36])
        .tensor("c", &[256], GgmlType::Q4_K, vec![0; 144])
        .build();
    let path = temp_path("ok");
    std::fs::write(&path, bytes).unwrap();
    let out = run(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8(out.stdout).unwrap();
    for needle in [
        "architecture: \"llama\"",
        "version: 3",
        "tensors: 3",
        "total tensor bytes: 212",
        "F32: 1 tensors, 32 bytes",
        "Q4_0: 1 tensors, 36 bytes",
        "Q4_K: 1 tensors, 144 bytes",
        "llama.context_length: 4096",
        "llama.block_count: 16",
        "llama.embedding_length: 2048",
        "tokenizer.ggml.model: \"gpt2\"",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
}

#[test]
fn inspect_exits_2_on_corrupt_file() {
    let path = temp_path("corrupt");
    std::fs::write(&path, b"GGUFthis is not a model").unwrap();
    let out = run(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("invalid_model"));
}

#[test]
fn inspect_exits_1_on_missing_file() {
    let out = run(&temp_path("does-not-exist"));
    assert_eq!(out.status.code(), Some(1));
}
