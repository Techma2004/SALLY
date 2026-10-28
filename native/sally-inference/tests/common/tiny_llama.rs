//! Shared fixture: a tiny synthetic Llama model (and tokenizer) built in memory.
#![allow(dead_code)]

use super::gguf_builder::GgufBuilder;
use sally_inference::architecture::llama::LlamaModel;
use sally_inference::architecture::Architecture;
use sally_inference::architecture::KvCache;
use sally_inference::cancel::CancelToken;
use sally_inference::error::EngineError;
use sally_inference::generate::{generate, GenParams, GenStats};
use sally_inference::gguf::{GgmlType, GgufFile};
use sally_inference::tensor_store::TensorStore;
use sally_inference::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

pub static COUNTER: AtomicU32 = AtomicU32::new(0);

pub struct TempFile(pub PathBuf);
impl TempFile {
    pub fn new(bytes: &[u8]) -> TempFile {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("sally-gen-{}-{n}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        TempFile(p)
    }
}
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---------------------------------------------------------------- tiny model fixture

#[derive(Clone)]
pub struct Cfg {
    pub layers: usize,
    pub embd: usize,
    pub heads: usize,
    pub kv: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub vocab: usize,
    pub ctx: usize,
    pub base: f32,
    pub eps: f32,
    pub rope_freqs: bool,
    pub tied: bool,
    pub q8: bool,
}

impl Cfg {
    pub fn base() -> Cfg {
        Cfg {
            layers: 2,
            embd: 32,
            heads: 4,
            kv: 2,
            head_dim: 8,
            ff: 64,
            vocab: 16,
            ctx: 64,
            base: 10000.0,
            eps: 1e-5,
            rope_freqs: false,
            tied: false,
            q8: false,
        }
    }
}

pub struct LayerW {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

pub struct Weights {
    pub tok_embd: Vec<f32>,
    pub out: Option<Vec<f32>>,
    pub out_norm: Vec<f32>,
    pub layers: Vec<LayerW>,
    pub freqs: Option<Vec<f32>>,
}

/// Q8_0 scale used by the fixture: 2^-8, exactly representable as f16 (0x1C00).
pub const Q8_D: f32 = 1.0 / 256.0;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((r >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

pub fn mat(rng: &mut Rng, n: usize, q8: bool) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let v = rng.next() * 0.8;
            if q8 {
                (v / Q8_D).round().clamp(-127.0, 127.0) * Q8_D
            } else {
                v
            }
        })
        .collect()
}

pub fn norm(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| 1.0 + rng.next() * 0.4).collect()
}

pub fn weights(cfg: &Cfg, seed: u64) -> Weights {
    let mut r = Rng(seed);
    let q_dim = cfg.heads * cfg.head_dim;
    let kv_dim = cfg.kv * cfg.head_dim;
    let q8 = cfg.q8;
    let layers = (0..cfg.layers)
        .map(|_| LayerW {
            attn_norm: norm(&mut r, cfg.embd),
            ffn_norm: norm(&mut r, cfg.embd),
            wq: mat(&mut r, q_dim * cfg.embd, q8),
            wk: mat(&mut r, kv_dim * cfg.embd, q8),
            wv: mat(&mut r, kv_dim * cfg.embd, q8),
            wo: mat(&mut r, cfg.embd * q_dim, q8),
            gate: mat(&mut r, cfg.ff * cfg.embd, q8),
            up: mat(&mut r, cfg.ff * cfg.embd, q8),
            down: mat(&mut r, cfg.embd * cfg.ff, q8),
        })
        .collect();
    Weights {
        tok_embd: mat(&mut r, cfg.vocab * cfg.embd, q8),
        out: (!cfg.tied).then(|| mat(&mut r, cfg.vocab * cfg.embd, q8)),
        out_norm: norm(&mut r, cfg.embd),
        layers,
        freqs: cfg.rope_freqs.then(|| vec![1.0, 1.0, 2.0, 8.0]),
    }
}

pub struct T {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub bytes: Vec<u8>,
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn q8_bytes(v: &[f32]) -> Vec<u8> {
    assert_eq!(v.len() % 32, 0);
    let mut out = Vec::new();
    for block in v.chunks(32) {
        out.extend_from_slice(&0x1C00u16.to_le_bytes());
        for x in block {
            out.push(((x / Q8_D).round() as i32).clamp(-127, 127) as i8 as u8);
        }
    }
    out
}

pub fn tensor_list(cfg: &Cfg, w: &Weights) -> Vec<T> {
    let mut ts = Vec::new();
    let mut lin = |name: String, rows: usize, cols: usize, v: &[f32]| {
        let (ty, bytes) = if cfg.q8 {
            (GgmlType::Q8_0, q8_bytes(v))
        } else {
            (GgmlType::F32, f32_bytes(v))
        };
        ts.push(T {
            name,
            dims: vec![cols as u64, rows as u64],
            ty,
            bytes,
        });
    };
    lin("token_embd.weight".into(), cfg.vocab, cfg.embd, &w.tok_embd);
    if let Some(o) = &w.out {
        lin("output.weight".into(), cfg.vocab, cfg.embd, o);
    }
    let q_dim = cfg.heads * cfg.head_dim;
    let kv_dim = cfg.kv * cfg.head_dim;
    for (l, lw) in w.layers.iter().enumerate() {
        lin(format!("blk.{l}.attn_q.weight"), q_dim, cfg.embd, &lw.wq);
        lin(format!("blk.{l}.attn_k.weight"), kv_dim, cfg.embd, &lw.wk);
        lin(format!("blk.{l}.attn_v.weight"), kv_dim, cfg.embd, &lw.wv);
        lin(
            format!("blk.{l}.attn_output.weight"),
            cfg.embd,
            q_dim,
            &lw.wo,
        );
        lin(
            format!("blk.{l}.ffn_gate.weight"),
            cfg.ff,
            cfg.embd,
            &lw.gate,
        );
        lin(format!("blk.{l}.ffn_up.weight"), cfg.ff, cfg.embd, &lw.up);
        lin(
            format!("blk.{l}.ffn_down.weight"),
            cfg.embd,
            cfg.ff,
            &lw.down,
        );
    }
    let mut vecs = |name: String, v: &[f32]| {
        ts.push(T {
            name,
            dims: vec![v.len() as u64],
            ty: GgmlType::F32,
            bytes: f32_bytes(v),
        });
    };
    vecs("output_norm.weight".into(), &w.out_norm);
    for (l, lw) in w.layers.iter().enumerate() {
        vecs(format!("blk.{l}.attn_norm.weight"), &lw.attn_norm);
        vecs(format!("blk.{l}.ffn_norm.weight"), &lw.ffn_norm);
    }
    if let Some(f) = &w.freqs {
        vecs("rope_freqs.weight".into(), f);
    }
    ts
}

pub fn meta_builder(cfg: &Cfg, skip: &[&str]) -> GgufBuilder {
    let mut b = GgufBuilder::new().kv_str("general.architecture", "llama");
    let uints: [(&str, usize); 6] = [
        ("llama.block_count", cfg.layers),
        ("llama.embedding_length", cfg.embd),
        ("llama.feed_forward_length", cfg.ff),
        ("llama.attention.head_count", cfg.heads),
        ("llama.attention.head_count_kv", cfg.kv),
        ("llama.context_length", cfg.ctx),
    ];
    for (k, v) in uints {
        if !skip.contains(&k) {
            b = b.kv_u32(k, v as u32);
        }
    }
    b = b.kv_f32("llama.rope.freq_base", cfg.base);
    if !skip.contains(&"llama.attention.layer_norm_rms_epsilon") {
        b = b.kv_f32("llama.attention.layer_norm_rms_epsilon", cfg.eps);
    }
    b
}

pub fn assemble(mut b: GgufBuilder, ts: &[T]) -> Vec<u8> {
    for t in ts {
        b = b.tensor(&t.name, &t.dims, t.ty, t.bytes.clone());
    }
    b.build()
}

pub fn model_bytes(cfg: &Cfg, w: &Weights) -> Vec<u8> {
    assemble(meta_builder(cfg, &[]), &tensor_list(cfg, w))
}

pub fn load(bytes: &[u8], threads: usize) -> Result<LlamaModel, EngineError> {
    let tmp = TempFile::new(bytes);
    let file = GgufFile::open(&tmp.0)?;
    let store = TensorStore::open(&file, &tmp.0)?;
    LlamaModel::load(store, &file.metadata, threads)
}

pub fn code(r: Result<LlamaModel, EngineError>) -> (&'static str, String) {
    match r {
        Ok(_) => panic!("expected an error, but the model loaded"),
        Err(e) => (e.code(), e.to_string()),
    }
}

pub fn run(model: &mut LlamaModel, cache: &mut KvCache, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .enumerate()
        .map(|(pos, t)| model.forward(*t, pos, cache).unwrap().to_vec())
        .collect()
}

/// GPT-2 bytes_to_unicode, written independently of the crate.
pub fn byte_chars() -> Vec<char> {
    let printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut table = vec!['\0'; 256];
    for &b in &printable {
        table[b as usize] = char::from_u32(b).unwrap();
    }
    let mut n = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            table[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

// 0..=255 byte tokens, he(256) ll(257) hell(258) hello(259), then BOS(260) .. eot(264).
pub const LM_VOCAB: usize = 265;

pub fn with_tokenizer(b: GgufBuilder) -> GgufBuilder {
    let mut tokens: Vec<String> = byte_chars().iter().map(|c| c.to_string()).collect();
    let mut types = vec![1; 256];
    for t in ["he", "ll", "hell", "hello"] {
        tokens.push(t.into());
        types.push(1);
    }
    for t in [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ] {
        tokens.push(t.into());
        types.push(3);
    }
    assert_eq!(tokens.len(), LM_VOCAB);
    let merges: Vec<String> = ["h e", "l l", "he ll", "hell o"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    b.kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str_array("tokenizer.ggml.tokens", &tokens)
        .kv_str_array("tokenizer.ggml.merges", &merges)
        .kv_i32_array("tokenizer.ggml.token_type", &types)
        .kv_u32("tokenizer.ggml.bos_token_id", 260)
        .kv_u32("tokenizer.ggml.eos_token_id", 264)
}

pub fn lm_cfg() -> Cfg {
    let mut c = Cfg::base();
    c.vocab = LM_VOCAB;
    c
}

pub fn lm_bytes() -> Vec<u8> {
    let cfg = lm_cfg();
    assemble(
        with_tokenizer(meta_builder(&cfg, &[])),
        &tensor_list(&cfg, &weights(&cfg, 11)),
    )
}

pub fn lm() -> (LlamaModel, Tokenizer) {
    let bytes = lm_bytes();
    let tmp = TempFile::new(&bytes);
    let file = GgufFile::open(&tmp.0).unwrap();
    let store = TensorStore::open(&file, &tmp.0).unwrap();
    let tok = Tokenizer::from_gguf(&file.metadata).unwrap();
    (LlamaModel::load(store, &file.metadata, 2).unwrap(), tok)
}

pub fn gen(
    model: &mut dyn Architecture,
    tok: &Tokenizer,
    cache: &mut KvCache,
    prompt: &[u32],
    params: &GenParams,
    cancel: &CancelToken,
) -> Result<(GenStats, String), EngineError> {
    let mut text = String::new();
    let stats = generate(model, tok, cache, prompt, params, cancel, &mut |t| {
        assert!(!t.is_empty(), "on_text must never get an empty string");
        text.push_str(t);
    })?;
    Ok((stats, text))
}

pub fn greedy(max_tokens: usize) -> GenParams {
    GenParams {
        max_tokens,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 1,
    }
}

pub const PROMPT: [u32; 3] = [260, 104, 105];
