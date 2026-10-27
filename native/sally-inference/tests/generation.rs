//! Model-level tests for the Llama adapter.
//!
//! Correctness is established against an independent naive f64 reference forward pass
//! written in this file (straight loops, no code shared with the engine). It is NOT yet
//! compared with llama.cpp on a real model; see `real_model_smoke`.
#[path = "common/gguf_builder.rs"]
mod gguf_builder;

use gguf_builder::GgufBuilder;
use sally_inference::architecture::llama::LlamaModel;
use sally_inference::architecture::{Architecture, KvCache};
use sally_inference::error::EngineError;
use sally_inference::gguf::{GgmlType, GgufFile};
use sally_inference::model_registry;
use sally_inference::tensor_store::TensorStore;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

struct TempFile(PathBuf);
impl TempFile {
    fn new(bytes: &[u8]) -> TempFile {
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
struct Cfg {
    layers: usize,
    embd: usize,
    heads: usize,
    kv: usize,
    head_dim: usize,
    ff: usize,
    vocab: usize,
    ctx: usize,
    base: f32,
    eps: f32,
    rope_freqs: bool,
    tied: bool,
    q8: bool,
}

impl Cfg {
    fn base() -> Cfg {
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

struct LayerW {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

struct Weights {
    tok_embd: Vec<f32>,
    out: Option<Vec<f32>>,
    out_norm: Vec<f32>,
    layers: Vec<LayerW>,
    freqs: Option<Vec<f32>>,
}

/// Q8_0 scale used by the fixture: 2^-8, exactly representable as f16 (0x1C00).
const Q8_D: f32 = 1.0 / 256.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((r >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn mat(rng: &mut Rng, n: usize, q8: bool) -> Vec<f32> {
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

fn norm(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| 1.0 + rng.next() * 0.4).collect()
}

fn weights(cfg: &Cfg, seed: u64) -> Weights {
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

struct T {
    name: String,
    dims: Vec<u64>,
    ty: GgmlType,
    bytes: Vec<u8>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn q8_bytes(v: &[f32]) -> Vec<u8> {
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

fn tensor_list(cfg: &Cfg, w: &Weights) -> Vec<T> {
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

fn meta_builder(cfg: &Cfg, skip: &[&str]) -> GgufBuilder {
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

fn assemble(mut b: GgufBuilder, ts: &[T]) -> Vec<u8> {
    for t in ts {
        b = b.tensor(&t.name, &t.dims, t.ty, t.bytes.clone());
    }
    b.build()
}

fn model_bytes(cfg: &Cfg, w: &Weights) -> Vec<u8> {
    assemble(meta_builder(cfg, &[]), &tensor_list(cfg, w))
}

fn load(bytes: &[u8], threads: usize) -> Result<LlamaModel, EngineError> {
    let tmp = TempFile::new(bytes);
    let file = GgufFile::open(&tmp.0)?;
    let store = TensorStore::open(&file, &tmp.0)?;
    LlamaModel::load(store, &file.metadata, threads)
}

fn code(r: Result<LlamaModel, EngineError>) -> (&'static str, String) {
    match r {
        Ok(_) => panic!("expected an error, but the model loaded"),
        Err(e) => (e.code(), e.to_string()),
    }
}

fn run(model: &mut LlamaModel, cache: &mut KvCache, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .enumerate()
        .map(|(pos, t)| model.forward(*t, pos, cache).unwrap().to_vec())
        .collect()
}

// ---------------------------------------------------------------- naive f64 reference

fn mv(w: &[f32], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(w.len(), rows * cols);
    (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] as f64 * x[c]).sum())
        .collect()
}

fn rms(x: &[f64], g: &[f32], eps: f32) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let s = 1.0 / (ms + eps as f64).sqrt();
    x.iter().zip(g).map(|(v, g)| v * s * *g as f64).collect()
}

fn ref_logits(cfg: &Cfg, w: &Weights, tokens: &[u32]) -> Vec<Vec<f64>> {
    let (hd, nh, nkv, e) = (cfg.head_dim, cfg.heads, cfg.kv, cfg.embd);
    let mut kc: Vec<Vec<Vec<f64>>> = vec![vec![]; cfg.layers];
    let mut vc: Vec<Vec<Vec<f64>>> = vec![vec![]; cfg.layers];
    let mut all = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let mut x: Vec<f64> = w.tok_embd[tok as usize * e..][..e]
            .iter()
            .map(|v| *v as f64)
            .collect();
        for l in 0..cfg.layers {
            let lw = &w.layers[l];
            let xb = rms(&x, &lw.attn_norm, cfg.eps);
            let mut q = mv(&lw.wq, nh * hd, e, &xb);
            let mut k = mv(&lw.wk, nkv * hd, e, &xb);
            let v = mv(&lw.wv, nkv * hd, e, &xb);
            for (vec, heads) in [(&mut q, nh), (&mut k, nkv)] {
                for h in 0..heads {
                    for i in 0..hd / 2 {
                        let ff = w.freqs.as_ref().map_or(1.0, |f| f[i] as f64);
                        let freq = (cfg.base as f64).powf(-2.0 * i as f64 / hd as f64) / ff;
                        let (s, c) = (pos as f64 * freq).sin_cos();
                        let (i0, i1) = (h * hd + 2 * i, h * hd + 2 * i + 1);
                        let (x0, x1) = (vec[i0], vec[i1]);
                        vec[i0] = x0 * c - x1 * s;
                        vec[i1] = x0 * s + x1 * c;
                    }
                }
            }
            kc[l].push(k);
            vc[l].push(v);
            let mut att = vec![0.0; nh * hd];
            for h in 0..nh {
                let g = h / (nh / nkv);
                let mut sc: Vec<f64> = (0..=pos)
                    .map(|t| {
                        (0..hd)
                            .map(|d| q[h * hd + d] * kc[l][t][g * hd + d])
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut z = 0.0;
                for s in sc.iter_mut() {
                    *s = (*s - m).exp();
                    z += *s;
                }
                for t in 0..=pos {
                    for d in 0..hd {
                        att[h * hd + d] += sc[t] / z * vc[l][t][g * hd + d];
                    }
                }
            }
            let o = mv(&lw.wo, e, nh * hd, &att);
            for i in 0..e {
                x[i] += o[i];
            }
            let xb = rms(&x, &lw.ffn_norm, cfg.eps);
            let gate = mv(&lw.gate, cfg.ff, e, &xb);
            let up = mv(&lw.up, cfg.ff, e, &xb);
            let hid: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let d = mv(&lw.down, e, cfg.ff, &hid);
            for i in 0..e {
                x[i] += d[i];
            }
        }
        let xb = rms(&x, &w.out_norm, cfg.eps);
        let out = w.out.as_ref().unwrap_or(&w.tok_embd);
        all.push(mv(out, cfg.vocab, e, &xb));
    }
    all
}

const SEQ: [u32; 6] = [3, 1, 7, 7, 0, 12];

fn assert_matches_reference(name: &str, cfg: &Cfg) {
    let w = weights(cfg, 0x5A11_7000 + cfg.layers as u64);
    let mut model = load(&model_bytes(cfg, &w), 2).unwrap();
    let mut cache = model.new_cache(16).unwrap();
    let got = run(&mut model, &mut cache, &SEQ);
    let want = ref_logits(cfg, &w, &SEQ);
    for (pos, (g, r)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.len(), r.len());
        for (i, (a, b)) in g.iter().zip(r).enumerate() {
            let tol = 1e-4 + 1e-4 * b.abs();
            assert!(
                (*a as f64 - b).abs() <= tol,
                "{name}: pos {pos} logit {i}: engine {a} vs reference {b}"
            );
        }
    }
    // The reference must not be degenerate: logits differ across vocabulary entries.
    let spread = want[0].iter().cloned().fold(f64::MIN, f64::max)
        - want[0].iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread > 1e-3,
        "{name}: degenerate fixture (spread {spread})"
    );
}

#[test]
fn tiny_model_matches_naive_reference() {
    assert_matches_reference("gqa", &Cfg::base());
}

#[test]
fn tiny_model_matches_reference_multi_head_attention() {
    let mut c = Cfg::base();
    c.kv = c.heads;
    assert_matches_reference("mha", &c);
}

#[test]
fn tiny_model_matches_reference_with_rope_freqs() {
    let mut c = Cfg::base();
    c.rope_freqs = true;
    assert_matches_reference("rope_freqs", &c);
}

#[test]
fn tiny_model_matches_reference_with_tied_embeddings() {
    let mut c = Cfg::base();
    c.tied = true;
    assert_matches_reference("tied", &c);
}

#[test]
fn tiny_model_matches_reference_with_q8_0_weights() {
    let mut c = Cfg::base();
    c.q8 = true;
    assert_matches_reference("q8_0", &c);
}

#[test]
fn rope_freqs_changes_the_result() {
    // Guards against a loader that reads rope_freqs.weight but never applies it.
    let plain = Cfg::base();
    let mut with = Cfg::base();
    with.rope_freqs = true;
    let w = weights(&plain, 99);
    let mut w2 = weights(&with, 99);
    w2.freqs = Some(vec![1.0, 1.0, 2.0, 8.0]);
    let seq = [3u32, 1, 7, 2];
    let mut m1 = load(&model_bytes(&plain, &w), 1).unwrap();
    let mut m2 = load(&model_bytes(&with, &w2), 1).unwrap();
    let (mut c1, mut c2) = (m1.new_cache(8).unwrap(), m2.new_cache(8).unwrap());
    let a = run(&mut m1, &mut c1, &seq);
    let b = run(&mut m2, &mut c2, &seq);
    assert_ne!(a[3], b[3]);
}

// ---------------------------------------------------------------- refusals

#[test]
fn registry_rejects_unknown_arch() {
    for arch in ["glm_moe_dsa", "qwen2", "deepseek2", ""] {
        let e = model_registry::lookup(arch).unwrap_err();
        assert_eq!(e.code(), "unsupported_architecture", "{arch}");
    }
    assert!(model_registry::lookup("llama").is_ok());
}

#[test]
fn non_llama_architecture_metadata_is_unsupported() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    let mut b = GgufBuilder::new().kv_str("general.architecture", "glm_moe_dsa");
    for t in tensor_list(&cfg, &w) {
        b = b.tensor(&t.name, &t.dims, t.ty, t.bytes);
    }
    let (c, msg) = code(load(&b.build(), 1));
    assert_eq!(c, "unsupported_architecture");
    assert!(msg.contains("glm_moe_dsa"), "{msg}");
}

#[test]
fn missing_tensor_is_invalid_model() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    let ts: Vec<T> = tensor_list(&cfg, &w)
        .into_iter()
        .filter(|t| t.name != "output_norm.weight")
        .collect();
    let (c, msg) = code(load(&assemble(meta_builder(&cfg, &[]), &ts), 1));
    assert_eq!(c, "invalid_model");
    assert!(msg.contains("output_norm.weight"), "{msg}");
}

#[test]
fn missing_metadata_is_invalid_model() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    for key in [
        "llama.block_count",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.context_length",
    ] {
        let bytes = assemble(meta_builder(&cfg, &[key]), &tensor_list(&cfg, &w));
        let (c, msg) = code(load(&bytes, 1));
        assert_eq!(c, "invalid_model", "{key}");
        assert!(msg.contains(key), "{msg}");
    }
}

#[test]
fn moe_tensors_are_unsupported() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    let mut ts = tensor_list(&cfg, &w);
    ts.push(T {
        name: "blk.0.ffn_gate_exps.weight".into(),
        dims: vec![32, 4],
        ty: GgmlType::F32,
        bytes: vec![0; 32 * 4 * 4],
    });
    let (c, msg) = code(load(&assemble(meta_builder(&cfg, &[]), &ts), 1));
    assert_eq!(c, "unsupported_architecture");
    assert!(msg.contains("ffn_gate_exps"), "{msg}");
}

#[test]
fn weight_biases_are_unsupported() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    let mut ts = tensor_list(&cfg, &w);
    ts.push(T {
        name: "blk.0.attn_q.bias".into(),
        dims: vec![32],
        ty: GgmlType::F32,
        bytes: vec![0; 32 * 4],
    });
    let (c, _) = code(load(&assemble(meta_builder(&cfg, &[]), &ts), 1));
    assert_eq!(c, "unsupported_architecture");
}

#[test]
fn shape_mismatch_is_invalid_model() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    let mut ts = tensor_list(&cfg, &w);
    let t = ts
        .iter_mut()
        .find(|t| t.name == "blk.1.attn_q.weight")
        .unwrap();
    t.dims = vec![32, 31];
    t.bytes.truncate(31 * 32 * 4);
    let (c, msg) = code(load(&assemble(meta_builder(&cfg, &[]), &ts), 1));
    assert_eq!(c, "invalid_model");
    assert!(msg.contains("blk.1.attn_q.weight"), "{msg}");
}

#[test]
fn unsupported_dtype_fails_at_load() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    let mut ts = tensor_list(&cfg, &w);
    let t = ts
        .iter_mut()
        .find(|t| t.name == "blk.0.ffn_down.weight")
        .unwrap();
    t.ty = GgmlType::Other(13);
    let (c, msg) = code(load(&assemble(meta_builder(&cfg, &[]), &ts), 1));
    assert_eq!(c, "unsupported_architecture");
    assert!(msg.contains("blk.0.ffn_down.weight"), "{msg}");
}

#[test]
fn rope_scaling_metadata_is_unsupported() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    for ty in ["yarn", "linear", "longrope"] {
        let b = meta_builder(&cfg, &[]).kv_str("llama.rope.scaling.type", ty);
        let (c, msg) = code(load(&assemble(b, &tensor_list(&cfg, &w)), 1));
        assert_eq!(c, "unsupported_architecture", "{ty}");
        assert!(msg.contains(ty), "{msg}");
    }
    // "none" is the explicit absence of scaling and is accepted.
    let b = meta_builder(&cfg, &[]).kv_str("llama.rope.scaling.type", "none");
    assert!(load(&assemble(b, &tensor_list(&cfg, &w)), 1).is_ok());
}

#[test]
fn expert_count_metadata_is_unsupported() {
    let cfg = Cfg::base();
    let w = weights(&cfg, 1);
    let b = meta_builder(&cfg, &[]).kv_u32("llama.expert_count", 8);
    let (c, _) = code(load(&assemble(b, &tensor_list(&cfg, &w)), 1));
    assert_eq!(c, "unsupported_architecture");
}

#[test]
fn zero_rope_factor_is_invalid_model() {
    let mut cfg = Cfg::base();
    cfg.rope_freqs = true;
    let mut w = weights(&cfg, 1);
    w.freqs = Some(vec![1.0, 0.0, 2.0, 8.0]);
    let (c, _) = code(load(&model_bytes(&cfg, &w), 1));
    assert_eq!(c, "invalid_model");
}

#[test]
fn heads_not_multiple_of_kv_heads_is_invalid_model() {
    let mut cfg = Cfg::base();
    cfg.kv = 3; // 4 heads is not a multiple of 3
    let w = weights(&Cfg::base(), 1);
    let (c, _) = code(load(&model_bytes(&cfg, &w), 1));
    assert_eq!(c, "invalid_model");
}

// ---------------------------------------------------------------- runtime behaviour

fn small_model(threads: usize) -> LlamaModel {
    let cfg = Cfg::base();
    load(&model_bytes(&cfg, &weights(&cfg, 7)), threads).unwrap()
}

#[test]
fn logits_are_deterministic_across_runs() {
    let (mut a, mut b) = (small_model(1), small_model(1));
    let (mut ca, mut cb) = (a.new_cache(8).unwrap(), b.new_cache(8).unwrap());
    assert_eq!(run(&mut a, &mut ca, &SEQ), run(&mut b, &mut cb, &SEQ));
}

#[test]
fn thread_counts_give_identical_logits() {
    let (mut a, mut b) = (small_model(1), small_model(3));
    let (mut ca, mut cb) = (a.new_cache(8).unwrap(), b.new_cache(8).unwrap());
    assert_eq!(run(&mut a, &mut ca, &SEQ), run(&mut b, &mut cb, &SEQ));
}

#[test]
fn kv_cache_overflow_is_resource_exhausted() {
    let mut m = small_model(1);
    let mut cache = m.new_cache(2).unwrap();
    m.forward(1, 0, &mut cache).unwrap();
    m.forward(2, 1, &mut cache).unwrap();
    let e = m.forward(3, 2, &mut cache).unwrap_err();
    assert_eq!(e.code(), "resource_exhausted");
    assert_eq!(cache.len(), 2, "a refused step must not change the cache");
}

#[test]
fn model_context_limit_is_resource_exhausted() {
    let mut cfg = Cfg::base();
    cfg.ctx = 3;
    let mut m = load(&model_bytes(&cfg, &weights(&cfg, 7)), 1).unwrap();
    let mut cache = m.new_cache(8).unwrap();
    for pos in 0..3 {
        m.forward(1, pos, &mut cache).unwrap();
    }
    let e = m.forward(1, 3, &mut cache).unwrap_err();
    assert_eq!(e.code(), "resource_exhausted");
}

#[test]
fn positions_must_be_contiguous() {
    let mut m = small_model(1);
    let mut cache = m.new_cache(8).unwrap();
    let e = m.forward(1, 1, &mut cache).unwrap_err();
    assert_eq!(e.code(), "invalid_request");
}

#[test]
fn token_outside_vocabulary_is_invalid_request() {
    let mut m = small_model(1);
    let mut cache = m.new_cache(8).unwrap();
    let e = m.forward(16, 0, &mut cache).unwrap_err();
    assert_eq!(e.code(), "invalid_request");
}

#[test]
fn mismatched_cache_shape_is_refused() {
    let mut m = small_model(1);
    let mut wrong = KvCache::new(1, 1, 1, 8);
    assert!(m.forward(1, 0, &mut wrong).is_err());
}

#[test]
fn rewinding_position_overwrites() {
    let mut m = small_model(2);
    let mut cache = m.new_cache(8).unwrap();
    run(&mut m, &mut cache, &[3, 1, 7, 7]);
    // Re-run position 2 with a different token: later entries are discarded.
    let rewound = m.forward(5, 2, &mut cache).unwrap().to_vec();
    assert_eq!(cache.len(), 3);

    let mut fresh_model = small_model(2);
    let mut fresh = fresh_model.new_cache(8).unwrap();
    let want = run(&mut fresh_model, &mut fresh, &[3, 1, 5]);
    assert_eq!(rewound, want[2]);
}

#[test]
fn kv_bytes_matches_allocation() {
    let m = small_model(1);
    let cache = m.new_cache(10).unwrap();
    assert_eq!(m.kv_bytes(10), cache.bytes());
    assert_eq!(m.kv_bytes(usize::MAX), usize::MAX);
    assert_eq!(m.vocab_size(), 16);
    assert_eq!(m.context_len(), 64);
}

// ---------------------------------------------------------------- real model (gated)

#[test]
fn real_model_smoke() {
    let Ok(path) = std::env::var("SALLY_TEST_MODEL") else {
        eprintln!("skipped: SALLY_TEST_MODEL not set");
        return;
    };
    use sally_inference::tokenizer::Tokenizer;
    let path = PathBuf::from(path);
    let file = GgufFile::open(&path).unwrap();
    let store = TensorStore::open(&file, &path).unwrap();
    let tok = Tokenizer::from_gguf(&file.metadata).unwrap();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut model = LlamaModel::load(store, &file.metadata, threads).unwrap();

    let generate = |model: &mut LlamaModel| -> String {
        let mut cache = model.new_cache(256).unwrap();
        let prompt = tok.encode("The capital of France is", true);
        let mut logits = Vec::new();
        for (pos, t) in prompt.iter().enumerate() {
            logits = model.forward(*t, pos, &mut cache).unwrap().to_vec();
        }
        let mut text = Vec::new();
        for pos in (prompt.len()..).take(8) {
            assert!(logits.iter().all(|v| v.is_finite()));
            let next = logits
                .iter()
                .enumerate()
                .fold(
                    (0, f32::MIN),
                    |b, (i, v)| if *v > b.1 { (i, *v) } else { b },
                )
                .0 as u32;
            if tok.is_eos(next) {
                break;
            }
            text.extend(tok.decode_piece(next));
            logits = model.forward(next, pos, &mut cache).unwrap().to_vec();
        }
        String::from_utf8_lossy(&text).into_owned()
    };
    let a = generate(&mut model);
    let b = generate(&mut model);
    assert_eq!(a, b, "greedy generation must repeat exactly");
    assert!(a.contains("Paris"), "unexpected continuation: {a:?}");
}
