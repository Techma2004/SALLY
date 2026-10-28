//! Dense Llama-family graph (RMSNorm, GQA attention with RoPE, SwiGLU).
//!
//! Everything the graph needs is checked at load time: missing tensors, shapes that
//! disagree with the metadata, unsupported tensor types, mixture-of-experts tensors,
//! weight biases and rope scaling are all refused with a typed error rather than being
//! ignored (ignoring any of them would silently change the model's behaviour).
//!
//! RoPE follows ggml's `GGML_ROPE_TYPE_NORMAL` (used for `LLM_ARCH_LLAMA`): adjacent
//! pairs `(x[2i], x[2i+1])` are rotated by `pos * freq_base^(-2i/rope_dim)`, divided by
//! the optional `rope_freqs.weight` factor of that pair (Llama 3.x ships its scaling that
//! way). Angles are accumulated in f32 exactly like `ggml_rope_cache_init`.

use super::{Architecture, KvCache};
use crate::backend::{dequant_row, is_supported, matvec};
use crate::error::EngineError;
use crate::gguf::{GgmlType, Metadata};
use crate::model_registry;
use crate::tensor_store::{TensorStore, TensorView};

const MAX_LAYERS: usize = 4096;

fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidModel(msg.into())
}

fn unsupported(msg: impl Into<String>) -> EngineError {
    EngineError::UnsupportedArchitecture(msg.into())
}

#[derive(Clone, Copy, Debug)]
struct Hp {
    n_layer: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    rope_dim: usize,
    n_ctx: usize,
    n_vocab: usize,
    eps: f32,
}

struct LayerNames {
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_output: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

struct LayerNorms {
    attn: Vec<f32>,
    ffn: Vec<f32>,
}

#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    xb: Vec<f32>,
    proj: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    att_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    rope: Vec<f32>,
    logits: Vec<f32>,
}

pub struct LlamaModel {
    store: TensorStore,
    threads: usize,
    hp: Hp,
    names: Vec<LayerNames>,
    norms: Vec<LayerNorms>,
    output_norm: Vec<f32>,
    has_output: bool,
    freq_factors: Option<Vec<f32>>,
    theta_scale: f32,
    s: Scratch,
}

/// Looks a tensor up, then checks (in this order) that its type is supported and that
/// its shape is `rows x cols`.
fn tensor<'a>(
    store: &'a TensorStore,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<TensorView<'a>, EngineError> {
    let v = store.view(name)?;
    if !is_supported(v.dtype) {
        return Err(unsupported(format!(
            "tensor {name:?} has unsupported type {:?}",
            v.dtype
        )));
    }
    if v.rows != rows || v.cols != cols {
        return Err(invalid(format!(
            "tensor {name:?} has shape {}x{} (rows x cols), expected {rows}x{cols}",
            v.rows, v.cols
        )));
    }
    Ok(v)
}

fn read_f32_vec(store: &TensorStore, name: &str, len: usize) -> Result<Vec<f32>, EngineError> {
    let v = tensor(store, name, 1, len)?;
    if !matches!(v.dtype, GgmlType::F32 | GgmlType::F16) {
        return Err(unsupported(format!(
            "tensor {name:?} must be F32 or F16, found {:?}",
            v.dtype
        )));
    }
    let mut out = vec![0.0f32; len];
    dequant_row(&v, 0, &mut out)?;
    if out.iter().any(|x| !x.is_finite()) {
        return Err(invalid(format!(
            "tensor {name:?} contains non-finite values"
        )));
    }
    Ok(out)
}

fn opt_uint(meta: &Metadata, key: &str) -> Result<Option<usize>, EngineError> {
    match meta.get(key) {
        None => Ok(None),
        Some(_) => match meta.get_u64(key).and_then(|v| usize::try_from(v).ok()) {
            Some(v) => Ok(Some(v)),
            None => Err(unsupported(format!(
                "metadata {key} is not a plain unsigned integer (per-layer values are not supported)"
            ))),
        },
    }
}

fn req_uint(meta: &Metadata, key: &str) -> Result<usize, EngineError> {
    match opt_uint(meta, key)? {
        Some(v) if v > 0 => Ok(v),
        _ => Err(invalid(format!("missing or zero metadata {key}"))),
    }
}

fn opt_f32(meta: &Metadata, key: &str) -> Result<Option<f32>, EngineError> {
    match meta.get(key) {
        None => Ok(None),
        Some(_) => meta
            .get_f32(key)
            .map(Some)
            .ok_or_else(|| invalid(format!("metadata {key} is not a float"))),
    }
}

impl LlamaModel {
    pub fn load(
        store: TensorStore,
        meta: &Metadata,
        threads: usize,
    ) -> Result<LlamaModel, EngineError> {
        let arch = meta.get_str("general.architecture").unwrap_or("");
        model_registry::lookup(arch)?;

        if opt_uint(meta, "llama.expert_count")?.unwrap_or(0) > 0 {
            return Err(unsupported(
                "mixture-of-experts llama models are not supported",
            ));
        }
        if let Some(t) = meta.get("llama.rope.scaling.type") {
            match t {
                crate::gguf::Value::Str(s) if s == "none" => {}
                crate::gguf::Value::Str(s) => {
                    return Err(unsupported(format!(
                    "rope scaling type {s:?} is not supported (only rope_freqs.weight factors are)"
                )))
                }
                _ => return Err(invalid("metadata llama.rope.scaling.type is not a string")),
            }
        }
        for name in store.tensor_names() {
            if name.contains("_exps") || name.contains("ffn_gate_inp") {
                return Err(unsupported(format!(
                    "tensor {name:?} belongs to a mixture-of-experts layer"
                )));
            }
            if name.ends_with(".bias") {
                return Err(unsupported(format!(
                    "tensor {name:?}: weight biases are not supported"
                )));
            }
        }

        let n_layer = req_uint(meta, "llama.block_count")?;
        if n_layer > MAX_LAYERS {
            return Err(invalid(format!("implausible block count {n_layer}")));
        }
        let n_embd = req_uint(meta, "llama.embedding_length")?;
        let n_ff = req_uint(meta, "llama.feed_forward_length")?;
        let n_head = req_uint(meta, "llama.attention.head_count")?;
        let n_ctx = req_uint(meta, "llama.context_length")?;
        let n_kv = opt_uint(meta, "llama.attention.head_count_kv")?.unwrap_or(n_head);
        if n_kv == 0 || n_head % n_kv != 0 {
            return Err(invalid(format!(
                "head count {n_head} is not a multiple of kv head count {n_kv}"
            )));
        }
        let key_len = opt_uint(meta, "llama.attention.key_length")?;
        let val_len = opt_uint(meta, "llama.attention.value_length")?;
        let head_dim = match (key_len, val_len) {
            (Some(k), Some(v)) if k != v => {
                return Err(unsupported("key and value head sizes differ"))
            }
            (Some(k), _) if k > 0 => k,
            _ => {
                if n_embd % n_head != 0 {
                    return Err(invalid(format!(
                        "embedding length {n_embd} is not divisible by head count {n_head}"
                    )));
                }
                n_embd / n_head
            }
        };
        let rope_dim = opt_uint(meta, "llama.rope.dimension_count")?.unwrap_or(head_dim);
        if rope_dim == 0 || rope_dim % 2 != 0 || rope_dim > head_dim {
            return Err(invalid(format!(
                "rope dimension count {rope_dim} must be even, nonzero and at most the head size {head_dim}"
            )));
        }
        let freq_base = opt_f32(meta, "llama.rope.freq_base")?.unwrap_or(10000.0);
        if !(freq_base.is_finite() && freq_base > 0.0) {
            return Err(invalid("llama.rope.freq_base must be positive"));
        }
        let eps = opt_f32(meta, "llama.attention.layer_norm_rms_epsilon")?
            .ok_or_else(|| invalid("missing metadata llama.attention.layer_norm_rms_epsilon"))?;
        if !(eps.is_finite() && eps >= 0.0) {
            return Err(invalid("rms epsilon must be finite and non-negative"));
        }

        let q_dim = n_head
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("attention size overflows"))?;
        let kv_dim = n_kv
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("attention size overflows"))?;

        let embd = store.view("token_embd.weight")?;
        let n_vocab = embd.rows;
        tensor(&store, "token_embd.weight", n_vocab, n_embd)?;
        let has_output = store.view("output.weight").is_ok();
        if has_output {
            tensor(&store, "output.weight", n_vocab, n_embd)?;
        }
        let output_norm = read_f32_vec(&store, "output_norm.weight", n_embd)?;

        let freq_factors = if store.view("rope_freqs.weight").is_ok() {
            let f = read_f32_vec(&store, "rope_freqs.weight", rope_dim / 2)?;
            if f.contains(&0.0) {
                return Err(invalid("rope_freqs.weight contains a zero factor"));
            }
            Some(f)
        } else {
            None
        };

        let mut names = Vec::new();
        let mut norms = Vec::new();
        for l in 0..n_layer {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let ln = LayerNames {
                attn_q: n("attn_q"),
                attn_k: n("attn_k"),
                attn_v: n("attn_v"),
                attn_output: n("attn_output"),
                ffn_gate: n("ffn_gate"),
                ffn_up: n("ffn_up"),
                ffn_down: n("ffn_down"),
            };
            let attn_norm = read_f32_vec(&store, &n("attn_norm"), n_embd)?;
            let ffn_norm = read_f32_vec(&store, &n("ffn_norm"), n_embd)?;
            tensor(&store, &ln.attn_q, q_dim, n_embd)?;
            tensor(&store, &ln.attn_k, kv_dim, n_embd)?;
            tensor(&store, &ln.attn_v, kv_dim, n_embd)?;
            tensor(&store, &ln.attn_output, n_embd, q_dim)?;
            tensor(&store, &ln.ffn_gate, n_ff, n_embd)?;
            tensor(&store, &ln.ffn_up, n_ff, n_embd)?;
            tensor(&store, &ln.ffn_down, n_embd, n_ff)?;
            names.push(ln);
            norms.push(LayerNorms {
                attn: attn_norm,
                ffn: ffn_norm,
            });
        }

        let hp = Hp {
            n_layer,
            n_head,
            n_kv,
            head_dim,
            rope_dim,
            n_ctx,
            n_vocab,
            eps,
        };
        // Sizes below are bounded by tensors that were just validated against the file.
        let s = Scratch {
            x: vec![0.0; n_embd],
            xb: vec![0.0; n_embd],
            proj: vec![0.0; n_embd],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: Vec::new(),
            att_out: vec![0.0; q_dim],
            gate: vec![0.0; n_ff],
            up: vec![0.0; n_ff],
            rope: vec![0.0; rope_dim],
            logits: vec![0.0; n_vocab],
        };
        Ok(LlamaModel {
            store,
            threads: threads.max(1),
            hp,
            names,
            norms,
            output_norm,
            has_output,
            freq_factors,
            theta_scale: freq_base.powf(-2.0 / rope_dim as f32),
            s,
        })
    }
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
        *o = v * scale * g;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn apply_rope(v: &mut [f32], heads: usize, head_dim: usize, rope: &[f32]) {
    for h in 0..heads {
        let base = h * head_dim;
        for (i, cs) in rope.chunks_exact(2).enumerate() {
            let (c, s) = (cs[0], cs[1]);
            let x0 = v[base + 2 * i];
            let x1 = v[base + 2 * i + 1];
            v[base + 2 * i] = x0 * c - x1 * s;
            v[base + 2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

impl Architecture for LlamaModel {
    fn forward(
        &mut self,
        token: u32,
        pos: usize,
        state: &mut KvCache,
    ) -> Result<&[f32], EngineError> {
        let hp = self.hp;
        if state.shape() != (hp.n_layer, hp.n_kv, hp.head_dim) {
            return Err(EngineError::Internal(
                "KV cache shape does not match the model".into(),
            ));
        }
        let tok = token as usize;
        if tok >= hp.n_vocab {
            return Err(EngineError::InvalidRequest(format!(
                "token id {token} is outside the vocabulary of {}",
                hp.n_vocab
            )));
        }
        if pos > state.len() {
            return Err(EngineError::InvalidRequest(format!(
                "position {pos} skips ahead of the {} cached positions",
                state.len()
            )));
        }
        if pos >= state.capacity() || pos >= hp.n_ctx {
            return Err(EngineError::ResourceExhausted(format!(
                "position {pos} exceeds the context limit (cache {}, model {})",
                state.capacity(),
                hp.n_ctx
            )));
        }

        let LlamaModel {
            store,
            threads,
            names,
            norms,
            output_norm,
            has_output,
            freq_factors,
            theta_scale,
            s,
            ..
        } = self;
        let threads = *threads;

        if s.att.len() < state.capacity() {
            s.att.resize(state.capacity(), 0.0);
        }

        // RoPE cos/sin for this position, shared by every layer and head.
        let mut theta = pos as f32;
        for i in 0..hp.rope_dim / 2 {
            let ff = freq_factors.as_ref().map_or(1.0, |f| f[i]);
            let t = theta / ff;
            s.rope[2 * i] = t.cos();
            s.rope[2 * i + 1] = t.sin();
            theta *= *theta_scale;
        }

        let embd = store.view("token_embd.weight")?;
        dequant_row(&embd, tok, &mut s.x)?;

        let kv_stride = hp.n_kv * hp.head_dim;
        let group = hp.n_head / hp.n_kv;
        let scale = 1.0 / (hp.head_dim as f32).sqrt();

        for l in 0..hp.n_layer {
            let ln = &names[l];
            rms_norm(&s.x, &norms[l].attn, hp.eps, &mut s.xb);
            matvec(&store.view(&ln.attn_q)?, &s.xb, &mut s.q, threads)?;
            matvec(&store.view(&ln.attn_k)?, &s.xb, &mut s.k, threads)?;
            matvec(&store.view(&ln.attn_v)?, &s.xb, &mut s.v, threads)?;
            apply_rope(&mut s.q, hp.n_head, hp.head_dim, &s.rope);
            apply_rope(&mut s.k, hp.n_kv, hp.head_dim, &s.rope);

            let (kbuf, vbuf) = state
                .layer_mut(l)
                .ok_or_else(|| EngineError::Internal("KV cache layer out of range".into()))?;
            let off = pos * kv_stride;
            kbuf[off..off + kv_stride].copy_from_slice(&s.k);
            vbuf[off..off + kv_stride].copy_from_slice(&s.v);

            for h in 0..hp.n_head {
                let kvh = h / group;
                let qh = &s.q[h * hp.head_dim..(h + 1) * hp.head_dim];
                let scores = &mut s.att[..=pos];
                for (t, sc) in scores.iter_mut().enumerate() {
                    let at = t * kv_stride + kvh * hp.head_dim;
                    *sc = dot(qh, &kbuf[at..at + hp.head_dim]) * scale;
                }
                softmax(scores);
                let out = &mut s.att_out[h * hp.head_dim..(h + 1) * hp.head_dim];
                out.fill(0.0);
                for (t, w) in scores.iter().enumerate() {
                    let at = t * kv_stride + kvh * hp.head_dim;
                    for (o, v) in out.iter_mut().zip(&vbuf[at..at + hp.head_dim]) {
                        *o += w * v;
                    }
                }
            }
            matvec(
                &store.view(&ln.attn_output)?,
                &s.att_out,
                &mut s.proj,
                threads,
            )?;
            for (x, p) in s.x.iter_mut().zip(&s.proj) {
                *x += p;
            }

            rms_norm(&s.x, &norms[l].ffn, hp.eps, &mut s.xb);
            matvec(&store.view(&ln.ffn_gate)?, &s.xb, &mut s.gate, threads)?;
            matvec(&store.view(&ln.ffn_up)?, &s.xb, &mut s.up, threads)?;
            for (g, u) in s.gate.iter_mut().zip(&s.up) {
                *g = *g / (1.0 + (-*g).exp()) * u;
            }
            matvec(&store.view(&ln.ffn_down)?, &s.gate, &mut s.proj, threads)?;
            for (x, p) in s.x.iter_mut().zip(&s.proj) {
                *x += p;
            }
        }

        state.set_len(pos + 1);
        rms_norm(&s.x, output_norm, hp.eps, &mut s.xb);
        let out_name = if *has_output {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        matvec(&store.view(out_name)?, &s.xb, &mut s.logits, threads)?;
        Ok(&s.logits)
    }

    fn context_len(&self) -> usize {
        self.hp.n_ctx
    }

    fn kv_bytes(&self, ctx: usize) -> usize {
        ctx.saturating_mul(self.hp.n_layer)
            .saturating_mul(self.hp.n_kv)
            .saturating_mul(self.hp.head_dim)
            .saturating_mul(2 * std::mem::size_of::<f32>())
    }

    fn vocab_size(&self) -> usize {
        self.hp.n_vocab
    }

    fn new_cache(&self, ctx: usize) -> Result<KvCache, EngineError> {
        KvCache::try_new(self.hp.n_layer, self.hp.n_kv, self.hp.head_dim, ctx)
    }

    fn scratch_bytes(&self) -> usize {
        let s = &self.s;
        [
            &s.x, &s.xb, &s.proj, &s.q, &s.k, &s.v, &s.att, &s.att_out, &s.gate, &s.up, &s.rope,
            &s.logits,
        ]
        .iter()
        .map(|v| v.capacity() * std::mem::size_of::<f32>())
        .sum()
    }
}
