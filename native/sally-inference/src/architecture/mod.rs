//! Model graphs. An [`Architecture`] turns one token at a time into logits, using a
//! caller-owned [`KvCache`] so the model itself holds no per-request state.

pub mod llama;

use crate::error::EngineError;

/// A loaded model that can run one forward step.
pub trait Architecture {
    /// Runs `token` at position `pos`, appending its keys and values to `state`, and
    /// returns the logits for the next token (valid until the next call).
    ///
    /// `pos` must be at most `state.len()` (positions are contiguous; passing an earlier
    /// position overwrites from there, which discards the later entries). Errors:
    /// `resource_exhausted` if `pos >= ` the cache capacity, `invalid_request` if `token`
    /// is outside the vocabulary.
    fn forward(
        &mut self,
        token: u32,
        pos: usize,
        state: &mut KvCache,
    ) -> Result<&[f32], EngineError>;

    /// Maximum context length the model was trained for.
    fn context_len(&self) -> usize;

    /// Bytes of KV cache needed for `ctx` positions (saturates at `usize::MAX`).
    fn kv_bytes(&self, ctx: usize) -> usize;

    fn vocab_size(&self) -> usize;

    /// Allocates an empty KV cache shaped for this model with room for `ctx` positions.
    fn new_cache(&self, ctx: usize) -> Result<KvCache, EngineError>;

    /// Bytes of reusable scratch buffers the model holds besides its weights.
    fn scratch_bytes(&self) -> usize;
}

/// Key/value cache for one sequence, f32 storage:
/// `[layer][position][kv_head][head_dim]` for keys and again for values.
#[derive(Debug)]
pub struct KvCache {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    ctx: usize,
    len: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvCache {
    /// Allocates a zeroed cache, with checked size arithmetic and fallible allocation.
    pub fn try_new(
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        ctx: usize,
    ) -> Result<KvCache, EngineError> {
        let too_big = || {
            EngineError::ResourceExhausted(format!(
                "KV cache for {layers} layers x {ctx} positions x {kv_heads} kv heads x {head_dim} dims is too large"
            ))
        };
        let elems = layers
            .checked_mul(ctx)
            .and_then(|n| n.checked_mul(kv_heads))
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or_else(too_big)?;
        // Total bytes for K and V together must be addressable too.
        elems
            .checked_mul(2 * std::mem::size_of::<f32>())
            .ok_or_else(too_big)?;
        let alloc = |n: usize| -> Result<Vec<f32>, EngineError> {
            let mut v: Vec<f32> = Vec::new();
            v.try_reserve_exact(n).map_err(|_| too_big())?;
            v.resize(n, 0.0);
            Ok(v)
        };
        Ok(KvCache {
            layers,
            kv_heads,
            head_dim,
            ctx,
            len: 0,
            k: alloc(elems)?,
            v: alloc(elems)?,
        })
    }

    /// Infallible constructor. If the size is not representable or cannot be allocated, the
    /// result has capacity 0, so every `forward` fails with `resource_exhausted` rather than
    /// panicking. Prefer [`try_new`](Self::try_new) when the error matters.
    pub fn new(layers: usize, kv_heads: usize, head_dim: usize, ctx: usize) -> KvCache {
        KvCache::try_new(layers, kv_heads, head_dim, ctx).unwrap_or(KvCache {
            layers,
            kv_heads,
            head_dim,
            ctx: 0,
            len: 0,
            k: Vec::new(),
            v: Vec::new(),
        })
    }

    /// Number of positions currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity in positions.
    pub fn capacity(&self) -> usize {
        self.ctx
    }

    /// Forgets all stored positions (the memory stays allocated).
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Keeps only the first `n` positions (no-op if `n >= len`).
    pub fn truncate(&mut self, n: usize) {
        self.len = self.len.min(n);
    }

    /// Bytes held by the key and value buffers.
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }

    pub(crate) fn shape(&self) -> (usize, usize, usize) {
        (self.layers, self.kv_heads, self.head_dim)
    }

    /// Whole key and value buffers of one layer (`ctx * kv_heads * head_dim` floats each).
    pub(crate) fn layer_mut(&mut self, layer: usize) -> Option<(&mut [f32], &mut [f32])> {
        let per = self
            .ctx
            .checked_mul(self.kv_heads)?
            .checked_mul(self.head_dim)?;
        let start = layer.checked_mul(per)?;
        let end = start.checked_add(per)?;
        let k = self.k.get_mut(start..end)?;
        let v = self.v.get_mut(start..end)?;
        Some((k, v))
    }

    pub(crate) fn set_len(&mut self, len: usize) {
        self.len = len.min(self.ctx);
    }
}
