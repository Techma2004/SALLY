//! Read-only qualification of a GGUF file: can this engine run it, and would it fit?
//! Only the header and tensor index are read, so it works on files far larger than RAM.
//! It reports what is missing; it never suggests a substitute architecture.

use crate::config::Config;
use crate::error::EngineError;
use crate::gguf::GgufFile;
use crate::model_registry;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct QualificationReport {
    pub architecture: String,
    pub supported: bool,
    pub unsupported_ops: Vec<String>,
    pub weight_bytes: u64,
    /// Bytes outside the routed-expert tensors; only reported for mixture-of-experts files.
    pub dense_bytes: Option<u64>,
    /// KV cache size at the configured context; unknown for architectures we cannot size.
    pub kv_bytes_at_ctx: Option<u64>,
    /// Weights + KV cache against the memory budget, assuming no expert streaming.
    pub fits_in_budget: bool,
    pub verdict: String,
}

impl QualificationReport {
    /// 0 qualified; 2 not supported; 1 supported but does not fit the budget.
    pub fn exit_code(&self) -> u8 {
        match (self.supported, self.fits_in_budget) {
            (true, true) => 0,
            (false, _) => 2,
            (true, false) => 1,
        }
    }

    pub fn render(&self) -> String {
        let mut s = format!(
            "architecture: {:?}\nsupported: {}\nweights: {} bytes\n",
            self.architecture, self.supported, self.weight_bytes
        );
        if let Some(d) = self.dense_bytes {
            s += &format!("dense (non-expert) weights: {d} bytes\n");
        }
        match self.kv_bytes_at_ctx {
            Some(k) => s += &format!("kv cache at configured context: {k} bytes\n"),
            None => s += "kv cache at configured context: unknown\n",
        }
        s += &format!("fits in memory budget: {}\n", self.fits_in_budget);
        for op in &self.unsupported_ops {
            s += &format!("missing: {op}\n");
        }
        s += &format!("verdict: {}\n", self.verdict);
        s
    }
}

fn kv_bytes(file: &GgufFile, arch: &str, ctx: u64) -> Option<u64> {
    let m = &file.metadata;
    let layers = u64::from(m.get_u32(&format!("{arch}.block_count"))?);
    let embd = u64::from(m.get_u32(&format!("{arch}.embedding_length"))?);
    let heads = u64::from(m.get_u32(&format!("{arch}.attention.head_count"))?);
    let kv_heads = m
        .get_u32(&format!("{arch}.attention.head_count_kv"))
        .map_or(heads, u64::from);
    let head_dim = m
        .get_u32(&format!("{arch}.attention.key_length"))
        .map_or(embd.checked_div(heads)?, u64::from);
    2u64.checked_mul(layers)?
        .checked_mul(ctx)?
        .checked_mul(kv_heads)?
        .checked_mul(head_dim)?
        .checked_mul(4)
}

pub fn qualify(path: &Path, cfg: &Config) -> Result<QualificationReport, EngineError> {
    let file = GgufFile::open(path)?;
    let arch = file.architecture().unwrap_or("").to_string();
    let weight_bytes: u64 = file.tensors.iter().map(|t| t.n_bytes).sum();
    let mut missing = Vec::new();

    let capability = model_registry::lookup(&arch).ok();
    if capability.is_none() {
        missing.push(format!("architecture {arch:?} has no adapter in this engine"));
    }

    let expert_bytes: u64 = file
        .tensors
        .iter()
        .filter(|t| t.name.contains("_exps."))
        .map(|t| t.n_bytes)
        .sum();
    let is_moe = expert_bytes > 0
        || file.metadata.get_u32(&format!("{arch}.expert_count")).unwrap_or(0) > 0;
    if is_moe {
        missing.push("mixture-of-experts routing and routed-expert tensors (ffn_*_exps)".into());
    }
    let has = |needle: &str| file.tensors.iter().any(|t| t.name.contains(needle));
    // Name-based heuristics: these tensor names are how llama.cpp-style GGUFs carry the features.
    if has("attn_kv_a_mqa") || has("attn_q_a.") || has("attn_kv_b.") {
        missing.push("multi-head latent attention (compressed KV projections)".into());
    }
    if has("indexer") {
        missing.push("sparse attention (token indexer tensors)".into());
    }
    if has(".bias") {
        missing.push("weight biases".into());
    }
    if let Some(cap) = capability {
        let mut bad: Vec<String> = file
            .tensors
            .iter()
            .filter(|t| !cap.supported_dtypes.contains(&t.dtype))
            .map(|t| format!("{:?}", t.dtype))
            .collect();
        bad.sort();
        bad.dedup();
        if !bad.is_empty() {
            missing.push(format!("tensor types not implemented: {}", bad.join(", ")));
        }
    }

    let supported = missing.is_empty();
    let ctx_limit = file
        .metadata
        .get_u32(&format!("{arch}.context_length"))
        .map_or(u64::MAX, u64::from);
    let ctx = (cfg.context_len as u64).min(ctx_limit);
    let kv = if capability.is_some() { kv_bytes(&file, &arch, ctx) } else { None };
    let budget = (cfg.memory_budget_mb as u64).saturating_mul(1024 * 1024);
    let need = weight_bytes.saturating_add(kv.unwrap_or(0));
    let fits = supported && need <= budget;

    let verdict = if !supported {
        format!(
            "NOT QUALIFIED: {} required capabilit{} missing; this engine will refuse the model.",
            missing.len(),
            if missing.len() == 1 { "y is" } else { "ies are" }
        )
    } else if !fits {
        format!(
            "NOT QUALIFIED for this budget: weights + KV cache need {need} bytes, budget is {budget} bytes. \
             Without expert streaming the weights would be paged from disk, which is not a supported operating mode."
        )
    } else {
        "supported by the adapter and within the memory budget; still unqualified until its \
         integration test passes on the target machine."
            .to_string()
    };

    Ok(QualificationReport {
        architecture: arch,
        supported,
        unsupported_ops: missing,
        weight_bytes,
        dense_bytes: is_moe.then(|| weight_bytes - expert_bytes),
        kv_bytes_at_ctx: kv,
        fits_in_budget: fits,
        verdict,
    })
}
