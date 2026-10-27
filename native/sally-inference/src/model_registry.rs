//! Allow-list of supported model architectures. Anything not listed is refused with
//! `unsupported_architecture`; there is no fallback to a "closest" architecture.

use crate::error::EngineError;
use crate::gguf::GgmlType;

#[derive(Debug)]
pub struct Capability {
    /// Value of `general.architecture`.
    pub architecture: &'static str,
    pub supported_dtypes: &'static [GgmlType],
    pub required_ops: &'static [&'static str],
    pub known_limitations: &'static [&'static str],
}

static LLAMA: Capability = Capability {
    architecture: "llama",
    supported_dtypes: &[
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
    ],
    required_ops: &["rms_norm", "matvec", "rope_norm", "gqa_attention", "swiglu"],
    known_limitations: &[
        "dense models only (no mixture of experts)",
        "no weight biases",
        "rope scaling only via the rope_freqs.weight tensor (no yarn, longrope or linear factor)",
        "per-layer variable head counts are not supported",
    ],
};

static ALLOWED: [&Capability; 1] = [&LLAMA];

pub fn lookup(arch: &str) -> Result<&'static Capability, EngineError> {
    ALLOWED
        .iter()
        .copied()
        .find(|c| c.architecture == arch)
        .ok_or_else(|| {
            EngineError::UnsupportedArchitecture(format!(
                "architecture {arch:?} is not supported (supported: llama)"
            ))
        })
}
