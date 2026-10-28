//! Tensor directory parsing and size computation.

use super::reader::{invalid, Reader};
use crate::error::EngineError;
use std::collections::HashSet;
use std::fmt;
use std::io::Read;

/// ggml's own limit on tensor rank.
pub const MAX_DIMS: u32 = 4;
pub const MAX_TENSOR_COUNT: u64 = 1 << 20;
/// Smallest encoded tensor info: 8 (name len) + 4 (n_dims) + 4 (type) + 8 (offset).
pub const MIN_TENSOR_INFO_BYTES: u64 = 24;
pub const MAX_TENSOR_NAME_LEN: u64 = 1024;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q6_K,
    Other(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4_K,
            14 => GgmlType::Q6_K,
            other => GgmlType::Other(other),
        }
    }

    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4_K => 12,
            GgmlType::Q6_K => 14,
            GgmlType::Other(id) => id,
        }
    }

    /// `(elements per block, bytes per block)`; `None` for types this crate
    /// does not know the layout of.
    pub fn block(self) -> Option<(u64, u64)> {
        match self {
            GgmlType::F32 => Some((1, 4)),
            GgmlType::F16 => Some((1, 2)),
            GgmlType::Q4_0 => Some((32, 18)),
            GgmlType::Q8_0 => Some((32, 34)),
            GgmlType::Q4_K => Some((256, 144)),
            GgmlType::Q6_K => Some((256, 210)),
            GgmlType::Other(_) => None,
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgmlType::F32 => write!(f, "F32"),
            GgmlType::F16 => write!(f, "F16"),
            GgmlType::Q4_0 => write!(f, "Q4_0"),
            GgmlType::Q8_0 => write!(f, "Q8_0"),
            GgmlType::Q4_K => write!(f, "Q4_K"),
            GgmlType::Q6_K => write!(f, "Q6_K"),
            GgmlType::Other(id) => write!(f, "Other({id})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset relative to the start of the data section.
    pub offset: u64,
    /// Size in bytes; `0` for `GgmlType::Other` (later stages reject those).
    pub n_bytes: u64,
}

/// Computes the byte size of a tensor of `dtype` with `dims`, checking for overflow
/// and for the block-size constraint (`dims[0]` is the contiguous dimension).
pub(crate) fn tensor_bytes(name: &str, dtype: GgmlType, dims: &[u64]) -> Result<u64, EngineError> {
    let mut elements: u64 = 1;
    for d in dims {
        elements = elements
            .checked_mul(*d)
            .ok_or_else(|| invalid(format!("tensor {name:?}: dimension product overflows")))?;
    }
    let Some((block_elems, block_bytes)) = dtype.block() else {
        return Ok(0);
    };
    if let Some(first) = dims.first() {
        if !first.is_multiple_of(block_elems) {
            return Err(invalid(format!(
                "tensor {name:?}: first dimension {first} is not a multiple of block size {block_elems} for {dtype}"
            )));
        }
    }
    if !elements.is_multiple_of(block_elems) {
        return Err(invalid(format!(
            "tensor {name:?}: {elements} elements is not a multiple of block size {block_elems} for {dtype}"
        )));
    }
    (elements / block_elems)
        .checked_mul(block_bytes)
        .ok_or_else(|| invalid(format!("tensor {name:?}: byte size overflows")))
}

pub(crate) fn read_tensor_infos<R: Read>(
    r: &mut Reader<R>,
    count: u64,
) -> Result<Vec<TensorInfo>, EngineError> {
    if count > MAX_TENSOR_COUNT {
        return Err(invalid(format!(
            "tensor count {count} exceeds limit of {MAX_TENSOR_COUNT}"
        )));
    }
    let needed = count
        .checked_mul(MIN_TENSOR_INFO_BYTES)
        .ok_or_else(|| invalid("tensor count overflow"))?;
    if needed > r.remaining() {
        return Err(invalid(format!(
            "tensor count {count} cannot fit in the remaining {} bytes",
            r.remaining()
        )));
    }
    let cap = usize::try_from(count).map_err(|_| invalid("tensor count too large"))?;
    let mut tensors = Vec::with_capacity(cap);
    let mut seen = HashSet::new();
    for _ in 0..cap {
        let name = r.string(MAX_TENSOR_NAME_LEN, "tensor name")?;
        let n_dims = r.u32()?;
        if n_dims > MAX_DIMS {
            return Err(invalid(format!(
                "tensor {name:?}: {n_dims} dimensions exceeds limit of {MAX_DIMS}"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let dtype = GgmlType::from_id(r.u32()?);
        let offset = r.u64()?;
        let n_bytes = tensor_bytes(&name, dtype, &dims)?;
        if !seen.insert(name.clone()) {
            return Err(invalid(format!("duplicate tensor name {name:?}")));
        }
        tensors.push(TensorInfo {
            name,
            dims,
            dtype,
            offset,
            n_bytes,
        });
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_ggml_layouts() {
        // 2 f16 header + 16 nibbles; 2 + 32; 2*2 + 12 + 128; 2 + 16 + 192.
        assert_eq!(tensor_bytes("t", GgmlType::F32, &[8, 2]).unwrap(), 64);
        assert_eq!(tensor_bytes("t", GgmlType::F16, &[8, 2]).unwrap(), 32);
        assert_eq!(tensor_bytes("t", GgmlType::Q4_0, &[64, 2]).unwrap(), 4 * 18);
        assert_eq!(tensor_bytes("t", GgmlType::Q8_0, &[64, 2]).unwrap(), 4 * 34);
        assert_eq!(
            tensor_bytes("t", GgmlType::Q4_K, &[256, 3]).unwrap(),
            3 * 144
        );
        assert_eq!(
            tensor_bytes("t", GgmlType::Q6_K, &[512, 1]).unwrap(),
            2 * 210
        );
    }

    #[test]
    fn rejects_non_block_multiple_and_allows_other() {
        assert!(tensor_bytes("t", GgmlType::Q4_K, &[100]).is_err());
        assert!(tensor_bytes("t", GgmlType::Q4_0, &[16, 4]).is_err());
        assert_eq!(tensor_bytes("t", GgmlType::Other(30), &[7]).unwrap(), 0);
        assert!(tensor_bytes("t", GgmlType::F32, &[u64::MAX, 3]).is_err());
    }

    #[test]
    fn type_ids_round_trip() {
        for id in [0, 1, 2, 8, 12, 14, 3, 30, u32::MAX] {
            assert_eq!(GgmlType::from_id(id).id(), id);
        }
    }
}
