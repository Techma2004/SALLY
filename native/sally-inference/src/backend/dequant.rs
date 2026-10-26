//! Block dequantization, layouts as in ggml (`ggml-common.h`, `ggml-quants.c`).

use crate::error::EngineError;
use crate::gguf::GgmlType;
use crate::tensor_store::TensorView;

fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidModel(msg.into())
}

pub(crate) fn unsupported(dtype: GgmlType) -> EngineError {
    EngineError::UnsupportedArchitecture(format!(
        "tensor type {dtype} (ggml type id {}) is not supported by the CPU backend",
        dtype.id()
    ))
}

pub(crate) fn is_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q4_0
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q6_K
    )
}

/// Bytes of one row of `cols` elements.
pub(crate) fn row_bytes(dtype: GgmlType, cols: usize) -> Result<usize, EngineError> {
    let (be, bb) = dtype.block().ok_or_else(|| unsupported(dtype))?;
    let cols = cols as u64;
    if !cols.is_multiple_of(be) {
        return Err(invalid(format!(
            "{cols} columns is not a multiple of the {dtype} block size {be}"
        )));
    }
    (cols / be)
        .checked_mul(bb)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid("row size overflows"))
}

/// IEEE 754 binary16 to binary32: subnormals, infinities and NaNs handled.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    match exp {
        0 => {
            // Zero or subnormal: value = man * 2^-24 (exact in f32).
            let mag = man as f32 * (1.0 / 16_777_216.0);
            f32::from_bits(mag.to_bits() | sign)
        }
        31 => f32::from_bits(sign | 0x7F80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

fn half(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// ggml `get_scale_min_k4`.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantizes exactly one row (`bytes`, a whole number of blocks) into `out`.
/// Callers guarantee `bytes.len() == row_bytes(dtype, out.len())`.
pub(crate) fn dequant_row_bytes(dtype: GgmlType, bytes: &[u8], out: &mut [f32]) {
    match dtype {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlType::Q4_0 => {
            for (o, b) in out.chunks_exact_mut(32).zip(bytes.chunks_exact(18)) {
                let d = half(b, 0);
                for j in 0..16 {
                    let q = b[2 + j];
                    o[j] = ((q & 0x0F) as i32 - 8) as f32 * d;
                    o[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Q8_0 => {
            for (o, b) in out.chunks_exact_mut(32).zip(bytes.chunks_exact(34)) {
                let d = half(b, 0);
                for j in 0..32 {
                    o[j] = (b[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_K => {
            for (o, b) in out.chunks_exact_mut(256).zip(bytes.chunks_exact(144)) {
                let d = half(b, 0);
                let min = half(b, 2);
                let scales = &b[4..16];
                let qs = &b[16..144];
                for chunk in 0..4 {
                    let (sc1, m1) = scale_min_k4(chunk * 2, scales);
                    let (sc2, m2) = scale_min_k4(chunk * 2 + 1, scales);
                    let (d1, m1) = (d * sc1 as f32, min * m1 as f32);
                    let (d2, m2) = (d * sc2 as f32, min * m2 as f32);
                    let q = &qs[chunk * 32..chunk * 32 + 32];
                    let y = &mut o[chunk * 64..chunk * 64 + 64];
                    for l in 0..32 {
                        y[l] = d1 * (q[l] & 0xF) as f32 - m1;
                        y[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            for (o, b) in out.chunks_exact_mut(256).zip(bytes.chunks_exact(210)) {
                let d = half(b, 208);
                for half_idx in 0..2 {
                    let ql = &b[half_idx * 64..half_idx * 64 + 64];
                    let qh = &b[128 + half_idx * 32..128 + half_idx * 32 + 32];
                    let sc = &b[192 + half_idx * 8..192 + half_idx * 8 + 8];
                    let y = &mut o[half_idx * 128..half_idx * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        // Unreachable for callers that check `is_supported`; leave `out` untouched.
        GgmlType::Other(_) => {}
    }
}

/// Dequantizes row `row` of `w` into `out` (`out.len()` must equal `w.cols`).
pub fn dequant_row(w: &TensorView<'_>, row: usize, out: &mut [f32]) -> Result<(), EngineError> {
    if !is_supported(w.dtype) {
        return Err(unsupported(w.dtype));
    }
    if row >= w.rows {
        return Err(invalid(format!("row {row} out of range ({} rows)", w.rows)));
    }
    if out.len() != w.cols {
        return Err(invalid(format!(
            "output length {} does not match {} columns",
            out.len(),
            w.cols
        )));
    }
    let rb = row_bytes(w.dtype, w.cols)?;
    let start = row
        .checked_mul(rb)
        .ok_or_else(|| invalid("row offset overflows"))?;
    let end = start
        .checked_add(rb)
        .ok_or_else(|| invalid("row offset overflows"))?;
    let bytes = w
        .data
        .get(start..end)
        .ok_or_else(|| invalid("row lies outside the tensor data"))?;
    dequant_row_bytes(w.dtype, bytes, out);
    Ok(())
}
