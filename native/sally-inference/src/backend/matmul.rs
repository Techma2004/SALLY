//! Matrix-vector product over quantized rows, parallel across rows.

use super::dequant::{dequant_row_bytes, is_supported, row_bytes, unsupported};
use crate::error::EngineError;
use crate::tensor_store::TensorView;

fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidModel(msg.into())
}

/// Computes `out[r] = dot(row r of w, x)`. Each row is dequantized into a
/// scratch buffer and dotted in f32 in a fixed order, so the result does not
/// depend on `threads` (0 is treated as 1).
pub fn matvec(
    w: &TensorView<'_>,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
) -> Result<(), EngineError> {
    if !is_supported(w.dtype) {
        return Err(unsupported(w.dtype));
    }
    if x.len() != w.cols {
        return Err(invalid(format!(
            "input length {} does not match {} columns",
            x.len(),
            w.cols
        )));
    }
    if out.len() != w.rows {
        return Err(invalid(format!(
            "output length {} does not match {} rows",
            out.len(),
            w.rows
        )));
    }
    let rb = row_bytes(w.dtype, w.cols)?;
    let total = w
        .rows
        .checked_mul(rb)
        .ok_or_else(|| invalid("tensor size overflows"))?;
    if w.data.len() != total {
        return Err(invalid(format!(
            "tensor data is {} bytes, expected {total}",
            w.data.len()
        )));
    }
    if w.rows == 0 {
        return Ok(());
    }
    let threads = threads.clamp(1, w.rows);
    if threads == 1 {
        compute_rows(w, rb, x, 0, out);
        return Ok(());
    }
    let chunk = w.rows.div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = out
            .chunks_mut(chunk)
            .enumerate()
            .map(|(i, o)| s.spawn(move || compute_rows(w, rb, x, i * chunk, o)))
            .collect();
        for h in handles {
            h.join()
                .map_err(|_| EngineError::Internal("matvec worker panicked".into()))?;
        }
        Ok(())
    })
}

/// Preconditions (checked by `matvec`): `w.data` holds `w.rows` full rows of
/// `rb` bytes, `x.len() == w.cols`, `first + out.len() <= w.rows`.
fn compute_rows(w: &TensorView<'_>, rb: usize, x: &[f32], first: usize, out: &mut [f32]) {
    let mut scratch = vec![0.0f32; w.cols];
    for (i, o) in out.iter_mut().enumerate() {
        let start = (first + i) * rb;
        dequant_row_bytes(w.dtype, &w.data[start..start + rb], &mut scratch);
        let mut acc = 0.0f32;
        for (a, b) in scratch.iter().zip(x) {
            acc += a * b;
        }
        *o = acc;
    }
}
