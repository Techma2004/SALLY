//! CPU kernels: scalar, correctness first. See docs/backend-capability-matrix.md.

mod dequant;
mod matmul;

pub use dequant::{dequant_row, f16_to_f32};
pub use matmul::matvec;

pub(crate) use dequant::row_bytes;
