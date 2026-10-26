//! Memory-mapped tensor storage. The mapping is read-only; every tensor view is
//! re-validated against the mapped length with checked arithmetic.

use crate::error::EngineError;
use crate::gguf::{GgmlType, GgufFile, TensorInfo};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct TensorStore {
    map: Mmap,
    data_offset: u64,
    tensors: HashMap<String, TensorInfo>,
}

/// A borrowed 2-D view of a tensor: `rows` rows of `cols` contiguous elements,
/// stored as ggml blocks of `dtype` in `data`.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidModel(msg.into())
}

/// The only `unsafe` in the crate.
fn map_readonly(file: &File) -> std::io::Result<Mmap> {
    // SAFETY: the mapping is read-only and private to this process. The usual
    // hazard of `Mmap::map` is another process truncating or rewriting the file
    // while it is mapped, which would be undefined behaviour (typically SIGBUS)
    // on access. That cannot be ruled out from inside this process, so model
    // files must not be modified while the engine runs. Within that contract all
    // access goes through `TensorStore::view`, which bounds-checks every range
    // against `Mmap::len` first and only hands out `&[u8]` slices of the map.
    unsafe { Mmap::map(file) }
}

impl TensorStore {
    pub fn open(file: &GgufFile, path: &Path) -> Result<TensorStore, EngineError> {
        let f = File::open(path)
            .map_err(|e| EngineError::Internal(format!("cannot open {}: {e}", path.display())))?;
        let map = map_readonly(&f)
            .map_err(|e| EngineError::Internal(format!("cannot map {}: {e}", path.display())))?;
        if (map.len() as u64) < file.file_len {
            return Err(invalid(format!(
                "file is {} bytes but was parsed as {} bytes (truncated after parsing?)",
                map.len(),
                file.file_len
            )));
        }
        let tensors = file
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();
        Ok(TensorStore {
            map,
            data_offset: file.data_offset,
            tensors,
        })
    }

    pub fn view(&self, name: &str) -> Result<TensorView<'_>, EngineError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("no tensor named {name:?}")))?;
        let cols_u64 = *t
            .dims
            .first()
            .ok_or_else(|| invalid(format!("tensor {name:?} has no dimensions")))?;
        let cols = usize::try_from(cols_u64)
            .map_err(|_| invalid(format!("tensor {name:?}: column count too large")))?;
        let mut rows: usize = 1;
        for d in &t.dims[1..] {
            let d = usize::try_from(*d)
                .map_err(|_| invalid(format!("tensor {name:?}: dimension too large")))?;
            rows = rows
                .checked_mul(d)
                .ok_or_else(|| invalid(format!("tensor {name:?}: row count overflows")))?;
        }

        let start = self
            .data_offset
            .checked_add(t.offset)
            .ok_or_else(|| invalid(format!("tensor {name:?}: offset overflows")))?;
        let end = start
            .checked_add(t.n_bytes)
            .ok_or_else(|| invalid(format!("tensor {name:?}: range overflows")))?;
        if end > self.map.len() as u64 {
            return Err(invalid(format!(
                "tensor {name:?}: range {start}..{end} lies outside the {} byte file",
                self.map.len()
            )));
        }
        // Both fit in usize because `end <= map.len()`.
        let data = &self.map[start as usize..end as usize];

        // Unknown dtypes have no known layout; the size cannot be checked and
        // the kernels reject them.
        if t.dtype.block().is_some() {
            let row_bytes = crate::backend::row_bytes(t.dtype, cols)?;
            let expected = rows
                .checked_mul(row_bytes)
                .ok_or_else(|| invalid(format!("tensor {name:?}: size overflows")))?;
            if data.len() != expected {
                return Err(invalid(format!(
                    "tensor {name:?}: {} bytes recorded but {rows} rows x {row_bytes} bytes = {expected}",
                    data.len()
                )));
            }
        }
        Ok(TensorView {
            dtype: t.dtype,
            rows,
            cols,
            data,
        })
    }
}
