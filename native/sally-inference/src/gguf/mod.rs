//! GGUF reader: parsing and validation only. Accepts versions 2 and 3
//! (little endian). Never loads tensor data; the file is read through a
//! bounds-tracked cursor, so multi-GB models are not pulled into memory.

mod metadata;
mod reader;
mod tensor_index;

pub use metadata::{Metadata, Value};
pub use tensor_index::{GgmlType, TensorInfo};

use crate::error::EngineError;
use reader::{invalid, Reader};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the tensor data section starts.
    pub data_offset: u64,
    pub alignment: u64,
    pub file_len: u64,
    /// Sum of all tensor sizes (checked).
    pub total_tensor_bytes: u64,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile, EngineError> {
        let file = File::open(path)
            .map_err(|e| EngineError::Internal(format!("cannot open {}: {e}", path.display())))?;
        let file_len = file
            .metadata()
            .map_err(|e| EngineError::Internal(format!("cannot stat {}: {e}", path.display())))?
            .len();
        GgufFile::from_reader(BufReader::with_capacity(64 * 1024, file), file_len)
    }

    /// Parses from any reader whose total length is `file_len`.
    pub fn from_reader<R: Read>(reader: R, file_len: u64) -> Result<GgufFile, EngineError> {
        let mut r = Reader::new(reader, file_len);
        let mut magic = [0u8; 4];
        for b in magic.iter_mut() {
            *b = r.u8()?;
        }
        if &magic != b"GGUF" {
            return Err(invalid("bad magic (not a GGUF file)"));
        }
        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(invalid(format!(
                "unsupported GGUF version {version} (only 2 and 3, little endian)"
            )));
        }
        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;
        let metadata = metadata::read_metadata(&mut r, kv_count)?;
        let tensors = tensor_index::read_tensor_infos(&mut r, tensor_count)?;

        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(_) => metadata
                .get_u64("general.alignment")
                .ok_or_else(|| invalid("general.alignment is not an unsigned integer"))?,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(invalid(format!(
                "general.alignment {alignment} is not a non-zero power of two"
            )));
        }
        let data_offset = align_up(r.pos(), alignment)?;
        // A file with no tensor bytes may omit the final padding; any tensor
        // with data then fails the range check below against a zero-length section.
        let data_len = file_len.saturating_sub(data_offset);

        let mut total: u64 = 0;
        for t in &tensors {
            if !t.offset.is_multiple_of(alignment) {
                return Err(invalid(format!(
                    "tensor {:?}: offset {} is not a multiple of alignment {alignment}",
                    t.name, t.offset
                )));
            }
            let end = t
                .offset
                .checked_add(t.n_bytes)
                .ok_or_else(|| invalid(format!("tensor {:?}: offset + size overflows", t.name)))?;
            if end > data_len {
                return Err(invalid(format!(
                    "tensor {:?}: range {}..{end} extends past the data section ({data_len} bytes)",
                    t.name, t.offset
                )));
            }
            total = total
                .checked_add(t.n_bytes)
                .ok_or_else(|| invalid("total tensor size overflows"))?;
        }

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
            file_len,
            total_tensor_bytes: total,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get_str("general.architecture")
    }

    /// Human-readable summary for `inspect`. Untrusted strings are printed with
    /// Debug escaping so control characters cannot reach the terminal.
    pub fn report(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let arch = self.architecture();
        let _ = writeln!(s, "GGUF file");
        let _ = writeln!(s, "  architecture: {}", opt_str(arch));
        let _ = writeln!(s, "  version: {}", self.version);
        let _ = writeln!(s, "  file size: {} bytes", self.file_len);
        let _ = writeln!(s, "  tensors: {}", self.tensors.len());
        let _ = writeln!(s, "  total tensor bytes: {}", self.total_tensor_bytes);
        let _ = writeln!(
            s,
            "  data offset: {} (alignment {})",
            self.data_offset, self.alignment
        );
        let mut hist: BTreeMap<GgmlType, (u64, u64)> = BTreeMap::new();
        for t in &self.tensors {
            let e = hist.entry(t.dtype).or_insert((0, 0));
            // Sums are bounded by the checked `total_tensor_bytes`; saturate for display only.
            e.0 = e.0.saturating_add(1);
            e.1 = e.1.saturating_add(t.n_bytes);
        }
        let _ = writeln!(s, "  tensor types:");
        for (ty, (n, bytes)) in &hist {
            let _ = writeln!(s, "    {ty}: {n} tensors, {bytes} bytes");
        }
        let _ = writeln!(s, "  metadata keys: {}", self.metadata.len());
        if let Some(name) = self.metadata.get_str("general.name") {
            let _ = writeln!(s, "  name: {name:?}");
        }
        if let Some(arch) = arch {
            for suffix in ["context_length", "block_count", "embedding_length"] {
                let key = format!("{arch}.{suffix}");
                match self.metadata.get_u64(&key) {
                    Some(v) => {
                        let _ = writeln!(s, "  {key}: {v}");
                    }
                    None => {
                        let _ = writeln!(s, "  {key}: (not set)");
                    }
                }
            }
        }
        let _ = writeln!(
            s,
            "  tokenizer.ggml.model: {}",
            opt_str(self.metadata.get_str("tokenizer.ggml.model"))
        );
        s
    }
}

fn opt_str(v: Option<&str>) -> String {
    match v {
        Some(s) => format!("{s:?}"),
        None => "(not set)".to_string(),
    }
}

fn align_up(pos: u64, alignment: u64) -> Result<u64, EngineError> {
    let mask = alignment
        .checked_sub(1)
        .ok_or_else(|| invalid("zero alignment"))?;
    let bumped = pos
        .checked_add(mask)
        .ok_or_else(|| invalid("data offset overflow"))?;
    Ok(bumped & !mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn header(version: u32, tensors: u64, kvs: u64) -> Vec<u8> {
        let mut v = b"GGUF".to_vec();
        v.extend_from_slice(&version.to_le_bytes());
        v.extend_from_slice(&tensors.to_le_bytes());
        v.extend_from_slice(&kvs.to_le_bytes());
        v
    }

    fn kv_array_key(v: &mut Vec<u8>) {
        v.extend_from_slice(&1u64.to_le_bytes());
        v.push(b'k');
        v.extend_from_slice(&9u32.to_le_bytes());
    }

    fn parse(b: &[u8]) -> Result<GgufFile, EngineError> {
        GgufFile::from_reader(Cursor::new(b), b.len() as u64)
    }

    #[test]
    fn version_2_accepted_empty_file() {
        let g = parse(&header(2, 0, 0)).unwrap();
        assert_eq!(g.version, 2);
        assert_eq!(g.data_offset, 32);
    }

    #[test]
    fn array_depth_two_ok_three_rejected() {
        // depth 2: array of arrays of u8 (one inner array, empty)
        let mut ok = header(3, 0, 1);
        kv_array_key(&mut ok);
        ok.extend_from_slice(&9u32.to_le_bytes()); // elem type = array
        ok.extend_from_slice(&1u64.to_le_bytes()); // one element
        ok.extend_from_slice(&0u32.to_le_bytes()); // inner elem type u8
        ok.extend_from_slice(&0u64.to_le_bytes()); // inner len 0
        assert!(parse(&ok).is_ok());

        // depth 3: array of arrays of arrays
        let mut bad = header(3, 0, 1);
        kv_array_key(&mut bad);
        bad.extend_from_slice(&9u32.to_le_bytes());
        bad.extend_from_slice(&1u64.to_le_bytes());
        bad.extend_from_slice(&9u32.to_le_bytes());
        bad.extend_from_slice(&1u64.to_le_bytes());
        bad.extend_from_slice(&0u32.to_le_bytes());
        bad.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(parse(&bad).unwrap_err().code(), "invalid_model");
    }

    #[test]
    fn huge_array_length_rejected() {
        let mut b = header(3, 0, 1);
        kv_array_key(&mut b);
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(parse(&b).unwrap_err().code(), "invalid_model");
    }
}
