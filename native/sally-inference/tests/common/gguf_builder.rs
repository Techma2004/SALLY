//! Builds synthetic GGUF v3 files for tests. Not part of the shipped crate.
#![allow(dead_code)]

use sally_inference::gguf::GgmlType;

pub const T_U32: u32 = 4;
pub const T_STRING: u32 = 8;
pub const T_ARRAY: u32 = 9;

struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
    /// `Some` = explicit offset, no data is written (for malformed files).
    explicit_offset: Option<u64>,
    data: Vec<u8>,
}

pub struct GgufBuilder {
    magic: [u8; 4],
    version: u32,
    alignment: u64,
    kvs: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<TensorEntry>,
    tensor_count: Option<u64>,
    kv_count: Option<u64>,
}

pub fn gguf_string(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

impl GgufBuilder {
    pub fn new() -> Self {
        GgufBuilder {
            magic: *b"GGUF",
            version: 3,
            alignment: 32,
            kvs: Vec::new(),
            tensors: Vec::new(),
            tensor_count: None,
            kv_count: None,
        }
    }

    pub fn magic(mut self, m: [u8; 4]) -> Self {
        self.magic = m;
        self
    }

    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Sets the padding alignment and writes `general.alignment`.
    pub fn alignment(mut self, a: u32) -> Self {
        self.alignment = a as u64;
        self.kv_u32("general.alignment", a)
    }

    pub fn tensor_count(mut self, n: u64) -> Self {
        self.tensor_count = Some(n);
        self
    }

    pub fn kv_count(mut self, n: u64) -> Self {
        self.kv_count = Some(n);
        self
    }

    pub fn kv_raw(mut self, key: &str, type_id: u32, value: Vec<u8>) -> Self {
        self.kvs.push((key.to_string(), type_id, value));
        self
    }

    pub fn kv_str(self, key: &str, value: &str) -> Self {
        self.kv_raw(key, T_STRING, gguf_string(value))
    }

    pub fn kv_u32(self, key: &str, value: u32) -> Self {
        self.kv_raw(key, T_U32, value.to_le_bytes().to_vec())
    }

    pub fn kv_str_array(self, key: &str, items: &[String]) -> Self {
        let mut v = T_STRING.to_le_bytes().to_vec();
        v.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for s in items {
            v.extend_from_slice(&gguf_string(s));
        }
        self.kv_raw(key, T_ARRAY, v)
    }

    pub fn kv_i32_array(self, key: &str, items: &[i32]) -> Self {
        let mut v = 5u32.to_le_bytes().to_vec();
        v.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for x in items {
            v.extend_from_slice(&x.to_le_bytes());
        }
        self.kv_raw(key, T_ARRAY, v)
    }

    pub fn tensor(mut self, name: &str, dims: &[u64], ty: GgmlType, bytes: Vec<u8>) -> Self {
        self.tensors.push(TensorEntry {
            name: name.to_string(),
            dims: dims.to_vec(),
            type_id: ty.id(),
            explicit_offset: None,
            data: bytes,
        });
        self
    }

    /// Tensor descriptor with arbitrary dims/type/offset and no data.
    pub fn tensor_raw(mut self, name: &str, dims: &[u64], type_id: u32, offset: u64) -> Self {
        self.tensors.push(TensorEntry {
            name: name.to_string(),
            dims: dims.to_vec(),
            type_id,
            explicit_offset: Some(offset),
            data: Vec::new(),
        });
        self
    }

    fn align_up(&self, n: usize) -> usize {
        let a = self.alignment as usize;
        n.div_ceil(a) * a
    }

    pub fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        let tc = self.tensor_count.unwrap_or(self.tensors.len() as u64);
        let kc = self.kv_count.unwrap_or(self.kvs.len() as u64);
        out.extend_from_slice(&tc.to_le_bytes());
        out.extend_from_slice(&kc.to_le_bytes());
        for (k, t, v) in &self.kvs {
            out.extend_from_slice(&gguf_string(k));
            out.extend_from_slice(&t.to_le_bytes());
            out.extend_from_slice(v);
        }
        let mut data = Vec::new();
        for t in &self.tensors {
            let offset = match t.explicit_offset {
                Some(o) => o,
                None => {
                    let o = self.align_up(data.len());
                    data.resize(o, 0);
                    data.extend_from_slice(&t.data);
                    o as u64
                }
            };
            out.extend_from_slice(&gguf_string(&t.name));
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.type_id.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }
        let padded = self.align_up(out.len());
        out.resize(padded, 0);
        out.extend_from_slice(&data);
        out
    }
}
