//! GGUF metadata key/value parsing with hard bounds on every length.

use super::reader::{invalid, Reader};
use crate::error::EngineError;
use std::collections::BTreeMap;
use std::io::Read;

pub const MAX_KEY_LEN: u64 = 65_535;
pub const MAX_STRING_LEN: u64 = 16 * 1024 * 1024;
/// Arrays may contain arrays, but not arrays of arrays of arrays.
pub const MAX_ARRAY_DEPTH: u32 = 2;
/// Total array elements allowed across the whole metadata section
/// (a 150k-token vocabulary needs a few hundred thousand).
pub const MAX_TOTAL_ARRAY_ELEMENTS: u64 = 4 * 1024 * 1024;
pub const MAX_KV_COUNT: u64 = 1 << 20;
/// Smallest possible encoded KV: 8 (key len) + 4 (type) + 1 (smallest value).
pub const MIN_KV_BYTES: u64 = 13;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    entries: BTreeMap<String, Value>,
}

impl Metadata {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Unsigned integer value that fits in `u32` (u8/u16/u32 storage accepted).
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        u32::try_from(self.get_u64(key)?).ok()
    }

    /// Unsigned integer value (u8/u16/u32/u64 storage accepted).
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.get(key)? {
            Value::U8(v) => Some(u64::from(*v)),
            Value::U16(v) => Some(u64::from(*v)),
            Value::U32(v) => Some(u64::from(*v)),
            Value::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            Value::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Array whose elements are all integers (any width, signed or unsigned).
    pub fn get_int_array(&self, key: &str) -> Option<Vec<i64>> {
        match self.get(key)? {
            Value::Array(items) => items
                .iter()
                .map(|v| match v {
                    Value::U8(x) => Some(i64::from(*x)),
                    Value::I8(x) => Some(i64::from(*x)),
                    Value::U16(x) => Some(i64::from(*x)),
                    Value::I16(x) => Some(i64::from(*x)),
                    Value::U32(x) => Some(i64::from(*x)),
                    Value::I32(x) => Some(i64::from(*x)),
                    Value::I64(x) => Some(*x),
                    Value::U64(x) => i64::try_from(*x).ok(),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    /// Array whose elements are all strings.
    pub fn get_str_array(&self, key: &str) -> Option<Vec<&str>> {
        match self.get(key)? {
            Value::Array(items) => items
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }
}

/// Smallest encoded size of one value of this type id, or `None` for unknown ids.
fn min_value_bytes(type_id: u32) -> Option<u64> {
    match type_id {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        8 => Some(8),  // u64 length prefix
        9 => Some(12), // u32 element type + u64 length
        10..=12 => Some(8),
        _ => None,
    }
}

struct Budget {
    elements_left: u64,
}

fn read_value<R: Read>(
    r: &mut Reader<R>,
    type_id: u32,
    depth: u32,
    budget: &mut Budget,
) -> Result<Value, EngineError> {
    Ok(match type_id {
        0 => Value::U8(r.u8()?),
        1 => Value::I8(r.i8()?),
        2 => Value::U16(r.u16()?),
        3 => Value::I16(r.i16()?),
        4 => Value::U32(r.u32()?),
        5 => Value::I32(r.i32()?),
        6 => Value::F32(r.f32()?),
        7 => match r.u8()? {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            b => return Err(invalid(format!("invalid bool byte {b}"))),
        },
        8 => Value::Str(r.string(MAX_STRING_LEN, "metadata string")?),
        9 => read_array(r, depth, budget)?,
        10 => Value::U64(r.u64()?),
        11 => Value::I64(r.i64()?),
        12 => Value::F64(r.f64()?),
        other => return Err(invalid(format!("unknown metadata value type {other}"))),
    })
}

fn read_array<R: Read>(
    r: &mut Reader<R>,
    depth: u32,
    budget: &mut Budget,
) -> Result<Value, EngineError> {
    let depth = depth
        .checked_add(1)
        .ok_or_else(|| invalid("array nesting overflow"))?;
    if depth > MAX_ARRAY_DEPTH {
        return Err(invalid(format!(
            "array nesting deeper than {MAX_ARRAY_DEPTH}"
        )));
    }
    let elem_type = r.u32()?;
    let len = r.u64()?;
    let min = min_value_bytes(elem_type)
        .ok_or_else(|| invalid(format!("unknown array element type {elem_type}")))?;
    // The elements must physically fit in the rest of the file.
    if len > r.remaining() / min {
        return Err(invalid(format!(
            "array length {len} exceeds remaining file size {}",
            r.remaining()
        )));
    }
    if len > budget.elements_left {
        return Err(invalid(format!(
            "metadata arrays exceed the limit of {MAX_TOTAL_ARRAY_ELEMENTS} total elements"
        )));
    }
    budget.elements_left = budget
        .elements_left
        .checked_sub(len)
        .ok_or_else(|| invalid("array element budget underflow"))?;
    let cap = usize::try_from(len).map_err(|_| invalid("array length too large"))?;
    let mut items = Vec::with_capacity(cap);
    for _ in 0..cap {
        items.push(read_value(r, elem_type, depth, budget)?);
    }
    Ok(Value::Array(items))
}

pub(crate) fn read_metadata<R: Read>(
    r: &mut Reader<R>,
    count: u64,
) -> Result<Metadata, EngineError> {
    if count > MAX_KV_COUNT {
        return Err(invalid(format!(
            "metadata count {count} exceeds limit of {MAX_KV_COUNT}"
        )));
    }
    let needed = count
        .checked_mul(MIN_KV_BYTES)
        .ok_or_else(|| invalid("metadata count overflow"))?;
    if needed > r.remaining() {
        return Err(invalid(format!(
            "metadata count {count} cannot fit in the remaining {} bytes",
            r.remaining()
        )));
    }
    let mut budget = Budget {
        elements_left: MAX_TOTAL_ARRAY_ELEMENTS,
    };
    let mut entries = BTreeMap::new();
    for _ in 0..count {
        let key = r.string(MAX_KEY_LEN, "metadata key")?;
        if key.is_empty() {
            return Err(invalid("empty metadata key"));
        }
        let type_id = r.u32()?;
        let value = read_value(r, type_id, 0, &mut budget)?;
        if entries.insert(key.clone(), value).is_some() {
            return Err(invalid(format!("duplicate metadata key {key:?}")));
        }
    }
    Ok(Metadata { entries })
}
