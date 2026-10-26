//! Bounds-tracked reader over any `Read`. Every read is checked against the
//! known total length first, so counts and lengths read from the file can be
//! validated against the bytes that actually remain before anything is allocated.

use crate::error::EngineError;
use std::io::{ErrorKind, Read};

pub(crate) fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidModel(msg.into())
}

pub(crate) struct Reader<R: Read> {
    inner: R,
    pos: u64,
    len: u64,
}

impl<R: Read> Reader<R> {
    pub(crate) fn new(inner: R, len: u64) -> Self {
        Reader { inner, pos: 0, len }
    }

    pub(crate) fn pos(&self) -> u64 {
        self.pos
    }

    /// Bytes left before the end of the file. `pos <= len` is an invariant of `fill`.
    pub(crate) fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn fill(&mut self, buf: &mut [u8]) -> Result<(), EngineError> {
        let n = buf.len() as u64;
        if n > self.remaining() {
            return Err(invalid(format!(
                "unexpected end of file at byte {} (need {n} more)",
                self.pos
            )));
        }
        self.inner.read_exact(buf).map_err(|e| {
            if e.kind() == ErrorKind::UnexpectedEof {
                invalid("unexpected end of file")
            } else {
                EngineError::Internal(format!("read error: {e}"))
            }
        })?;
        self.pos = self
            .pos
            .checked_add(n)
            .ok_or_else(|| invalid("file offset overflow"))?;
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], EngineError> {
        let mut b = [0u8; N];
        self.fill(&mut b)?;
        Ok(b)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.array::<1>()?[0])
    }
    pub(crate) fn i8(&mut self) -> Result<i8, EngineError> {
        Ok(i8::from_le_bytes(self.array()?))
    }
    pub(crate) fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    pub(crate) fn i16(&mut self) -> Result<i16, EngineError> {
        Ok(i16::from_le_bytes(self.array()?))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    pub(crate) fn i32(&mut self) -> Result<i32, EngineError> {
        Ok(i32::from_le_bytes(self.array()?))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    pub(crate) fn i64(&mut self) -> Result<i64, EngineError> {
        Ok(i64::from_le_bytes(self.array()?))
    }
    pub(crate) fn f32(&mut self) -> Result<f32, EngineError> {
        Ok(f32::from_le_bytes(self.array()?))
    }
    pub(crate) fn f64(&mut self) -> Result<f64, EngineError> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    /// Reads a length-prefixed UTF-8 string. The length is checked against
    /// `max_len` and the remaining file bytes before the buffer is allocated.
    pub(crate) fn string(&mut self, max_len: u64, what: &str) -> Result<String, EngineError> {
        let len = self.u64()?;
        if len > max_len {
            return Err(invalid(format!(
                "{what} length {len} exceeds limit of {max_len} bytes"
            )));
        }
        if len > self.remaining() {
            return Err(invalid(format!(
                "{what} length {len} exceeds remaining file size {}",
                self.remaining()
            )));
        }
        let n = usize::try_from(len).map_err(|_| invalid(format!("{what} length too large")))?;
        let mut buf = vec![0u8; n];
        self.fill(&mut buf)?;
        String::from_utf8(buf).map_err(|_| invalid(format!("{what} is not valid UTF-8")))
    }
}
