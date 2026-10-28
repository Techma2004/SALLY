//! Byte budget for the engine's own working memory (KV cache, scratch buffers, request
//! state). Model weights are memory-mapped and file-backed, so they are reported (see
//! [`crate::telemetry`]) rather than charged: the OS pages them in and out, which is a
//! speed cost, not an allocation this budget can bound.

use crate::architecture::{Architecture, KvCache};
use crate::error::EngineError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct Inner {
    limit: usize,
    used: AtomicUsize,
}

/// Cloneable handle to one shared budget; safe to use from several threads.
#[derive(Clone, Debug)]
pub struct Budget(Arc<Inner>);

/// Bytes held against a [`Budget`]; returned to it on drop.
#[derive(Debug)]
pub struct Reservation {
    budget: Budget,
    bytes: usize,
    what: &'static str,
}

/// `1536` -> `"1.5 KiB"`; below 1 KiB it prints plain bytes.
pub fn fmt_bytes(n: usize) -> String {
    const UNITS: [(&str, f64); 3] = [
        ("GiB", 1024.0 * 1024.0 * 1024.0),
        ("MiB", 1024.0 * 1024.0),
        ("KiB", 1024.0),
    ];
    for (name, size) in UNITS {
        if n as f64 >= size {
            return format!("{:.1} {name}", n as f64 / size);
        }
    }
    format!("{n} B")
}

impl Budget {
    pub fn new(limit_bytes: usize) -> Budget {
        Budget(Arc::new(Inner {
            limit: limit_bytes,
            used: AtomicUsize::new(0),
        }))
    }

    pub fn limit(&self) -> usize {
        self.0.limit
    }

    pub fn used(&self) -> usize {
        self.0.used.load(Ordering::SeqCst)
    }

    pub fn remaining(&self) -> usize {
        self.0.limit.saturating_sub(self.used())
    }

    /// Claims `bytes` for `what`, or fails with `resource_exhausted` naming the resource
    /// and the numbers. Nothing is claimed on failure.
    pub fn reserve(&self, what: &'static str, bytes: usize) -> Result<Reservation, EngineError> {
        let mut cur = self.0.used.load(Ordering::SeqCst);
        loop {
            let next = match cur.checked_add(bytes) {
                Some(n) if n <= self.0.limit => n,
                _ => {
                    return Err(EngineError::ResourceExhausted(format!(
                        "{what} needs {} but only {} of the {} memory budget remain",
                        fmt_bytes(bytes),
                        fmt_bytes(self.0.limit.saturating_sub(cur)),
                        fmt_bytes(self.0.limit),
                    )))
                }
            };
            match self
                .0
                .used
                .compare_exchange_weak(cur, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => {
                    return Ok(Reservation {
                        budget: self.clone(),
                        bytes,
                        what,
                    })
                }
                Err(actual) => cur = actual,
            }
        }
    }
}

impl Reservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn what(&self) -> &'static str {
        self.what
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.0.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

/// Allocates a KV cache for `ctx` positions only if the budget can hold it. The reservation
/// must live as long as the cache. A refusal says how much context would fit.
pub fn open_cache(
    model: &dyn Architecture,
    ctx: usize,
    budget: &Budget,
) -> Result<(KvCache, Reservation), EngineError> {
    let need = model.kv_bytes(ctx);
    let reservation = budget.reserve("KV cache", need).map_err(|_| {
        let per_pos = model.kv_bytes(1).max(1);
        EngineError::ResourceExhausted(format!(
            "a {ctx}-token KV cache needs {} but only {} of the {} memory budget remain; \
             at most {} tokens of context fit (lower --context or raise the budget)",
            fmt_bytes(need),
            fmt_bytes(budget.remaining()),
            fmt_bytes(budget.limit()),
            budget.remaining() / per_pos,
        ))
    })?;
    let cache = model.new_cache(ctx)?;
    Ok((cache, reservation))
}
