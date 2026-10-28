//! Cooperative cancellation shared between a request handler and the generation loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Cheap to clone; every clone observes the same flag.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> CancelToken {
        CancelToken::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
