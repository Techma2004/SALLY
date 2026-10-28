//! Process memory readings and the per-request metrics record. Nothing here ever holds
//! request text, so logs built from it cannot leak prompts.

use crate::error::EngineError;
use crate::generate::GenStats;

/// Reads a `Key:   1234 kB` line from `/proc` style text and returns bytes.
fn status_kb(text: &str, key: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with(key))?;
    let rest = line.strip_prefix(key)?.strip_prefix(':')?;
    let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
    kb.checked_mul(1024)
}

fn proc_kb(path: &str, key: &str) -> Option<u64> {
    status_kb(&std::fs::read_to_string(path).ok()?, key)
}

/// Resident set size of this process (`VmRSS`), if readable.
pub fn rss_bytes() -> Option<u64> {
    proc_kb("/proc/self/status", "VmRSS")
}

/// Peak resident set size of this process (`VmHWM`), if readable.
pub fn peak_rss_bytes() -> Option<u64> {
    proc_kb("/proc/self/status", "VmHWM")
}

/// Memory the OS says can be given to programs without swapping (`MemAvailable`).
pub fn mem_available_bytes() -> Option<u64> {
    proc_kb("/proc/meminfo", "MemAvailable")
}

/// One request's measurements, logged as a single JSON line.
#[derive(Debug, Clone)]
pub struct Metrics {
    pub load_ms: f64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub first_token_ms: Option<f64>,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub peak_rss_bytes: Option<u64>,
    pub kv_bytes: usize,
    pub finish: String,
    /// Error code when the request failed.
    pub error: Option<String>,
}

impl Metrics {
    pub fn from_stats(stats: &GenStats, load_ms: f64, kv_bytes: usize) -> Metrics {
        Metrics {
            load_ms,
            prompt_tokens: stats.prompt_tokens,
            generated_tokens: stats.generated_tokens(),
            first_token_ms: stats.first_token_secs.map(|s| s * 1000.0),
            prefill_tok_s: stats.prefill_tokens_per_sec(),
            decode_tok_s: stats.decode_tokens_per_sec(),
            peak_rss_bytes: peak_rss_bytes(),
            kv_bytes,
            finish: stats.finish.as_str().to_string(),
            error: None,
        }
    }

    pub fn failed(e: &EngineError, load_ms: f64) -> Metrics {
        Metrics {
            load_ms,
            prompt_tokens: 0,
            generated_tokens: 0,
            first_token_ms: None,
            prefill_tok_s: 0.0,
            decode_tok_s: 0.0,
            peak_rss_bytes: peak_rss_bytes(),
            kv_bytes: 0,
            finish: "error".into(),
            error: Some(e.code().to_string()),
        }
    }

    /// Single-line JSON, no request text.
    pub fn log_json(&self) -> String {
        serde_json::json!({
            "event": "generation",
            "load_ms": self.load_ms,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "first_token_ms": self.first_token_ms,
            "prefill_tok_s": self.prefill_tok_s,
            "decode_tok_s": self.decode_tok_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "kv_bytes": self.kv_bytes,
            "finish": self.finish,
            "error": self.error,
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::status_kb;

    #[test]
    fn parses_kb_lines() {
        let t = "Name:\tx\nVmRSS:\t  1234 kB\nVmHWM:\t5 kB\n";
        assert_eq!(status_kb(t, "VmRSS"), Some(1234 * 1024));
        assert_eq!(status_kb(t, "VmHWM"), Some(5 * 1024));
        assert_eq!(status_kb(t, "VmSwap"), None);
    }

    #[test]
    fn rejects_garbage_and_overflow() {
        assert_eq!(status_kb("VmRSS: abc kB", "VmRSS"), None);
        assert_eq!(status_kb("VmRSS:", "VmRSS"), None);
        assert_eq!(status_kb(&format!("VmRSS: {} kB", u64::MAX), "VmRSS"), None);
    }
}
