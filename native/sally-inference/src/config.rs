use crate::error::EngineError;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub model_path: PathBuf,
    pub context_len: usize,
    pub max_output_tokens: usize,
    pub max_body_bytes: usize,
    pub queue_len: usize,
    pub request_timeout_secs: u64,
    pub bind: SocketAddr,
    pub memory_budget_mb: usize,
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model_path: PathBuf::new(),
            context_len: 2048,
            max_output_tokens: 512,
            max_body_bytes: 1_048_576,
            queue_len: 2,
            request_timeout_secs: 120,
            bind: SocketAddr::from(([127, 0, 0, 1], 8088)),
            memory_budget_mb: 2048,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .max(1),
        }
    }
}

fn parse<T: FromStr>(key: &str, value: &str) -> Result<T, EngineError> {
    value
        .trim()
        .parse::<T>()
        .map_err(|_| EngineError::Internal(format!("invalid value for {key}: {value:?}")))
}

impl Config {
    pub fn load(
        path: Option<&Path>,
        overrides: &[(String, String)],
    ) -> Result<Config, EngineError> {
        let mut cfg = match path {
            Some(p) => {
                let text = std::fs::read_to_string(p).map_err(|e| {
                    EngineError::Internal(format!("cannot read config {}: {e}", p.display()))
                })?;
                serde_json::from_str::<Config>(&text).map_err(|e| {
                    EngineError::Internal(format!("invalid config {}: {e}", p.display()))
                })?
            }
            None => Config::default(),
        };
        for (key, value) in overrides {
            cfg.apply(key, value)?;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn apply(&mut self, key: &str, value: &str) -> Result<(), EngineError> {
        match key {
            "model_path" => self.model_path = PathBuf::from(value),
            "context_len" => self.context_len = parse(key, value)?,
            "max_output_tokens" => self.max_output_tokens = parse(key, value)?,
            "max_body_bytes" => self.max_body_bytes = parse(key, value)?,
            "queue_len" => self.queue_len = parse(key, value)?,
            "request_timeout_secs" => self.request_timeout_secs = parse(key, value)?,
            "bind" => self.bind = parse(key, value)?,
            "memory_budget_mb" => self.memory_budget_mb = parse(key, value)?,
            "threads" => self.threads = parse(key, value)?,
            _ => return Err(EngineError::Internal(format!("unknown config key: {key}"))),
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        let limits: [(&str, u64); 7] = [
            ("context_len", self.context_len as u64),
            ("max_output_tokens", self.max_output_tokens as u64),
            ("max_body_bytes", self.max_body_bytes as u64),
            ("queue_len", self.queue_len as u64),
            ("request_timeout_secs", self.request_timeout_secs),
            ("memory_budget_mb", self.memory_budget_mb as u64),
            ("threads", self.threads as u64),
        ];
        for (name, v) in limits {
            if v == 0 {
                return Err(EngineError::ResourceExhausted(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bind_is_loopback() {
        assert!(Config::default().bind.ip().is_loopback());
    }

    #[test]
    fn rejects_zero_context() {
        let err = Config {
            context_len: 0,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert_eq!(err.code(), "resource_exhausted");
    }

    #[test]
    fn override_applies() {
        let c = Config::load(None, &[("context_len".into(), "512".into())]).unwrap();
        assert_eq!(c.context_len, 512);
    }

    #[test]
    fn unknown_override_key_errors() {
        let err = Config::load(None, &[("nope".into(), "1".into())]).unwrap_err();
        assert_eq!(err.code(), "internal");
    }

    fn temp_file(tag: &str, contents: Option<&str>) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sally-inference-cfg-{}-{tag}.json",
            std::process::id()
        ));
        match contents {
            Some(c) => std::fs::write(&path, c).unwrap(),
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
        path
    }

    #[test]
    fn load_reads_partial_json_file() {
        let path = temp_file("partial", Some(r#"{"context_len": 256}"#));
        let c = Config::load(Some(&path), &[]).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(c.context_len, 256);
        let d = Config::default();
        assert_eq!(c.max_output_tokens, d.max_output_tokens);
        assert_eq!(c.bind, d.bind);
        assert_eq!(c.queue_len, d.queue_len);
    }

    #[test]
    fn load_rejects_unknown_json_field() {
        let path = temp_file("unknown", Some(r#"{"nope": 1}"#));
        let err = Config::load(Some(&path), &[]).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.code(), "internal");
    }

    #[test]
    fn load_missing_file_errors() {
        let path = temp_file("missing", None);
        let err = Config::load(Some(&path), &[]).unwrap_err();
        assert_eq!(err.code(), "internal");
    }
}
