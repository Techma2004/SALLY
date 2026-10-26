use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    UnsupportedArchitecture(String),
    InvalidModel(String),
    ResourceExhausted(String),
    Cancelled,
    Internal(String),
}

impl EngineError {
    pub fn code(&self) -> &'static str {
        match self {
            EngineError::UnsupportedArchitecture(_) => "unsupported_architecture",
            EngineError::InvalidModel(_) => "invalid_model",
            EngineError::ResourceExhausted(_) => "resource_exhausted",
            EngineError::Cancelled => "cancelled",
            EngineError::Internal(_) => "internal",
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::UnsupportedArchitecture(m) => write!(f, "unsupported architecture: {m}"),
            EngineError::InvalidModel(m) => write!(f, "invalid model: {m}"),
            EngineError::ResourceExhausted(m) => write!(f, "resource exhausted: {m}"),
            EngineError::Cancelled => write!(f, "cancelled"),
            EngineError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}
