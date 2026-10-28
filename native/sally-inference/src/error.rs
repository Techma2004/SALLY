use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    UnsupportedArchitecture(String),
    InvalidModel(String),
    ResourceExhausted(String),
    Cancelled,
    /// Bad client input (e.g. an unknown chat role), not a model problem.
    InvalidRequest(String),
    Internal(String),
}

impl EngineError {
    pub fn code(&self) -> &'static str {
        match self {
            EngineError::UnsupportedArchitecture(_) => "unsupported_architecture",
            EngineError::InvalidModel(_) => "invalid_model",
            EngineError::ResourceExhausted(_) => "resource_exhausted",
            EngineError::Cancelled => "cancelled",
            EngineError::InvalidRequest(_) => "invalid_request",
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
            EngineError::InvalidRequest(m) => write!(f, "invalid request: {m}"),
            EngineError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_mapping_is_stable() {
        let cases = [
            (
                EngineError::UnsupportedArchitecture("x".into()),
                "unsupported_architecture",
            ),
            (EngineError::InvalidModel("x".into()), "invalid_model"),
            (
                EngineError::ResourceExhausted("x".into()),
                "resource_exhausted",
            ),
            (EngineError::Cancelled, "cancelled"),
            (EngineError::InvalidRequest("x".into()), "invalid_request"),
            (EngineError::Internal("x".into()), "internal"),
        ];
        for (e, code) in cases {
            assert_eq!(e.code(), code);
        }
        assert_eq!(
            EngineError::InvalidRequest("bad role".into()).to_string(),
            "invalid request: bad role"
        );
    }
}
