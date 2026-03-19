use std::fmt;

use crate::spec::ThinkingConfig;

// ── Request errors ──────────────────────────────────────────────────

/// Errors that can occur when building a provider request.
#[derive(Debug, Clone)]
pub enum RequestError {
    /// The provider does not support the given thinking config.
    UnsupportedThinkingConfig {
        provider: &'static str,
        config: ThinkingConfig,
    },
    /// Response failed to deserialize.
    ResponseParse {
        detail: String,
    },
    /// Expected field missing in response.
    MissingField {
        field: &'static str,
    },
    /// Provider does not support this operation.
    Unsupported {
        provider: &'static str,
        operation: &'static str,
    },
    /// Request/response serialization failed.
    Serialization {
        detail: String,
    },
    /// Empty response (no choices/candidates).
    EmptyResponse,
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestError::UnsupportedThinkingConfig { provider, config } => {
                write!(f, "{provider}: unsupported thinking config {config:?}")
            }
            RequestError::ResponseParse { detail } => {
                write!(f, "response parse error: {detail}")
            }
            RequestError::MissingField { field } => {
                write!(f, "missing field '{field}' in response")
            }
            RequestError::Unsupported { provider, operation } => {
                write!(f, "{provider}: {operation} not supported")
            }
            RequestError::Serialization { detail } => {
                write!(f, "serialization error: {detail}")
            }
            RequestError::EmptyResponse => {
                write!(f, "empty response: no choices/candidates")
            }
        }
    }
}

impl std::error::Error for RequestError {}

// ── Transport ───────────────────────────────────────────────────────

pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: serde_json::Value,
}

/// Raw HTTP fetch — implementors only handle transport.
#[trait_variant::make(Send)]
pub trait Fetch: Sync {
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String>;
}

impl<F> Fetch for std::sync::Arc<F>
where
    F: Fetch,
{
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
        (**self).fetch(request).await
    }
}
