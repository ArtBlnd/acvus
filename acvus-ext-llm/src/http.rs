//! HTTP transport abstraction for LLM providers.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Errors that can occur during provider request/response handling.
#[derive(Debug, Clone)]
pub enum RequestError {
    /// Response failed to deserialize.
    ResponseParse { detail: String },
    /// Expected field missing in response.
    MissingField { field: &'static str },
    /// Empty response (no choices/candidates).
    EmptyResponse,
    /// Request serialization failed.
    Serialization { detail: String },
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestError::ResponseParse { detail } => {
                write!(f, "response parse error: {detail}")
            }
            RequestError::MissingField { field } => {
                write!(f, "missing field '{field}' in response")
            }
            RequestError::EmptyResponse => {
                write!(f, "empty response: no choices/candidates")
            }
            RequestError::Serialization { detail } => {
                write!(f, "serialization error: {detail}")
            }
        }
    }
}

impl std::error::Error for RequestError {}

/// An HTTP request ready to be sent.
pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: serde_json::Value,
}

/// Raw HTTP fetch - implementors only handle transport.
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

/// A `Fetch` behind one concrete type: what a provider's handler holds as
/// its state.
#[derive(Clone)]
pub struct FetchClient(Arc<dyn DynFetch>);

impl FetchClient {
    pub fn new<F>(fetch: F) -> Self
    where
        F: Fetch + Send + Sync + 'static,
    {
        Self(Arc::new(fetch))
    }
}

impl Fetch for FetchClient {
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
        self.0.fetch_boxed(request).await
    }
}

trait DynFetch: Send + Sync {
    fn fetch_boxed<'a>(
        &'a self,
        request: &'a HttpRequest,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>>;
}

impl<F> DynFetch for F
where
    F: Fetch + Send + Sync,
{
    fn fetch_boxed<'a>(
        &'a self,
        request: &'a HttpRequest,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>> {
        Box::pin(self.fetch(request))
    }
}
