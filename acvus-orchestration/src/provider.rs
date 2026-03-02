use futures::future::BoxFuture;

/// Raw fetch trait: takes a JSON request body and returns a JSON response.
///
/// The orchestration crate handles message formatting and response parsing.
/// Implementors only need to handle HTTP transport (endpoint, auth, headers).
pub trait Fetch: Send + Sync {
    fn fetch<'a>(
        &'a self,
        body: serde_json::Value,
    ) -> BoxFuture<'a, Result<serde_json::Value, String>>;
}
