use acvus_mir::ty::Ty;
use rustc_hash::FxHashMap;

use crate::dsl::MessageSpec;

/// Google AI context caching node spec.
#[derive(Debug, Clone)]
pub struct GoogleAICacheSpec {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<MessageSpec>,
    /// TTL string, e.g. "300s", "1h".
    pub ttl: String,
    /// Provider-specific cache config (e.g. display_name for Gemini).
    pub cache_config: FxHashMap<String, serde_json::Value>,
}

impl GoogleAICacheSpec {
    pub fn output_ty(&self) -> Ty {
        Ty::String
    }
}
