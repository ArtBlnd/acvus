use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::compile::CompiledMessage;
use crate::dsl::MessageSpec;
use crate::error::OrchError;

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

/// Compiled Google AI context caching node.
#[derive(Debug, Clone)]
pub struct CompiledGoogleAICache {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<CompiledMessage>,
    pub ttl: String,
    pub cache_config: FxHashMap<String, serde_json::Value>,
}

/// Compile a Google AI cache node spec.
pub fn compile_google_cache(
    interner: &Interner,
    spec: &GoogleAICacheSpec,
    registry: &ContextTypeRegistry,
) -> Result<(CompiledGoogleAICache, FxHashSet<Astr>), Vec<OrchError>> {
    let elem_ty = crate::dsl::message_elem_ty(interner);
    let (compiled_messages, keys) = crate::compile::compile_messages(
        interner,
        &spec.messages,
        registry,
        &elem_ty,
    )?;
    Ok((
        CompiledGoogleAICache {
            endpoint: spec.endpoint.clone(),
            api_key: spec.api_key.clone(),
            model: spec.model.clone(),
            messages: compiled_messages,
            ttl: spec.ttl.clone(),
            cache_config: spec.cache_config.clone(),
        },
        keys,
    ))
}
