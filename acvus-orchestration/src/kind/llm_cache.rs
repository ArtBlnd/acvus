use std::collections::{HashMap, HashSet};

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;

use crate::compile::CompiledMessage;
use crate::dsl::MessageSpec;
use crate::error::OrchError;
use crate::provider::ApiKind;

/// LLM cache node spec — cached model call.
#[derive(Debug, Clone)]
pub struct LlmCacheSpec {
    pub api: ApiKind,
    pub provider: String,
    pub model: String,
    pub messages: Vec<MessageSpec>,
    /// TTL string, e.g. "300s", "1h".
    pub ttl: String,
    /// Provider-specific cache config (e.g. display_name for Gemini).
    pub cache_config: HashMap<String, serde_json::Value>,
}

impl LlmCacheSpec {
    pub fn output_ty(&self) -> Ty {
        Ty::String
    }
}

/// Compiled LLM cache node.
#[derive(Debug, Clone)]
pub struct CompiledLlmCache {
    pub api: ApiKind,
    pub provider: String,
    pub model: String,
    pub messages: Vec<CompiledMessage>,
    pub ttl: String,
    pub cache_config: HashMap<String, serde_json::Value>,
}

/// Compile an LLM cache node spec.
pub fn compile_llm_cache(
    spec: &LlmCacheSpec,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<(CompiledLlmCache, HashSet<String>), Vec<OrchError>> {
    let elem_ty = spec.api.message_elem_ty();
    let (compiled_messages, keys) =
        crate::compile::compile_messages(&spec.messages, context_types, registry, &elem_ty)?;
    Ok((
        CompiledLlmCache {
            api: spec.api.clone(),
            provider: spec.provider.clone(),
            model: spec.model.clone(),
            messages: compiled_messages,
            ttl: spec.ttl.clone(),
            cache_config: spec.cache_config.clone(),
        },
        keys,
    ))
}
