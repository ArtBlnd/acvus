use rust_decimal::Decimal;
use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashSet;

use crate::compile::{self, CompiledMessage, CompiledScript};
use crate::dsl::MessageSpec;
use crate::error::OrchError;

use super::{ToolBinding, CompiledToolBinding, compile_tool_bindings, MaxTokens, ThinkingConfig};

/// Anthropic LLM node spec.
#[derive(Debug, Clone)]
pub struct AnthropicSpec {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<MessageSpec>,
    pub tools: Vec<ToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub top_k: Option<u32>,
    pub max_tokens: MaxTokens,
    pub thinking: Option<ThinkingConfig>,
    pub cache_key: Option<String>,
}

impl AnthropicSpec {
    pub fn output_ty(&self, interner: &Interner) -> Ty {
        Ty::List(Box::new(crate::dsl::message_elem_ty(interner)))
    }
}

/// Compiled Anthropic LLM node.
#[derive(Debug, Clone)]
pub struct CompiledAnthropic {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<CompiledMessage>,
    pub tools: Vec<CompiledToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub top_k: Option<u32>,
    pub max_tokens: MaxTokens,
    pub thinking: Option<ThinkingConfig>,
    pub cache_key: Option<CompiledScript>,
}

pub fn compile_anthropic(
    interner: &Interner,
    spec: &AnthropicSpec,
    registry: &ContextTypeRegistry,
) -> Result<(CompiledAnthropic, FxHashSet<Astr>), Vec<OrchError>> {
    let elem_ty = crate::dsl::message_elem_ty(interner);
    let (compiled_messages, mut all_keys) =
        compile::compile_messages(interner, &spec.messages, registry, &elem_ty)?;
    let compiled_tools = compile_tool_bindings(&spec.tools)?;
    let compiled_cache_key = match &spec.cache_key {
        Some(ck) => {
            let (expr, _ck_ty) =
                compile::compile_script_with_hint(interner, ck, registry, Some(&Ty::String))
                    .map_err(|e| vec![e])?;
            all_keys.extend(expr.context_keys.iter().cloned());
            Some(expr)
        }
        None => None,
    };
    Ok((
        CompiledAnthropic {
            endpoint: spec.endpoint.clone(),
            api_key: spec.api_key.clone(),
            model: spec.model.clone(),
            messages: compiled_messages,
            tools: compiled_tools,
            temperature: spec.temperature,
            top_p: spec.top_p,
            top_k: spec.top_k,
            max_tokens: spec.max_tokens.clone(),
            thinking: spec.thinking.clone(),
            cache_key: compiled_cache_key,
        },
        all_keys,
    ))
}
