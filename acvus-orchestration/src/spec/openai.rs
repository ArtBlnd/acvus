use rust_decimal::Decimal;
use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashSet;

use crate::compile::{self, CompiledMessage, CompiledScript};
use crate::dsl::MessageSpec;
use crate::error::OrchError;

use super::{ToolBinding, CompiledToolBinding, compile_tool_bindings, MaxTokens};

/// OpenAI-compatible LLM node spec.
#[derive(Debug, Clone)]
pub struct OpenAICompatibleSpec {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<MessageSpec>,
    pub tools: Vec<ToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub max_tokens: MaxTokens,
    pub cache_key: Option<String>,
}

impl OpenAICompatibleSpec {
    pub fn output_ty(&self, interner: &Interner) -> Ty {
        Ty::List(Box::new(crate::dsl::message_elem_ty(interner)))
    }
}

/// Compiled OpenAI-compatible LLM node.
#[derive(Debug, Clone)]
pub struct CompiledOpenAICompatible {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<CompiledMessage>,
    pub tools: Vec<CompiledToolBinding>,
    pub temperature: Option<Decimal>,
    pub top_p: Option<Decimal>,
    pub max_tokens: MaxTokens,
    pub cache_key: Option<CompiledScript>,
}

pub fn compile_openai(
    interner: &Interner,
    spec: &OpenAICompatibleSpec,
    registry: &ContextTypeRegistry,
) -> Result<(CompiledOpenAICompatible, FxHashSet<Astr>), Vec<OrchError>> {
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
        CompiledOpenAICompatible {
            endpoint: spec.endpoint.clone(),
            api_key: spec.api_key.clone(),
            model: spec.model.clone(),
            messages: compiled_messages,
            tools: compiled_tools,
            temperature: spec.temperature,
            top_p: spec.top_p,
            max_tokens: spec.max_tokens.clone(),
            cache_key: compiled_cache_key,
        },
        all_keys,
    ))
}
