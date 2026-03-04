use std::collections::{BTreeMap, HashMap, HashSet};

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;

use crate::compile::{CompiledMessage, CompiledScript};
use crate::dsl::MessageSpec;
use crate::error::{OrchError, OrchErrorKind};

/// LLM node spec — model call with messages, tools, generation params.
#[derive(Debug, Clone)]
pub struct LlmSpec {
    pub provider: String,
    pub model: String,
    pub messages: Vec<MessageSpec>,
    pub tools: Vec<ToolBinding>,
    pub generation: GenerationParams,
    pub cache_key: Option<String>,
    /// Total input token budget shared across budgeted iterators.
    pub max_tokens: Option<u32>,
}

impl LlmSpec {
    pub fn output_ty(&self) -> Ty {
        Ty::Object(BTreeMap::from([
            ("role".into(), Ty::String),
            ("content".into(), Ty::String),
            ("content_type".into(), Ty::String),
        ]))
    }
}

/// Generation parameters for model calls.
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub grounding: bool,
}

/// Tool binding — binds a tool name to a target node with typed parameters.
#[derive(Debug, Clone)]
pub struct ToolBinding {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: HashMap<String, String>,
}

/// A compiled tool binding with resolved types.
#[derive(Debug, Clone)]
pub struct CompiledToolBinding {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: HashMap<String, Ty>,
}

/// Compiled LLM node.
#[derive(Debug, Clone)]
pub struct CompiledLlm {
    pub provider: String,
    pub model: String,
    pub messages: Vec<CompiledMessage>,
    pub tools: Vec<CompiledToolBinding>,
    pub generation: GenerationParams,
    pub cache_key: Option<CompiledScript>,
    pub max_tokens: Option<u32>,
}

/// The element type that bodyless iterators must produce for LLM nodes.
/// Each element is used directly as a message: `{role, content, content_type}`.
pub(crate) fn message_elem_ty() -> Ty {
    Ty::Object(BTreeMap::from([
        ("role".into(), Ty::String),
        ("content".into(), Ty::String),
        ("content_type".into(), Ty::String),
    ]))
}

/// Parse a type name string into a `Ty`.
pub(crate) fn parse_type_name(name: &str) -> Option<Ty> {
    match name {
        "string" => Some(Ty::String),
        "int" => Some(Ty::Int),
        "float" => Some(Ty::Float),
        "bool" => Some(Ty::Bool),
        _ => None,
    }
}

/// Compile an LLM node spec.
pub fn compile_llm(
    spec: &LlmSpec,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<(CompiledLlm, HashSet<String>), Vec<OrchError>> {
    let elem_ty = message_elem_ty();
    let (compiled_messages, mut all_keys) =
        crate::compile::compile_messages(&spec.messages, context_types, registry, &elem_ty)?;
    let compiled_tools = compile_tool_bindings(&spec.tools)?;
    let compiled_cache_key = match &spec.cache_key {
        Some(ck) => {
            let (expr, ck_ty) =
                crate::compile::compile_script(ck, context_types, registry).map_err(|e| vec![e])?;
            crate::compile::expect_ty("cache_key", &ck_ty, &Ty::String).map_err(|e| vec![e])?;
            all_keys.extend(expr.context_keys.iter().cloned());
            Some(expr)
        }
        None => None,
    };
    Ok((
        CompiledLlm {
            provider: spec.provider.clone(),
            model: spec.model.clone(),
            messages: compiled_messages,
            tools: compiled_tools,
            generation: spec.generation.clone(),
            cache_key: compiled_cache_key,
            max_tokens: spec.max_tokens,
        },
        all_keys,
    ))
}

/// Compile tool bindings, converting param type name strings to `Ty`.
fn compile_tool_bindings(
    tools: &[ToolBinding],
) -> Result<Vec<CompiledToolBinding>, Vec<OrchError>> {
    let mut compiled = Vec::new();
    let mut errors = Vec::new();

    for tool in tools {
        let mut params = HashMap::new();
        for (param_name, type_name) in &tool.params {
            let Some(ty) = parse_type_name(type_name) else {
                errors.push(OrchError::new(OrchErrorKind::ToolParamType {
                    tool: tool.name.clone(),
                    param: param_name.clone(),
                    type_name: type_name.clone(),
                }));
                continue;
            };
            params.insert(param_name.clone(), ty);
        }
        compiled.push(CompiledToolBinding {
            name: tool.name.clone(),
            description: tool.description.clone(),
            node: tool.node.clone(),
            params,
        });
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(compiled)
}
