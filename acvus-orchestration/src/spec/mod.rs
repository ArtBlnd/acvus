mod anthropic;
mod expression;
mod google;
mod google_cache;
mod openai;
mod plain;

pub use anthropic::{AnthropicSpec, CompiledAnthropic, compile_anthropic};
pub use expression::{CompiledExpression, ExpressionSpec, compile_expression};
pub use google::{CompiledGoogleAI, GoogleAISpec, compile_google};
pub use google_cache::{CompiledGoogleAICache, GoogleAICacheSpec, compile_google_cache};
pub use openai::{CompiledOpenAICompatible, OpenAICompatibleSpec, compile_openai};
pub use plain::{CompiledPlain, PlainSpec, compile_plain};

use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::compile::{CompiledMessage, CompiledScript};
use crate::dsl::MessageSpec;
use crate::error::{OrchError, OrchErrorKind};

// ── Shared types ────────────────────────────────────────────────────

/// Token limits for LLM calls.
#[derive(Debug, Clone, Default)]
pub struct MaxTokens {
    /// Total input token budget shared across budgeted iterators.
    pub input: Option<u32>,
    /// Maximum output tokens for the model response.
    pub output: Option<u32>,
}

/// Thinking / reasoning configuration for models that support it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum ThinkingConfig {
    Off,
    Low,
    Medium,
    High,
    Custom(u32),
}

/// Tool parameter info (pre-compilation).
#[derive(Debug, Clone)]
pub struct ToolParamInfo {
    pub ty: String,
    pub description: Option<String>,
}

/// Tool binding — binds a tool name to a target node with typed parameters.
#[derive(Debug, Clone)]
pub struct ToolBinding {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: FxHashMap<String, ToolParamInfo>,
}

/// Compiled tool parameter info with resolved type.
#[derive(Debug, Clone)]
pub struct CompiledToolParamInfo {
    pub ty: Ty,
    pub description: Option<String>,
}

/// A compiled tool binding with resolved types.
#[derive(Debug, Clone)]
pub struct CompiledToolBinding {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: FxHashMap<String, CompiledToolParamInfo>,
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

/// Compile tool bindings, converting param type name strings to `Ty`.
pub(crate) fn compile_tool_bindings(
    tools: &[ToolBinding],
) -> Result<Vec<CompiledToolBinding>, Vec<OrchError>> {
    let mut compiled = Vec::new();
    let mut errors = Vec::new();

    for tool in tools {
        let mut params = FxHashMap::default();
        for (param_name, info) in &tool.params {
            let Some(ty) = parse_type_name(&info.ty) else {
                errors.push(OrchError::new(OrchErrorKind::ToolParamType {
                    tool: tool.name.clone(),
                    param: param_name.clone(),
                    type_name: info.ty.clone(),
                }));
                continue;
            };
            params.insert(param_name.clone(), CompiledToolParamInfo {
                ty,
                description: info.description.clone(),
            });
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

// ── Node kind enums ─────────────────────────────────────────────────

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Plain(PlainSpec),
    OpenAICompatible(OpenAICompatibleSpec),
    Anthropic(AnthropicSpec),
    GoogleAI(GoogleAISpec),
    GoogleAICache(GoogleAICacheSpec),
    Expression(ExpressionSpec),
    Iterator(IteratorSpec),
}

/// Per-item transform for an iterator source.
#[derive(Debug, Clone)]
pub enum SourceTransform {
    /// Template interpolation — string with `{{ }}` blocks.
    /// Compiled via `compile_template`.
    Template(Astr),
    /// Script expression — arbitrary value transform.
    /// Compiled via `compile_script`.
    Script(Astr),
}

/// Per-item entry: condition (filter) + transform (map).
///
/// When processing each item, entries are evaluated in order.
/// The first entry whose condition matches has its transform applied.
/// Items with no matching entry are skipped.
#[derive(Debug, Clone)]
pub struct IteratorEntry {
    /// Condition script -> Bool. None = always match.
    pub condition: Option<Astr>,
    /// Transform to apply when condition matches.
    pub transform: SourceTransform,
}

/// A single source in an iterator node.
///
/// Evaluates `expr` to get a value, then:
/// - If Iterator/List/Deque: applies skip(`start`) -> take(`end`)
/// - If scalar: single item
///
/// For each item, `entries` are evaluated in order (first-match):
/// - No entries = pass-through (item yielded as-is)
/// - One entry (condition=None) = simple map
/// - Multiple entries = conditional map (first match wins, no match -> skip)
///
/// Results are tagged with `name` as `{name: String, item: T}`.
#[derive(Debug, Clone)]
pub struct IteratorSource {
    pub name: String,
    pub expr: Astr,
    pub entries: Vec<IteratorEntry>,
    /// Skip N items from the start. Script -> Int.
    pub start: Option<Astr>,
    /// Take up to N items. Script -> Option<Int>. None = exhaust.
    pub end: Option<Astr>,
}

/// Spec for a composite iterator node.
///
/// Pulls items from multiple sources, tags each with the source name,
/// and yields `{name: String, item: T}` objects.
///
/// `unordered=false`: sequential -- exhaust source A, then B, etc.
/// `unordered=true`: concurrent -- yield from whichever source is ready first.
#[derive(Debug, Clone)]
pub struct IteratorSpec {
    pub sources: Vec<IteratorSource>,
    pub unordered: bool,
}

impl NodeKind {
    /// The raw output type produced by this node kind (before self.bind).
    ///
    /// Returns `None` for kinds whose output type is inferred during compilation
    /// (e.g. Iterator, Expr with no explicit type annotation).
    pub fn raw_output_ty(&self, interner: &Interner) -> Option<Ty> {
        match self {
            NodeKind::Plain(spec) => Some(spec.output_ty()),
            NodeKind::OpenAICompatible(spec) => Some(spec.output_ty(interner)),
            NodeKind::Anthropic(spec) => Some(spec.output_ty(interner)),
            NodeKind::GoogleAI(spec) => Some(spec.output_ty(interner)),
            NodeKind::GoogleAICache(spec) => Some(spec.output_ty()),
            NodeKind::Expression(spec) => spec.output_ty.clone(),
            NodeKind::Iterator(_) => None,
        }
    }

    /// Returns tools if this is an LLM-kind node, empty slice otherwise.
    pub fn tools(&self) -> &[ToolBinding] {
        match self {
            NodeKind::OpenAICompatible(s) => &s.tools,
            NodeKind::Anthropic(s) => &s.tools,
            NodeKind::GoogleAI(s) => &s.tools,
            _ => &[],
        }
    }

    /// Returns messages if this is an LLM-kind node, empty slice otherwise.
    pub fn messages(&self) -> &[MessageSpec] {
        match self {
            NodeKind::OpenAICompatible(s) => &s.messages,
            NodeKind::Anthropic(s) => &s.messages,
            NodeKind::GoogleAI(s) => &s.messages,
            NodeKind::GoogleAICache(s) => &s.messages,
            _ => &[],
        }
    }
}

/// Compiled per-item transform.
#[derive(Debug, Clone)]
pub enum CompiledSourceTransform {
    Template(CompiledScript),
    Script(CompiledScript),
}

/// Compiled entry: condition + transform.
#[derive(Debug, Clone)]
pub struct CompiledIteratorEntry {
    pub condition: Option<CompiledScript>,
    pub transform: CompiledSourceTransform,
}

/// Compiled iterator source.
#[derive(Debug, Clone)]
pub struct CompiledIteratorSource {
    pub name: String,
    pub expr: CompiledScript,
    pub entries: Vec<CompiledIteratorEntry>,
    pub start: Option<CompiledScript>,
    pub end: Option<CompiledScript>,
}

/// Compiled node kind — mirrors `NodeKind` but with compiled data.
#[derive(Debug, Clone)]
pub enum CompiledNodeKind {
    Plain(CompiledPlain),
    OpenAICompatible(CompiledOpenAICompatible),
    Anthropic(CompiledAnthropic),
    GoogleAI(CompiledGoogleAI),
    GoogleAICache(CompiledGoogleAICache),
    Expression(CompiledExpression),
    Iterator {
        sources: Vec<CompiledIteratorSource>,
        unordered: bool,
    },
}

impl CompiledNodeKind {
    pub fn messages(&self) -> &[CompiledMessage] {
        match self {
            Self::OpenAICompatible(c) => &c.messages,
            Self::Anthropic(c) => &c.messages,
            Self::GoogleAI(c) => &c.messages,
            Self::GoogleAICache(c) => &c.messages,
            Self::Plain(_) | Self::Expression(_) | Self::Iterator { .. } => &[],
        }
    }

    /// Returns tools if this is an LLM-kind node, empty slice otherwise.
    pub fn tools(&self) -> &[CompiledToolBinding] {
        match self {
            Self::OpenAICompatible(c) => &c.tools,
            Self::Anthropic(c) => &c.tools,
            Self::GoogleAI(c) => &c.tools,
            _ => &[],
        }
    }
}
