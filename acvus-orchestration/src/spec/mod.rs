mod anthropic;
mod expression;
mod google;
mod google_cache;
mod openai;
mod plain;

pub use anthropic::AnthropicSpec;
pub use expression::ExpressionSpec;
pub use google::GoogleAISpec;
pub use google_cache::GoogleAICacheSpec;
pub use openai::OpenAICompatibleSpec;
pub use plain::PlainSpec;

use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

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
}

impl NodeKind {
    pub fn output_ty(&self, interner: &Interner) -> Ty {
        match self {
            NodeKind::Plain(spec) => spec.output_ty(),
            NodeKind::OpenAICompatible(spec) => spec.output_ty(interner),
            NodeKind::Anthropic(spec) => spec.output_ty(interner),
            NodeKind::GoogleAI(spec) => spec.output_ty(interner),
            NodeKind::GoogleAICache(spec) => spec.output_ty(),
            NodeKind::Expression(spec) => spec.output_ty.clone().unwrap_or(Ty::infer()),
        }
    }
}
