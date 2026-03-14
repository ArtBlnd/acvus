mod expr;
mod llm;
mod llm_cache;
mod plain;

pub use expr::{CompiledExpr, ExprSpec, compile_expr};
pub(crate) use llm::parse_type_name;
pub use llm::{
    CompiledLlm, CompiledToolBinding, CompiledToolParamInfo, GenerationParams, LlmSpec, MaxTokens,
    ThinkingConfig, ToolBinding, ToolParamInfo, compile_llm,
};
pub use llm_cache::{CompiledLlmCache, LlmCacheSpec, compile_llm_cache};
pub use plain::{CompiledPlain, PlainSpec, compile_plain};

use acvus_utils::Interner;

use crate::compile::CompiledMessage;

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Plain(PlainSpec),
    Llm(LlmSpec),
    LlmCache(LlmCacheSpec),
    Expr(ExprSpec),
}

impl NodeKind {
    /// The raw output type produced by this node kind (before self.bind).
    pub fn raw_output_ty(&self, interner: &Interner) -> acvus_mir::ty::Ty {
        match self {
            NodeKind::Plain(spec) => spec.output_ty(),
            NodeKind::Llm(spec) => spec.output_ty(interner),
            NodeKind::LlmCache(spec) => spec.output_ty(),
            NodeKind::Expr(spec) => spec.output_ty.clone(),
        }
    }
}

/// Compiled node kind — mirrors `NodeKind` but with compiled data.
#[derive(Debug, Clone)]
pub enum CompiledNodeKind {
    Plain(CompiledPlain),
    Llm(CompiledLlm),
    LlmCache(CompiledLlmCache),
    Expr(CompiledExpr),
}

impl CompiledNodeKind {
    pub fn messages(&self) -> &[CompiledMessage] {
        match self {
            Self::Plain(_) => &[],
            Self::Llm(llm) => &llm.messages,
            Self::LlmCache(cache) => &cache.messages,
            Self::Expr(_) => &[],
        }
    }
}
