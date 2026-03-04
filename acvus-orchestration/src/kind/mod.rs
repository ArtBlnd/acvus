mod llm;
mod llm_cache;
mod plain;

pub(crate) use llm::parse_type_name;
pub use llm::{
    CompiledLlm, CompiledToolBinding, GenerationParams, LlmSpec, ToolBinding, compile_llm,
};
pub use llm_cache::{CompiledLlmCache, LlmCacheSpec, compile_llm_cache};
pub use plain::{CompiledPlain, PlainSpec, compile_plain};

use crate::compile::CompiledMessage;

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Plain(PlainSpec),
    Llm(LlmSpec),
    LlmCache(LlmCacheSpec),
}

impl NodeKind {
    /// The output type produced by this node kind.
    pub fn output_ty(&self) -> acvus_mir::ty::Ty {
        match self {
            NodeKind::Plain(spec) => spec.output_ty(),
            NodeKind::Llm(spec) => spec.output_ty(),
            NodeKind::LlmCache(spec) => spec.output_ty(),
        }
    }
}

/// Compiled node kind — mirrors `NodeKind` but with compiled data.
#[derive(Debug, Clone)]
pub enum CompiledNodeKind {
    Plain(CompiledPlain),
    Llm(CompiledLlm),
    LlmCache(CompiledLlmCache),
}

impl CompiledNodeKind {
    pub fn messages(&self) -> &[CompiledMessage] {
        match self {
            Self::Plain(_) => &[],
            Self::Llm(llm) => &llm.messages,
            Self::LlmCache(cache) => &cache.messages,
        }
    }
}
