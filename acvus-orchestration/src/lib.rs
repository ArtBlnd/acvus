mod compile;
mod dag;
mod dsl;
mod error;
mod executor;
pub(crate) mod kind;
mod message;
mod provider;
mod storage;

pub use acvus_mir_pass::analysis::reachable_context::ContextKeyPartition;
pub use compile::{
    CompiledBlock, CompiledHistory, CompiledMessage, CompiledNode, CompiledScript, compile_node,
    compile_nodes, compile_script,
};
pub use dag::{Dag, build_dag};
pub use dsl::{HistorySpec, MessageSpec, NodeSpec, Strategy, StrategyMode, TokenBudget};
pub use error::{OrchError, OrchErrorKind};
pub use executor::{Executor, value_to_literal};
pub use kind::{
    CompiledExpr, CompiledLlm, CompiledLlmCache, CompiledNodeKind, CompiledPlain,
    CompiledToolBinding, ExprSpec, GenerationParams, LlmCacheSpec, LlmSpec, MaxTokens, NodeKind,
    PlainSpec, ToolBinding,
};
pub use message::{
    Content, ContentItem, Message, ModelResponse, Output, ToolCall, ToolSpec, Usage,
};
pub use provider::{
    ApiKind, Fetch, HttpRequest, LlmModelKind, ProviderConfig, build_cache_request, build_request,
    create_llm_model, parse_cache_response, parse_response,
};
pub use storage::{HashMapStorage, Storage};
