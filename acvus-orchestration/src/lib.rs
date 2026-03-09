mod compile;
mod convert;
mod dag;
mod display;
mod dsl;
mod error;
pub(crate) mod kind;
mod message;
pub mod node;
mod provider;
pub mod resolve;
mod state;
mod storage;

pub use acvus_mir_pass::analysis::reachable_context::ContextKeyPartition;
pub use compile::{
    CompiledBlock, CompiledMessage, CompiledNode, CompiledScript, CompiledSelf, CompiledStrategy,
    ExternalContextEnv, NodeLocalTypes, compile_node, compile_nodes, compile_nodes_with_env,
    compile_script, compute_external_context_env,
};
pub use convert::{json_to_value, value_to_literal};
pub use dag::{Dag, build_dag};
pub use display::{
    CompiledDisplayEntry, CompiledIterableDisplay, CompiledStaticDisplay, DisplayEntrySpec,
    IterableDisplaySpec, RenderedDisplayEntry, StaticDisplaySpec, compile_iterable_display,
    compile_static_display, render_display, render_display_with_idx,
};
pub use dsl::{MessageSpec, NodeSpec, SelfSpec, Strategy, TokenBudget};
pub use error::{OrchError, OrchErrorDisplay, OrchErrorKind};
pub use kind::{
    CompiledExpr, CompiledLlm, CompiledLlmCache, CompiledNodeKind, CompiledPlain,
    CompiledToolBinding, ExprSpec, GenerationParams, LlmCacheSpec, LlmSpec, MaxTokens, NodeKind,
    PlainSpec, ToolBinding,
};
pub use message::{
    Content, ContentItem, Message, ModelResponse, Output, ToolCall, ToolSpec, Usage,
};
pub use node::{ExprNode, LlmCacheNode, LlmNode, Node, PlainNode, build_node_table};
pub use provider::{
    ApiKind, Fetch, HttpRequest, LlmModelKind, ProviderConfig, build_cache_request, build_request,
    create_llm_model, parse_cache_response, parse_response,
};
pub use resolve::{ResolveError, ResolveState, Resolved, Resolver};
pub use state::State;
pub use storage::{HashMapStorage, Storage};
