mod blob;
mod blob_journal;
mod convert;
mod dsl;
mod error;
pub mod http;
mod message;
pub mod executor;
pub(crate) mod spec;
mod storage;
pub mod unit;

/// Lowering: NodeSpec[] → CompilationGraph + Vec<NodeMeta>.
pub mod lower;

/// MIR graph types re-exported for phased API consumers.
pub use acvus_mir::graph::{
    CompilationGraph, CompiledGraph as MirCompiledGraph, CompiledUnit as MirCompiledUnit,
    ContextId, ContextIdTable, GraphError, ResolvedGraph, ScopeId, FunctionId,
};

pub use acvus_mir::analysis::reachable_context::ContextKeyPartition;
pub use blob::{BlobHash, BlobStore, MemBlobStore};
pub use blob_journal::BlobStoreJournal;
pub use convert::{json_to_value, value_to_literal};
pub use dsl::{
    ContextScope, Execution, FnParam, MessageSpec, NodeLocalTypes, NodeSpec, Persistency, Strategy,
    TokenBudget,
};
pub use error::{OrchError, OrchErrorDisplay, OrchErrorKind};
pub use http::{Fetch, HttpRequest, RequestError};
// NodeMeta removed — lowering now produces CompilationGraph, not runtime Units.
pub use message::{
    Content, ContentItem, Message, ModelResponse, Output, ToolCall, ToolCallExtras, ToolSpec,
    ToolSpecParam, Usage,
};
pub use executor::{ExecutionError, ExecutionState, Executor, LoopState, ParkedDiag, Resolved};
pub use spec::{
    AnthropicSpec, ExpressionSpec, GoogleAICacheSpec,
    GoogleAISpec, MaxTokens, NodeKind, OpenAICompatibleSpec, PlainSpec, ThinkingConfig,
    ToolBinding, ToolParamInfo,
};
pub use storage::{
    EntryMut, EntryRef, HistoryEntry, Journal, JournalError, PatchDiff, Prune, TreeEntryMut,
    TreeEntryRef, TreeExport, TreeJournal, TreeNodeExport,
};
