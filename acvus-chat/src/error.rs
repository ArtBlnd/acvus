use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("entrypoint not found: {0}")]
    EntrypointNotFound(String),

    #[error("unknown provider: {0}")]
    UnknownProvider(String),

    #[error("unresolved context: @{0}")]
    UnresolvedContext(String),

    #[error("unexpected emit type: expected String, got {0}")]
    EmitType(String),

    #[error("fetch error for node {node}: {detail}")]
    Fetch { node: String, detail: String },

    #[error("parse error for node {node}: {detail}")]
    Parse { node: String, detail: String },

    #[error("tool not found: node {node} requested tool {tool}")]
    ToolNotFound { node: String, tool: String },

    #[error("tool target node not found: tool {tool} targets {target}")]
    ToolTargetNotFound { tool: String, target: String },

    #[error("tool call limit exceeded for node {0}")]
    ToolCallLimitExceeded(String),
}

/// Shorthand for formatting a `Value` variant name without pulling in the full debug repr.
pub(crate) fn value_type_name(v: &impl fmt::Debug) -> String {
    format!("{v:?}")
}
