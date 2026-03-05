#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("entrypoint not found: {0}")]
    EntrypointNotFound(String),

    #[error("dependency cycle detected: {0}")]
    CycleDetected(String),

    #[error("history node unreachable from entrypoint: {0}")]
    HistoryNodeUnreachable(String),

    #[error("unresolved context: @{0}")]
    UnresolvedContext(String),

    #[error("resolve error: {0}")]
    Resolve(String),
}
