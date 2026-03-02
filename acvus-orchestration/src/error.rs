use std::fmt;

use acvus_mir::error::MirError;

#[derive(Debug)]
pub struct OrchError {
    pub kind: OrchErrorKind,
}

#[derive(Debug)]
pub enum OrchErrorKind {
    // Config
    InvalidConfig(String),

    // Template loading
    TemplateLoad { path: String, error: String },

    // Compile
    TemplateParse { block: usize, error: String },
    TemplateCompile { block: usize, errors: Vec<MirError> },

    // DAG
    CycleDetected { nodes: Vec<String> },
    UnresolvedDependency { node: String, key: String },

    // Runtime
    FuelExhausted,
    ModelError(String),
    ToolNotFound(String),
}

impl OrchError {
    pub fn new(kind: OrchErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for OrchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            OrchErrorKind::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            OrchErrorKind::TemplateLoad { path, error } => {
                write!(f, "failed to load template '{path}': {error}")
            }
            OrchErrorKind::TemplateParse { block, error } => {
                write!(f, "template parse error in block {block}: {error}")
            }
            OrchErrorKind::TemplateCompile { block, errors } => {
                write!(f, "compile errors in block {block}: {} error(s)", errors.len())
            }
            OrchErrorKind::CycleDetected { nodes } => {
                write!(f, "cycle detected: {}", nodes.join(" -> "))
            }
            OrchErrorKind::UnresolvedDependency { node, key } => {
                write!(f, "unresolved dependency: node '{node}' requires key '{key}'")
            }
            OrchErrorKind::FuelExhausted => write!(f, "fuel exhausted"),
            OrchErrorKind::ModelError(msg) => write!(f, "model error: {msg}"),
            OrchErrorKind::ToolNotFound(name) => write!(f, "tool not found: {name}"),
        }
    }
}

impl std::error::Error for OrchError {}
