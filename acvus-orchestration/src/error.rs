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

    // Compile
    TemplateParse { block: usize, error: String },
    TemplateCompile { block: usize, errors: Vec<MirError> },
    ScriptParse { error: String },

    // DAG
    CycleDetected { nodes: Vec<String> },
    UnresolvedDependency { node: String, key: String },

    // Tool
    ToolTargetNotFound { tool: String, target: String },
    ToolParamType { tool: String, param: String, type_name: String },

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
            OrchErrorKind::TemplateParse { block, error } => {
                write!(f, "template parse error in block {block}: {error}")
            }
            OrchErrorKind::TemplateCompile { block, errors } => {
                write!(f, "compile errors in block {block}: {} error(s)", errors.len())
            }
            OrchErrorKind::ScriptParse { error } => {
                write!(f, "script parse error: {error}")
            }
            OrchErrorKind::CycleDetected { nodes } => {
                write!(f, "cycle detected: {}", nodes.join(" -> "))
            }
            OrchErrorKind::UnresolvedDependency { node, key } => {
                write!(f, "unresolved dependency: node '{node}' requires key '{key}'")
            }
            OrchErrorKind::ToolTargetNotFound { tool, target } => {
                write!(f, "tool '{tool}' references unknown node '{target}'")
            }
            OrchErrorKind::ToolParamType { tool, param, type_name } => {
                write!(f, "tool '{tool}' param '{param}': unknown type '{type_name}'")
            }
            OrchErrorKind::FuelExhausted => write!(f, "fuel exhausted"),
            OrchErrorKind::ModelError(msg) => write!(f, "model error: {msg}"),
            OrchErrorKind::ToolNotFound(name) => write!(f, "tool not found: {name}"),
        }
    }
}

impl std::error::Error for OrchError {}
