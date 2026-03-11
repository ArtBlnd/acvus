use std::fmt;

use acvus_mir::error::MirError;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[derive(Debug)]
pub struct OrchError {
    pub kind: OrchErrorKind,
}

#[derive(Debug)]
pub enum OrchErrorKind {
    // Config
    InvalidConfig(String),

    // Compile
    TemplateParse {
        block: usize,
        error: String,
    },
    TemplateCompile {
        block: usize,
        errors: Vec<MirError>,
    },
    ScriptParse {
        error: String,
    },
    ScriptCompile {
        context: String,
        errors: Vec<MirError>,
    },
    ScriptTypeMismatch {
        context: String,
        expected: Ty,
        got: Ty,
    },

    // DAG
    CycleDetected {
        nodes: Vec<String>,
    },
    UnresolvedDependency {
        node: String,
        key: String,
    },

    // Tool
    ToolTargetNotFound {
        tool: String,
        target: String,
    },
    ToolParamType {
        tool: String,
        param: String,
        type_name: String,
    },

    // Function
    FnParamConflict {
        node: String,
        param: String,
    },

    // Runtime
    FuelExhausted,
    ModelError(String),
    ToolNotFound(String),
}

impl OrchError {
    pub fn new(kind: OrchErrorKind) -> Self {
        Self { kind }
    }

    pub fn display<'a>(&'a self, interner: &'a Interner) -> OrchErrorDisplay<'a> {
        OrchErrorDisplay {
            error: self,
            interner,
        }
    }
}

pub struct OrchErrorDisplay<'a> {
    error: &'a OrchError,
    interner: &'a Interner,
}

impl<'a> fmt::Display for OrchErrorDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let interner = self.interner;
        match &self.error.kind {
            OrchErrorKind::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            OrchErrorKind::TemplateParse { block, error } => {
                write!(f, "template parse error in block {block}: {error}")
            }
            OrchErrorKind::TemplateCompile { block, errors } => {
                writeln!(f, "compile errors in block {block}:")?;
                for e in errors {
                    writeln!(
                        f,
                        "  [{}..{}] {}",
                        e.span.start,
                        e.span.end,
                        e.display(interner)
                    )?;
                }
                Ok(())
            }
            OrchErrorKind::ScriptParse { error } => {
                write!(f, "script parse error: {error}")
            }
            OrchErrorKind::ScriptCompile { context, errors } => {
                writeln!(f, "script compile errors ({context}):")?;
                for e in errors {
                    writeln!(
                        f,
                        "  [{}..{}] {}",
                        e.span.start,
                        e.span.end,
                        e.display(interner)
                    )?;
                }
                Ok(())
            }
            OrchErrorKind::ScriptTypeMismatch {
                context,
                expected,
                got,
            } => {
                write!(
                    f,
                    "script type mismatch ({context}): expected {}, got {}",
                    expected.display(interner),
                    got.display(interner)
                )
            }
            OrchErrorKind::CycleDetected { nodes } => {
                write!(f, "cycle detected: {}", nodes.join(" -> "))
            }
            OrchErrorKind::UnresolvedDependency { node, key } => {
                write!(
                    f,
                    "unresolved dependency: node '{node}' requires key '{key}'"
                )
            }
            OrchErrorKind::ToolTargetNotFound { tool, target } => {
                write!(f, "tool '{tool}' references unknown node '{target}'")
            }
            OrchErrorKind::ToolParamType {
                tool,
                param,
                type_name,
            } => {
                write!(
                    f,
                    "tool '{tool}' param '{param}': unknown type '{type_name}'"
                )
            }
            OrchErrorKind::FnParamConflict { node, param } => {
                write!(f, "function node '{node}': param '{param}' conflicts with existing context key")
            }
            OrchErrorKind::FuelExhausted => write!(f, "fuel exhausted"),
            OrchErrorKind::ModelError(msg) => write!(f, "model error: {msg}"),
            OrchErrorKind::ToolNotFound(name) => write!(f, "tool not found: {name}"),
        }
    }
}
