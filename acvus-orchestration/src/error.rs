use std::fmt;

use acvus_mir::error::MirError;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};

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

    // Registry
    RegistryConflict {
        key: Astr,
        tier_a: &'static str,
        tier_b: &'static str,
    },

    // Persistency
    UnpureStoredType {
        node: Astr,
        persistency: String,
        ty: Ty,
    },
    /// Node has persistent strategy but output type is not storable
    /// (contains Fn, Opaque, or effectful Iterator/Sequence).
    NonStorableOutput {
        node: String,
        ty: Ty,
    },

    // Persistency — initial_value required
    MissingInitialValue {
        node: Astr,
        persistency: &'static str,
    },

    // Runtime
    FuelExhausted,
    ModelError(String),
    ToolNotFound(String),

    // Provider
    UnknownProvider {
        provider: String,
    },
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
                write!(
                    f,
                    "function node '{node}': param '{param}' conflicts with existing context key"
                )
            }
            OrchErrorKind::RegistryConflict {
                key,
                tier_a,
                tier_b,
            } => {
                write!(
                    f,
                    "context type conflict: key '{}' appears in both '{}' and '{}'",
                    interner.resolve(*key),
                    tier_a,
                    tier_b
                )
            }
            OrchErrorKind::UnpureStoredType {
                node,
                persistency,
                ty,
            } => {
                write!(
                    f,
                    "node '{}' has {} persistency but its stored type {} is not pure (cannot be persisted)",
                    interner.resolve(*node),
                    persistency,
                    ty.display(interner)
                )
            }
            OrchErrorKind::NonStorableOutput { node, ty } => {
                write!(
                    f,
                    "node '{node}' has persistent strategy but output type {} is not storable \
                     (contains Fn, Opaque, or effectful Iterator/Sequence)",
                    ty.display(interner)
                )
            }
            OrchErrorKind::MissingInitialValue { node, persistency } => {
                write!(
                    f,
                    "node '{}' has {} persistency but no initial_value — \
                     Sequence and Patch modes require initial_value",
                    interner.resolve(*node),
                    persistency,
                )
            }
            OrchErrorKind::UnknownProvider { provider } => {
                write!(f, "unknown provider: {provider}")
            }
            OrchErrorKind::FuelExhausted => write!(f, "fuel exhausted"),
            OrchErrorKind::ModelError(msg) => write!(f, "model error: {msg}"),
            OrchErrorKind::ToolNotFound(name) => write!(f, "tool not found: {name}"),
        }
    }
}
