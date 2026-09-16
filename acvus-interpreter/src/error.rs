use std::fmt;

use acvus_extern::ExternError;

// -- CollectionOp - which collection operation failed ----------------

/// Which collection operation triggered an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionOp {
    Find,
    Reduce,
    First,
    Last,
}

impl fmt::Display for CollectionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CollectionOp::Find => write!(f, "find"),
            CollectionOp::Reduce => write!(f, "reduce"),
            CollectionOp::First => write!(f, "first"),
            CollectionOp::Last => write!(f, "last"),
        }
    }
}

// -- RuntimeError ----------------------------------------------------

/// Runtime error during template/script execution.
///
/// NOT recoverable by retry - indicates a bug or invalid data. A type
/// mismatch is never one of these: the type checker rules it out, and the
/// interpreter panics where it would have been observed.
#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub kind: RuntimeErrorKind,
    /// The instruction that failed, set by the run loop (RFC-0031).
    pub span: Option<acvus_ast::Span>,
}

#[derive(Debug, Clone)]
pub enum RuntimeErrorKind {
    IntegerOverflow,
    /// Division by zero.
    DivisionByZero,
    /// Index out of bounds.
    IndexOutOfBounds {
        index: i64,
        len: usize,
    },
    /// Operation on empty collection.
    EmptyCollection {
        op: CollectionOp,
    },
    /// Object field not found. Field name is resolved to String at
    /// construction time so Display works without an interner.
    MissingField {
        field: std::string::String,
    },
    /// External function call failed.
    ExternCallFailed {
        /// Resolved function name.
        name: std::string::String,
        /// Error from the external function.
        source: std::string::String,
    },
    /// LLM/HTTP fetch or provider error.
    FetchFailed {
        /// Human-readable error detail from the provider/transport.
        source: std::string::String,
    },
    /// Tool call iteration limit exceeded.
    ToolCallLimitExceeded {
        limit: usize,
    },
    /// Assert expression evaluated to false.
    AssertFailed,
    /// Internal interpreter error (compiler bug or invalid state).
    Internal {
        message: std::string::String,
    },
}

// -- Constructors ----------------------------------------------------

impl RuntimeError {
    pub fn integer_overflow() -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::IntegerOverflow,
        }
    }

    pub fn division_by_zero() -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::DivisionByZero,
        }
    }

    pub fn index_out_of_bounds(index: i64, len: usize) -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::IndexOutOfBounds { index, len },
        }
    }

    pub fn empty_collection(op: CollectionOp) -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::EmptyCollection { op },
        }
    }

    pub fn missing_field(field: impl Into<std::string::String>) -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::MissingField {
                field: field.into(),
            },
        }
    }

    pub fn extern_call(
        name: impl Into<std::string::String>,
        source: impl Into<std::string::String>,
    ) -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::ExternCallFailed {
                name: name.into(),
                source: source.into(),
            },
        }
    }

    pub fn fetch(source: impl Into<std::string::String>) -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::FetchFailed {
                source: source.into(),
            },
        }
    }

    pub fn tool_call_limit(limit: usize) -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::ToolCallLimitExceeded { limit },
        }
    }

    pub fn assert_failed() -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::AssertFailed,
        }
    }

    pub fn internal(message: impl Into<std::string::String>) -> Self {
        Self {
            span: None,
            kind: RuntimeErrorKind::Internal {
                message: message.into(),
            },
        }
    }
}

// -- Display ---------------------------------------------------------

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            RuntimeErrorKind::IntegerOverflow => write!(f, "integer overflow"),
            RuntimeErrorKind::DivisionByZero => write!(f, "division by zero"),
            RuntimeErrorKind::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds (len {len})")
            }
            RuntimeErrorKind::EmptyCollection { op } => {
                write!(f, "{op}: empty collection")
            }
            RuntimeErrorKind::MissingField { field } => {
                write!(f, "missing field: {field}")
            }
            RuntimeErrorKind::ExternCallFailed { name, source } => {
                write!(f, "extern call '{name}' failed: {source}")
            }
            RuntimeErrorKind::FetchFailed { source } => write!(f, "fetch failed: {source}"),
            RuntimeErrorKind::ToolCallLimitExceeded { limit } => {
                write!(f, "tool call limit exceeded ({limit} rounds)")
            }
            RuntimeErrorKind::AssertFailed => write!(f, "assert failed"),
            RuntimeErrorKind::Internal { message } => write!(f, "internal: {message}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ExternError> for RuntimeError {
    fn from(e: ExternError) -> Self {
        let kind = match e {
            ExternError::Call { name, message } => RuntimeErrorKind::ExternCallFailed {
                name,
                source: message,
            },
            ExternError::Internal { message } => RuntimeErrorKind::Internal { message },
        };
        Self { kind, span: None }
    }
}
