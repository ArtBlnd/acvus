use std::fmt;

use acvus_ast::{BinOp, UnaryOp};

// ── ValueKind — lightweight discriminant for error reporting ─────────

/// What kind of runtime value was encountered.
///
/// Used in error variants to report type mismatches without carrying
/// the full `Value`. Derived from `Value` via `ValueKind::of`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Int,
    Float,
    String,
    Bool,
    Unit,
    Range,
    Byte,
    List,
    Deque,
    Object,
    Tuple,
    Variant,
    Fn,
    ExternFn,
    Iterator,
    Sequence,
    Handle,
    Opaque,
}

impl fmt::Display for ValueKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueKind::Int => write!(f, "Int"),
            ValueKind::Float => write!(f, "Float"),
            ValueKind::String => write!(f, "String"),
            ValueKind::Bool => write!(f, "Bool"),
            ValueKind::Unit => write!(f, "Unit"),
            ValueKind::Range => write!(f, "Range"),
            ValueKind::Byte => write!(f, "Byte"),
            ValueKind::List => write!(f, "List"),
            ValueKind::Deque => write!(f, "Deque"),
            ValueKind::Object => write!(f, "Object"),
            ValueKind::Tuple => write!(f, "Tuple"),
            ValueKind::Variant => write!(f, "Variant"),
            ValueKind::Fn => write!(f, "Fn"),
            ValueKind::ExternFn => write!(f, "ExternFn"),
            ValueKind::Iterator => write!(f, "Iterator"),
            ValueKind::Sequence => write!(f, "Sequence"),
            ValueKind::Handle => write!(f, "Handle"),
            ValueKind::Opaque => write!(f, "Opaque"),
        }
    }
}

// ── CollectionOp — which collection operation failed ────────────────

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

// ── RuntimeError ────────────────────────────────────────────────────

/// Runtime error during template/script execution.
///
/// Propagated through the coroutine as `Stepped::Error`.
/// NOT recoverable by retry — indicates a bug or invalid data.
#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub kind: RuntimeErrorKind,
}

#[derive(Debug, Clone)]
pub enum RuntimeErrorKind {
    /// Binary operator applied to incompatible types.
    BinOpMismatch {
        op: BinOp,
        left: ValueKind,
        right: ValueKind,
    },
    /// Unary operator applied to incompatible type.
    UnaryOpMismatch { op: UnaryOp, operand: ValueKind },
    /// Value was not the expected kind.
    UnexpectedType {
        /// What operation required this type (static label).
        operation: &'static str,
        /// What kind(s) were acceptable.
        expected: &'static [ValueKind],
        /// What was actually found.
        got: ValueKind,
    },
    /// NaN encountered in ordered comparison.
    NanComparison,
    /// Division by zero.
    DivisionByZero,
    /// Index out of bounds.
    IndexOutOfBounds { index: i64, len: usize },
    /// Operation on empty collection.
    EmptyCollection { op: CollectionOp },
    /// Object field not found. Field name is resolved to String at
    /// construction time so Display works without an interner.
    MissingField { field: std::string::String },
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
    ToolCallLimitExceeded { limit: usize },
    /// Assert expression evaluated to false.
    AssertFailed,
    /// Internal interpreter error (compiler bug or invalid state).
    Internal { message: std::string::String },
}

// ── Constructors ────────────────────────────────────────────────────

impl RuntimeError {
    pub fn bin_op_mismatch(op: BinOp, left: ValueKind, right: ValueKind) -> Self {
        Self {
            kind: RuntimeErrorKind::BinOpMismatch { op, left, right },
        }
    }

    pub fn unary_op_mismatch(op: UnaryOp, operand: ValueKind) -> Self {
        Self {
            kind: RuntimeErrorKind::UnaryOpMismatch { op, operand },
        }
    }

    pub fn unexpected_type(
        operation: &'static str,
        expected: &'static [ValueKind],
        got: ValueKind,
    ) -> Self {
        Self {
            kind: RuntimeErrorKind::UnexpectedType {
                operation,
                expected,
                got,
            },
        }
    }

    pub fn nan_comparison() -> Self {
        Self {
            kind: RuntimeErrorKind::NanComparison,
        }
    }

    pub fn division_by_zero() -> Self {
        Self {
            kind: RuntimeErrorKind::DivisionByZero,
        }
    }

    pub fn index_out_of_bounds(index: i64, len: usize) -> Self {
        Self {
            kind: RuntimeErrorKind::IndexOutOfBounds { index, len },
        }
    }

    pub fn empty_collection(op: CollectionOp) -> Self {
        Self {
            kind: RuntimeErrorKind::EmptyCollection { op },
        }
    }

    pub fn missing_field(field: impl Into<std::string::String>) -> Self {
        Self {
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
            kind: RuntimeErrorKind::ExternCallFailed {
                name: name.into(),
                source: source.into(),
            },
        }
    }

    pub fn fetch(source: impl Into<std::string::String>) -> Self {
        Self {
            kind: RuntimeErrorKind::FetchFailed {
                source: source.into(),
            },
        }
    }

    pub fn tool_call_limit(limit: usize) -> Self {
        Self {
            kind: RuntimeErrorKind::ToolCallLimitExceeded { limit },
        }
    }

    pub fn assert_failed() -> Self {
        Self {
            kind: RuntimeErrorKind::AssertFailed,
        }
    }

    pub fn internal(message: impl Into<std::string::String>) -> Self {
        Self {
            kind: RuntimeErrorKind::Internal {
                message: message.into(),
            },
        }
    }
}

// ── Display ─────────────────────────────────────────────────────────

fn fmt_expected(expected: &[ValueKind], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match expected {
        [] => write!(f, "(none)"),
        [single] => write!(f, "{single}"),
        [a, b] => write!(f, "{a} or {b}"),
        many => {
            for (i, k) in many.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if i == many.len() - 1 {
                    write!(f, "or ")?;
                }
                write!(f, "{k}")?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            RuntimeErrorKind::BinOpMismatch { op, left, right } => {
                write!(f, "{op:?}: incompatible types {left} and {right}")
            }
            RuntimeErrorKind::UnaryOpMismatch { op, operand } => {
                write!(f, "{op:?}: incompatible type {operand}")
            }
            RuntimeErrorKind::UnexpectedType {
                operation,
                expected,
                got,
            } => {
                write!(f, "{operation}: expected ")?;
                fmt_expected(expected, f)?;
                write!(f, ", got {got}")
            }
            RuntimeErrorKind::NanComparison => write!(f, "NaN in ordered comparison"),
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
