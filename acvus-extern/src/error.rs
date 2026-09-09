//! What a handler can report. A runtime's own error type absorbs it.

use std::fmt;

use crate::extern_value::ExternTypeName;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternError {
    /// The named ExternFn refused its input.
    Call { name: String, message: String },
    /// An extension value of another type reached a handler.
    UnexpectedExtern {
        expected: ExternTypeName,
        got: ExternTypeName,
    },
    /// A move-only extension value was still shared when a handler took it.
    SharedMoveOnly { type_name: ExternTypeName },
    /// An object argument lacks a declared field.
    MissingField { field: String },
    /// A declaration and its runtime disagree.
    Internal { message: String },
}

impl ExternError {
    pub fn call(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Call {
            name: name.into(),
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

impl fmt::Display for ExternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Call { name, message } => write!(f, "extern call '{name}' failed: {message}"),
            Self::UnexpectedExtern { expected, got } => {
                write!(f, "expected extension type {expected}, got {got}")
            }
            Self::SharedMoveOnly { type_name } => {
                write!(
                    f,
                    "move-only extension value of type {type_name} is still shared"
                )
            }
            Self::MissingField { field } => write!(f, "missing field: {field}"),
            Self::Internal { message } => write!(f, "internal: {message}"),
        }
    }
}

impl std::error::Error for ExternError {}
