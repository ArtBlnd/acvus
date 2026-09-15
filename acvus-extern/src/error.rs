//! What a handler can report. A runtime's own error type absorbs it.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternError {
    /// The named ExternFn refused its input.
    Call { name: String, message: String },
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
            Self::Internal { message } => write!(f, "internal: {message}"),
        }
    }
}

impl std::error::Error for ExternError {}
