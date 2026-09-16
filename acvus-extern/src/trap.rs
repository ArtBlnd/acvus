//! A trap stops the run: a contract the runtime cannot honor, never a
//! failure the program could act on — that is a `Result` (RFC-0038). A
//! runtime's own error type absorbs it.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Trap {
    /// The named ExternFn cannot honor a value the checker admitted.
    Call { name: String, message: String },
    /// A declaration and its runtime disagree.
    Internal { message: String },
}

impl Trap {
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

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Call { name, message } => write!(f, "trap in '{name}': {message}"),
            Self::Internal { message } => write!(f, "internal: {message}"),
        }
    }
}

impl std::error::Error for Trap {}
