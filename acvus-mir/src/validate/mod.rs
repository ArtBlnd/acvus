pub mod borrow_check;
pub mod init_check;
pub mod move_check;
mod type_check;

pub use move_check::is_move_only;
pub use type_check::{ValidationError, ValidationErrorKind};

use std::fmt;

use acvus_utils::Interner;

use crate::error::{MirError, MirErrorKind};
use crate::ir::{MirModule, ValOrigin};

/// The checks that hold of a MIR module at any point in the pipeline. The
/// move check is deliberately not among them: a move is a property of the
/// shape the source wrote, which optimization erases, so it runs once, in
/// pass 0 of `graph::optimize` (RFC-0029).
pub fn validate(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = type_check::check_types(module);
    errors.extend(borrow_check::check_borrows(module));
    errors
}

impl ValidationError {
    /// Convert this validation error into a [`MirError`] for unified error reporting.
    pub fn into_mir_error(self) -> MirError {
        let message = match &self.kind {
            ValidationErrorKind::TypeMismatch {
                inst_name,
                desc,
                expected,
                actual,
            } => {
                format!(
                    "type mismatch at {inst_name}, {desc}: expected {expected:?}, actual {actual:?}"
                )
            }
            ValidationErrorKind::MissingType { value_id } => {
                format!("Val({value_id}) has no type entry")
            }
            ValidationErrorKind::OrderEdge { inst_name, pure } => {
                if *pure {
                    format!("{inst_name}: a Pure call carries an Order")
                } else {
                    format!("{inst_name}: an effectful call carries no Order")
                }
            }
            ValidationErrorKind::ArityMismatch {
                inst_name,
                expected,
                got,
            } => {
                format!("arity mismatch at {inst_name}: expected {expected}, got {got}")
            }
            ValidationErrorKind::InvalidConstructor {
                inst_name,
                expected_constructor,
                actual,
            } => {
                format!("{inst_name}: expected {expected_constructor}, got {actual:?}")
            }
            ValidationErrorKind::UseAfterMove {
                value_id,
                moved_at,
                ty,
                origin: _,
            } => {
                format!(
                    "use of move-only value Val({value_id}) after move (moved at inst #{moved_at}), type: {ty:?}"
                )
            }
            ValidationErrorKind::BorrowConflict { storage, reference } => {
                format!("{storage} is used while reference Val({reference}) to it is live")
            }
            ValidationErrorKind::ContextMovedOut { moved_at, .. } => {
                format!(
                    "a context is moved out at [{}..{}] and not assigned again before the run ends",
                    moved_at.start, moved_at.end
                )
            }
        };

        MirError {
            kind: MirErrorKind::ValidationCheck {
                scope: self.scope,
                inst_index: self.inst_index,
                message,
            },
            span: self.span,
        }
    }
}

/// A [`ValidationError`] rendered for a reader: one sentence per kind, in the
/// program's words, naming its subject as the source wrote it.
pub struct ValidationErrorDisplay<'a> {
    error: &'a ValidationError,
    interner: &'a Interner,
}

impl ValidationError {
    pub fn display<'a>(&'a self, interner: &'a Interner) -> ValidationErrorDisplay<'a> {
        ValidationErrorDisplay {
            error: self,
            interner,
        }
    }
}

impl fmt::Display for ValidationErrorDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.error.kind {
            ValidationErrorKind::UseAfterMove {
                value_id, origin, ..
            } => {
                let subject = match origin {
                    Some(ValOrigin::Named(name)) => {
                        format!("`{}`", self.interner.resolve(*name))
                    }
                    Some(ValOrigin::Context(name)) => {
                        format!("`@{}`", self.interner.resolve(*name))
                    }
                    Some(ValOrigin::ExternParam(name)) => {
                        format!("`${}`", self.interner.resolve(*name))
                    }
                    Some(
                        ValOrigin::Field(..)
                        | ValOrigin::RefField(..)
                        | ValOrigin::Call(_)
                        | ValOrigin::Expr,
                    )
                    | None => format!("Val({value_id})"),
                };
                write!(f, "use of {subject} after it was moved")
            }
            ValidationErrorKind::ContextMovedOut { context, .. } => write!(
                f,
                "context @{} is moved out here and not assigned again before the run ends",
                self.interner.resolve(*context)
            ),
            ValidationErrorKind::TypeMismatch {
                inst_name,
                desc,
                expected,
                actual,
            } => write!(
                f,
                "{inst_name} takes {desc} as {expected:?}, and it is {actual:?}"
            ),
            ValidationErrorKind::MissingType { value_id } => {
                write!(f, "Val({value_id}) has no type")
            }
            ValidationErrorKind::OrderEdge { inst_name, pure } => {
                if *pure {
                    write!(f, "{inst_name} is a pure call and carries an Order")
                } else {
                    write!(f, "{inst_name} is an effectful call and carries no Order")
                }
            }
            ValidationErrorKind::ArityMismatch {
                inst_name,
                expected,
                got,
            } => write!(f, "{inst_name} takes {expected} operands and got {got}"),
            ValidationErrorKind::InvalidConstructor {
                inst_name,
                expected_constructor,
                actual,
            } => write!(
                f,
                "{inst_name} takes {expected_constructor} and got {actual:?}"
            ),
            ValidationErrorKind::BorrowConflict { storage, reference } => write!(
                f,
                "{storage} is touched while the reference Val({reference}) to it is live"
            ),
        }
    }
}
