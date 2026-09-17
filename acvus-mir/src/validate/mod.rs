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

/// Run all validation passes on a MIR module.
/// Returns errors found. Empty vec means valid.
pub fn validate(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = type_check::check_types(module);
    errors.extend(move_check::check_moves(module));
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

/// A [`ValidationError`] rendered for a reader. A `UseAfterMove` names its
/// subject as the source wrote it; every other kind keeps its `Debug` form.
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
            kind => write!(f, "{kind:?}"),
        }
    }
}
