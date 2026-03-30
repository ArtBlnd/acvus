pub mod init_check;
pub mod move_check;
mod type_check;

pub use move_check::is_move_only;
pub use type_check::{ValidationError, ValidationErrorKind};

use crate::error::{MirError, MirErrorKind};
use crate::graph::{ContextPolicy, QualifiedRef};
use crate::ir::MirModule;
use crate::ty::Ty;
use rustc_hash::FxHashMap;

/// Run all validation passes on a MIR module.
/// Returns errors found. Empty vec means valid.
pub fn validate(
    module: &MirModule,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    policies: &FxHashMap<QualifiedRef, ContextPolicy>,
) -> Vec<ValidationError> {
    let mut errors = type_check::check_types(module, fn_types, policies);
    errors.extend(move_check::check_moves(module));
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
            } => {
                format!(
                    "use of move-only value Val({value_id}) after move (moved at inst #{moved_at}), type: {ty:?}"
                )
            }
            ValidationErrorKind::NotMaterializable { ty } => {
                format!("cannot store non-materializable type to context: {ty:?}")
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
