pub mod borrow_check;
pub mod exhaustive;
pub mod init_check;
pub mod move_check;
pub mod type_check;

pub use exhaustive::{Known, known_variants};
pub use move_check::is_move_only;
pub use type_check::{ValidationError, ValidationErrorKind};

use std::fmt;

use acvus_utils::Interner;

use crate::ir::{MirModule, ValOrigin};

/// Every rule a `MirModule` answers on its own, for a module no phase of
/// `graph::optimize` was run over — the hand-built and directly lowered
/// modules of `acvus-mir-test` and `acvus-interpreter-test`. The move check is
/// not among them and cannot be: a move is a property of the shape the source
/// wrote, which optimization erases (RFC-0029).
///
/// The pipeline does not call this, and a call added from it would report every
/// refusal pass 0 already reported a second time.
/// `acvus-cli/tests/cli.rs::a_write_while_a_reference_is_live_is_refused_once`
/// is what fails then.
pub fn validate(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = type_check::check_types(module);
    errors.extend(borrow_check::check_borrows(module));
    errors.extend(exhaustive::check_exhaustive(module));
    errors
}

/// The subject as the source wrote it, where its origin is a name the source
/// wrote. A field, a call's result and a bare expression have no name of their
/// own, and each caller says what it puts there instead.
fn written_as(interner: &Interner, origin: Option<&ValOrigin>) -> Option<String> {
    match origin? {
        ValOrigin::Named(name) => Some(format!("`{}`", interner.resolve(*name))),
        ValOrigin::Context(name) => Some(format!("`@{}`", interner.resolve(*name))),
        ValOrigin::ExternParam(name) => Some(format!("`${}`", interner.resolve(*name))),
        ValOrigin::Field(..) | ValOrigin::RefField(..) | ValOrigin::Call(_) | ValOrigin::Expr => {
            None
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
            } => write!(
                f,
                "{} is used here after it was moved",
                written_as(self.interner, origin.as_ref())
                    .unwrap_or_else(|| format!("Val({value_id})"))
            ),
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
                "{inst_name} takes {desc} as {}, and it is {}",
                expected.display(self.interner),
                actual.display(self.interner)
            ),
            ValidationErrorKind::MissingType { value_id } => {
                write!(f, "Val({value_id}) has no type")
            }
            ValidationErrorKind::ErrorType { value_id, origin } => write!(
                f,
                "{} reaches the machine with no type, and no refusal said why",
                written_as(self.interner, origin.as_ref())
                    .unwrap_or_else(|| format!("Val({value_id})"))
            ),
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
                "{inst_name} takes {expected_constructor} and got {}",
                actual.display(self.interner)
            ),
            ValidationErrorKind::NonExhaustiveMatch { over } => {
                write!(f, "non-exhaustive match: {}; add a `_` arm", over.shown())
            }
            ValidationErrorKind::MatchMissesBoolArm { missing } => write!(
                f,
                "non-exhaustive match: `{missing}` is not covered; add that arm or a `_` arm"
            ),
            ValidationErrorKind::DiamondArmMissesJoin { side, join } => write!(
                f,
                "an `if`'s {side} arm does not reach the join L{} it is written to rejoin at",
                join.0
            ),
            ValidationErrorKind::DemotedDiamondMeetsAgain { join } => write!(
                f,
                "a branch demoted from an `if` has arms that meet again at L{}: \
                 a pass running after `optimize::rejoin` re-formed them and left \
                 the terminator a `jump_if`",
                join.0
            ),
            ValidationErrorKind::ForRangeWidths { at, hi } => write!(
                f,
                "a `for` over `{}..{}` needs one integer width at both bounds",
                at.display(self.interner),
                hi.display(self.interner)
            ),
            ValidationErrorKind::MatchMissesVariants { enum_name, missing } => {
                let written = missing
                    .iter()
                    .map(|tag| match enum_name {
                        Some(name) => format!(
                            "`{}::{}`",
                            self.interner.resolve(*name),
                            self.interner.resolve(*tag)
                        ),
                        None => format!("`{}`", self.interner.resolve(*tag)),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "non-exhaustive match: {written} is not covered; add that arm or a `_` arm"
                )
            }
            ValidationErrorKind::MatchMissesBuiltinVariants {
                enum_name,
                arity,
                covered,
            } => write!(
                f,
                "non-exhaustive match: an `{enum_name}` has {arity} variants and the arms cover {covered}; add the missing arm or a `_` arm"
            ),
            ValidationErrorKind::ReferenceToLocalLeavesBody { storage } => {
                let named = written_as(self.interner, storage.as_ref())
                    .unwrap_or_else(|| "a local".to_string());
                write!(f, "a reference to {named} cannot leave the body")
            }
            ValidationErrorKind::NotAParameter { value_id } => write!(
                f,
                "Val({value_id}) is named as a parameter's storage and is not a parameter of the body"
            ),
            ValidationErrorKind::BorrowConflict { storage, touch, .. } => {
                let named = written_as(self.interner, storage.as_ref())
                    .unwrap_or_else(|| "the storage".to_string());
                write!(
                    f,
                    "{named} is {} here while a reference to it is live",
                    touch.word()
                )
            }
        }
    }
}
