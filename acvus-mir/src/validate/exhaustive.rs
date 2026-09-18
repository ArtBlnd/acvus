//! A `match` is exhaustive, and it is decided by the scrutinee's type
//! (RFC-0051 §3-§4).
//!
//! The language's enums are structural and unify by union, so a value's
//! variant set grows everywhere the value flows — and that union is what
//! the solver leaves on the value's type. A MIR value's type is settled:
//! `Ty` is `TyTerm<Concrete>`, whose `Var` is `Infallible`. So the variant
//! list on the scrutinee's `Ty::Enum` is the closed set for that value,
//! wherever the value came from, and [`known_variants`] reads it off the
//! type. Maranget's matrix over nested positions is the widening still
//! owed; it changes neither the rule nor its place.
//!
//! The pass reads `InstKind::Switch`, the one shape that names a `match`.
//! An `if let` is two arms and always exhaustive, and never wears one.

use std::collections::BTreeSet;

use acvus_utils::Astr;

use crate::ir::{InstKind, MirBody, MirModule, ValueId};
use crate::ty::Ty;
use crate::validate::type_check::{ValidationError, ValidationErrorKind};

/// A builtin enum whose type names its variants: the count is the whole
/// answer, because typeck has already refused any arm outside the set
/// (RFC-0039).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Builtin {
    pub name: &'static str,
    pub arity: usize,
}

const OPTION: Builtin = Builtin {
    name: "Option",
    arity: 2,
};
const RESULT: Builtin = Builtin {
    name: "Result",
    arity: 2,
};

/// What the scrutinee's type says about the variants it can hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Known {
    /// The type is an enum: these tags and no others.
    Closed(BTreeSet<Astr>),
    /// The type is an `Option` or a `Result`. This stage holds no interner
    /// and so names no tag itself; the count is what it checks against.
    ClosedBuiltin(Builtin),
    /// The type names no variant set. `typeck` refuses an arm naming a
    /// variant the scrutinee's type does not have (RFC-0051 §2), so no
    /// script reaches this answer; what reaches it is a `Switch` a pass or a
    /// test builds over a tag of some other type, and it is refused rather
    /// than assumed closed.
    Open,
}

pub fn check_exhaustive(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    check_body(&module.main, "main", &mut errors);
    for (label, closure) in &module.closures {
        check_body(closure, &format!("closure({label:?})"), &mut errors);
    }
    errors
}

fn check_body(body: &MirBody, scope: &str, errors: &mut Vec<ValidationError>) {
    for (at, inst) in body.insts.iter().enumerate() {
        let InstKind::Switch { tag, arms, default } = &inst.kind else {
            continue;
        };
        // A catch-all is always the way through: `default` is present
        // exactly when the source wrote one.
        if default.is_some() {
            continue;
        }
        let covered: BTreeSet<Astr> = arms.iter().map(|(tag, _, _)| *tag).collect();
        let kind = match known_variants(body, *tag) {
            Known::Open => ValidationErrorKind::NonExhaustiveMatch,
            Known::ClosedBuiltin(builtin) if covered.len() < builtin.arity => {
                ValidationErrorKind::MatchMissesBuiltinVariants {
                    enum_name: builtin.name,
                    arity: builtin.arity,
                    covered: covered.len(),
                }
            }
            Known::ClosedBuiltin(_) => continue,
            Known::Closed(set) => {
                let missing: Vec<Astr> = set.difference(&covered).copied().collect();
                if missing.is_empty() {
                    continue;
                }
                ValidationErrorKind::MatchMissesVariants {
                    enum_name: enum_name_of(body, *tag),
                    missing,
                }
            }
        };
        errors.push(ValidationError {
            scope: scope.to_string(),
            inst_index: at,
            span: inst.span,
            kind,
        });
    }
}

/// The name the source gave the enum the value carries, for the refusal to
/// write `E::B` rather than a bare tag.
fn enum_name_of(body: &MirBody, value: ValueId) -> Option<Astr> {
    match scrutinee_ty(body, value)? {
        Ty::Enum { name, .. } => Some(*name),
        _ => None,
    }
}

/// The variants the value a `Switch` reads its tag from can hold
/// (RFC-0051 §4). The value's type is the answer: a structural enum's type
/// is the union of every construction the value can flow from, so its
/// variant list is closed for this value wherever the value came from.
pub fn known_variants(body: &MirBody, value: ValueId) -> Known {
    let Some(ty) = scrutinee_ty(body, value) else {
        return Known::Open;
    };
    match ty {
        Ty::Option(_) => Known::ClosedBuiltin(OPTION),
        Ty::Result(..) => Known::ClosedBuiltin(RESULT),
        Ty::Enum { variants, .. } => Known::Closed(variants.keys().copied().collect()),
        _ => Known::Open,
    }
}

/// The type of the value a `Switch` reads its tag from. A place scrutinee is
/// lent for the tag read, and what it lends is what names the variants.
fn scrutinee_ty(body: &MirBody, value: ValueId) -> Option<&Ty> {
    match body.val_types.get(&value)? {
        Ty::Ref(_, inner) => Some(&inner.ty),
        ty => Some(ty),
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;
    use acvus_utils::Interner;

    use super::*;
    use crate::ir::{Inst, Label};

    /// `typeck` refuses every source that would reach [`Known::Open`], so the
    /// body is built here.
    #[test]
    fn a_switch_over_a_tag_whose_type_names_no_variants_is_refused() {
        let interner = Interner::new();
        let mut body = MirBody::default();
        let tag = body.val_factory.next();
        body.val_types.insert(tag, Ty::I64);
        body.insts.push(Inst {
            span: Span::ZERO,
            kind: InstKind::Switch {
                tag,
                arms: vec![(interner.intern("A"), Label(0), Vec::new())],
                default: None,
            },
        });

        let mut errors = Vec::new();
        check_body(&body, "main", &mut errors);
        let [error] = errors.as_slice() else {
            panic!("one refusal, got {errors:?}");
        };
        assert!(
            matches!(error.kind, ValidationErrorKind::NonExhaustiveMatch),
            "{:?}",
            error.kind
        );
    }
}
