//! Action (extern-fn §3): what a call's update of a place is. That an
//! assignment keeps the later value is the language's, so it is never
//! declared.

use acvus_utils::QualifiedRef;

use super::{Functions, Place, Stated, fits};
use crate::decl::Refused;
use crate::ty::{Mutability, Poly, TyId, TyKind, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// The declaration is a combine (extern-fn §3): an associative
    /// operation on `S`, value shaped `fn(S, S) -> S` or update shaped
    /// `fn(&mut S, S)`, one meaning `s = f(s, p)`. Its laws are stated here
    /// only; an update names it and reads them here.
    Combine(Combine),
    Update(Update),
    /// A call's update of `place` is the inverse of `other`'s update of the
    /// same place: one after the other leaves the place as it was.
    InverseOf { place: Place, other: QualifiedRef },
    /// The declaration, of type `fn(&T, &T) -> int`, is a total order on
    /// `T`: the sign of its answer compares its two arguments, and equal
    /// values are one value. It yields the combines that keep the least
    /// and the greatest of two `T`s.
    TotalOrder,
}

/// A combine's laws: an identity, a declared `fn() -> S`, where one is
/// stated; commutative or not; and an absorbing element, a declared
/// `fn() -> S` the combine returns whatever the other operand, where one
/// is stated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combine {
    pub identity: Option<QualifiedRef>,
    pub commutative: bool,
    pub absorbing: Option<QualifiedRef>,
}

/// A call's update of `place` is the action of `combine`, a declaration
/// stating `Action::Combine`, whose laws it reads there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Update {
    pub place: Place,
    pub combine: QualifiedRef,
}

/// Refuses each action that is not well formed at `stated`'s signature.
pub(super) fn refuse_ill_formed(
    stated: &Stated<'_>,
    actions: &[Action],
    functions: &Functions<'_>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    for action in actions {
        match action {
            Action::Combine(combine) => refuse_ill_formed_combine(stated, combine, functions, types, refused),
            Action::Update(update) => refuse_ill_formed_update(stated, update, functions, types, refused),
            Action::InverseOf { place, other } => refuse_ill_formed_inverse(stated, place, *other, functions, types, refused),
            Action::TotalOrder if !stated.is_total_order_shape(types) => {
                refused.push(Refused::TotalOrderDoesNotFit { declared });
            }
            Action::TotalOrder => {}
        }
    }
}

fn refuse_ill_formed_update(
    stated: &Stated<'_>,
    update: &Update,
    functions: &Functions<'_>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    if !stated.place_well_formed(types, &update.place, true, refused) {
        return;
    }
    let Some(combine) = functions.ty(update.combine).filter(|_| states_combine(functions, update.combine)) else {
        refused.push(Refused::UpdateByNoCombine { declared, combine: update.combine });
        return;
    };
    if let Some(held) = stated.held_at(types, &update.place) {
        let lent_mut = types.intern(TyKind::Ref(Mutability::Mut, held));
        let unit = types.intern(TyKind::Unit);
        let fitting = fits(types, combine, &[lent_mut, held], unit) || fits(types, combine, &[held, held], held);
        if !fitting {
            refused.push(Refused::CombineDoesNotFit { declared, combine: update.combine });
        }
    }
}

/// Whether `function`, or one of its impls where it is a generic function,
/// is declared a combine.
fn states_combine(functions: &Functions<'_>, function: QualifiedRef) -> bool {
    functions
        .stated_facts(function)
        .into_iter()
        .any(|facts| facts.action.iter().any(|action| matches!(action, Action::Combine(_))))
}

/// A combine is an operation on one `S`, value shaped or update shaped,
/// and its identity and absorbing element, where stated, are declared
/// `fn() -> S`.
fn refuse_ill_formed_combine(
    stated: &Stated<'_>,
    combine: &Combine,
    functions: &Functions<'_>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    let Some(state) = operation_state(types, &stated.params, stated.ret) else {
        refused.push(Refused::CombineNotAnOperation { declared });
        return;
    };
    let mut is_element = |element: QualifiedRef| {
        let element_ty = functions.ty(element);
        element_ty.is_some_and(|ty| fits(types, ty, &[], state))
    };
    if let Some(identity) = combine.identity
        && !is_element(identity)
    {
        refused.push(Refused::IdentityDoesNotFit { declared, identity });
    }
    if let Some(absorbing) = combine.absorbing
        && !is_element(absorbing)
    {
        refused.push(Refused::AbsorbingDoesNotFit { declared, absorbing });
    }
}

/// An inverse is of an update: `other` states an update of the same place
/// by a combine.
fn refuse_ill_formed_inverse(
    stated: &Stated<'_>,
    place: &Place,
    other: QualifiedRef,
    functions: &Functions<'_>,
    types: &Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    if !stated.place_well_formed(types, place, true, refused) {
        return;
    }
    let updates_the_place = functions.stated_facts(other).into_iter().any(|facts| {
        facts.action.iter().any(|action| matches!(action, Action::Update(update) if update.place == *place))
    });
    if !updates_the_place {
        refused.push(Refused::InverseNotAnUpdateAtThePlace { declared: stated.declared, inverse: other });
    }
}

/// `S` of an operation `fn(S, S) -> S` or `fn(&mut S, S) -> ()`; `None` for
/// a function of any other type.
fn operation_state(types: &Types<Poly>, params: &[TyId<Poly>], ret: TyId<Poly>) -> Option<TyId<Poly>> {
    let (state, operand) = super::step_state(types, params, ret)?;
    (state == operand).then_some(state)
}
