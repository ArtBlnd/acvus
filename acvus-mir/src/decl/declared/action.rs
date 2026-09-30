//! Action (extern-fn §3): what a call's update of a place is. That an
//! assignment keeps the later value is the language's, so it is never
//! declared.

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use super::{Place, Stated};
use crate::decl::Refused;
use crate::decl::Scheme;
use crate::decl::impl_fit::is_impl_of;
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
    functions: &FxHashMap<QualifiedRef, Scheme>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    for action in actions {
        match action {
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
    functions: &FxHashMap<QualifiedRef, Scheme>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    if !stated.place_well_formed(types, &update.place, true, refused) {
        return;
    }
    let combine = functions.get(&update.combine).map(|combine| combine.ty);
    let Some((combine, own_state)) = combine.and_then(|ty| Some((ty, combine_state(types, ty)?))) else {
        refused.push(Refused::CombineDoesNotFit { declared, combine: update.combine });
        return;
    };
    let state = match stated.held_at(types, &update.place) {
        Some(held) => {
            let lent_mut = types.intern(TyKind::Ref(Mutability::Mut, held));
            let unit = types.intern(TyKind::Unit);
            if !fits(types, combine, &[lent_mut, held], unit) {
                refused.push(Refused::CombineDoesNotFit { declared, combine: update.combine });
            }
            held
        }
        None => own_state,
    };
    if let Some(identity) = update.identity {
        let identity_ty = functions.get(&identity).map(|scheme| scheme.ty);
        if !identity_ty.is_some_and(|ty| fits(types, ty, &[], state)) {
            refused.push(Refused::IdentityDoesNotFit { declared, identity });
        }
    }
}

/// An inverse is of an update: `other` states an update of the same place
/// by a combine.
fn refuse_ill_formed_inverse(
    stated: &Stated<'_>,
    place: &Place,
    other: QualifiedRef,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    types: &Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    if !stated.place_well_formed(types, place, true, refused) {
        return;
    }
    let updates_the_place = functions.get(&other).is_some_and(|scheme| {
        super::stated_facts(scheme).any(|facts| {
            facts.action.iter().any(|action| matches!(action, Action::Update(update) if update.place == *place))
        })
    });
    if !updates_the_place {
        refused.push(Refused::InverseNotAnUpdateAtThePlace { declared: stated.declared, inverse: other });
    }
}

/// `S` of a combine of type `fn(&mut S, S) -> ()`; `None` for a function of
/// any other type.
fn combine_state(types: &Types<Poly>, ty: TyId<Poly>) -> Option<TyId<Poly>> {
    let TyKind::Fn { params, ret, .. } = types.kind(ty) else {
        return None;
    };
    let [first, second] = params[..] else {
        return None;
    };
    let TyKind::Ref(Mutability::Mut, lent) = types.kind(first) else {
        return None;
    };
    (*lent == second && matches!(types.kind(*ret), TyKind::Unit)).then_some(second)
}

/// Whether the declared function type `general` is `fn(params) -> ret` at
/// some choice of its own variables (extern-fn §6's matching). The effect,
/// flows, calls and holds of `general` are its own, so only the
/// parameters and the result are compared.
fn fits(types: &mut Types<Poly>, general: TyId<Poly>, params: &[TyId<Poly>], ret: TyId<Poly>) -> bool {
    let TyKind::Fn { effect, flows, calls, holds, .. } = types.kind(general).clone() else {
        return false;
    };
    let specific = types.intern(TyKind::Fn { params: params.into(), ret, effect, flows, calls, holds, values: () });
    is_impl_of(types, general, specific)
}
