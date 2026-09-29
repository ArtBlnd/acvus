//! Action (extern-fn §3): what a call's update of a place is. That an
//! assignment keeps the later value is the language's, so it is never
//! declared.

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use super::{Place, Stated};
use crate::decl::Refused;
use crate::decl::Scheme;
use crate::decl::impl_fit::is_impl_of;
use crate::ty::{Mutability, Poly, TyId, TyKind, TypeArg, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Update(Update),
    /// The declaration, of type `fn(&T, &T) -> int`, is a total order on
    /// `T`: the sign of its answer compares its two arguments, and equal
    /// values are one value. It yields the combines that keep the least
    /// and the greatest of two `T`s.
    TotalOrder,
}

/// A call's update of `place` is the action of `combine`, a declared
/// `fn(s: &mut S, part: S)`: associative, with `identity`, a declared
/// `fn() -> S`, where one is stated, and commutative where stated. Where
/// `inverse_of` is stated, that function's update of the same place and
/// this one, one after the other, leave the place as it was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Update {
    pub place: Place,
    pub combine: QualifiedRef,
    pub identity: Option<QualifiedRef>,
    pub commutative: bool,
    pub inverse_of: Option<QualifiedRef>,
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
            let lent_mut = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(held)));
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
    if let Some(inverse) = update.inverse_of {
        let updates_the_place = functions.get(&inverse).is_some_and(|scheme| {
            super::stated_facts(scheme).any(|facts| {
                facts.action.iter().any(|action| matches!(action, Action::Update(other) if other.place == update.place))
            })
        });
        if !updates_the_place {
            refused.push(Refused::InverseNotAnUpdateAtThePlace { declared, inverse });
        }
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
    (lent.ty() == second && matches!(types.kind(*ret), TyKind::Unit)).then_some(second)
}

/// Whether the declared function type `general` is `fn(params) -> ret` at
/// some choice of its own variables (extern-fn §6's matching). The effect,
/// flows, calls and holds of `general` are its own, so only the
/// parameters and the result are compared.
fn fits(types: &mut Types<Poly>, general: TyId<Poly>, params: &[TyId<Poly>], ret: TyId<Poly>) -> bool {
    let TyKind::Fn { effect, flows, calls, holds, .. } = types.kind(general).clone() else {
        return false;
    };
    let specific = types.intern(TyKind::Fn { params: params.into(), ret, effect, flows, calls, holds });
    is_impl_of(types, general, specific)
}
