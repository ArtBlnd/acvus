//! Access (extern-fn §3): which places a call touches, and how.

use acvus_utils::QualifiedRef;

use super::{Element, Functions, Place, Stated, fits, fits_part};
use crate::decl::{Refused, Scheme};
use crate::ty::{EffectTerm, Mutability, Poly, Reissue, TyKind, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    /// A call touches these places, each in its mode, and no other place.
    Touches(Box<[Touch]>),
    /// The declaration, an impl of `eq`, is an equivalence: two keys of its
    /// type name one place exactly when it calls them equal.
    Equivalence,
    /// The call's result is what `place` held when the call began, or,
    /// where `through` names one, that declared pure `fn(&H) -> R` of it
    /// (extern-fn §3): a call that stores into a place may give back what
    /// it held.
    Gives { place: Place, through: Option<QualifiedRef> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Touch {
    pub place: Place,
    pub mode: Mode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Read,
    /// A `&` or `&mut` to the place is given.
    View(Mutability),
    /// An absent entry becomes the value of the parameter `default`, a
    /// present one stays, and a `&mut` to it is given.
    Open { default: usize },
    Store(Stored),
}

/// What a store leaves the place holding, whatever it held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// The value of this parameter.
    Value(usize),
    Emptied,
}

impl Mode {
    /// Whether the call reads what the place held.
    pub fn reads(self) -> bool {
        match self {
            Mode::Read | Mode::View(_) | Mode::Open { .. } => true,
            Mode::Store(_) => false,
        }
    }

    /// Whether the call may change what the place holds.
    pub fn writes(self) -> bool {
        match self {
            Mode::Read | Mode::View(Mutability::Shared) => false,
            Mode::View(Mutability::Mut) | Mode::Open { .. } | Mode::Store(_) => true,
        }
    }
}

/// Refuses each access that is not well formed at `stated`'s signature.
pub(super) fn refuse_ill_formed(
    stated: &Stated<'_>,
    access: &[Access],
    functions: &Functions<'_>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    let declared = stated.declared;
    if access.iter().filter(|form| matches!(form, Access::Touches(_))).count() > 1 {
        refused.push(Refused::TwoAccesses { declared });
    }
    if access.iter().filter(|form| matches!(form, Access::Gives { .. })).count() > 1 {
        refused.push(Refused::TwoResultsGiven { declared });
    }
    for form in access {
        match form {
            Access::Touches(touches) => {
                for touch in touches.iter() {
                    refuse_ill_formed_touch(stated, types, touch, refused);
                }
            }
            Access::Equivalence if stated.impl_of != Some(stated.core.eq) => {
                refused.push(Refused::EquivalenceNotOnEq { declared });
            }
            Access::Equivalence => {}
            Access::Gives { place, through } => refuse_ill_formed_gives(stated, place, *through, functions, types, refused),
        }
    }
}

/// The result is what the place held, or `through` of it: `through` is a
/// declared `fn(&H) -> R`, `H` what the place holds and `R` the call's
/// result; without it, the result's type is what the place holds. Where
/// the signature does not say what the place holds, only `R` is compared.
fn refuse_ill_formed_gives(
    stated: &Stated<'_>,
    place: &Place,
    through: Option<QualifiedRef>,
    functions: &Functions<'_>,
    types: &mut Types<Poly>,
    refused: &mut Vec<Refused>,
) {
    if !stated.place_well_formed(types, place, false, refused) {
        return;
    }
    let declared = stated.declared;
    let held = stated.held_at(types, place);
    let Some(through) = through else {
        if held.is_some_and(|held| held != stated.ret) {
            refused.push(Refused::ResultNotWhatThePlaceHeld { declared, param: place.param });
        }
        return;
    };
    let Some(ty) = functions.ty(through) else {
        refused.push(Refused::GivesThroughDoesNotFit { declared, through });
        return;
    };
    let fitting = match held {
        Some(held) => {
            let lent = types.intern(TyKind::Ref(Mutability::Shared, held));
            fits(types, ty, &[lent], stated.ret)
        }
        None => match types.kind(ty) {
            TyKind::Fn { params, ret, .. } => {
                let ret = *ret;
                matches!(params[..], [param] if matches!(types.kind(param), TyKind::Ref(Mutability::Shared, _)))
                    && fits_part(types, ty, ret, stated.ret)
            }
            _ => false,
        },
    };
    if !fitting {
        refused.push(Refused::GivesThroughDoesNotFit { declared, through });
    }
    // A generic function states no effect, so it is declared pure only
    // where it is a declaration (extern-fn §6).
    if !functions.declaration(through).is_some_and(|scheme| is_declared_pure(types, scheme)) {
        refused.push(Refused::GivesThroughNotPure { declared, through });
    }
}

/// Whether a call of `scheme` is pure: its own effect is known pure and it
/// runs no work its parameters give (extern-fn §2).
fn is_declared_pure(types: &Types<Poly>, scheme: &Scheme) -> bool {
    let own_pure = match types.kind(scheme.ty) {
        TyKind::Fn { effect: EffectTerm::Known(effect), .. } => effect.reissue() == Reissue::Pure,
        _ => false,
    };
    own_pure && scheme.runs.is_empty()
}

fn refuse_ill_formed_touch(stated: &Stated<'_>, types: &Types<Poly>, touch: &Touch, refused: &mut Vec<Refused>) {
    if !stated.place_well_formed(types, &touch.place, touch.mode.writes(), refused) {
        return;
    }
    let declared = stated.declared;
    match touch.mode {
        Mode::Read | Mode::View(_) | Mode::Store(Stored::Emptied) => {}
        Mode::Open { default } => {
            stated.param(default, refused);
            if !matches!(touch.place.element, Some(Element::Key(_))) {
                refused.push(Refused::OpenOfNoEntry { declared, param: touch.place.param });
            }
        }
        Mode::Store(Stored::Value(value)) => {
            stated.param(value, refused);
        }
    }
}
