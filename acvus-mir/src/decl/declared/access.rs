//! Access (extern-fn §3): which places a call touches, and how.

use super::{Element, Place, Stated};
use crate::decl::Refused;
use crate::ty::{Mutability, Poly, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    /// A call touches these places, each in its mode, and no other place.
    Touches(Box<[Touch]>),
    /// The declaration, an impl of `eq`, is an equivalence: two keys of its
    /// type name one place exactly when it calls them equal.
    Equivalence,
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
pub(super) fn refuse_ill_formed(stated: &Stated<'_>, types: &Types<Poly>, access: &[Access], refused: &mut Vec<Refused>) {
    let declared = stated.declared;
    if access.iter().filter(|form| matches!(form, Access::Touches(_))).count() > 1 {
        refused.push(Refused::TwoAccesses { declared });
    }
    for form in access {
        match form {
            Access::Touches(touches) => {
                for touch in touches.iter() {
                    refuse_ill_formed_touch(stated, types, touch, refused);
                }
            }
            Access::Equivalence if stated.declared.function != stated.core.eq => {
                refused.push(Refused::EquivalenceNotOnEq { declared });
            }
            Access::Equivalence => {}
        }
    }
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
