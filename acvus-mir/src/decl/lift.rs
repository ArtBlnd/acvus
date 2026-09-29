//! Which generic functions have an impl at a transparent structured type,
//! and how it follows from its parts' (types §1). Derived once, from the
//! declaration alone, by `verify`: the lang items' generic functions and
//! an extern's alike.

use super::facts::NamedLaw;
use super::scheme::Scheme;
use crate::ty::{Mutability, Poly, Types};

/// A generic function's impl at a transparent structured type, where one
/// follows from the impls at its parts.
#[derive(Debug, Clone, PartialEq)]
pub enum Lift {
    /// Its meaning maps one to one onto the parts: each part's result
    /// rebuilds the structure in its shape, and no law is relied on
    /// (`clone`).
    Rebuild(LiftShape),
    /// Its meaning combines the parts' answers, by the combination the
    /// language defines for `law`: an impl exists only where every part's
    /// impl states `law`, and it states `law` in turn (`eq`, where `law` is
    /// an equivalence).
    Combine { shape: LiftShape, law: NamedLaw },
}

impl Lift {
    pub fn shape(&self) -> &LiftShape {
        match self {
            Lift::Rebuild(shape) | Lift::Combine { shape, .. } => shape,
        }
    }
}

/// Where the structured type stands in the generic function's signature:
/// the one type variable its impls choose, and how each parameter takes it.
#[derive(Debug, Clone, PartialEq)]
pub struct LiftShape {
    pub var: u32,
    pub params: Box<[Taking]>,
}

/// How a parameter takes the structured type: as `T`, `&T` or `&mut T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Taking {
    Value,
    Ref(Mutability),
}

/// `scheme`'s lift, where it has one. A generic function lifts where its
/// impls choose one type variable `T`, every parameter takes `T` as `T`,
/// `&T` or `&mut T`, its requirements name no law, and either its result
/// is `T` (`Lift::Rebuild`), or its result is the answer the combination
/// of a law its impls may state combines (`Lift::Combine`).
pub(super) fn lift_of(types: &Types<Poly>, scheme: &Scheme) -> Option<Lift> {
    todo!("the signature read against the rule of types §1")
}
