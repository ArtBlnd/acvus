//! Which generic functions have an impl at a transparent structured type,
//! and how it follows from its parts' (types §1). Derived once, from the
//! declaration alone, by `verify`: the lang items' generic functions and
//! an extern's alike.

use super::declared::NamedFact;
use super::scheme::Scheme;
use crate::ty::{IntTy, Mutability, Poly, TyId, TyKind, Types};

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
    Combine { shape: LiftShape, law: NamedFact },
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
    const VAR: u32 = 0;
    if scheme.bounds.len() != 1 || scheme.requires.iter().any(|required| required.fact.is_some()) {
        return None;
    }
    let TyKind::Fn { params, ret, .. } = types.kind(scheme.ty) else {
        return None;
    };
    let params: Box<[Taking]> = params.iter().map(|param| taking(types, *param, VAR)).collect::<Option<_>>()?;
    let shape = LiftShape { var: VAR, params };
    match types.kind(*ret) {
        TyKind::Var(var) if *var == VAR => Some(Lift::Rebuild(shape)),
        TyKind::Bool if *shape.params == [Taking::Ref(Mutability::Shared); 2] => {
            Some(Lift::Combine { shape, law: NamedFact::Equivalence })
        }
        // The lexicographic combination of total orders is a total order.
        TyKind::Int(IntTy::I64) if *shape.params == [Taking::Ref(Mutability::Shared); 2] => {
            Some(Lift::Combine { shape, law: NamedFact::TotalOrder })
        }
        _ => None,
    }
}

/// How `param` takes the type variable `var`, where it takes it.
fn taking(types: &Types<Poly>, param: TyId<Poly>, var: u32) -> Option<Taking> {
    let is_var = |ty: TyId<Poly>| matches!(types.kind(ty), TyKind::Var(at) if *at == var);
    match types.kind(param) {
        _ if is_var(param) => Some(Taking::Value),
        TyKind::Ref(mutability, target) if is_var(target.ty()) => Some(Taking::Ref(*mutability)),
        _ => None,
    }
}
