//! The facts a declaration states beyond its signature, in their four kinds
//! (extern-fn §3): Access, Action, Source and Relation. Each kind's forms
//! are exactly the ledger's, and a form is added only there. `verify`
//! refuses a form that is not well formed at the signature it is stated
//! on; what a form means is read in one place per kind (D82): Access by
//! `passes::analysis::Loans`, Relation by `passes::domain::affine`.

mod access;
mod action;
mod relation;
mod source;

#[cfg(test)]
mod tests;

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

pub use access::{Access, Mode, Stored, Touch};
pub use action::{Action, Combine, Update};
pub use relation::{Atom, Before, Comparison, Relation, Subject, Term};
pub use source::{Source, Step, Stream};

/// A declared fact a requirement names at its type (extern-fn §6), which
/// only an impl stating it resolves: one form of §3's kinds, by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamedFact {
    /// A `core::eq` impl declared an equivalence (`Access::Equivalence`).
    Equivalence,
    /// A `core::cmp` impl declared a total order (`Action::TotalOrder`).
    TotalOrder,
    /// An impl declared a combine (`Action::Combine`).
    Combine,
}

impl NamedFact {
    /// Whether an impl declaring `facts` states this fact.
    pub fn stated_by(self, facts: &DeclaredFacts) -> bool {
        match self {
            NamedFact::Equivalence => facts.access.iter().any(|access| matches!(access, Access::Equivalence)),
            NamedFact::TotalOrder => facts.action.iter().any(|action| matches!(action, Action::TotalOrder)),
            NamedFact::Combine => facts.action.iter().any(|action| matches!(action, Action::Combine(_))),
        }
    }
}

use super::impl_fit::is_impl_of;
use super::{Core, DeclaredFacts, GenericFunction, Impls, Refused, Scheme};
use crate::ty::{Declared, IntTy, Mutability, Poly, TyId, TyKind, Types};

/// A parameter's storage whole, or one element of it. A parameter is
/// numbered as a call's arguments are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Place {
    pub param: usize,
    pub element: Option<Element>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    /// The entry at the key parameter this numbers.
    Key(usize),
    /// The element at the `u64` parameter this numbers.
    Index(usize),
}

/// The functions a declared fact names: a declaration that is no impl, or
/// a generic function, whose impls state its facts. An impl is reached
/// only through its generic function (extern-fn §6), so a fact names none.
pub(in crate::decl) struct Functions<'d> {
    functions: &'d FxHashMap<QualifiedRef, Scheme>,
    generic_functions: &'d FxHashMap<QualifiedRef, GenericFunction>,
    impls: &'d FxHashMap<QualifiedRef, Impls>,
}

impl<'d> Functions<'d> {
    pub(in crate::decl) fn new(
        functions: &'d FxHashMap<QualifiedRef, Scheme>,
        generic_functions: &'d FxHashMap<QualifiedRef, GenericFunction>,
        impls: &'d FxHashMap<QualifiedRef, Impls>,
    ) -> Self {
        Functions { functions, generic_functions, impls }
    }

    /// The declaration `function` names, where it is one and no impl.
    fn declaration(&self, function: QualifiedRef) -> Option<&'d Scheme> {
        self.functions.get(&function).filter(|scheme| scheme.impl_of.is_none())
    }

    /// The signature `function` names: a declaration's, or a generic
    /// function's.
    fn ty(&self, function: QualifiedRef) -> Option<TyId<Poly>> {
        match self.generic_functions.get(&function) {
            Some(generic) => Some(generic.ty),
            None => self.declaration(function).map(|scheme| scheme.ty),
        }
    }

    /// Every set of facts stated for a call of `function`: a declaration's
    /// own, or each of a generic function's impls'.
    fn stated_facts(&self, function: QualifiedRef) -> Vec<&'d DeclaredFacts> {
        match self.impls.get(&function) {
            Some(impls) => impls
                .concrete
                .iter()
                .chain(&impls.generic)
                .filter_map(|r#impl| self.functions.get(r#impl))
                .map(|scheme| &scheme.facts)
                .collect(),
            None => self.declaration(function).map(|scheme| &scheme.facts).into_iter().collect(),
        }
    }
}

/// Refuses every declared fact that is not well formed at the signature of
/// the declaration it is stated on.
pub(in crate::decl) fn refuse_ill_formed(
    types: &mut Types<Poly>,
    core: &Core,
    functions: &Functions<'_>,
    declared_order: &[QualifiedRef],
    refused: &mut Vec<Refused>,
) {
    for function in declared_order {
        let Some(scheme) = functions.functions.get(function) else {
            continue;
        };
        let declared = Declared { function: *function };
        let impl_of = scheme.impl_of.map(|impl_of| impl_of.generic_function);
        let Some(stated) = Stated::of(declared, impl_of, core, types, scheme.ty) else {
            continue;
        };
        let facts = &scheme.facts;
        access::refuse_ill_formed(&stated, &facts.access, functions, types, refused);
        action::refuse_ill_formed(&stated, &facts.action, functions, types, refused);
        source::refuse_ill_formed(&stated, &facts.source, functions, types, refused);
        relation::refuse_ill_formed(&stated, types, &facts.relations, refused);
    }
}

/// The signature a set of facts is stated on.
struct Stated<'a> {
    declared: Declared,
    /// The generic function the declaration is an impl of, where it is
    /// one.
    impl_of: Option<QualifiedRef>,
    core: &'a Core,
    params: Box<[TyId<Poly>]>,
    ret: TyId<Poly>,
}

impl<'a> Stated<'a> {
    /// `None` where `ty` is no function type, which `verify` refuses on its
    /// own.
    fn of(
        declared: Declared,
        impl_of: Option<QualifiedRef>,
        core: &'a Core,
        types: &Types<Poly>,
        ty: TyId<Poly>,
    ) -> Option<Self> {
        let TyKind::Fn { params, ret, .. } = types.kind(ty) else {
            return None;
        };
        Some(Stated { declared, impl_of, core, params: params.clone(), ret: *ret })
    }

    /// The type of parameter `at`; where the signature has none, refused.
    fn param(&self, at: usize, refused: &mut Vec<Refused>) -> Option<TyId<Poly>> {
        let ty = self.params.get(at).copied();
        if ty.is_none() {
            refused.push(Refused::FactNamesNoParameter { declared: self.declared, param: at });
        }
        ty
    }

    /// How `ty` lends and what, where it is a reference.
    fn lent(&self, types: &Types<Poly>, ty: TyId<Poly>) -> Option<(Mutability, TyId<Poly>)> {
        match types.kind(ty) {
            TyKind::Ref(mutability, lent) => Some((*mutability, *lent)),
            _ => None,
        }
    }

    fn is_integer(&self, types: &Types<Poly>, ty: TyId<Poly>) -> bool {
        matches!(types.kind(ty), TyKind::Int(_))
    }

    /// Whether a value of `ty`, or what it lends, has an element count: a
    /// slice, a `str`, an array, a `String`, or an extern-defined type,
    /// whose count is its declarer's.
    fn has_length(&self, types: &Types<Poly>, ty: TyId<Poly>) -> bool {
        let held = self.lent(types, ty).map_or(ty, |(_, lent)| lent);
        matches!(
            types.kind(held),
            TyKind::Slice(_) | TyKind::Str | TyKind::Array(..) | TyKind::String | TyKind::UserDefined { .. }
        )
    }

    /// Whether `place` is well formed, refusing what is not: its parameter
    /// is a reference, `&mut` where the call may write it, and an index
    /// that names its element is a `u64` (operators §3).
    fn place_well_formed(&self, types: &Types<Poly>, place: &Place, writes: bool, refused: &mut Vec<Refused>) -> bool {
        let declared = self.declared;
        let Some(ty) = self.param(place.param, refused) else {
            return false;
        };
        let Some((mutability, _)) = self.lent(types, ty) else {
            refused.push(Refused::PlaceNotLent { declared, param: place.param });
            return false;
        };
        if writes && mutability == Mutability::Shared {
            refused.push(Refused::PlaceNotLentMutably { declared, param: place.param });
            return false;
        }
        match place.element {
            None => true,
            Some(Element::Key(key)) => self.param(key, refused).is_some(),
            Some(Element::Index(index)) => match self.param(index, refused) {
                None => false,
                Some(ty) if matches!(types.kind(ty), TyKind::Int(IntTy::U64)) => true,
                Some(_) => {
                    refused.push(Refused::IndexNotAnIndex { declared, param: index });
                    false
                }
            },
        }
    }

    /// The type `place` holds, where the signature says it: the whole of
    /// what its parameter lends, or an element of a slice or an array. An
    /// entry's type is its container's declarer's.
    ///
    /// # Panics
    /// `place` is not well formed.
    fn held_at(&self, types: &Types<Poly>, place: &Place) -> Option<TyId<Poly>> {
        let (_, lent) = self.lent(types, self.params[place.param]).expect("a well-formed place is lent");
        match (place.element, types.kind(lent)) {
            (None, _) => Some(lent),
            (Some(Element::Index(_)), TyKind::Slice(element) | TyKind::Array(element, _)) => Some(*element),
            (Some(Element::Index(_) | Element::Key(_)), _) => None,
        }
    }

    /// `fn(&T, &T) -> int`, both `T` one type.
    fn is_total_order_shape(&self, types: &Types<Poly>) -> bool {
        let [a, b] = self.params[..] else {
            return false;
        };
        let shared = |ty| match types.kind(ty) {
            TyKind::Ref(Mutability::Shared, lent) => Some(*lent),
            _ => None,
        };
        matches!((shared(a), shared(b)), (Some(a), Some(b)) if a == b) && self.is_integer(types, self.ret)
    }
}

/// The state and the operand of a step `fn(S, P) -> S` or
/// `fn(&mut S, P) -> ()`, each meaning `s = f(s, p)`; `None` for a function
/// of any other shape.
fn step_state(types: &Types<Poly>, params: &[TyId<Poly>], ret: TyId<Poly>) -> Option<(TyId<Poly>, TyId<Poly>)> {
    let [first, operand] = params[..] else {
        return None;
    };
    match (types.kind(first), types.kind(ret)) {
        (TyKind::Ref(Mutability::Mut, state), TyKind::Unit) => Some((*state, operand)),
        _ if first == ret => Some((first, operand)),
        _ => None,
    }
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

/// Whether `general_part`, a part of the declared function type `general`,
/// is `part` at some choice of `general`'s own variables, the rest of
/// `general` left unconstrained: the fit of one position where the
/// signature says nothing of the others.
fn fits_part(types: &mut Types<Poly>, general: TyId<Poly>, general_part: TyId<Poly>, part: TyId<Poly>) -> bool {
    let TyKind::Fn { effect, flows, calls, holds, .. } = types.kind(general).clone() else {
        return false;
    };
    let general = types.intern(TyKind::Fn { params: Box::new([]), ret: general_part, effect, flows, calls, holds, values: () });
    fits(types, general, &[], part)
}
