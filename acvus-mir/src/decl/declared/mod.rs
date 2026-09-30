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
        }
    }
}

use super::{Core, DeclaredFacts, Refused, Scheme, TyVarBound};
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

/// Refuses every declared fact that is not well formed at the signature it
/// is stated on, and facts stated on a declaration that has impls, which
/// no call reaches (`Scheme`). The generic impl's signature is its
/// function's own.
pub(super) fn refuse_ill_formed(
    types: &mut Types<Poly>,
    core: &Core,
    functions: &FxHashMap<QualifiedRef, Scheme>,
    declared_order: &[QualifiedRef],
    refused: &mut Vec<Refused>,
) {
    for function in declared_order {
        let Some(scheme) = functions.get(function) else {
            continue;
        };
        if scheme.impls.is_some() && !scheme.facts.states_nothing() {
            refused.push(Refused::FactsBesideImpls { function: *function });
        }
        for StatedOn { declared, ty, bounds, facts } in stated_signatures(*function, scheme) {
            let Some(stated) = Stated::of(declared, core, types, ty, bounds) else {
                continue;
            };
            access::refuse_ill_formed(&stated, types, &facts.access, refused);
            action::refuse_ill_formed(&stated, &facts.action, functions, types, refused);
            source::refuse_ill_formed(&stated, types, &facts.source, refused);
            relation::refuse_ill_formed(&stated, types, &facts.relations, refused);
        }
    }
}

/// A set of facts and the signature it is stated on.
struct StatedOn<'s> {
    declared: Declared,
    ty: TyId<Poly>,
    bounds: &'s [TyVarBound],
    facts: &'s DeclaredFacts,
}

/// Each set of facts with its signature: the declaration's own and its
/// generic impl's, both at its own type and bounds, then each concrete
/// impl's, at the impl's own type and bounds.
fn stated_signatures(function: QualifiedRef, scheme: &Scheme) -> impl Iterator<Item = StatedOn<'_>> {
    let own = Declared { function, r#impl: None };
    let at_own = move |facts| StatedOn { declared: own, ty: scheme.ty, bounds: &scheme.bounds, facts };
    let generic = scheme.impls.iter().filter_map(|impls| impls.generic.as_ref()).map(move |generic| at_own(&generic.facts));
    let concrete =
        scheme.impls.iter().flat_map(|impls| impls.concrete.iter().enumerate()).map(move |(at, r#impl)| StatedOn {
            declared: Declared { function, r#impl: Some(at) },
            ty: r#impl.ty,
            bounds: &r#impl.bounds,
            facts: &r#impl.facts,
        });
    std::iter::once(at_own(&scheme.facts)).chain(generic).chain(concrete)
}

/// Every set of facts stated on `scheme` or on one of its impls.
fn stated_facts(scheme: &Scheme) -> impl Iterator<Item = &DeclaredFacts> {
    let impls = scheme.impls.iter();
    let generic = impls.clone().filter_map(|impls| impls.generic.as_ref()).map(|generic| &generic.facts);
    let concrete = impls.flat_map(|impls| impls.concrete.iter()).map(|r#impl| &r#impl.facts);
    std::iter::once(&scheme.facts).chain(generic).chain(concrete)
}

/// The signature a set of facts is stated on.
struct Stated<'a> {
    declared: Declared,
    core: &'a Core,
    params: Box<[TyId<Poly>]>,
    ret: TyId<Poly>,
    bounds: &'a [TyVarBound],
}

impl<'a> Stated<'a> {
    /// `None` where `ty` is no function type, which `verify` refuses on its
    /// own.
    fn of(declared: Declared, core: &'a Core, types: &Types<Poly>, ty: TyId<Poly>, bounds: &'a [TyVarBound]) -> Option<Self> {
        let TyKind::Fn { params, ret, .. } = types.kind(ty) else {
            return None;
        };
        Some(Stated { declared, core, params: params.clone(), ret: *ret, bounds })
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

    /// An integer type, or a type variable bounded to integers.
    fn is_integer(&self, types: &Types<Poly>, ty: TyId<Poly>) -> bool {
        match types.kind(ty) {
            TyKind::Int(_) => true,
            TyKind::Var(at) => matches!(self.bounds.get(*at as usize), Some(TyVarBound::Integer { .. })),
            _ => false,
        }
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
            (Some(Element::Index(_)), TyKind::Slice(element) | TyKind::Array(element, _, _)) => Some(*element),
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
