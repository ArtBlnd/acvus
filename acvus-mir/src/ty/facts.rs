//! Facts of a concrete type alone: what copies (ownership §1) and where a
//! value holds loans (ownership §5).

use acvus_utils::QualifiedRef;

use super::kind::{Mutability, ObjectTy, TyKind};
use super::{Concrete, Ty, Types};

/// The lifetimes each extern-defined type declares (extern-fn §8). The
/// declarations answer; a type does not carry the count.
pub trait LifetimesOf {
    fn lifetimes_of(&self, id: QualifiedRef) -> usize;
}

/// Whether a value of `ty` copies: a primitive, a shared reference, or a
/// structural tuple, array or object whose every part copies (ownership §1).
pub fn copies(types: &Types<Concrete>, ty: Ty) -> bool {
    acvus_utils::grow(|| copies_level(types, ty))
}

fn copies_level(types: &Types<Concrete>, ty: Ty) -> bool {
    match types.kind(ty) {
        TyKind::Int(_) | TyKind::Float | TyKind::Bool | TyKind::Char | TyKind::Unit => true,
        TyKind::Ref(Mutability::Shared, _) => true,
        TyKind::Tuple(items) => items.iter().all(|item| copies(types, *item)),
        TyKind::Array(element, _) => copies(types, *element),
        TyKind::Object(ObjectTy::Structural(fields, ())) => fields.iter().all(|(_, field)| copies(types, *field)),
        TyKind::String
        | TyKind::Never
        | TyKind::Order
        | TyKind::Handle(_)
        | TyKind::Slice(_)
        | TyKind::Str
        | TyKind::Object(ObjectTy::Declared(..))
        | TyKind::Enum { .. }
        | TyKind::Fn { .. }
        | TyKind::UserDefined { .. }
        | TyKind::Ref(Mutability::Mut, _) => false,
        TyKind::Var(var) => match *var {},
    }
}

/// How many positions a value of `ty` holds loans at (ownership §5): one
/// per reference and then its target's, one per function value, one per
/// lifetime an extern-defined type declares and then its arguments', the
/// parts' of a structural type, and one element's for a sequence. A started
/// call holds its arguments' loans at one position until it is taken
/// (effects §6).
pub fn positions(types: &Types<Concrete>, lifetimes: &impl LifetimesOf, ty: Ty) -> usize {
    acvus_utils::grow(|| positions_level(types, lifetimes, ty))
}

fn positions_level(types: &Types<Concrete>, lifetimes: &impl LifetimesOf, ty: Ty) -> usize {
    match types.kind(ty) {
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Order
        | TyKind::Str => 0,
        TyKind::Ref(_, target) => 1 + positions(types, lifetimes, target.ty()),
        TyKind::Fn { .. } => 1,
        TyKind::Handle(result) => 1 + positions(types, lifetimes, *result),
        TyKind::Array(element, _) | TyKind::Slice(element) => positions(types, lifetimes, *element),
        TyKind::Tuple(items) => items.iter().map(|item| positions(types, lifetimes, *item)).sum(),
        TyKind::Object(object) => object
            .fields()
            .iter()
            .map(|(_, field)| positions(types, lifetimes, *field))
            .sum(),
        TyKind::Enum { variants, .. } => variants
            .iter()
            .filter_map(|(_, payload)| *payload)
            .map(|payload| positions(types, lifetimes, payload))
            .sum(),
        TyKind::UserDefined { id, type_args, .. } => {
            lifetimes.lifetimes_of(*id)
                + type_args
                    .iter()
                    .map(|arg| positions(types, lifetimes, arg.ty()))
                    .sum::<usize>()
        }
        TyKind::Var(var) => match *var {},
    }
}
