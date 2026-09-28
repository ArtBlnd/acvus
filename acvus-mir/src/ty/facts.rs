//! Facts of a concrete type alone: what copies (ownership §1) and where a
//! value holds loans (ownership §5).

use acvus_utils::QualifiedRef;

use super::kind::{Holds, Mutability, ObjectTy, TyKind};
use super::{Concrete, Ty, Types};

/// The lifetimes each extern-defined type declares (extern-fn §8). The
/// declarations answer; a type does not carry the count.
pub trait LifetimesOf {
    fn lifetimes_of(&self, id: QualifiedRef) -> usize;
}

/// Whether a value of `ty` copies: a primitive, a shared reference, or a
/// structural tuple, array, object, `Option` or `Result` whose every part
/// copies (ownership §1).
pub fn copies(types: &Types<Concrete>, ty: Ty) -> bool {
    acvus_utils::grow(|| copies_level(types, ty))
}

fn copies_level(types: &Types<Concrete>, ty: Ty) -> bool {
    match types.kind(ty) {
        TyKind::Int(_) | TyKind::Float | TyKind::Bool | TyKind::Char | TyKind::Unit => true,
        TyKind::Ref(Mutability::Shared, _) => true,
        TyKind::Tuple(items) => items.iter().all(|item| copies(types, *item)),
        TyKind::Array(element, _) | TyKind::Option(element) => copies(types, *element),
        TyKind::Result(ok, err) => copies(types, *ok) && copies(types, *err),
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

/// What one position of a type is (ownership §5, effects §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionKind {
    /// A reference's own position, and how many positions its target
    /// has; the target's follow it.
    Ref(Mutability, usize),
    /// A function value's one position: what its captures hold.
    Captures(Holds),
    /// One lifetime an extern-defined type declares.
    Lifetime,
    /// A started call's arguments, held until it is taken.
    InFlight,
}

/// Every position a value of `ty` holds loans at, in order (ownership §5):
/// a reference's own before its target's; a function value's one; a
/// started call's one before its result's; an extern-defined type's
/// lifetimes before its arguments'; a structural type's parts in order,
/// `Result`'s `Ok` before its `Err`; one element's for a sequence. This is
/// the only walk of positions: `positions` and `position_kinds` read it.
pub fn for_each_position(
    types: &Types<Concrete>,
    lifetimes: &impl LifetimesOf,
    ty: Ty,
    on_position: &mut impl FnMut(PositionKind),
) {
    acvus_utils::grow(|| for_each_position_level(types, lifetimes, ty, on_position))
}

fn for_each_position_level(
    types: &Types<Concrete>,
    lifetimes: &impl LifetimesOf,
    ty: Ty,
    on_position: &mut impl FnMut(PositionKind),
) {
    match types.kind(ty) {
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Order
        | TyKind::Str => {}
        TyKind::Ref(mutability, target) => {
            let mut width = 0;
            for_each_position(types, lifetimes, target.ty(), &mut |_| width += 1);
            on_position(PositionKind::Ref(*mutability, width));
            for_each_position(types, lifetimes, target.ty(), on_position);
        }
        TyKind::Fn { holds, .. } => on_position(PositionKind::Captures(*holds)),
        TyKind::Handle(result) => {
            on_position(PositionKind::InFlight);
            for_each_position(types, lifetimes, *result, on_position);
        }
        TyKind::Array(element, _) | TyKind::Slice(element) | TyKind::Option(element) => {
            for_each_position(types, lifetimes, *element, on_position);
        }
        TyKind::Result(ok, err) => {
            for_each_position(types, lifetimes, *ok, on_position);
            for_each_position(types, lifetimes, *err, on_position);
        }
        TyKind::Tuple(items) => {
            for item in items.iter() {
                for_each_position(types, lifetimes, *item, on_position);
            }
        }
        TyKind::Object(object) => {
            for (_, field) in object.fields() {
                for_each_position(types, lifetimes, *field, on_position);
            }
        }
        TyKind::Enum { variants, .. } => {
            for payload in variants.iter().filter_map(|(_, payload)| *payload) {
                for_each_position(types, lifetimes, payload, on_position);
            }
        }
        TyKind::UserDefined { id, type_args, .. } => {
            for _ in 0..lifetimes.lifetimes_of(*id) {
                on_position(PositionKind::Lifetime);
            }
            for arg in type_args.iter() {
                for_each_position(types, lifetimes, arg.ty(), on_position);
            }
        }
        TyKind::Var(var) => match *var {},
    }
}

/// The kind of every position of `ty`, in order.
pub fn position_kinds(types: &Types<Concrete>, lifetimes: &impl LifetimesOf, ty: Ty) -> Vec<PositionKind> {
    let mut kinds = Vec::new();
    for_each_position(types, lifetimes, ty, &mut |kind| kinds.push(kind));
    kinds
}

/// How many positions a value of `ty` holds loans at.
pub fn positions(types: &Types<Concrete>, lifetimes: &impl LifetimesOf, ty: Ty) -> usize {
    let mut count = 0;
    for_each_position(types, lifetimes, ty, &mut |_| count += 1);
    count
}
