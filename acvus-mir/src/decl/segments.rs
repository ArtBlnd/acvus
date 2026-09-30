//! The segments of a declared type (`labels`): what an extern's lifetimes
//! label, and what a call of it lays over its concrete positions
//! (`call_flows`).

use super::Declarations;
use crate::ty::{LifetimesOf, Poly, TyId, TyKind, Types};

/// One segment of a declared type, in the order `segments` reads them
/// (`labels`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Segment {
    /// A reference's own position.
    Reference,
    /// One lifetime an extern-defined type declares.
    Lifetime,
    /// A function value's one position (ownership §5).
    FunctionValue,
    /// Every position of what instantiates this type variable.
    TypeVar(u32),
}

impl Segment {
    /// Whether position `k` of `self` takes position `k` of `from` alone:
    /// both are one type variable's part (`labels`). Otherwise every
    /// position of `self` takes every position of `from`.
    pub fn aligned_with(self, from: Segment) -> bool {
        matches!((self, from), (Segment::TypeVar(to), Segment::TypeVar(from)) if to == from)
    }
}

/// The segments of the declared type `ty`, in order: a reference's own
/// position before its target's, an extern-defined type's lifetimes before
/// its arguments', a structural type's parts in order, and one element's
/// for a sequence. A function value's one position is one segment.
///
/// # Panics
/// `ty` holds a `Handle` or an `Order`, which no declaration writes.
pub fn segments(poly: &Types<Poly>, decls: &Declarations, ty: TyId<Poly>) -> Vec<Segment> {
    let mut out = Vec::new();
    push_segments(poly, decls, ty, &mut out);
    out
}

fn push_segments(poly: &Types<Poly>, decls: &Declarations, ty: TyId<Poly>, out: &mut Vec<Segment>) {
    acvus_utils::grow(|| push_segments_level(poly, decls, ty, out))
}

fn push_segments_level(poly: &Types<Poly>, decls: &Declarations, ty: TyId<Poly>, out: &mut Vec<Segment>) {
    match poly.kind(ty) {
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Str => {}
        TyKind::Fn { .. } => out.push(Segment::FunctionValue),
        TyKind::Order | TyKind::Handle(_) | TyKind::Closures(_) => panic!("a declared type holds no Handle, Order or Closures"),
        TyKind::Var(var) => out.push(Segment::TypeVar(*var)),
        TyKind::Ref(_, target) => {
            out.push(Segment::Reference);
            push_segments(poly, decls, *target, out);
        }
        TyKind::UserDefined { id, type_args, .. } => {
            out.extend(std::iter::repeat_n(Segment::Lifetime, decls.lifetimes_of(*id)));
            for arg in type_args {
                push_segments(poly, decls, *arg, out);
            }
        }
        TyKind::Array(element, _, _) | TyKind::Slice(element) | TyKind::Option(element, _) => {
            push_segments(poly, decls, *element, out)
        }
        TyKind::Result(ok, err, _) => {
            push_segments(poly, decls, *ok, out);
            push_segments(poly, decls, *err, out);
        }
        TyKind::Tuple(items, _) => {
            for item in items {
                push_segments(poly, decls, *item, out);
            }
        }
        TyKind::Object { fields, .. } => {
            for (_, field) in fields {
                push_segments(poly, decls, *field, out);
            }
        }
        TyKind::Enum { variants, .. } => {
            for payload in variants.iter().filter_map(|(_, payload)| *payload) {
                push_segments(poly, decls, payload, out);
            }
        }
    }
}
