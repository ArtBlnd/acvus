//! The parts of a value or of a place's content, named by paths
//! (ownership §6): what a partial move leaves behind is ended part by part.
//!
//! A holder's parts are its units: the coarsest paths such that every path
//! the body moves out of it or stores into it is a union of units. A holder
//! nothing cuts has the one unit `[]`, itself whole.
//!
//! A unit never names a variant's payload: moving a variant's payload out
//! moves everything the variant owns, since its tag copies and a variant
//! has one payload, so a path through a payload moves the region up to that
//! variant (`owned_region`).

use super::{PathSeg, known_len};
use crate::ty::{Concrete, ObjectTy, Ty, TyKind, Types};

pub(crate) type Path = Vec<PathSeg>;

/// Whether `outer` names `inner` or a region holding it.
pub(crate) fn covers(outer: &[PathSeg], inner: &[PathSeg]) -> bool {
    inner.starts_with(outer)
}

/// Whether the regions `a` and `b` name share anything.
pub(crate) fn overlap(a: &[PathSeg], b: &[PathSeg]) -> bool {
    covers(a, b) || covers(b, a)
}

/// The region a move of `path` empties: `path` up to the first payload it
/// steps into, since that variant then owns nothing more.
pub(crate) fn owned_region(path: &[PathSeg]) -> &[PathSeg] {
    let end = path.iter().position(|seg| *seg == PathSeg::Payload).unwrap_or(path.len());
    &path[..end]
}

/// The type of the part of a `ty` at `path`.
///
/// # Panics
/// `path` steps into a type without that part, or into a payload, whose
/// type the path alone does not name: callers pass `owned_region`s.
pub(crate) fn part_ty(types: &Types<Concrete>, ty: Ty, path: &[PathSeg]) -> Ty {
    let mut at = ty;
    for seg in path {
        at = match (types.kind(at), seg) {
            (TyKind::Tuple(items, _), PathSeg::Index(index)) => items[*index],
            (TyKind::Array(element, _, _), PathSeg::Index(_)) => *element,
            (TyKind::Object(object), PathSeg::Field(field)) => field_ty(object, *field),
            (kind, seg) => panic!("{seg:?} names no part of {kind:?}; a path names only parts its type has"),
        };
    }
    at
}

fn field_ty(object: &ObjectTy<Concrete>, field: acvus_utils::Astr) -> Ty {
    object
        .fields()
        .iter()
        .find(|(name, _)| *name == field)
        .map(|(_, ty)| *ty)
        .unwrap_or_else(|| panic!("{field:?} is no field of {object:?}; a path names only fields its type has"))
}

/// The steps from a `ty` to each of its immediate parts.
///
/// # Panics
/// `ty` has no parts a path names: a cut steps only into a product.
fn children(types: &Types<Concrete>, ty: Ty) -> Vec<PathSeg> {
    match types.kind(ty) {
        TyKind::Tuple(items, _) => (0..items.len()).map(PathSeg::Index).collect(),
        TyKind::Array(_, len, _) => (0..known_len(len)).map(PathSeg::Index).collect(),
        TyKind::Object(object) => object.fields().iter().map(|(name, _)| PathSeg::Field(*name)).collect(),
        kind => panic!("a cut steps only into a tuple, an array or an object, not {kind:?}"),
    }
}

/// The units of a holder of type `ty` that the body cuts at `cuts`.
///
/// Cost: O(cuts · depth · the widest product on a cut's path).
pub(crate) fn units(types: &Types<Concrete>, ty: Ty, cuts: &[Path]) -> Vec<Path> {
    let mut units: Vec<Path> = vec![Vec::new()];
    for cut in cuts {
        let Some(split) = units.iter().position(|unit| unit.len() < cut.len() && covers(unit, cut)) else {
            continue;
        };
        let unit = units.swap_remove(split);
        for depth in unit.len()..cut.len() {
            let above = &cut[..depth];
            let siblings = children(types, part_ty(types, ty, above)).into_iter().filter(|seg| *seg != cut[depth]);
            units.extend(siblings.map(|seg| {
                let mut sibling = above.to_vec();
                sibling.push(seg);
                sibling
            }));
        }
        units.push(cut.clone());
    }
    units
}
