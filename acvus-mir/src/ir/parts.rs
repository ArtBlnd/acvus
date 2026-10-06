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
use crate::ty::structural::{PartKey, Structure, structure};
use crate::ty::{Concrete, Ty, TyKind, Types};

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
    let end = path.iter().position(|seg| matches!(seg, PathSeg::Payload(_))).unwrap_or(path.len());
    &path[..end]
}

/// The type of the part `seg` names of a value of `holder`, where it has
/// that part: a tuple's element at its position, an array's or a slice's
/// element at any index, a field, or the payload of a variant (types §1).
/// The one place a step of a path is typed.
pub(crate) fn step_ty(types: &Types<Concrete>, holder: Ty, seg: PathSeg) -> Option<Ty> {
    if let (PathSeg::Index(_), TyKind::Slice(element)) = (seg, types.kind(holder)) {
        return Some(*element);
    }
    let structure = structure(types, holder)?;
    let key = match (seg, structure) {
        (PathSeg::Index(at), Structure::Tuple(..)) => PartKey::Index(at),
        (PathSeg::Index(_), Structure::Array { .. }) => PartKey::Element,
        (PathSeg::Index(_), _) => return None,
        (PathSeg::Field(name), _) => PartKey::Field(name),
        (PathSeg::Payload(variant), _) => PartKey::Payload(variant),
    };
    structure.part(key)
}

/// The type of the part of a `ty` at `path`.
///
/// # Panics
/// `path` steps into a part its type does not have: a path names only
/// parts its type has.
pub(crate) fn part_ty(types: &Types<Concrete>, ty: Ty, path: &[PathSeg]) -> Ty {
    path.iter().fold(ty, |at, seg| {
        step_ty(types, at, *seg).unwrap_or_else(|| panic!("{seg:?} names no part of {:?}; a path names only parts its type has", types.kind(at)))
    })
}

/// The steps from a `ty` to each of its immediate parts.
///
/// # Panics
/// `ty` has no parts a path names: a cut steps only into a product.
fn children(types: &Types<Concrete>, ty: Ty) -> Vec<PathSeg> {
    product_parts(types, ty).unwrap_or_else(|| panic!("a cut steps only into a tuple, an array or an object, not {:?}", types.kind(ty)))
}

/// The steps from a `ty` to each of its immediate parts, where it is a
/// product.
fn product_parts(types: &Types<Concrete>, ty: Ty) -> Option<Vec<PathSeg>> {
    match types.kind(ty) {
        TyKind::Tuple(items, _) => Some((0..items.len()).map(PathSeg::Index).collect()),
        TyKind::Array(_, len, _) => Some((0..known_len(len)).map(PathSeg::Index).collect()),
        TyKind::Object { fields, .. } => Some(fields.iter().map(|(name, _)| PathSeg::Field(*name)).collect()),
        _ => None,
    }
}

/// Whether `units` can cut a holder of type `ty` at `path`: every step of
/// it is into a part of a product.
pub(crate) fn is_cut_of(types: &Types<Concrete>, ty: Ty, path: &[PathSeg]) -> bool {
    let mut at = ty;
    for seg in path {
        let Some(parts) = product_parts(types, at) else {
            return false;
        };
        if !parts.contains(seg) {
            return false;
        }
        at = part_ty(types, at, std::slice::from_ref(seg));
    }
    true
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
