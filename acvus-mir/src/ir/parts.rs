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

use super::cfg::Op;
use super::{PathSeg, ValueId, known_len};
use crate::ty::structural::{PartKey, element, structure};
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
/// that part: a sequence's element at any index (`element`), a tuple's
/// element at its position, a field, or the payload of a variant (types
/// §1). The one place a step of a path is typed.
pub(crate) fn step_ty(types: &Types<Concrete>, holder: Ty, seg: PathSeg) -> Option<Ty> {
    match seg {
        PathSeg::Index(at) => element(types, holder).or_else(|| structure(types, holder)?.part(PartKey::Index(at))),
        PathSeg::Field(name) => structure(types, holder)?.part(PartKey::Field(name)),
        PathSeg::Payload(variant) => structure(types, holder)?.part(PartKey::Payload(variant)),
    }
}

/// The step to the part `key` names, where one step names it: a tuple's
/// element at its position, a field, a variant's payload; `None` for an
/// array's elements, which every index names (ownership §5). `step_ty`
/// reads a step back to its part.
pub(crate) fn step(key: PartKey) -> Option<PathSeg> {
    match key {
        PartKey::Index(at) => Some(PathSeg::Index(at)),
        PartKey::Field(name) => Some(PathSeg::Field(name)),
        PartKey::Payload(variant) => Some(PathSeg::Payload(variant)),
        PartKey::Element => None,
    }
}

/// The operation that takes the part at `path` out of `src`, a value of
/// `holder`: a `FieldGet` of a path of fields, a `TupleIndex` or an
/// `ArrayIndex` of one index, as `holder` is a tuple or an array, and an
/// `UnwrapVariant` of one payload; `None` for any other path, which no one
/// operation takes. `Op::extracted` reads one back to its path.
pub(crate) fn extraction(types: &Types<Concrete>, holder: Ty, src: ValueId, path: &[PathSeg]) -> Option<Op> {
    match path {
        [PathSeg::Field(field), rest @ ..] => {
            let rest = rest
                .iter()
                .map(|seg| match seg {
                    PathSeg::Field(name) => Some(*name),
                    PathSeg::Index(_) | PathSeg::Payload(_) => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Op::FieldGet { object: src, field: *field, rest })
        }
        [PathSeg::Index(index)] => match types.kind(holder) {
            TyKind::Tuple(..) => Some(Op::TupleIndex { tuple: src, index: *index }),
            TyKind::Array(..) => Some(Op::ArrayIndex { array: src, index: *index }),
            _ => None,
        },
        [PathSeg::Payload(variant)] => Some(Op::UnwrapVariant { src, variant: *variant }),
        _ => None,
    }
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
/// product: a field's own step (`Structure::fields`), and each index of an
/// array, whose every element a cut may move apart (ownership §6).
fn product_parts(types: &Types<Concrete>, ty: Ty) -> Option<Vec<PathSeg>> {
    if let TyKind::Array(_, len, _) = types.kind(ty) {
        return Some((0..known_len(len)).map(PathSeg::Index).collect());
    }
    let fields = structure(types, ty)?.fields()?;
    Some(fields.iter().map(|field| step(field.key).expect("a field has one step")).collect())
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

#[cfg(test)]
mod tests {
    use acvus_utils::{Interner, LocalIdOps, QualifiedRef};

    use super::{extraction, step, step_ty};
    use crate::ir::cfg::Op;
    use crate::ir::{PathSeg, ValueId};
    use crate::ty::structural::{PartKey, VariantKey, structure};
    use crate::ty::{Concrete, IntTy, LenTerm, TyKind, Types};

    /// A part's own step leads back to it: of each part `parts` lists, the
    /// step `step` names it by is typed by `step_ty` as that part, but an
    /// array's elements, which no one step names and every index reaches.
    #[test]
    fn a_parts_step_is_typed_as_that_part() {
        let names = Interner::new();
        let (a, full) = (names.intern("a"), names.intern("Full"));
        let mut types = Types::<Concrete>::new();
        let (int, string) = (types.intern(TyKind::Int(IntTy::I64)), types.intern(TyKind::String));
        let structured = [
            types.intern(TyKind::Tuple(Box::new([string, int]), ())),
            types.intern(TyKind::Object { fields: Box::new([(a, int)]), label: () }),
            types.intern(TyKind::Enum { name: QualifiedRef::root(names.intern("E")), variants: Box::new([(full, Some(int))]), label: () }),
            types.intern(TyKind::Option(string, ())),
            types.intern(TyKind::Result(string, int, ())),
        ];
        for ty in structured {
            for part in structure(&types, ty).expect("a structural type").parts() {
                let seg = step(part.key).expect("a part other than an array's elements has one step");
                assert_eq!(step_ty(&types, ty, seg), Some(part.ty), "{:?} of {:?}", part.key, types.kind(ty));
            }
        }
        assert_eq!(step(PartKey::Element), None);
        let array = types.intern(TyKind::Array(string, LenTerm::Known(3), ()));
        let slice = types.intern(TyKind::Slice(int));
        assert_eq!(step_ty(&types, array, PathSeg::Index(2)), Some(string));
        assert_eq!(step_ty(&types, slice, PathSeg::Index(7)), Some(int));
    }

    /// An extraction takes the part its path names, and `Op::extracted`
    /// reads that path back: fields by a `FieldGet`, one index of a tuple or
    /// an array by a `TupleIndex` or an `ArrayIndex`, one payload by an
    /// `UnwrapVariant`; no one operation takes any other path.
    #[test]
    fn an_extraction_names_the_path_it_takes() {
        let names = Interner::new();
        let (a, b) = (names.intern("a"), names.intern("b"));
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let tuple = types.intern(TyKind::Tuple(Box::new([int, int]), ()));
        let array = types.intern(TyKind::Array(int, LenTerm::Known(2), ()));
        let inner = types.intern(TyKind::Object { fields: Box::new([(b, int)]), label: () });
        let object = types.intern(TyKind::Object { fields: Box::new([(a, inner)]), label: () });
        let option = types.intern(TyKind::Option(int, ()));
        let src = ValueId::from_raw(0);
        let taken = [
            (object, vec![PathSeg::Field(a), PathSeg::Field(b)]),
            (tuple, vec![PathSeg::Index(1)]),
            (array, vec![PathSeg::Index(1)]),
            (option, vec![PathSeg::Payload(VariantKey::Some)]),
        ];
        for (holder, path) in taken {
            let op = extraction(&types, holder, src, &path).expect("one operation takes the path");
            assert_eq!(op.extracted(), Some((src, path.clone())), "{op:?}");
        }
        assert!(matches!(extraction(&types, tuple, src, &[PathSeg::Index(0)]), Some(Op::TupleIndex { .. })));
        assert!(matches!(extraction(&types, array, src, &[PathSeg::Index(0)]), Some(Op::ArrayIndex { .. })));
        let refused = [
            (object, vec![PathSeg::Field(a), PathSeg::Index(0)]),
            (tuple, vec![PathSeg::Index(0), PathSeg::Index(1)]),
            (object, vec![PathSeg::Index(0)]),
            (option, Vec::new()),
        ];
        for (holder, path) in refused {
            assert_eq!(extraction(&types, holder, src, &path).map(|op| op.extracted()), None, "{path:?}");
        }
    }
}
