//! Equality and clone at a structural type are its components', one by one
//! (RFC-0020).

use acvus_utils::{Astr, Interner};

use crate::ty::{Concrete, Phase, Ty, TyId, TyKind, Types};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructuralSignature {
    Eq,
    Clone,
}

/// A component is named rather than numbered because an object or an enum
/// can still gain members after its structural impl settled, and the
/// solver opens a decision for each member it gains beside the ones it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Component {
    Field(Astr),
    TupleElement(usize),
    ArrayElement,
    VariantPayload(Astr),
    Some,
    Ok,
    Err,
}

/// `!` is structural with no component: no value of it exists, so there is
/// nothing to compare or copy.
pub fn components<V>(types: &Types<V>, ty: TyId<V>) -> Option<Vec<(Component, TyId<V>)>>
where
    V: Phase,
{
    let out = match types.kind(ty) {
        TyKind::Object(object) => object
            .fields()
            .iter()
            .map(|(name, field)| (Component::Field(*name), *field))
            .collect(),
        TyKind::Tuple(elements, _) => elements
            .iter()
            .enumerate()
            .map(|(at, element)| (Component::TupleElement(at), *element))
            .collect(),
        TyKind::Array(element, _, _) => vec![(Component::ArrayElement, *element)],
        TyKind::Enum { variants, .. } => variants
            .iter()
            .filter_map(|(tag, payload)| payload.map(|payload| (Component::VariantPayload(*tag), payload)))
            .collect(),
        TyKind::Option(payload, _) => vec![(Component::Some, *payload)],
        TyKind::Result(ok, err, _) => vec![(Component::Ok, *ok), (Component::Err, *err)],
        TyKind::Never => Vec::new(),
        _ => return None,
    };
    Some(out)
}

/// Obligation across artifacts: fields sort by their spelled names, which is
/// `acvus_interpreter::layout::sorted_fields`'s order and so the position of
/// each field in a heap object (RFC-0050 rule 8). Variants sort the same way.
pub fn ordered_components(types: &Types<Concrete>, ty: Ty, interner: &Interner) -> Option<Vec<(Component, Ty)>> {
    let mut out = components(types, ty)?;
    out.sort_by(|(a, _), (b, _)| match (a, b) {
        (Component::Field(a), Component::Field(b))
        | (Component::VariantPayload(a), Component::VariantPayload(b)) => {
            interner.resolve(*a).cmp(interner.resolve(*b))
        }
        (Component::TupleElement(a), Component::TupleElement(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal,
    });
    Some(out)
}

pub fn language_owned(types: &Types<Concrete>, ty: Ty) -> bool {
    matches!(
        types.kind(ty),
        TyKind::Int(_)
            | TyKind::Float
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Unit
            | TyKind::String
            | TyKind::Str
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leaf {
    pub path: Vec<Component>,
    pub ty: Ty,
}

/// Obligation across artifacts: the lowering writes the registry impl of
/// each leaf into `InstKind::StructuralEq::leaves` and
/// `InstKind::StructuralClone::leaves` in this order, and
/// `acvus_interpreter::prepare` finds each leaf's impl by its position
/// here.
pub fn structural_leaves(types: &Types<Concrete>, ty: Ty, interner: &Interner) -> Vec<Leaf> {
    let mut out = Vec::new();
    walk(types, ty, interner, &mut Vec::new(), &mut out);
    out
}

fn walk(types: &Types<Concrete>, ty: Ty, interner: &Interner, path: &mut Vec<Component>, out: &mut Vec<Leaf>) {
    acvus_utils::grow(|| walk_level(types, ty, interner, path, out))
}

fn walk_level(types: &Types<Concrete>, ty: Ty, interner: &Interner, path: &mut Vec<Component>, out: &mut Vec<Leaf>) {
    let Some(parts) = ordered_components(types, ty, interner) else {
        return;
    };
    for (component, part) in parts {
        path.push(component);
        if !language_owned(types, part) {
            match components(types, part) {
                Some(_) => walk(types, part, interner, path, out),
                None => out.push(Leaf {
                    path: path.clone(),
                    ty: part,
                }),
            }
        }
        path.pop();
    }
}
