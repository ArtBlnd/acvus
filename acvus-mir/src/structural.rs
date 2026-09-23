//! Equality and clone at a structural type are its components', one by one
//! (RFC-0020).

use acvus_utils::{Astr, Interner};

use crate::ty::{Phase, Ty, TyTerm};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructuralSignature {
    Eq,
    Clone,
}

/// A component is named rather than numbered because an object or an enum
/// can still gain members after its structural instance settled, and the
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
pub fn components<V>(ty: &TyTerm<V>) -> Option<Vec<(Component, &TyTerm<V>)>>
where
    V: Phase,
{
    let out = match ty {
        TyTerm::Object(object) => object
            .iter()
            .map(|(name, field)| (Component::Field(*name), field))
            .collect(),
        TyTerm::Tuple(elements) => elements
            .iter()
            .enumerate()
            .map(|(at, element)| (Component::TupleElement(at), element))
            .collect(),
        TyTerm::Array(element, _) => vec![(Component::ArrayElement, &**element)],
        TyTerm::Enum { variants, .. } => variants
            .iter()
            .filter_map(|(tag, payload)| {
                payload
                    .as_deref()
                    .map(|payload| (Component::VariantPayload(*tag), payload))
            })
            .collect(),
        TyTerm::Option(payload) => vec![(Component::Some, &**payload)],
        TyTerm::Result(ok, err) => vec![(Component::Ok, &**ok), (Component::Err, &**err)],
        TyTerm::Never => Vec::new(),
        _ => return None,
    };
    Some(out)
}

/// Obligation across artifacts: fields sort by their spelled names, which is
/// `acvus_interpreter::layout::sorted_fields`'s order and so the position of
/// each field in a heap object (RFC-0050 rule 8). Variants sort the same way.
pub fn ordered_components<'t>(ty: &'t Ty, interner: &Interner) -> Option<Vec<(Component, &'t Ty)>> {
    let mut out = components(ty)?;
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

pub fn language_owned(ty: &Ty) -> bool {
    matches!(
        ty,
        TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Bool
            | TyTerm::Char
            | TyTerm::Unit
            | TyTerm::String
            | TyTerm::Str
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leaf<'t> {
    pub path: Vec<Component>,
    pub ty: &'t Ty,
}

/// Obligation across artifacts: the lowering writes the registry instance of
/// each leaf into `InstKind::StructuralEq::leaves` and
/// `InstKind::StructuralClone::leaves` in this order, and
/// `acvus_interpreter::prepare` finds each leaf's instance by its position
/// here.
pub fn structural_leaves<'t>(ty: &'t Ty, interner: &Interner) -> Vec<Leaf<'t>> {
    let mut out = Vec::new();
    walk(ty, interner, &mut Vec::new(), &mut out);
    out
}

fn walk<'t>(ty: &'t Ty, interner: &Interner, path: &mut Vec<Component>, out: &mut Vec<Leaf<'t>>) {
    let Some(parts) = ordered_components(ty, interner) else {
        return;
    };
    for (component, part) in parts {
        path.push(component);
        if !language_owned(part) {
            match components(part) {
                Some(_) => walk(part, interner, path, out),
                None => out.push(Leaf {
                    path: path.clone(),
                    ty: part,
                }),
            }
        }
        path.pop();
    }
}
