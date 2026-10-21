//! The join of two closed types, the error join at a `Result`'s error part,
//! and the order `≤` the join defines (l.80–91).

use std::collections::BTreeMap;

use crate::decl::{Decls, Node};
use crate::ty::{Constructor, ERR, Mutability, Name, PartKey, Term, Ty, Variance};

#[cfg(test)]
mod tests;

/// The language's enum whose variants only the error join makes (l.91).
pub const ERROR: &str = "Error";

/// Where a join gives no result.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JoinError {
    /// The two nodes have no common ancestor.
    NoCommonAncestor(Node, Node),
    /// No cast steps the type up to the node (`holes.md`, H1).
    NotRaised(Ty, Node),
    /// Invariant parts of one name are not equal (l.74).
    InvariantPart(PartKey, Ty, Ty),
    /// Two or more error types remain, and they are not all types of distinct
    /// names (l.95).
    ErrorElements(Vec<Ty>),
}

/// `s ⊔ t`, for types that agree with `decls.types()`.
pub fn join(decls: &Decls, s: &Ty, t: &Ty) -> Result<Ty, JoinError> {
    match (s.term(), t.term()) {
        (Term::Never, _) => return Ok(t.clone()),
        (_, Term::Never) => return Ok(s.clone()),
        (Term::Ref(Mutability::Exclusive, target), Term::Ref(Mutability::Shared, _)) => {
            return join(decls, &shared(target), t);
        }
        (Term::Ref(Mutability::Shared, _), Term::Ref(Mutability::Exclusive, target)) => {
            return join(decls, s, &shared(target));
        }
        _ => {}
    }
    let (a, b) = (Node::of(s), Node::of(t));
    let Some(nca) = decls.tree().nca(&a, &b) else {
        return Err(JoinError::NoCommonAncestor(a, b));
    };
    let s = raised(decls, s, &nca)?;
    let t = raised(decls, t, &nca)?;
    combine(decls, &s, &t)
}

/// `e₁ ⊔ᴱ e₂`.
pub fn error_join(decls: &Decls, e1: &Ty, e2: &Ty) -> Result<Ty, JoinError> {
    let mut by_node: BTreeMap<Node, Ty> = BTreeMap::new();
    for element in error_elements(e1).into_iter().chain(error_elements(e2)) {
        let node = Node::of(&element);
        let joined = match by_node.remove(&node) {
            Some(held) => join(decls, &held, &element)?,
            None => element,
        };
        by_node.insert(node, joined);
    }
    let elements: Vec<Ty> = by_node.into_values().collect();
    match elements.as_slice() {
        [] => Ok(never()),
        [one] => Ok(one.clone()),
        _ => {
            let mut variants = BTreeMap::new();
            for element in &elements {
                let Some(name) = type_name(element.term()) else {
                    return Err(JoinError::ErrorElements(elements));
                };
                if variants.insert(name.clone(), element.clone()).is_some() {
                    return Err(JoinError::ErrorElements(elements));
                }
            }
            Ok(Ty::new(Term::Enum(Name::new(ERROR), variants))
                .expect("an enum's payloads are closed types at payload parts"))
        }
    }
}

/// `s ≤ t`, that is `s ⊔ t = t`.
pub fn le(decls: &Decls, s: &Ty, t: &Ty) -> bool {
    join(decls, s, t).as_ref() == Ok(t)
}

/// `⟦e⟧`: the error part's type as a set of error types (l.94).
fn error_elements(e: &Ty) -> Vec<Ty> {
    match e.term() {
        Term::Never => Vec::new(),
        Term::Enum(_, variants) if is_error(e.term()) => variants.values().cloned().collect(),
        _ => vec![e.clone()],
    }
}

/// Whether the term is the language's enum `Error`.
pub(crate) fn is_error<P>(term: &Term<P>) -> bool {
    matches!(term, Term::Enum(name, _) if name.as_str() == ERROR)
}

/// The name of an enum, a struct or an extern-defined type, which names the
/// variant of `Error` it lies in.
pub(crate) fn type_name<P>(term: &Term<P>) -> Option<&Name> {
    match term {
        Term::Enum(name, _) | Term::Struct(name, _) | Term::Extern { name, .. } => Some(name),
        _ => None,
    }
}

fn raised(decls: &Decls, ty: &Ty, node: &Node) -> Result<Ty, JoinError> {
    decls
        .raise(ty, node)
        .map(|(_, raised)| raised)
        .ok_or_else(|| JoinError::NotRaised(ty.clone(), node.clone()))
}

/// Two types of one node: their members united and their parts joined, each
/// covariant part by `⊔`, a `Result`'s error part by `⊔ᴱ`, and each invariant
/// part equal.
fn combine(decls: &Decls, s: &Ty, t: &Ty) -> Result<Ty, JoinError> {
    let constructor = s.term().constructor();
    debug_assert_eq!(constructor, t.term().constructor());
    let part = |key: &PartKey, a: &Ty, b: &Ty| match constructor.variance() {
        Variance::Covariant if is_error_part(&constructor, key) => error_join(decls, a, b),
        Variance::Covariant => join(decls, a, b),
        Variance::Invariant if a == b => Ok(a.clone()),
        Variance::Invariant => Err(JoinError::InvariantPart(key.clone(), a.clone(), b.clone())),
    };
    let term = match (s.term(), t.term()) {
        (Term::Object(a), Term::Object(b)) => Term::Object(united(a, b, PartKey::Field, part)?),
        (Term::Enum(name, a), Term::Enum(_, b)) => {
            Term::Enum(name.clone(), united(a, b, PartKey::Payload, part)?)
        }
        (term, _) => term.clone().try_map(|key, a| {
            let b = t
                .term()
                .part(&key)
                .expect("types that agree with their declarations have one constructor's parts");
            part(&key, &a, b)
        })?,
    };
    Ok(Ty::new(term).expect("joined parts lie at the places of the parts joined"))
}

/// The members of both, each part present in both combined by `part`.
fn united<K, F>(
    a: &BTreeMap<Name, Ty>,
    b: &BTreeMap<Name, Ty>,
    key: K,
    part: F,
) -> Result<BTreeMap<Name, Ty>, JoinError>
where
    K: Fn(Name) -> PartKey,
    F: Fn(&PartKey, &Ty, &Ty) -> Result<Ty, JoinError>,
{
    let mut united = a.clone();
    for (member, ty) in b {
        let joined = match a.get(member) {
            Some(held) => part(&key(member.clone()), held, ty)?,
            None => ty.clone(),
        };
        united.insert(member.clone(), joined);
    }
    Ok(united)
}

/// Whether the part is a `Result`'s error part, where the error join is taken.
pub(crate) fn is_error_part(constructor: &Constructor, key: &PartKey) -> bool {
    *constructor == Constructor::Result
        && constructor.payload_part(&Name::new(ERR)).as_ref() == Some(key)
}

fn shared(target: &Ty) -> Ty {
    Ty::new(Term::Ref(Mutability::Shared, target.clone()))
        .expect("the target of a reference is never a reference")
}

fn never() -> Ty {
    Ty::new(Term::Never).expect("`!` has no parts")
}
