//! The join of two closed types, the join of parts `⊔ₚ`, made without casts,
//! and the order `≤` the join defines (l.80–91).

use std::collections::BTreeMap;

use crate::decl::{Decls, Node};
use crate::ty::{Mutability, Name, PartKey, Term, Ty};

#[cfg(test)]
mod tests;

/// The language's enum whose variants only `?` makes (l.91).
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
    /// Two parts of a structural type have no one constructor and neither is
    /// `!`, so no join reaches both without casts (l.89).
    PartConstructors(Ty, Ty),
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
    combine(&s, &t)
}

/// `s ≤ t`, that is `s ⊔ t = t`.
pub fn le(decls: &Decls, s: &Ty, t: &Ty) -> bool {
    join(decls, s, t).as_ref() == Ok(t)
}

/// `s ⊔ₚ t` (l.89): of one constructor, combined as by `⊔`; otherwise only
/// `!` joins, and no cast raises either, nor is a reference taken as shared.
fn part_join(s: &Ty, t: &Ty) -> Result<Ty, JoinError> {
    match (s.term(), t.term()) {
        (Term::Never, _) => Ok(t.clone()),
        (_, Term::Never) => Ok(s.clone()),
        (a, b) if a.constructor() == b.constructor() => combine(s, t),
        _ => Err(JoinError::PartConstructors(s.clone(), t.clone())),
    }
}

/// Whether the term is the language's enum `Error`.
pub(crate) fn is_error<P>(term: &Term<P>) -> bool {
    matches!(term, Term::Enum(name, _) if name.as_str() == ERROR)
}

fn raised(decls: &Decls, ty: &Ty, node: &Node) -> Result<Ty, JoinError> {
    decls
        .raise(ty, node)
        .map(|(_, raised)| raised)
        .ok_or_else(|| JoinError::NotRaised(ty.clone(), node.clone()))
}

/// Two types of one constructor: their members united, each part of a
/// structural type joined by `⊔ₚ`, and each other part equal (l.86, l.89).
fn combine(s: &Ty, t: &Ty) -> Result<Ty, JoinError> {
    let constructor = s.term().constructor();
    debug_assert_eq!(constructor, t.term().constructor());
    let structural = constructor.is_structural();
    let part = |key: &PartKey, a: &Ty, b: &Ty| {
        if structural {
            part_join(a, b)
        } else if a == b {
            Ok(a.clone())
        } else {
            Err(JoinError::InvariantPart(key.clone(), a.clone(), b.clone()))
        }
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

fn shared(target: &Ty) -> Ty {
    Ty::new(Term::Ref(Mutability::Shared, target.clone()))
        .expect("the target of a reference is never a reference")
}
