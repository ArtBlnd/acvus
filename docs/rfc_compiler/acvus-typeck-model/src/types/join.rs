//! The join of two closed types, the join of parts `⊔ₚ`, made without casts,
//! and the order `≤` the join defines.

use std::collections::BTreeMap;

use crate::decl::{Decls, Node};
use crate::ty::{Constructor, Mutability, Name, PartKey, Term, Ty};

/// The language's enum whose variants only `?` makes.
pub const ERROR: &str = "Error";

/// Where a join gives no result.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JoinError {
    /// The two nodes have no common ancestor.
    NoCommonAncestor(Node, Node),
    /// No cast steps the type up to the node.
    NotRaised(Ty, Node),
    /// Invariant parts of one name are not equal.
    InvariantPart(PartKey, Ty, Ty),
    /// Two parts of a structural type have no one constructor and neither is
    /// `!`, so no join reaches both without casts.
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
    if let (Some(a), Some(b)) = (identity(&s), identity(&t))
        && a != b
    {
        let above = without_identity(decls, &nca)
            .ok_or_else(|| JoinError::NoCommonAncestor(Node::of(&s), Node::of(&t)))?;
        return join(
            decls,
            &raised(decls, &s, &above)?,
            &raised(decls, &t, &above)?,
        );
    }
    combine(&s, &t)
}

/// The identity an extern-defined type carries.
pub(crate) fn identity(ty: &Ty) -> Option<&Ty> {
    match ty.term() {
        Term::Extern { identity, .. } => identity.as_ref(),
        _ => None,
    }
}

/// The nearest node at or above `node` whose types carry no identity, to
/// which two different identities are raised.
pub(crate) fn without_identity(decls: &Decls, node: &Node) -> Option<Node> {
    decls
        .tree()
        .ancestors(node)
        .into_iter()
        .find(|node| match node {
            Node::Owned(Constructor::Extern(name)) => decls
                .types()
                .extern_decl(name)
                .is_none_or(|decl| !decl.identity),
            _ => true,
        })
}

/// `s ≤ t`, that is `s ⊔ t = t`.
pub fn le(decls: &Decls, s: &Ty, t: &Ty) -> bool {
    join(decls, s, t).as_ref() == Ok(t)
}

/// `s ⊔ₚ t`: of one constructor, combined as by `⊔`; otherwise only
/// `!` joins, and no cast raises either, nor is a reference taken as shared.
fn part_join(s: &Ty, t: &Ty) -> Result<Ty, JoinError> {
    match (s.term(), t.term()) {
        (Term::Never, _) => Ok(t.clone()),
        (_, Term::Never) => Ok(s.clone()),
        (a, b) if a.constructor() == b.constructor() => combine(s, t),
        _ => Err(JoinError::PartConstructors(s.clone(), t.clone())),
    }
}

/// Whether the term is the language's enum `Error`, with variants or
/// `Error⟨E⟩`.
pub(crate) fn is_error<P>(term: &Term<P>) -> bool {
    matches!(term, Term::Enum(name, _) if name.as_str() == ERROR)
        || matches!(term, Term::ErrorOf(_))
}

fn raised(decls: &Decls, ty: &Ty, node: &Node) -> Result<Ty, JoinError> {
    decls
        .raise(ty, node)
        .map(|(_, raised)| raised)
        .ok_or_else(|| JoinError::NotRaised(ty.clone(), node.clone()))
}

/// Two types of one constructor: their members united, each part of a
/// structural type joined by `⊔ₚ`, and each other part equal.
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

#[cfg(test)]
mod tests {
    use super::{JoinError, join, le};
    use crate::decl::Node;
    use crate::fixture::*;
    use crate::ty::{IntTy, Mutability, PartKey, Term, Ty};

    #[test]
    fn never_is_below_every_type() {
        let decls = decls();
        for t in [int(IntTy::U8), named("A"), never()] {
            assert_eq!(join(&decls, &never(), &t), Ok(t.clone()));
            assert_eq!(join(&decls, &t, &never()), Ok(t.clone()));
        }
    }

    #[test]
    fn types_of_two_nodes_are_raised_to_their_nearest_common_ancestor() {
        let decls = decls();
        assert_eq!(join(&decls, &named("A"), &named("C")), Ok(named("B")));
        assert_eq!(join(&decls, &named("A"), &named("B")), Ok(named("B")));
        assert_eq!(
            join(&decls, &int(IntTy::U8), &int(IntTy::U16)),
            Err(JoinError::NoCommonAncestor(
                Node::of(&int(IntTy::U8)),
                Node::of(&int(IntTy::U16))
            ))
        );
    }

    #[test]
    fn an_exclusive_reference_joins_a_shared_one_as_shared() {
        let decls = decls();
        let mut_a = reference(Mutability::Exclusive, named("A"));
        let shared_b = reference(Mutability::Shared, named("B"));
        let mut_b = reference(Mutability::Exclusive, named("B"));
        assert_eq!(join(&decls, &mut_a, &shared_b), Ok(shared_b.clone()));
        assert_eq!(join(&decls, &shared_b, &mut_a), Ok(shared_b.clone()));
        assert_eq!(join(&decls, &mut_a, &mut_b), Ok(mut_b));
    }

    #[test]
    fn members_are_united_and_structural_parts_joined_without_casts() {
        let decls = decls();
        let x_a = object(&[("x", named("A"))]);
        let x_a_y = object(&[("x", named("A")), ("y", int(IntTy::U8))]);
        assert_eq!(join(&decls, &x_a, &x_a_y), Ok(x_a_y.clone()));
        let x_c_y = object(&[("x", named("C")), ("y", int(IntTy::U8))]);
        assert_eq!(
            join(&decls, &x_a, &x_c_y),
            Err(JoinError::PartConstructors(named("A"), named("C")))
        );
        let v = enumeration("E", &[("V", named("A"))]);
        let w = enumeration("E", &[("W", never())]);
        assert_eq!(
            join(&decls, &v, &w),
            Ok(enumeration("E", &[("V", named("A")), ("W", never())]))
        );
        assert!(join(&decls, &v, &enumeration("F", &[("V", named("A"))])).is_err());
        let option = |payload| ty(Term::Option(payload));
        assert_eq!(
            join(&decls, &option(named("A")), &option(named("C"))),
            Err(JoinError::PartConstructors(named("A"), named("C")))
        );
        assert_eq!(
            join(&decls, &option(never()), &option(named("A"))),
            Ok(option(named("A")))
        );
        assert_eq!(
            join(
                &decls,
                &option(option(named("A"))),
                &option(option(never()))
            ),
            Ok(option(option(named("A"))))
        );
        let mut_a = reference(Mutability::Exclusive, named("A"));
        let shared_a = reference(Mutability::Shared, named("A"));
        assert_eq!(
            join(&decls, &option(mut_a.clone()), &option(shared_a.clone())),
            Err(JoinError::PartConstructors(mut_a, shared_a))
        );
        assert_eq!(
            join(
                &decls,
                &option(boxed(named("A"))),
                &option(boxed(named("C")))
            ),
            Err(JoinError::InvariantPart(
                PartKey::Arg(0),
                named("A"),
                named("C")
            ))
        );
    }

    #[test]
    fn invariant_parts_of_one_name_must_be_equal() {
        let decls = decls();
        assert_eq!(
            join(&decls, &boxed(named("A")), &boxed(named("C"))),
            Err(JoinError::InvariantPart(
                PartKey::Arg(0),
                named("A"),
                named("C")
            ))
        );
        assert_eq!(
            join(&decls, &boxed(named("A")), &boxed(named("A"))),
            Ok(boxed(named("A")))
        );
        let shared = |target| reference(Mutability::Shared, target);
        assert!(
            join(
                &decls,
                &shared(boxed(named("A"))),
                &shared(boxed(named("B")))
            )
            .is_err()
        );
    }

    #[test]
    fn an_error_part_is_joined_as_every_other_structural_part() {
        let decls = decls();
        let u8_ = int(IntTy::U8);
        let ok_a = result(u8_.clone(), named("A"));
        let ok_c = result(u8_.clone(), named("C"));
        assert_eq!(
            join(&decls, &ok_a, &ok_c),
            Err(JoinError::PartConstructors(named("A"), named("C")))
        );
        assert_eq!(
            join(&decls, &result(u8_.clone(), never()), &ok_a),
            Ok(ok_a.clone())
        );
        assert_eq!(join(&decls, &ok_a, &ok_a), Ok(ok_a.clone()));
        let a_c = error(&[named("A"), named("C")]);
        let b = error(&[named("B")]);
        assert_eq!(
            join(&decls, &result(u8_.clone(), a_c), &result(u8_.clone(), b)),
            Ok(result(u8_, error(&[named("A"), named("B"), named("C")])))
        );
    }

    #[test]
    fn le_is_the_order_the_join_defines() {
        let decls = decls();
        assert!(le(&decls, &named("A"), &named("B")));
        assert!(!le(&decls, &named("B"), &named("A")));
        assert!(le(
            &decls,
            &object(&[("x", named("A"))]),
            &object(&[("x", named("A")), ("y", never())])
        ));
        assert!(!le(
            &decls,
            &object(&[("x", named("A"))]),
            &object(&[("x", named("B"))])
        ));
        assert!(le(&decls, &never(), &named("A")));
        assert!(le(
            &decls,
            &reference(Mutability::Exclusive, named("A")),
            &reference(Mutability::Shared, named("B"))
        ));
        assert!(!le(
            &decls,
            &reference(Mutability::Shared, named("A")),
            &reference(Mutability::Exclusive, named("A"))
        ));
    }

    #[test]
    fn the_join_is_commutative_associative_and_idempotent_over_the_universe() {
        let decls = decls();
        let types = universe();
        let join_ok = |s: &Ty, t: &Ty| join(&decls, s, t).ok();
        for s in &types {
            assert_eq!(join_ok(s, s).as_ref(), Some(s), "{s} ⊔ {s}");
            for t in &types {
                assert_eq!(join_ok(s, t), join_ok(t, s), "{s} ⊔ {t}");
                for u in &types {
                    let left = join_ok(s, t).and_then(|st| join_ok(&st, u));
                    let right = join_ok(t, u).and_then(|tu| join_ok(s, &tu));
                    assert_eq!(left, right, "({s} ⊔ {t}) ⊔ {u} against {s} ⊔ ({t} ⊔ {u})");
                }
            }
        }
    }
}
