//! Lower bounds: rule 2 of the judgment, the head the values entering an
//! equivalence class join at.
//!
//! The join reads the language's own orderings alone: `!` is below every
//! head, two references join at the lesser mutability, and the members of an
//! object or an enum are united. It takes no cast and reads no declaration,
//! so that adding a declaration changes no head it gives.
//!
//! This layer imports no `State` and no other layer.

use std::collections::BTreeSet;

use crate::ty::{Constructor, Ty, lesser};

use super::Head;

/// The lower bounds of a class have no join.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NoJoin;

/// The join of two references' heads (rule 2): `Ref` at the lesser of their
/// mutabilities (`&mut` with `&` is `&`). The targets are parts, which the
/// head does not hold: they pass into the joined reference's target part
/// when each lower bound is decomposed against it.
///
/// Returns `None` where either head is no reference. It reads no target and
/// takes no cast: a reference is never raised to another node.
pub fn join_references(a: &Head, b: &Head) -> Option<Head> {
    let (Constructor::Ref(m), Constructor::Ref(n)) = (&a.constructor, &b.constructor) else {
        return None;
    };
    let mutability = lesser(Some(*m), Some(*n))?;
    Some(Head::reference(mutability))
}

/// The join of two heads (rule 2):
/// - `!` (`Never`) is below every head, so a `!` lower bound counts as
///   nothing;
/// - two references join by `join_references`;
/// - two heads of one other constructor join at it, the members of an
///   object or an enum united, and those of any other constructor equal.
///
/// Returns `None` where the constructors differ (`Ref(m)` and `Ref(n)`
/// counting as one), or where two heads of a constructor other than an
/// object or an enum have different members. It takes no cast, reads no
/// declaration and makes no head neither side gives.
pub fn head_join(a: &Head, b: &Head) -> Option<Head> {
    match (&a.constructor, &b.constructor) {
        (Constructor::Never, _) => Some(b.clone()),
        (_, Constructor::Never) => Some(a.clone()),
        (Constructor::Ref(_), Constructor::Ref(_)) => join_references(a, b),
        (c, d) if c != d => None,
        (Constructor::Object | Constructor::Enum(_), _) => Some(Head::owned(
            a.constructor.clone(),
            a.members.union(&b.members).cloned().collect(),
        )),
        _ if a.members == b.members => Some(Head::owned(a.constructor.clone(), a.members.clone())),
        _ => None,
    }
}

/// The join (rule 2) of the heads of the lower bounds `lowers` and the part
/// lower bounds `part_lowers` of one class, every head joined by
/// `head_join`.
///
/// The caller guarantees that `lowers` and `part_lowers` are not both empty,
/// and the function panics otherwise. Returns `Err(NoJoin)` where two heads
/// have no join. It reads no requirement, occurrence or domain, and takes no
/// cast.
pub fn join(lowers: &[Head], part_lowers: &[Head]) -> Result<Head, NoJoin> {
    let mut entering = lowers.iter().chain(part_lowers);
    let Some(first) = entering.next() else {
        panic!("a join is taken of at least one lower bound");
    };
    entering
        .try_fold(first.clone(), |joined, next| head_join(&joined, next))
        .ok_or(NoJoin)
}

/// Rule 2 where the values entering a class carry identities (D3): whether
/// `identities`, those of its lower bounds and part lower bounds, are fewer
/// than two distinct ones. Where it is `false`, the lower bounds have no
/// join, `no_join`: no class holds two identities, and no cast is looked up
/// to make them one. It takes no cast and reads no declaration.
pub fn identities_agree(identities: &BTreeSet<Ty>) -> bool {
    identities.len() < 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{IntTy, Member, Mutability, Name, Scalar};

    fn scalar_head(scalar: Scalar) -> Head {
        Head::owned(Constructor::Scalar(scalar), BTreeSet::new())
    }

    #[test]
    fn never_is_bottom_for_head_join() {
        let never = Head::owned(Constructor::Never, BTreeSet::new());
        let i32_head = scalar_head(Scalar::Int(IntTy::I32));
        assert_eq!(head_join(&never, &i32_head), Some(i32_head.clone()));
        assert_eq!(head_join(&i32_head, &never), Some(i32_head));
    }

    #[test]
    fn references_join_at_lesser_mutability() {
        let shared = Head::reference(Mutability::Shared);
        let exclusive = Head::reference(Mutability::Exclusive);
        assert_eq!(head_join(&exclusive, &exclusive), Some(exclusive.clone()));
        assert_eq!(head_join(&exclusive, &shared), Some(shared.clone()));
        assert_eq!(head_join(&shared, &exclusive), Some(shared));
    }

    #[test]
    fn object_heads_unite_fields() {
        let a = Head::owned(Constructor::Object, [Member::Field(Name::new("x"))].into());
        let b = Head::owned(Constructor::Object, [Member::Field(Name::new("y"))].into());
        let joined = head_join(&a, &b).expect("objects join");
        assert_eq!(
            joined.members,
            [Member::Field(Name::new("x")), Member::Field(Name::new("y"))].into()
        );
    }

    #[test]
    fn incompatible_constructors_have_no_join() {
        let i32_head = scalar_head(Scalar::Int(IntTy::I32));
        let bool_head = scalar_head(Scalar::Bool);
        assert_eq!(head_join(&i32_head, &bool_head), None);
        assert_eq!(join(&[i32_head], &[bool_head]), Err(NoJoin));
    }

    #[test]
    fn two_identities_never_agree() {
        let identity = |name: &str| {
            Ty::new(crate::ty::Term::Identity(Name::new(name))).expect("an identity is closed")
        };
        assert!(identities_agree(&BTreeSet::new()));
        assert!(identities_agree(&[identity("a")].into()));
        assert!(!identities_agree(&[identity("a"), identity("b")].into()));
    }

    #[test]
    fn one_lower_bound_joins_at_its_own_head() {
        let i32_head = scalar_head(Scalar::Int(IntTy::I32));
        assert_eq!(join(std::slice::from_ref(&i32_head), &[]), Ok(i32_head));
    }
}
