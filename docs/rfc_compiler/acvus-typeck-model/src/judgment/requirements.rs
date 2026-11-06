//! Requirements: the structure constraints and the domains recorded on an
//! equivalence class.
//!
//! A requirement never gives a class a head. A structure constraint checks
//! the head that the class's lower bounds or occurrences give, adding the
//! members it names; the domains are read as one domain, which fallback
//! reads. This layer imports no `State` and no other layer.

use std::collections::BTreeSet;

use crate::class::{Conflict, StructureConstraint};
use crate::domain::Domain;
use crate::ty::{Constructor, Member, Name};

use super::Head;

/// The constructor and members a structure constraint requires of the head
/// of its class, where it names a constructor:
/// - `Field(f, _)` requires `Object` with member `Field(f)`.
/// - `Payload { enum_name: Some(e), variant: v, .. }` requires `Enum(e)` with
///   member `Variant(v)`.
/// - `Payload { enum_name: None, variant: "Some" | "None", .. }` requires
///   `Option`.
/// - `Payload { enum_name: None, variant: "Ok" | "Err", .. }` requires `Result`.
/// - `Target(_)`, `Component(_, _)` and `Element(_)` name no single
///   constructor, and require none here.
///
/// It is read only by `check`, to compare a given head with the requirement
/// and to add the members the requirement names; it never becomes a head.
fn required_head(structure_constraint: &StructureConstraint) -> Option<Head> {
    match structure_constraint {
        StructureConstraint::Field(field, _) => Some(Head::owned(
            Constructor::Object,
            [Member::Field(field.clone())].into(),
        )),
        StructureConstraint::Payload {
            enum_name: Some(enum_name),
            variant,
            ..
        } => Some(Head::owned(
            Constructor::Enum(enum_name.clone()),
            [Member::Variant(variant.clone())].into(),
        )),
        StructureConstraint::Payload {
            enum_name: None,
            variant,
            ..
        } => match variant.as_str() {
            crate::ty::SOME | crate::ty::NONE => Some(Head::owned(
                Constructor::Option,
                [
                    Member::Variant(Name::new(crate::ty::SOME)),
                    Member::Variant(Name::new(crate::ty::NONE)),
                ]
                .into(),
            )),
            crate::ty::OK | crate::ty::ERR => Some(Head::owned(
                Constructor::Result,
                [
                    Member::Variant(Name::new(crate::ty::OK)),
                    Member::Variant(Name::new(crate::ty::ERR)),
                ]
                .into(),
            )),
            _ => None,
        },
        StructureConstraint::Target(_)
        | StructureConstraint::Component(_, _)
        | StructureConstraint::Element(_) => None,
    }
}

/// Checks `head`, which the lower bounds or the occurrences of a class give,
/// against every structure constraint in `requirements`, and returns it with
/// the members they name added.
///
/// - `!` satisfies every structure constraint as it is.
/// - A struct satisfies a field constraint here; unification checks its
///   declared fields when its structure is recorded.
/// - A structure constraint that names no constructor (`ref`, `component`,
///   `index`) is satisfied here; decomposition checks it against the recorded
///   structure.
/// - Any other structure constraint requires `head`'s constructor to be the
///   one it names, and adds the member it names where `head` is an object or
///   an enum.
///
/// Returns `Err(Conflict::StructureConstraint(requirement, constructor))` for
/// the first requirement whose constructor is not `head`'s. It takes a head
/// and never makes one: with no head given, requirements decide nothing.
pub fn check(head: Head, requirements: &[StructureConstraint]) -> Result<Head, Conflict> {
    requirements.iter().try_fold(head, satisfy)
}

/// `head` with the members `requirement` names added, or the conflict where
/// `requirement` names another constructor (see `check`).
fn satisfy(mut head: Head, requirement: &StructureConstraint) -> Result<Head, Conflict> {
    if head.constructor == Constructor::Never {
        return Ok(head);
    }
    if matches!(head.constructor, Constructor::Struct(_))
        && matches!(requirement, StructureConstraint::Field(..))
    {
        return Ok(head);
    }
    let Some(required) = required_head(requirement) else {
        return Ok(head);
    };
    if head.constructor != required.constructor {
        return Err(Conflict::StructureConstraint(
            requirement.clone(),
            head.constructor,
        ));
    }
    if matches!(head.constructor, Constructor::Object | Constructor::Enum(_)) {
        head.members.extend(required.members);
    }
    Ok(head)
}

/// Whether `domain` admits the constructor of `head`.
pub fn admits(head: &Head, domain: &Domain) -> bool {
    domain.admits(&head.constructor)
}

/// The one domain the domains of a class read as: their intersection
/// (`meet`). `None` where `domains` is empty, and `Domain::Only(∅)` where two
/// of them admit no constructor in common.
pub fn domain_of<I>(domains: I) -> Option<Domain>
where
    I: IntoIterator<Item = Domain>,
{
    domains
        .into_iter()
        .reduce(|met, domain| met.meet(&domain).unwrap_or(Domain::Only(BTreeSet::new())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::{Arena, ClassName};
    use crate::ty::{IntTy, Scalar};

    fn field(name: &str) -> StructureConstraint {
        let mut arena = Arena::default();
        let part = arena.intern(ClassName::Text(Name::new("t")));
        StructureConstraint::Field(Name::new(name), part)
    }

    #[test]
    fn a_field_requirement_adds_its_member_to_an_object_head() {
        let head = Head::owned(Constructor::Object, [Member::Field(Name::new("x"))].into());
        let checked = check(head, &[field("y")]).expect("an object satisfies a field");
        assert_eq!(
            checked.members,
            [Member::Field(Name::new("x")), Member::Field(Name::new("y"))].into()
        );
    }

    #[test]
    fn a_field_requirement_conflicts_with_a_scalar_head() {
        let head = Head::owned(Constructor::Scalar(Scalar::Bool), BTreeSet::new());
        assert!(matches!(
            check(head, &[field("x")]),
            Err(Conflict::StructureConstraint(
                StructureConstraint::Field(..),
                _
            ))
        ));
    }

    #[test]
    fn never_satisfies_every_requirement() {
        let never = Head::owned(Constructor::Never, BTreeSet::new());
        assert_eq!(check(never.clone(), &[field("x")]), Ok(never));
    }

    #[test]
    fn domain_of_intersects_multiple_domains() {
        let a = Domain::Only(
            [
                Constructor::Scalar(Scalar::Int(IntTy::I32)),
                Constructor::Scalar(Scalar::Int(IntTy::U32)),
            ]
            .into(),
        );
        let b = Domain::Only(
            [
                Constructor::Scalar(Scalar::Int(IntTy::I32)),
                Constructor::Scalar(Scalar::Int(IntTy::I64)),
            ]
            .into(),
        );
        let met = domain_of(vec![a, b]).expect("non-empty domains");
        assert!(met.admits(&Constructor::Scalar(Scalar::Int(IntTy::I32))));
        assert!(!met.admits(&Constructor::Scalar(Scalar::Int(IntTy::U32))));
        assert!(!met.admits(&Constructor::Scalar(Scalar::Int(IntTy::I64))));
    }
}
