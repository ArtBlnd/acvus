use super::*;
use crate::lit::{LitVar, Literal};
use crate::ty::{IntTy, Scalar};

fn setup(names: &[&str]) -> (Arena, Classes, Vec<ClassId>) {
    let mut arena = Arena::default();
    let ids = names
        .iter()
        .map(|name| arena.intern(ClassName::Text(Name::new(name))))
        .collect();
    let mut classes = Classes::default();
    classes.grow(&arena);
    (arena, classes, ids)
}

fn u8_() -> Term<ClassId> {
    Term::Scalar(Scalar::Int(IntTy::U8))
}

#[test]
fn unifying_two_structures_unifies_their_parts_of_one_name() {
    let types = DeclaredTypes::default();
    let (_, mut classes, ids) = setup(&["A", "B", "P", "Q"]);
    let [a, b, p, q] = ids[..] else {
        unreachable!()
    };
    classes
        .give_structure(a, Term::Option(p), StructureSource::Solved, &types)
        .unwrap();
    classes
        .give_structure(b, Term::Option(q), StructureSource::Written, &types)
        .unwrap();
    classes.unify(a, b, &types).unwrap();
    assert!(classes.same(p, q));
    assert_eq!(
        classes.held(a).structure.as_ref().unwrap().1,
        StructureSource::Written
    );
}

#[test]
fn structures_of_two_constructors_conflict_and_leave_both_apart() {
    let types = DeclaredTypes::default();
    let (_, mut classes, ids) = setup(&["A", "B", "P"]);
    let [a, b, p] = ids[..] else { unreachable!() };
    classes
        .give_structure(a, Term::Option(p), StructureSource::Solved, &types)
        .unwrap();
    classes
        .give_structure(b, u8_(), StructureSource::Written, &types)
        .unwrap();
    assert!(matches!(
        classes.unify(a, b, &types),
        Err(Conflict::Structures(..))
    ));
    assert!(!classes.same(a, b));
    assert!(classes.held(a).structure.is_some());
}

#[test]
fn a_literal_domain_is_held_by_a_scalar_admitting_its_values() {
    let types = DeclaredTypes::default();
    let (_, mut classes, ids) = setup(&["L", "U", "M"]);
    let [l, u, m] = ids[..] else { unreachable!() };
    classes
        .give_domain(
            l,
            Domain::of_literal(&LitVar::of(Literal::Int(255))),
            &types,
        )
        .unwrap();
    classes
        .give_structure(u, u8_(), StructureSource::Written, &types)
        .unwrap();
    classes.unify(l, u, &types).unwrap();
    classes
        .give_domain(
            m,
            Domain::of_literal(&LitVar::of(Literal::Int(256))),
            &types,
        )
        .unwrap();
    assert!(matches!(
        classes.unify(m, u, &types),
        Err(Conflict::Domain(..))
    ));
}

#[test]
fn structure_constraints_of_one_part_name_one_class_and_are_satisfied_by_the_structure() {
    let types = DeclaredTypes::default();
    let (_, mut classes, ids) = setup(&["A", "T1", "T2", "X"]);
    let [a, t1, t2, x] = ids[..] else {
        unreachable!()
    };
    classes
        .give_structure_constraint(a, StructureConstraint::Field(Name::new("x"), t1), &types)
        .unwrap();
    classes
        .give_structure_constraint(a, StructureConstraint::Field(Name::new("x"), t2), &types)
        .unwrap();
    assert!(classes.same(t1, t2));
    let fields = [(Name::new("x"), x)].into();
    classes
        .give_structure(a, Term::Object(fields), StructureSource::Solved, &types)
        .unwrap();
    assert!(classes.same(t1, x));
    let variant = StructureConstraint::Payload {
        enum_name: None,
        variant: Name::new("Ok"),
        payload: t1,
    };
    assert!(matches!(
        classes.give_structure_constraint(a, variant, &types),
        Err(Conflict::StructureConstraint(..))
    ));
}
