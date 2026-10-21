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
