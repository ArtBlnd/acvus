use super::*;
use crate::decl::{DeclsBuilder, Role};
use crate::sig::{DeclaredTypes, ExternDecl, SigTy, Signature};
use crate::ty::{IntTy, Scalar};

fn name(text: &str) -> Name {
    Name::new(text)
}

fn ty(term: Term<Ty>) -> Ty {
    Ty::new(term).expect("the term is a closed type")
}

fn int(int: IntTy) -> Ty {
    ty(Term::Scalar(Scalar::Int(int)))
}

fn named(text: &str) -> Ty {
    ty(Term::Extern {
        name: name(text),
        args: vec![],
        identity: None,
    })
}

fn boxed(arg: Ty) -> Ty {
    ty(Term::Extern {
        name: name("Box"),
        args: vec![arg],
        identity: None,
    })
}

fn reference(mutability: Mutability, target: Ty) -> Ty {
    ty(Term::Ref(mutability, target))
}

fn object(fields: &[(&str, Ty)]) -> Ty {
    ty(Term::Object(
        fields
            .iter()
            .map(|(field, ty)| (name(field), ty.clone()))
            .collect(),
    ))
}

fn enumeration(text: &str, variants: &[(&str, Ty)]) -> Ty {
    ty(Term::Enum(
        name(text),
        variants
            .iter()
            .map(|(variant, payload)| (name(variant), payload.clone()))
            .collect(),
    ))
}

fn result(ok: Ty, err: Ty) -> Ty {
    ty(Term::Result { ok, err })
}

fn error(elements: &[Ty]) -> Ty {
    let variants = elements
        .iter()
        .map(|element| (type_name(element).unwrap().clone(), element.clone()))
        .collect();
    ty(Term::Enum(name(ERROR), variants))
}

/// `A`, `B` and `C` of no argument and `Box` of one; the casts `A → B` and
/// `C → B`, and `&A → &B` and `&mut A → &mut B`.
fn decls() -> Decls {
    let mut types = DeclaredTypes::default();
    for text in ["A", "B", "C"] {
        types
            .declare_extern(
                name(text),
                ExternDecl {
                    arity: 0,
                    identity: false,
                },
            )
            .unwrap();
    }
    types
        .declare_extern(
            name("Box"),
            ExternDecl {
                arity: 1,
                identity: false,
            },
        )
        .unwrap();
    let mut builder = DeclsBuilder::new(types);
    let plain = |text: &str| {
        SigTy::term(Term::Extern {
            name: name(text),
            args: vec![],
            identity: None,
        })
    };
    let to_ref = |mutability, target| SigTy::term(Term::Ref(mutability, target));
    let casts = [
        (plain("A"), plain("B")),
        (plain("C"), plain("B")),
        (
            to_ref(Mutability::Shared, plain("A")),
            to_ref(Mutability::Shared, plain("B")),
        ),
        (
            to_ref(Mutability::Exclusive, plain("A")),
            to_ref(Mutability::Exclusive, plain("B")),
        ),
    ];
    for (i, (from, to)) in casts.into_iter().enumerate() {
        let sig = Signature::new(vec![], false, vec![from], to, builder.types()).unwrap();
        builder.declare(name(&format!("cast{i}")), sig, Role::Cast);
    }
    builder.finish().unwrap()
}

fn never() -> Ty {
    ty(Term::Never)
}

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
fn members_are_united_and_covariant_parts_joined() {
    let decls = decls();
    let x_a = object(&[("x", named("A"))]);
    let x_c_y = object(&[("x", named("C")), ("y", int(IntTy::U8))]);
    assert_eq!(
        join(&decls, &x_a, &x_c_y),
        Ok(object(&[("x", named("B")), ("y", int(IntTy::U8))]))
    );
    let v = enumeration("E", &[("V", named("A"))]);
    let w = enumeration("E", &[("W", never())]);
    assert_eq!(
        join(&decls, &v, &w),
        Ok(enumeration("E", &[("V", named("A")), ("W", never())]))
    );
    assert!(join(&decls, &v, &enumeration("F", &[("V", named("A"))])).is_err());
    assert_eq!(
        join(
            &decls,
            &ty(Term::Option(named("A"))),
            &ty(Term::Option(named("C")))
        ),
        Ok(ty(Term::Option(named("B"))))
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
fn error_parts_join_into_the_error_enum_by_node() {
    let decls = decls();
    let u8_ = int(IntTy::U8);
    let ok_a = result(u8_.clone(), named("A"));
    let ok_c = result(u8_.clone(), named("C"));
    let a_c = error(&[named("A"), named("C")]);
    assert_eq!(
        join(&decls, &ok_a, &ok_c),
        Ok(result(u8_.clone(), a_c.clone()))
    );
    assert_eq!(
        join(&decls, &result(u8_.clone(), never()), &ok_a),
        Ok(ok_a.clone())
    );
    assert_eq!(join(&decls, &ok_a, &ok_a), Ok(ok_a.clone()));
    let a_b_c = error(&[named("A"), named("B"), named("C")]);
    assert_eq!(error_join(&decls, &a_c, &named("B")), Ok(a_b_c));
    assert_eq!(
        error_join(&decls, &named("A"), &u8_),
        Err(JoinError::ErrorElements(vec![named("A"), u8_.clone()]))
    );
    let enum_a = enumeration("A", &[("V", never())]);
    assert!(matches!(
        error_join(&decls, &named("A"), &enum_a),
        Err(JoinError::ErrorElements(_))
    ));
}

#[test]
fn le_is_the_order_the_join_defines() {
    let decls = decls();
    assert!(le(&decls, &named("A"), &named("B")));
    assert!(!le(&decls, &named("B"), &named("A")));
    assert!(le(
        &decls,
        &object(&[("x", named("A"))]),
        &object(&[("x", named("B")), ("y", never())])
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

/// Types up to one constructor deep over leaves that the cast tree, the
/// reference row and the error join each relate.
fn universe() -> Vec<Ty> {
    let leaves = [never(), int(IntTy::U8), named("A"), named("B"), named("C")];
    let mut types: Vec<Ty> = leaves.to_vec();
    for mutability in [Mutability::Shared, Mutability::Exclusive] {
        for target in ["A", "B", "C"] {
            types.push(reference(mutability, named(target)));
        }
    }
    for leaf in &leaves {
        types.push(ty(Term::Option(leaf.clone())));
        types.push(object(&[("x", leaf.clone())]));
        types.push(object(&[("x", leaf.clone()), ("y", int(IntTy::U8))]));
        types.push(enumeration("E", &[("V", leaf.clone())]));
        types.push(enumeration("E", &[("W", leaf.clone())]));
        types.push(result(int(IntTy::U8), leaf.clone()));
        types.push(boxed(leaf.clone()));
    }
    types.push(result(int(IntTy::U8), error(&[named("A"), named("C")])));
    types
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
