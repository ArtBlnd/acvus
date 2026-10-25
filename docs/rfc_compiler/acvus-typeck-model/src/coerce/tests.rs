use super::*;
use crate::decl::{DeclsBuilder, Role};
use crate::domain::Domain;
use crate::fixture::*;
use crate::join::le;
use crate::lit::{LitVar, Literal};
use crate::sig::{DeclaredTypes, ExternDecl, SigTy, SigVar, Signature};
use crate::ty::{IntTy, Scalar};

fn read(ty: &Ty) -> ReadValue {
    ReadValue::of_ty(ty)
}

fn cast_id(i: usize) -> DeclId {
    DeclId {
        name: name(&format!("cast{i}")),
        overload: 0,
    }
}

fn casts(ids: &[usize]) -> Vec<DeclId> {
    ids.iter().copied().map(cast_id).collect()
}

fn rank(from: &Ty, to: &Ty) -> Rank {
    coerce(&decls(), from, to).unwrap().rank().unwrap()
}

fn plain_sig(text: &str) -> SigTy {
    SigTy::term(Term::Extern {
        name: name(text),
        args: vec![],
        identity: None,
    })
}

/// `A`, `B` and `D` with the casts `A → B` and `B → D`, and the conversions
/// from `u16` to `u8` named by `conversions`.
fn chain(conversions: &[&str]) -> Decls {
    let mut types = DeclaredTypes::default();
    for text in ["A", "B", "D"] {
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
    let mut builder = DeclsBuilder::new(types);
    for (i, (from, to)) in [("A", "B"), ("B", "D")].into_iter().enumerate() {
        let sig = Signature::new(
            vec![],
            false,
            vec![plain_sig(from)],
            plain_sig(to),
            builder.types(),
        )
        .unwrap();
        builder.declare(name(&format!("cast{i}")), sig, Role::Cast);
    }
    for text in conversions {
        let sig = Signature::new(
            vec![],
            false,
            vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U16)))],
            SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8))),
            builder.types(),
        )
        .unwrap();
        builder.declare(name(text), sig, Role::Conversion);
    }
    builder.finish().unwrap()
}

#[test]
fn a_type_passing_into_itself_needs_no_coercion() {
    assert_eq!(
        coerce(&decls(), &named("A"), &named("A")),
        Ok(Coercion::Identity)
    );
    assert_eq!(rank(&named("A"), &named("A")), Rank::empty());
}

#[test]
fn a_cast_steps_up_the_tree_and_a_reborrow_comes_before_it() {
    let decls = decls();
    assert_eq!(
        coerce(&decls, &named("A"), &named("B")),
        Ok(Coercion::Composition {
            reborrow: false,
            casts: casts(&[0]),
            structural: None,
        })
    );
    let mut_a = reference(Mutability::Exclusive, named("A"));
    assert_eq!(
        coerce(&decls, &mut_a, &reference(Mutability::Shared, named("B"))),
        Ok(Coercion::Composition {
            reborrow: true,
            casts: casts(&[2]),
            structural: None,
        })
    );
    assert_eq!(
        coerce(
            &decls,
            &mut_a,
            &reference(Mutability::Exclusive, named("B"))
        ),
        Ok(Coercion::Composition {
            reborrow: false,
            casts: casts(&[3]),
            structural: None,
        })
    );
    assert_eq!(
        coerce(&decls, &mut_a, &reference(Mutability::Shared, named("A"))),
        Ok(Coercion::Composition {
            reborrow: true,
            casts: Vec::new(),
            structural: None,
        })
    );
    assert_eq!(
        coerce(&decls, &named("B"), &named("A")),
        Err(CoerceError::None)
    );
}

#[test]
fn an_object_drops_fields_and_changes_those_it_keeps_by_their_part_rows() {
    let decls = decls();
    let v = enumeration("E", &[("V", never())]);
    let vw = enumeration("E", &[("V", never()), ("W", never())]);
    let from = object(&[("x", v), ("y", int(IntTy::U8))]);
    let to = object(&[("x", vw)]);
    let coercion = coerce(&decls, &from, &to).unwrap();
    assert_eq!(
        coercion,
        Coercion::Composition {
            reborrow: false,
            casts: Vec::new(),
            structural: Some(Structural::Shape {
                dropped: [Member::Field(name("y"))].into(),
                added: BTreeSet::new(),
                parts: vec![(
                    PartKey::Field(name("x")),
                    Coercion::Composition {
                        reborrow: false,
                        casts: Vec::new(),
                        structural: Some(Structural::Shape {
                            dropped: BTreeSet::new(),
                            added: [Member::Variant(name("W"))].into(),
                            parts: vec![(PartKey::Payload(name("V")), Coercion::Identity)],
                        }),
                    }
                )],
            }),
        }
    );
    let rank = coercion.rank().unwrap();
    assert_eq!(rank.highest(), Some(Stage::Structural));
    assert_eq!(rank.casts(), 0);
    assert_eq!(
        coerce(&decls, &to, &from),
        Err(CoerceError::None),
        "no coercion gains a field"
    );
    assert_eq!(
        coerce(
            &decls,
            &object(&[("x", named("A"))]),
            &object(&[("x", named("B"))])
        ),
        Err(CoerceError::None),
        "no cast at a part"
    );
}

#[test]
fn an_enum_widens_and_no_value_is_wrapped_in_an_error() {
    let decls = decls();
    let v = enumeration("E", &[("V", named("A"))]);
    let vw = enumeration("E", &[("V", named("A")), ("W", never())]);
    assert!(matches!(
        coerce(&decls, &v, &vw),
        Ok(Coercion::Composition {
            structural: Some(Structural::Shape { .. }),
            ..
        })
    ));
    let b_c = error(&[named("B"), named("C")]);
    assert_eq!(
        coerce(&decls, &named("A"), &b_c),
        Err(CoerceError::None),
        "no row wraps a value in an `Error`, which `?` alone makes"
    );
    let b_d = error(&[named("B"), named("D")]);
    assert_eq!(
        coerce(&chain(&[]), &named("A"), &b_d),
        Err(CoerceError::None),
        "no composition reaches an `Error`"
    );
}

#[test]
fn never_coerces_to_every_type_and_at_parts() {
    let decls = decls();
    assert_eq!(coerce(&decls, &never(), &named("A")), Ok(Coercion::Never));
    let none = ty(Term::Option(never()));
    let some_u8 = ty(Term::Option(int(IntTy::U8)));
    let coercion = coerce(&decls, &none, &some_u8).unwrap();
    assert_eq!(
        coercion,
        Coercion::Composition {
            reborrow: false,
            casts: Vec::new(),
            structural: Some(Structural::Shape {
                dropped: BTreeSet::new(),
                added: BTreeSet::new(),
                parts: vec![(PartKey::Payload(name("Some")), Coercion::Never)],
            }),
        }
    );
    assert_eq!(coercion.rank(), Some(Rank::empty()));
}

#[test]
fn a_conversion_is_taken_only_where_no_composition_reaches_and_exactly_one_applies() {
    let u16_ = int(IntTy::U16);
    let u8_ = int(IntTy::U8);
    assert_eq!(coerce(&chain(&[]), &u16_, &u8_), Err(CoerceError::None));
    let one = chain(&["narrow"]);
    let narrow = DeclId {
        name: name("narrow"),
        overload: 0,
    };
    assert_eq!(
        coerce(&one, &u16_, &u8_),
        Ok(Coercion::Conversion(narrow.clone()))
    );
    assert_eq!(coerce(&one, &u16_, &u8_).unwrap().rank(), None);
    let two = chain(&["narrow", "truncate"]);
    assert!(matches!(
        coerce(&two, &u16_, &u8_),
        Err(CoerceError::Conversions(found)) if found.len() == 2
    ));
    assert_eq!(
        coerce_read(&one, &read(&u16_), &read(&u8_)),
        Err(CoerceError::None),
        "narrowing takes no conversion"
    );
    let some = |t: &Ty| ty(Term::Option(t.clone()));
    assert_eq!(
        coerce(&one, &some(&u16_), &some(&u8_)),
        Err(CoerceError::None),
        "no conversion at a part"
    );
}

#[test]
fn ranks_compare_by_highest_stage_then_casts_then_lower_stages() {
    let rank = |decls: &Decls, from: Ty, to: Ty| coerce(decls, &from, &to).unwrap().rank().unwrap();
    let chain = chain(&[]);
    let one_cast = rank(&chain, named("A"), named("B"));
    let two_casts = rank(&chain, named("A"), named("D"));
    let decls = decls();
    let mut_a = reference(Mutability::Exclusive, named("A"));
    let shared_a = reference(Mutability::Shared, named("A"));
    let shared_b = reference(Mutability::Shared, named("B"));
    let v = enumeration("E", &[("V", never())]);
    let vw = enumeration("E", &[("V", never()), ("W", never())]);
    let reborrow = rank(&decls, mut_a.clone(), shared_a);
    let reborrow_cast = rank(&decls, mut_a, shared_b);
    let structural = rank(&decls, v, vw.clone());
    let mut types = DeclaredTypes::default();
    types
        .declare_extern(
            name("A"),
            ExternDecl {
                arity: 0,
                identity: false,
            },
        )
        .unwrap();
    let mut builder = DeclsBuilder::new(types);
    let up = Signature::new(
        vec![],
        false,
        vec![plain_sig("A")],
        SigTy::term(Term::Enum(
            name("E"),
            [(name("V"), SigTy::term(Term::Never))].into(),
        )),
        builder.types(),
    )
    .unwrap();
    builder.declare(name("up"), up, Role::Cast);
    let enum_decls = builder.finish().unwrap();
    let cast_structural = rank(&enum_decls, named("A"), vw.clone());
    assert!(Rank::empty() < reborrow);
    assert!(reborrow < one_cast);
    assert!(one_cast < two_casts);
    assert!(reborrow < reborrow_cast);
    assert!(one_cast < reborrow_cast);
    assert!(reborrow_cast < structural);
    assert!(two_casts < structural);
    assert!(structural < cast_structural);
}

#[test]
fn a_position_that_decides_nothing_reaches_every_type() {
    let decls = decls();
    let u8_ = read(&int(IntTy::U8));
    assert_eq!(
        coerce_read(&decls, &ReadValue::Undecided, &u8_),
        Ok(Coercion::Undecided)
    );
    let candidate = ReadValue::of_sig(&SigTy::term(Term::Option(SigTy::Var(SigVar::Type(0)))));
    assert_eq!(
        coerce_read(&decls, &read(&ty(Term::Option(named("A")))), &candidate),
        Ok(Coercion::Undecided)
    );
    let big = ReadValue::Domain(Domain::of_literal(&LitVar::of(Literal::Int(300))));
    assert_eq!(coerce_read(&decls, &big, &u8_), Err(CoerceError::None));
    assert_eq!(
        coerce_read(&decls, &big, &read(&int(IntTy::U16))),
        Ok(Coercion::Undecided)
    );
    assert_eq!(
        coerce_read(&decls, &big, &read(&named("A"))),
        Err(CoerceError::None)
    );
    let float = ReadValue::Domain(Domain::of_literal(&LitVar::of(Literal::float(1.5))));
    assert_eq!(coerce_read(&decls, &float, &big), Err(CoerceError::None));
}

#[test]
fn along_le_gains_fields_unchanged_and_takes_no_cast_or_wrap_at_a_part() {
    let decls = decls();
    let from = object(&[("x", named("A"))]);
    let to = object(&[("x", named("A")), ("y", int(IntTy::U8))]);
    assert_eq!(
        along_le(&decls, &from, &to),
        Ok(Coercion::Composition {
            reborrow: false,
            casts: Vec::new(),
            structural: Some(Structural::Shape {
                dropped: BTreeSet::new(),
                added: [Member::Field(name("y"))].into(),
                parts: vec![(PartKey::Field(name("x")), Coercion::Identity)],
            }),
        })
    );
    assert_eq!(
        along_le(&decls, &from, &object(&[("x", named("B"))])),
        Err(CoerceError::None),
        "no cast at a part"
    );
    let u8_ = int(IntTy::U8);
    let a_c = error(&[named("A"), named("C")]);
    let ok_a = result(u8_.clone(), named("A"));
    assert_eq!(
        along_le(&decls, &ok_a, &result(u8_.clone(), a_c.clone())),
        Err(CoerceError::None),
        "no wrap at a part"
    );
    let Ok(Coercion::Composition {
        structural: Some(Structural::Shape { parts, .. }),
        ..
    }) = along_le(
        &decls,
        &result(u8_.clone(), error(&[named("A")])),
        &result(u8_, a_c.clone()),
    )
    else {
        panic!("`Result⟨u8, Error{{A(A)}}⟩ ≤ Result⟨u8, Error{{A(A), C(C)}}⟩` by the enum row");
    };
    assert!(matches!(
        &parts[1],
        (_, Coercion::Composition { structural: Some(Structural::Shape { added, .. }), .. })
            if *added == [Member::Variant(name("C"))].into()
    ));
    assert_eq!(along_le(&decls, &named("A"), &a_c), Err(CoerceError::None));
}

#[test]
fn along_le_reaches_exactly_where_the_join_orders_over_the_universe() {
    let decls = decls();
    let types = universe();
    for s in &types {
        for t in &types {
            assert_eq!(
                along_le(&decls, s, t).is_ok(),
                le(&decls, s, t),
                "{s} ≤ {t}"
            );
        }
    }
}
