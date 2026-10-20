use super::*;
use crate::ty::{IntTy, Mutability, Scalar};

fn name(text: &str) -> Name {
    Name::new(text)
}

fn ty(term: Term<Ty>) -> Ty {
    Ty::new(term).expect("the term is a closed type")
}

fn u8_() -> Ty {
    ty(Term::Scalar(Scalar::Int(IntTy::U8)))
}

fn u16_() -> Ty {
    ty(Term::Scalar(Scalar::Int(IntTy::U16)))
}

fn var(i: usize) -> SigTy {
    SigTy::Var(SigVar::Type(i))
}

fn sig_ref(mutability: Mutability, target: SigTy) -> SigTy {
    SigTy::term(Term::Ref(mutability, target))
}

fn sig_extern(text: &str, args: Vec<SigTy>, identity: Option<SigTy>) -> SigTy {
    SigTy::term(Term::Extern {
        name: name(text),
        args,
        identity,
    })
}

/// `Vec` of one argument and no identity, `F` of no argument and an identity,
/// `Idx` of no argument and an identity, and the struct `Pair` of `l` and `r`.
fn types() -> DeclaredTypes {
    let mut types = DeclaredTypes::default();
    types
        .declare_extern(
            name("Vec"),
            ExternDecl {
                arity: 1,
                identity: false,
            },
        )
        .unwrap();
    types
        .declare_extern(
            name("F"),
            ExternDecl {
                arity: 0,
                identity: true,
            },
        )
        .unwrap();
    types
        .declare_extern(
            name("Idx"),
            ExternDecl {
                arity: 0,
                identity: true,
            },
        )
        .unwrap();
    types
        .declare_struct(
            name("Pair"),
            StructDecl {
                fields: vec![name("l"), name("r")],
            },
        )
        .unwrap();
    types
}

#[test]
fn a_name_is_declared_once_in_each_kind() {
    let mut types = types();
    assert_eq!(
        types.declare_extern(
            name("Vec"),
            ExternDecl {
                arity: 0,
                identity: false
            }
        ),
        Err(DeclError::Redeclared(name("Vec")))
    );
    assert_eq!(
        types.declare_struct(name("Pair"), StructDecl { fields: vec![] }),
        Err(DeclError::Redeclared(name("Pair")))
    );
    assert!(
        types
            .declare_struct(name("Vec"), StructDecl { fields: vec![] })
            .is_ok()
    );
}

#[test]
fn a_term_agrees_with_its_declaration() {
    let types = types();
    let vec_u8 = Term::Extern {
        name: name("Vec"),
        args: vec![u8_()],
        identity: None,
    };
    assert_eq!(types.check(&vec_u8), Ok(()));
    let bare_vec: Term<Ty> = Term::Extern {
        name: name("Vec"),
        args: vec![],
        identity: None,
    };
    assert_eq!(
        types.check(&bare_vec),
        Err(DeclError::Arity {
            constructor: Constructor::Extern(name("Vec")),
            declared: 1,
            written: 0,
        })
    );
    let f_without_identity: Term<Ty> = Term::Extern {
        name: name("F"),
        args: vec![],
        identity: None,
    };
    assert_eq!(
        types.check(&f_without_identity),
        Err(DeclError::Identity {
            name: name("F"),
            declared: true,
        })
    );
    let unknown: Term<Ty> = Term::Extern {
        name: name("Map"),
        args: vec![],
        identity: None,
    };
    assert_eq!(
        types.check(&unknown),
        Err(DeclError::Undeclared(Constructor::Extern(name("Map"))))
    );
    let pair = Term::Struct(name("Pair"), vec![u8_(), u16_()]);
    assert_eq!(types.check(&pair), Ok(()));
    let short_pair = Term::Struct(name("Pair"), vec![u8_()]);
    assert!(matches!(
        types.check(&short_pair),
        Err(DeclError::Arity { .. })
    ));
    assert_eq!(types.check(&Term::<Ty>::Unit), Ok(()));
}

#[test]
fn check_ty_looks_at_every_term() {
    let types = types();
    let inner = ty(Term::Extern {
        name: name("Vec"),
        args: vec![],
        identity: None,
    });
    let outer = ty(Term::Option(inner));
    assert!(types.check(outer.term()).is_ok());
    assert!(matches!(
        types.check_ty(&outer),
        Err(DeclError::Arity { .. })
    ));
}

#[test]
fn a_signature_is_checked_at_every_term_it_writes() {
    let types = types();
    let vec_t = sig_extern("Vec", vec![var(0)], None);
    let push = Signature::new(
        vec![name("T")],
        false,
        vec![sig_ref(Mutability::Exclusive, vec_t.clone()), var(0)],
        SigTy::term(Term::Unit),
        &types,
    );
    assert!(push.is_ok());
    assert_eq!(
        Signature::new(vec![], false, vec![var(0)], SigTy::term(Term::Unit), &types),
        Err(SigError::VarOutOfRange(0))
    );
    let f_p = sig_extern("F", vec![], Some(SigTy::Var(SigVar::Identity)));
    assert_eq!(
        Signature::new(vec![], false, vec![], f_p.clone(), &types),
        Err(SigError::IdentityUnnamed)
    );
    assert_eq!(
        Signature::new(
            vec![],
            true,
            vec![SigTy::Var(SigVar::Identity)],
            f_p.clone(),
            &types
        ),
        Err(SigError::IdentityPlace)
    );
    let f_t = sig_extern("F", vec![], Some(var(0)));
    assert_eq!(
        Signature::new(vec![name("T")], false, vec![], f_t, &types),
        Err(SigError::IdentityPlace)
    );
    let own_identity = sig_extern("F", vec![], Some(SigTy::term(Term::Identity(name("ι")))));
    assert_eq!(
        Signature::new(vec![], false, vec![], own_identity, &types),
        Err(SigError::IdentityPlace)
    );
    let ref_ref = sig_ref(Mutability::Shared, sig_ref(Mutability::Shared, var(0)));
    assert_eq!(
        Signature::new(
            vec![name("T")],
            false,
            vec![ref_ref],
            SigTy::term(Term::Unit),
            &types
        ),
        Err(SigError::RefTarget)
    );
    let bare_vec = sig_extern("Vec", vec![], None);
    assert!(matches!(
        Signature::new(
            vec![],
            false,
            vec![bare_vec],
            SigTy::term(Term::Unit),
            &types
        ),
        Err(SigError::Decl(DeclError::Arity { .. }))
    ));
}

#[test]
fn an_identity_only_the_result_carries_is_existential() {
    let types = types();
    let f_p = sig_extern("F", vec![], Some(SigTy::Var(SigVar::Identity)));
    let idx_p = sig_extern("Idx", vec![], Some(SigTy::Var(SigVar::Identity)));
    let open = Signature::new(vec![], true, vec![], f_p.clone(), &types).unwrap();
    assert!(open.existential_identity());
    let index_of = Signature::new(
        vec![],
        true,
        vec![sig_ref(Mutability::Shared, f_p)],
        idx_p,
        &types,
    )
    .unwrap();
    assert!(!index_of.existential_identity());
    let unit = Signature::new(vec![], true, vec![], SigTy::term(Term::Unit), &types).unwrap();
    assert!(!unit.existential_identity());
}

#[test]
fn substitution_puts_each_bound_variable_in_its_place() {
    let vec_t = sig_extern("Vec", vec![var(0)], None);
    let args = SigArgs {
        types: vec![Some(u8_())],
        identity: None,
    };
    let expected = ty(Term::Extern {
        name: name("Vec"),
        args: vec![u8_()],
        identity: None,
    });
    assert_eq!(vec_t.substitute(&args), Ok(expected));
    let unbound: SigArgs = SigArgs {
        types: vec![None],
        identity: None,
    };
    assert_eq!(
        vec_t.substitute(&unbound),
        Err(SubstError::Unbound(SigVar::Type(0)))
    );
}

#[test]
fn a_reference_substituted_under_a_reference_is_one_reference_exclusive_only_where_both_are() {
    let target = u8_();
    let shared_u = ty(Term::Ref(Mutability::Shared, target.clone()));
    let exclusive_u = ty(Term::Ref(Mutability::Exclusive, target.clone()));
    let at = |t: &Ty| SigArgs {
        types: vec![Some(t.clone())],
        identity: None,
    };
    let shared_t = sig_ref(Mutability::Shared, var(0));
    let exclusive_t = sig_ref(Mutability::Exclusive, var(0));
    assert_eq!(shared_t.substitute(&at(&shared_u)), Ok(shared_u.clone()));
    assert_eq!(shared_t.substitute(&at(&exclusive_u)), Ok(shared_u.clone()));
    assert_eq!(exclusive_t.substitute(&at(&shared_u)), Ok(shared_u.clone()));
    assert_eq!(
        exclusive_t.substitute(&at(&exclusive_u)),
        Ok(exclusive_u.clone())
    );
}

#[test]
fn matching_finds_what_substitution_gives_back() {
    let vec_t = sig_extern("Vec", vec![var(0)], None);
    let param = sig_ref(Mutability::Exclusive, vec_t);
    let closed = ty(Term::Ref(
        Mutability::Exclusive,
        ty(Term::Extern {
            name: name("Vec"),
            args: vec![u8_()],
            identity: None,
        }),
    ));
    let mut args = SigArgs {
        types: vec![None],
        identity: None,
    };
    assert!(param.match_ty(&closed, &mut args));
    assert_eq!(args.types, vec![Some(u8_())]);
    assert_eq!(param.substitute(&args), Ok(closed));
}

#[test]
fn matching_binds_a_variable_under_a_reference_to_the_target() {
    let shared_t = sig_ref(Mutability::Shared, var(0));
    let closed = ty(Term::Ref(Mutability::Shared, u8_()));
    let mut args = SigArgs {
        types: vec![None],
        identity: None,
    };
    assert!(shared_t.match_ty(&closed, &mut args));
    assert_eq!(args.types, vec![Some(u8_())]);
}

#[test]
fn matching_fails_at_another_constructor_or_a_variable_met_twice_and_keeps_the_bindings() {
    let pair_t = SigTy::term(Term::Tuple(vec![var(0), var(0)]));
    let mut args = SigArgs {
        types: vec![None],
        identity: None,
    };
    let same = ty(Term::Tuple(vec![u8_(), u8_()]));
    let different = ty(Term::Tuple(vec![u8_(), u16_()]));
    assert!(!pair_t.match_ty(&different, &mut args));
    assert_eq!(args.types, vec![None]);
    assert!(pair_t.match_ty(&same, &mut args));
    assert!(!pair_t.match_ty(&ty(Term::Option(u8_())), &mut args));
    assert_eq!(args.types, vec![Some(u8_())]);
    let shared_t = sig_ref(Mutability::Shared, var(0));
    let mut fresh = SigArgs {
        types: vec![None],
        identity: None,
    };
    assert!(!shared_t.match_ty(&ty(Term::Ref(Mutability::Exclusive, u8_())), &mut fresh));
}

#[test]
fn matching_binds_the_identity_variable_to_an_identity() {
    let f_p = sig_extern("F", vec![], Some(SigTy::Var(SigVar::Identity)));
    let closed = ty(Term::Extern {
        name: name("F"),
        args: vec![],
        identity: Some(ty(Term::Identity(name("ι₀")))),
    });
    let mut args = SigArgs {
        types: vec![],
        identity: None,
    };
    assert!(f_p.match_ty(&closed, &mut args));
    assert_eq!(args.identity, Some(ty(Term::Identity(name("ι₀")))));
}

#[test]
fn matching_compares_an_objects_fields() {
    let written = SigTy::term(Term::Object([(name("x"), var(0))].into()));
    let closed = ty(Term::Object(
        [(name("x"), u8_()), (name("y"), u8_())].into(),
    ));
    let mut args = SigArgs {
        types: vec![None],
        identity: None,
    };
    assert!(!written.match_ty(&closed, &mut args));
}
