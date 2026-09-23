//! A declared bound on an ExternFn's type variable, at the contract: the
//! solver admits only the declared types, and defers until it knows.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{
    InstanceSig, Instances, ParamTerm, Poly, PolyBuilder, RequirementSig, Task, Ty, TyTerm,
    TyVarBound, TypeArg, TypeRegistry, UserDefinedDecl, try_freeze_poly,
};
use acvus_utils::{Freeze, Interner};

/// `add: Fn(T, T) -> T` with `T: OneOf([Int, Float])`.
fn add_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    Function {
        qref: QualifiedRef::root(i.intern("add")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::one_of(vec![TyTerm::I64, TyTerm::Float])],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![
                ParamTerm::<Poly>::new(i.intern("a"), t.clone()),
                ParamTerm::<Poly>::new(i.intern("b"), t.clone()),
            ],
            ret: Box::new(t),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
        },
    }
}

/// `Counter` and `Doubled<_>`, the two types the instance set stands at.
fn types(i: &Interner) -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    for (name, params) in [("Counter", 0), ("Doubled", 1)] {
        registry
            .register(UserDefinedDecl {
                qref: QualifiedRef::root(i.intern(name)),
                type_params: vec![TyVarBound::Any; params],
                effect_params: 0,
                identity_params: 0,
                specializable: vec![false; params],
            })
            .expect("one declaration per name");
    }
    registry
}

fn user(i: &Interner, name: &str, args: Vec<acvus_mir::ty::PolyTy>) -> acvus_mir::ty::PolyTy {
    TyTerm::UserDefined {
        id: QualifiedRef::root(i.intern(name)),
        type_args: args.into_iter().map(TypeArg::uniform).collect(),
        effect_args: vec![],
        identity_args: vec![],
    }
}

fn advance_ref(i: &Interner) -> QualifiedRef {
    QualifiedRef::qualified(i.intern("probe"), i.intern("advance"))
}

/// `probe::advance: Fn(S) -> S`, a shared signature with an instance at
/// `Counter` and one at `Doubled<_>`.
fn advance_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let s = pb.fresh_ty_var();
    let at = |ty: acvus_mir::ty::PolyTy| InstanceSig {
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(i.intern("it"), ty.clone())],
            ret: Box::new(ty),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
        },
        admits: Task::Sync,
        task: Task::Sync,
        requires: vec![],
        effect_bounds: vec![],
    };
    let mut inner = PolyBuilder::new();
    Function {
        qref: advance_ref(i),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::one_of(vec![
                user(i, "Counter", vec![]),
                user(i, "Doubled", vec![inner.fresh_ty_var()]),
            ])],
            effect_bounds: vec![],
            instances: Instances {
                concrete: vec![
                    at(user(i, "Counter", vec![])),
                    at(user(i, "Doubled", vec![inner.fresh_ty_var()])),
                ],
                generic: false,
            },
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(i.intern("it"), s.clone())],
            ret: Box::new(s),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
        },
    }
}

/// `drain: Fn(T) -> T` requiring an instance of `probe::advance` at `T`
/// (RFC-0068 rule 5): the requirement's pattern is `advance`'s type at `T`.
fn drain_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    let signature = |ty: acvus_mir::ty::PolyTy| TyTerm::Fn {
        params: vec![ParamTerm::<Poly>::new(i.intern("it"), ty.clone())],
        ret: Box::new(ty),
        captures: vec![],
        effect: acvus_mir::ty::Effect::PURE.into(),
    };
    Function {
        qref: QualifiedRef::root(i.intern("drain")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::Any],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![RequirementSig {
                signature: advance_ref(i),
                pattern: signature(t.clone()),
                calls: Task::Sync,
            }],
        },
        ty: signature(t),
    }
}

/// `wrap: Fn(T) -> Doubled<T>` with no bound on `T`: builds the pattern
/// `Doubled<_>` stands at, for any inner type.
fn wrap_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    Function {
        qref: QualifiedRef::root(i.intern("wrap")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::Any],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(i.intern("it"), t.clone())],
            ret: Box::new(user(i, "Doubled", vec![t])),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
        },
    }
}

fn script_fn(i: &Interner, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("f")),
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(i, source).expect("parse"),
        )),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
        },
    }
}

/// The script's return type, or every error the checker reported.
fn check(i: &Interner, source: &str) -> Result<Ty, Vec<String>> {
    let f = script_fn(i, source);
    let qref = f.qref;
    let graph = CompilationGraph {
        functions: Freeze::new(vec![add_fn(i), advance_fn(i), drain_fn(i), wrap_fn(i), f]),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(types(i)),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: None,
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext);
    if inf.has_errors() {
        return Err(inf
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
            .collect());
    }
    match &inf.outcomes[&qref].meta().ty {
        Ty::Fn { ret, .. } => Ok((**ret).clone()),
        other => panic!("expected Fn, got {other:?}"),
    }
}

#[test]
fn a_declared_member_is_admitted() {
    let i = Interner::new();
    assert_eq!(check(&i, "add(1, 2)").unwrap(), Ty::I64);
    assert_eq!(check(&i, "add(1.5, 2.5)").unwrap(), Ty::Float);
}

#[test]
fn a_type_outside_the_bound_is_rejected_where_it_was_called() {
    let i = Interner::new();
    let errs = check(&i, "add(\"a\", \"b\")").unwrap_err();
    assert!(
        errs.iter()
            .any(|e| e.contains("outside the declared bound") && e.contains("&str")),
        "{errs:?}"
    );
}

#[test]
fn the_bound_waits_for_the_argument_to_resolve() {
    let i = Interner::new();
    assert_eq!(
        check(&i, "let g = |x| -> add(x, 1); g(41)").unwrap(),
        Ty::I64
    );
    let errs = check(&i, "let g = |x| -> add(x, x); g(\"a\")").unwrap_err();
    assert!(
        errs.iter()
            .any(|e| e.contains("outside the declared bound")),
        "{errs:?}"
    );
}

#[test]
fn members_do_not_mix() {
    let i = Interner::new();
    let errs = check(&i, "add(1, 2.5)").unwrap_err();
    assert!(errs.iter().any(|e| e.contains("type mismatch")), "{errs:?}");
}

/// A requirement is decided as a call of the signature it names, so the
/// refusal names the signature, the declaration that required it, the
/// call it could not place, and the instances it could have reached.
#[test]
fn a_refusal_names_the_signature_the_requirement_asked_of() {
    let i = Interner::new();
    let errs = check(&i, "drain(7)").unwrap_err();
    assert!(
        errs.contains(
            &"no instance of probe::advance required by drain has the call type Fn(i64) -> \
              i64; the instances it could reach are\n  Fn(Counter) -> Counter\n  Fn(Doubled<U>) \
              -> Doubled<U>"
                .to_string()
        ),
        "{errs:?}"
    );
}

/// The pattern instance is reached through a type built for it, and the
/// requirement's variable is bound from the instance: `drain(wrap(7))` is
/// `Doubled<i64>`.
#[test]
fn a_pattern_instance_is_reached_through_the_pattern() {
    let i = Interner::new();
    assert_eq!(
        check(&i, "drain(wrap(7))").expect("checks"),
        try_freeze_poly(&user(&i, "Doubled", vec![TyTerm::I64])).expect("ground")
    );
}
