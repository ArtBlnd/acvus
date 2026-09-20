//! A declared bound on an ExternFn's type variable, at the contract: the
//! solver admits only the declared types, and defers until it knows.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{
    InnerBound, InstanceSets, InstanceShape, ParamTerm, Poly, PolyBuilder, Ty, TyTerm, TyVarBound,
    TypeArg, TypeRegistry, UserDefinedDecl,
};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// `add: Fn(T, T) -> T` with `T: OneOf([Int, Float])`.
fn add_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    Function {
        qref: QualifiedRef::root(i.intern("add")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::one_of(vec![TyTerm::I64, TyTerm::Float])],
            instances: Default::default(),
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
        registry.register(UserDefinedDecl {
            qref: QualifiedRef::root(i.intern(name)),
            type_params: vec![TyVarBound::Any; params],
            effect_params: 0,
            identity_params: 0,
            specializable: vec![false; params],
        });
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

/// The instances of `probe::advance`: one at `Counter`, and one at
/// `Doubled<_>` whose own `where` clause requires an instance of the same
/// signature of what fills its one type argument.
fn advance_instances(i: &Interner) -> Arc<InstanceSets> {
    let mut inner = PolyBuilder::new();
    let mut by_signature = FxHashMap::default();
    by_signature.insert(
        advance_ref(i),
        vec![
            InstanceShape {
                ty: user(i, "Counter", vec![]),
                requires: vec![],
            },
            InstanceShape {
                ty: user(i, "Doubled", vec![inner.fresh_ty_var()]),
                requires: vec![InnerBound {
                    signature: advance_ref(i),
                    at: vec![0],
                }],
            },
        ],
    );
    Arc::new(InstanceSets::new(by_signature))
}

/// `drain: Fn(T) -> T` where `T` is required to have an instance of
/// `probe::advance` (RFC-0067 Decision 1).
fn drain_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    let mut inner = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("drain")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::instances_of(
                advance_ref(i),
                vec![
                    user(i, "Counter", vec![]),
                    user(i, "Doubled", vec![inner.fresh_ty_var()]),
                ],
                advance_instances(i),
            )],
            instances: Default::default(),
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(i.intern("it"), t.clone())],
            ret: Box::new(t),
            captures: vec![],
            effect: acvus_mir::ty::Effect::PURE.into(),
        },
    }
}

/// `wrap: Fn(T) -> Doubled<T>` with no bound on `T`: the second
/// constructor, which builds the pattern for an inner type that has no
/// instance.
fn wrap_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    Function {
        qref: QualifiedRef::root(i.intern("wrap")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::Any],
            instances: Default::default(),
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
        functions: Freeze::new(vec![add_fn(i), drain_fn(i), wrap_fn(i), f]),
        contexts: Freeze::new(vec![]),
        entry: None,
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(
        i,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(types(i)),
    );
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

/// A bound whose shapes are an instance set names the signature that asked
/// for them, so the refusal says what is missing rather than listing the
/// types that happen to be allowed.
#[test]
fn a_refusal_names_the_signature_the_bound_required() {
    let i = Interner::new();
    let errs = check(&i, "drain(7)").unwrap_err();
    assert!(
        errs.contains(
            &"no instance of probe::advance at i64; instances exist at Counter, Doubled<'0>"
                .to_string()
        ),
        "{errs:?}"
    );
}

/// The instance at a pattern carries its own bound, so a ground type whose
/// inner has no instance is refused at check rather than found missing at
/// `prepare`. `wrap` builds the pattern without the bound; `Doubled<i64>`
/// matches the shape and fails the requirement under it.
#[test]
fn a_pattern_instances_own_bound_holds_at_the_pattern() {
    let i = Interner::new();
    let errs = check(&i, "drain(wrap(7))").unwrap_err();
    assert!(
        errs.contains(
            &"no instance of probe::advance at Doubled<i64>; instances exist at Counter, \
              Doubled<'0>"
                .to_string()
        ),
        "{errs:?}"
    );
    assert_eq!(
        check(&i, "drain(wrap(wrap(7)))").unwrap_err().len(),
        1,
        "the recursion refuses the inner pattern too"
    );
}
