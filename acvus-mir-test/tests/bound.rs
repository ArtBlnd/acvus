//! A declared bound on an ExternFn's type variable, at the contract: the
//! solver admits only the declared types, and defers until it knows.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{ParamTerm, Poly, PolyBuilder, Ty, TyTerm, TyVarBound, TypeRegistry};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

/// `add: Fn(T, T) -> T` with `T: OneOf([Int, Float])`.
fn add_fn(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let t = pb.fresh_ty_var();
    Function {
        qref: QualifiedRef::root(i.intern("add")),
        kind: FnKind::Extern {
            bounds: vec![TyVarBound::OneOf(vec![TyTerm::I64, TyTerm::Float])],
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
        functions: Freeze::new(vec![add_fn(i), f]),
        contexts: Freeze::new(vec![]),
        entry: None,
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(
        i,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(TypeRegistry::new()),
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
