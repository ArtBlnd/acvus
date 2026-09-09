//! Effect chain tests.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{Effect, ParamTerm, Poly, PolyBuilder, PolyParam, Ty, TyTerm, lift_to_poly};
use acvus_mir_test::compile_multi_fn_optimized;
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

fn extern_fn(i: &Interner, name: &str, effect: Effect) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern { bounds: vec![] },
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(lift_to_poly(&Ty::Int)),
            captures: vec![],
            effect: effect.into(),
        },
    }
}

fn local_fn(i: &Interner, name: &str, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern(name)),
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

fn infer_effects(i: &Interner, functions: Vec<Function>) -> FxHashMap<String, Effect> {
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(
        i,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(acvus_mir::ty::TypeRegistry::new()),
    );
    assert!(!inf.has_errors(), "infer errors: {:?}", inf.errors());
    inf.outcomes
        .iter()
        .map(|(qref, outcome)| {
            let effect = outcome.meta().ty.effect().expect("function type");
            (i.resolve(qref.name).to_string(), effect)
        })
        .collect()
}

#[test]
fn function_effect_is_the_join_of_its_calls() {
    let i = Interner::new();
    let effects = infer_effects(
        &i,
        vec![
            extern_fn(&i, "fetch", Effect::Opaque),
            extern_fn(&i, "pure_fn", Effect::Pure),
            local_fn(&i, "wrap_io", "fetch()"),
            local_fn(&i, "wrap_pure", "pure_fn()"),
            local_fn(&i, "both", "wrap_pure() + wrap_io()"),
            local_fn(&i, "none", "1 + 2"),
        ],
    );
    assert_eq!(effects["wrap_io"], Effect::Opaque);
    assert_eq!(effects["wrap_pure"], Effect::Pure);
    assert_eq!(effects["both"], Effect::Opaque);
    assert_eq!(effects["none"], Effect::Pure);
}

#[test]
fn lambda_effect_counts_only_when_called() {
    let i = Interner::new();
    let effects = infer_effects(
        &i,
        vec![
            extern_fn(&i, "fetch", Effect::Opaque),
            local_fn(&i, "defines", "f = |_x| -> fetch(); 1"),
            local_fn(&i, "calls", "f = |_x| -> fetch(); f(1)"),
        ],
    );
    assert_eq!(effects["defines"], Effect::Pure);
    assert_eq!(effects["calls"], Effect::Opaque);
}

fn params(i: &Interner, names: &[&str]) -> Vec<PolyParam> {
    names
        .iter()
        .map(|n| ParamTerm::<Poly>::new(i.intern(n), lift_to_poly(&Ty::Int)))
        .collect()
}

#[test]
fn opaque_call_is_spawn_split_and_pure_call_is_not() {
    let i = Interner::new();
    let io = compile_multi_fn_optimized(
        &i,
        ("main", "wrap(1)"),
        &[("wrap", "fetch() + $x", params(&i, &["x"]))],
        &[],
        &[extern_fn(&i, "fetch", Effect::Opaque)],
    )
    .unwrap();
    assert!(io.contains("spawn"), "an Opaque call must be split:\n{io}");

    let pure = compile_multi_fn_optimized(
        &i,
        ("main", "wrap(1)"),
        &[("wrap", "pure_fn() + $x", params(&i, &["x"]))],
        &[],
        &[extern_fn(&i, "pure_fn", Effect::Pure)],
    )
    .unwrap();
    assert!(
        !pure.contains("spawn"),
        "a Pure call must not be split:\n{pure}"
    );
}
