//! A declared effect floor, `E: Suspends` (RFC-0011 rule 5): the task of
//! the variable is verified when it freezes, so a function value that
//! settles to `Sync` is refused rather than lifted to `Async`.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{
    Effect, EffectTerm, EffectVarBound, InstanceSig, Instances, ParamTerm, Poly, PolyBuilder,
    PolyTy, Task, Ty, TyTerm, lift_to_poly,
};
use acvus_mir_test::compile_script_ir_with;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

const BELOW_SYNC: &str =
    "a function whose task is Sync was given where one that suspends is required";

fn declared(i: &Interner, name: &str, ty: PolyTy, effect_bounds: Vec<EffectVarBound>) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds,
            instances: Default::default(),
            requires: vec![],
        },
        ty,
    }
}

fn fn_ty(i: &Interner, params: &[(&str, PolyTy)], effect: EffectTerm<Poly>) -> PolyTy {
    TyTerm::Fn {
        params: params
            .iter()
            .map(|(n, t)| ParamTerm::<Poly>::new(i.intern(n), t.clone()))
            .collect(),
        ret: Box::new(lift_to_poly(&Ty::I64)),
        captures: vec![],
        effect,
        flows: acvus_mir::ty::Flows::Every.into(),
    }
}

fn each(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let e = pb.fresh_effect_var();
    let callback = fn_ty(i, &[("x", lift_to_poly(&Ty::I64))], e.clone());
    declared(
        i,
        "each",
        fn_ty(i, &[("f", callback)], e),
        vec![EffectVarBound::Suspends],
    )
}

fn pull(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let e = pb.fresh_effect_var();
    let callback = fn_ty(i, &[("x", lift_to_poly(&Ty::I64))], e.clone());
    let ty = fn_ty(i, &[("f", callback)], e);
    Function {
        qref: QualifiedRef::root(i.intern("pull")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Instances {
                concrete: vec![InstanceSig {
                    ty: ty.clone(),
                    admits: Task::Heavy,
                    task: Task::Async,
                    requires: vec![],
                    effect_bounds: vec![EffectVarBound::Suspends],
                    laws: Default::default(),
                    ensures: Vec::new(),
                    reaches: Default::default(),
                }],
                generic: None,
            },
            requires: vec![],
        },
        ty,
    }
}

fn nullary(i: &Interner, name: &str, effect: Effect) -> Function {
    declared(i, name, fn_ty(i, &[], effect.into()), vec![])
}

fn externs(i: &Interner) -> Vec<Function> {
    vec![
        each(i),
        pull(i),
        nullary(i, "fetch", Effect::PURE.at_task(Task::Async)),
        nullary(i, "crunch", Effect::PURE.at_task(Task::Heavy)),
    ]
}

fn check(i: &Interner, source: &str) -> Result<String, String> {
    let context = FxHashMap::from_iter([(i.intern("x"), Ty::I64)]);
    compile_script_ir_with(i, source, &context, &externs(i))
}

fn assert_refused(source: &str) {
    let i = Interner::new();
    let err = check(&i, source).expect_err("the checker refuses a function that does not suspend");
    assert!(err.contains(BELOW_SYNC), "{err}");
}

fn assert_admitted(source: &str) {
    let i = Interner::new();
    if let Err(err) = check(&i, source) {
        panic!("admitted, got: {err}");
    }
}

#[test]
fn a_lambda_calling_an_async_extern_is_admitted() {
    assert_admitted("each(|_x| -> fetch())");
}

#[test]
fn a_lambda_calling_a_heavy_extern_is_admitted() {
    assert_admitted("each(|_x| -> crunch())");
}

#[test]
fn a_pure_lambda_is_refused() {
    assert_refused("each(|x| -> x + 1)");
}

#[test]
fn an_async_effect_settled_after_the_lambda_is_passed_is_admitted() {
    assert_admitted("let outer = |g| -> { g(1); each(g) }; outer(|_y| -> fetch())");
}

#[test]
fn a_sync_effect_settled_after_the_lambda_is_passed_is_refused() {
    assert_refused("let outer = |g| -> { g(1); each(g) }; outer(|y| -> y + 1)");
}

#[test]
fn a_suspending_instance_called_with_an_async_lambda_is_admitted() {
    assert_admitted("pull(|_x| -> fetch())");
}

#[test]
fn a_suspending_instance_called_with_a_pure_lambda_is_refused() {
    assert_refused("pull(|x| -> x + 1)");
}
