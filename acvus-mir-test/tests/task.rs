//! A call's task is an effect (RFC-0046), at the types a caller sees.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
    lower as graph_lower,
};
use acvus_mir::ir::{Callee, InstKind};
use acvus_mir::ty::{
    Effect, EffectTerm, InstanceSig, Instances, ParamTerm, Poly, PolyBuilder, PolyTy, Task, Ty,
    TyTerm, TypeRegistry, lift_to_poly,
};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

fn fn_ty(i: &Interner, params: &[(&str, PolyTy)], ret: PolyTy, effect: EffectTerm<Poly>) -> PolyTy {
    TyTerm::Fn {
        params: params
            .iter()
            .map(|(n, t)| ParamTerm::<Poly>::new(i.intern(n), t.clone()))
            .collect(),
        ret: Box::new(ret),
        captures: vec![],
        effect,
    }
}

fn extern_fn(i: &Interner, name: &str, ty: PolyTy) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty,
    }
}

fn nullary(i: &Interner, name: &str, effect: Effect) -> Function {
    extern_fn(
        i,
        name,
        fn_ty(i, &[], lift_to_poly(&Ty::I64), effect.into()),
    )
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

fn graph_of(functions: Vec<Function>) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(TypeRegistry::new()),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: None,
    }
}

fn inferred(
    i: &Interner,
    graph: &CompilationGraph,
) -> (extract::ExtractResult, infer::InferResult) {
    let ext = extract::extract(i, graph);
    let inf = infer::infer(i, graph, &ext);
    (ext, inf)
}

fn effects(i: &Interner, functions: Vec<Function>) -> FxHashMap<String, Effect> {
    let graph = graph_of(functions);
    let (_, inf) = inferred(i, &graph);
    assert!(!inf.has_errors(), "infer errors: {:?}", inf.errors());
    inf.outcomes
        .iter()
        .map(|(qref, outcome)| {
            let effect = outcome.meta().ty.effect().expect("function type");
            (i.resolve(qref.name).to_string(), effect)
        })
        .collect()
}

fn refusals(i: &Interner, functions: Vec<Function>) -> Vec<String> {
    let graph = graph_of(functions);
    let (_, inf) = inferred(i, &graph);
    inf.errors()
        .into_iter()
        .flat_map(|(_, errs)| {
            errs.iter()
                .map(|e| e.display(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn instances_taken(i: &Interner, functions: Vec<Function>, name: &str) -> Vec<usize> {
    let target = QualifiedRef::root(i.intern(name));
    let graph = graph_of(functions);
    let (ext, inf) = inferred(i, &graph);
    assert!(!inf.has_errors(), "infer errors: {:?}", inf.errors());
    let lowered = graph_lower::lower(i, &graph, &ext.view(), &inf);
    let module = lowered.module(target).expect("a module for the target");
    module
        .main
        .insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::FunctionCall {
                callee: Callee::Extern { instance, .. },
                ..
            }
            | InstKind::Spawn {
                callee: Callee::Extern { instance, .. },
                ..
            } => Some(*instance),
            _ => None,
        })
        .collect()
}

fn hof(i: &Interner, name: &str, param_task: Task) -> Function {
    let callback = fn_ty(
        i,
        &[("x", lift_to_poly(&Ty::I64))],
        lift_to_poly(&Ty::I64),
        Effect::PURE.at_task(param_task).into(),
    );
    extern_fn(
        i,
        name,
        fn_ty(
            i,
            &[("f", callback)],
            lift_to_poly(&Ty::I64),
            Effect::PURE.into(),
        ),
    )
}

fn relay(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let e = pb.fresh_effect_var();
    let callback = fn_ty(
        i,
        &[("x", lift_to_poly(&Ty::I64))],
        lift_to_poly(&Ty::I64),
        e,
    );
    extern_fn(
        i,
        "relay",
        fn_ty(i, &[("f", callback.clone())], callback, Effect::PURE.into()),
    )
}

fn sync_or_async(i: &Interner) -> Function {
    let mut pb = PolyBuilder::new();
    let e = pb.fresh_effect_var();
    let callback = fn_ty(
        i,
        &[("x", lift_to_poly(&Ty::I64))],
        lift_to_poly(&Ty::I64),
        e.clone(),
    );
    let ty = fn_ty(i, &[("f", callback)], lift_to_poly(&Ty::I64), e);
    Function {
        qref: QualifiedRef::root(i.intern("sync_or_async")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Instances {
                concrete: vec![
                    InstanceSig {
                        ty: ty.clone(),
                        admits: Task::Sync,
                        task: Task::Sync,
                        requires: vec![],
                        effect_bounds: vec![],
                        laws: Default::default(),
                        ensures: Vec::new(),
                    },
                    InstanceSig {
                        ty: ty.clone(),
                        admits: Task::Heavy,
                        task: Task::Async,
                        requires: vec![],
                        effect_bounds: vec![],
                        laws: Default::default(),
                        ensures: Vec::new(),
                    },
                ],
                generic: None,
            },
            requires: vec![],
        },
        ty,
    }
}

#[test]
fn a_bodys_task_is_the_join_of_its_calls() {
    let i = Interner::new();
    let effects = effects(
        &i,
        vec![
            nullary(&i, "tick", Effect::PURE),
            nullary(&i, "fetch", Effect::OPAQUE.at_task(Task::Async)),
            nullary(&i, "hash", Effect::PURE.at_task(Task::Heavy)),
            local_fn(&i, "only_sync", "tick() + tick()"),
            local_fn(&i, "awaits", "tick() + fetch()"),
            local_fn(&i, "offloads", "tick() + hash()"),
            local_fn(&i, "heavy_over_async", "fetch() + hash()"),
        ],
    );
    assert_eq!(effects["only_sync"].task, Task::Sync);
    assert_eq!(effects["awaits"].task, Task::Async);
    assert_eq!(effects["offloads"].task, Task::Heavy);
    assert_eq!(effects["heavy_over_async"].task, Task::Heavy);
}

#[test]
fn a_pure_call_may_be_heavy() {
    let i = Interner::new();
    let effects = effects(
        &i,
        vec![
            nullary(&i, "hash", Effect::PURE.at_task(Task::Heavy)),
            local_fn(&i, "offloads", "hash()"),
        ],
    );
    assert!(effects["offloads"].is_pure());
    assert_eq!(effects["offloads"].task, Task::Heavy);
}

#[test]
fn a_closures_task_is_its_bodys() {
    let i = Interner::new();
    let effects = effects(
        &i,
        vec![
            nullary(&i, "fetch", Effect::PURE.at_task(Task::Async)),
            local_fn(&i, "defines", "let f = |_x| -> fetch(); 1"),
            local_fn(&i, "calls", "let f = |_x| -> fetch(); f(1)"),
        ],
    );
    assert_eq!(effects["defines"].task, Task::Sync);
    assert_eq!(effects["calls"].task, Task::Async);
}

#[test]
fn demotion_is_the_join() {
    let i = Interner::new();
    let effects = effects(
        &i,
        vec![
            hof(&i, "needs_async", Task::Async),
            local_fn(&i, "passes_sync", "needs_async(|x| -> x + 1)"),
        ],
    );
    assert_eq!(effects["passes_sync"].task, Task::Sync);
}

#[test]
fn a_task_above_what_the_position_fixes_is_refused() {
    let i = Interner::new();
    let errors = refusals(
        &i,
        vec![
            hof(&i, "needs_sync", Task::Sync),
            nullary(&i, "fetch", Effect::PURE.at_task(Task::Async)),
            local_fn(&i, "passes_async", "needs_sync(|_x| -> fetch())"),
        ],
    );
    assert!(
        errors
            .iter()
            .any(|e| e.contains("a function whose task is Async where Sync is required")),
        "{errors:?}"
    );
}

#[test]
fn a_task_settled_after_the_callback_is_passed_is_refused() {
    let i = Interner::new();
    let errors = refusals(
        &i,
        vec![
            hof(&i, "needs_sync", Task::Sync),
            relay(&i),
            nullary(&i, "fetch", Effect::PURE.at_task(Task::Async)),
            local_fn(
                &i,
                "passes_async_later",
                "let outer = |g| -> { g(1); needs_sync(relay(g)) }; outer(|_y| -> fetch())",
            ),
        ],
    );
    assert!(
        errors
            .iter()
            .any(|e| e.contains("a function whose task is Async where Sync is required")),
        "{errors:?}"
    );
}

#[test]
fn the_instance_is_the_tightest_that_admits_the_task() {
    let i = Interner::new();
    let sync = instances_taken(
        &i,
        vec![
            sync_or_async(&i),
            local_fn(&i, "target", "sync_or_async(|x| -> x + 1)"),
        ],
        "target",
    );
    assert_eq!(sync, vec![0], "a Sync callback takes the Sync instance");

    let i = Interner::new();
    let awaited = instances_taken(
        &i,
        vec![
            sync_or_async(&i),
            nullary(&i, "fetch", Effect::PURE.at_task(Task::Async)),
            local_fn(&i, "target", "sync_or_async(|_x| -> fetch())"),
        ],
        "target",
    );
    assert_eq!(
        awaited,
        vec![1],
        "an Async callback takes the instance that awaits"
    );
}
