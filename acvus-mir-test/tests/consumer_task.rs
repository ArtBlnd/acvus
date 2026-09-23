//! RFC-0046: a consumer's frozen effect is the instance it runs.
//!
//! A call settles on a signature through a decision's join. That join read
//! the call's effect as a value flowing into the declared position - at
//! most what the position allows, which is RFC-0046 rule 7's demotion - and so a
//! call to an *overloaded* name took none of its callee's effect: `find`,
//! `last` and `contains` share their bare name with `str::find`,
//! `vec::last` and `str::contains`, and over a suspending pipeline the
//! solver took their asynchronous instance while the call froze to `Pure`.
//! The three programs below are that reproduction; the rest of the table
//! are the consumers whose names are theirs alone and were already right.

use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::{Callee, InstKind, MirBody, MirModule};
use acvus_mir::ty::{
    Effect, EffectTerm, Instances, ParamTerm, Poly, PolyBuilder, PolyTy, Task, Ty, TyTerm,
    TypeRegistry, lift_to_poly,
};
use acvus_mir_test::{lowered_script_module, optimized_script_module};
use acvus_utils::{Freeze, Interner};

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

fn extern_at(i: &Interner, qref: QualifiedRef, param: Ty, effect: Effect) -> Function {
    Function {
        qref,
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty: fn_ty(
            i,
            &[("x", lift_to_poly(&param))],
            lift_to_poly(&Ty::I64),
            effect.into(),
        ),
    }
}

/// The suspending stage every pipeline below is built from: the shape the
/// interpreter's `task_instances.rs` calls `passed_through`.
fn passed_through(i: &Interner) -> Function {
    extern_at(
        i,
        QualifiedRef::root(i.intern("passed_through")),
        Ty::I64,
        Effect::OPAQUE.at_task(Task::Async),
    )
}

// -- The table ----------------------------------------------------------

struct Pipeline {
    consumer: &'static str,
    source: &'static str,
    /// The task this pipeline runs at, which is the task its consumer's
    /// call must carry and its body must be.
    task: Task,
}

const fn pipeline(consumer: &'static str, source: &'static str) -> Pipeline {
    Pipeline {
        consumer,
        source,
        task: Task::Async,
    }
}

/// A pipeline whose every stage is pure.
const fn synchronous(consumer: &'static str, source: &'static str) -> Pipeline {
    Pipeline {
        consumer,
        source,
        task: Task::Sync,
    }
}

/// The three names that are shared with another declaration; they are the
/// reproduction, and they are read by the same assertion as the rest.
const PIPELINES: [Pipeline; 19] = [
    pipeline(
        "find",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | find(|x| -> *x > 2) { v } else { 0 - 1 }",
    ),
    pipeline(
        "last",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | last { v } else { 0 - 1 }",
    ),
    pipeline(
        "contains",
        "if range(0, 5) | map(|x| -> passed_through(x)) | contains(3) { 1 } else { 0 }",
    ),
    pipeline(
        "next",
        "let b = range(0, 5) | map(|x| -> passed_through(x)); if let Some(v) = next(&mut b) { v } else { 0 - 1 }",
    ),
    pipeline(
        "nth",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | nth(2) { v } else { 0 - 1 }",
    ),
    pipeline(
        "reduce",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | reduce(|a, b| -> a + b) { v } else { 0 - 1 }",
    ),
    pipeline(
        "min",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | min { v } else { 0 - 1 }",
    ),
    pipeline(
        "max",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | max { v } else { 0 - 1 }",
    ),
    pipeline(
        "min_by_key",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | min_by_key(|x| -> 0 - *x) { v } else { 0 - 1 }",
    ),
    pipeline(
        "max_by_key",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | max_by_key(|x| -> 0 - *x) { v } else { 0 - 1 }",
    ),
    pipeline(
        "collect",
        "let d = range(0, 5) | map(|x| -> passed_through(x)) | collect; d.len()",
    ),
    pipeline(
        "product",
        "range(1, 5) | map(|x| -> passed_through(x)) | product",
    ),
    pipeline(
        "any",
        "if range(0, 5) | map(|x| -> passed_through(x)) | any(|x| -> *x == 3) { 1 } else { 0 }",
    ),
    pipeline(
        "all",
        "if range(0, 5) | map(|x| -> passed_through(x)) | all(|x| -> *x < 9) { 1 } else { 0 }",
    ),
    pipeline(
        "position",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | position(|x| -> *x == 4) { v } else { 0 - 1 }",
    ),
    pipeline("sum", "range(0, 5) | map(|x| -> passed_through(x)) | sum"),
    pipeline(
        "fold",
        "range(0, 5) | map(|x| -> passed_through(x)) | fold(0, |a, b| -> a + b)",
    ),
    pipeline(
        "count",
        "range(0, 5) | map(|x| -> passed_through(x)) | count",
    ),
    synchronous(
        "join",
        r#"let j = into_iter(vec(["a".to_string(), "b".to_string(), "c".to_string()])) | map(|s| -> s) | join("-".to_string()); j.len()"#,
    ),
];

/// A pipeline of pure stages: the same consumers, at `Task::Sync`.
const SYNCHRONOUS: [Pipeline; 3] = [
    synchronous(
        "find",
        "if let Some(v) = range(0, 5) | find(|x| -> *x > 2) { v } else { 0 - 1 }",
    ),
    synchronous(
        "last",
        "if let Some(v) = range(0, 5) | last { v } else { 0 - 1 }",
    ),
    synchronous("contains", "if range(0, 5) | contains(3) { 1 } else { 0 }"),
];

fn call_effect_of(module: &MirModule, i: &Interner, consumer: &str) -> Effect {
    module
        .main
        .insts
        .iter()
        .find_map(|inst| match &inst.kind {
            InstKind::FunctionCall {
                callee: Callee::Extern { id, .. },
                callee_ty,
                ..
            } if i.resolve(id.name) == consumer => Some(
                callee_ty
                    .effect()
                    .expect("a callee type is a function type"),
            ),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a call to `{consumer}` in the entry body"))
}

#[test]
fn a_consumer_over_a_suspending_pipeline_carries_the_pipelines_task() {
    for Pipeline {
        consumer,
        source,
        task,
    } in PIPELINES
    {
        let i = Interner::new();
        let module = lowered_script_module(&i, source, &[passed_through(&i)]).expect("lowers");
        assert_eq!(
            call_effect_of(&module, &i, consumer).task,
            task,
            "iter::{consumer}: the call's frozen effect is the task it runs"
        );
        assert_eq!(
            module.main.task, task,
            "iter::{consumer}: the body's task is the join over its calls"
        );
    }
}

#[test]
fn a_consumer_over_a_synchronous_pipeline_stays_sync() {
    for Pipeline {
        consumer,
        source,
        task,
    } in SYNCHRONOUS
    {
        let i = Interner::new();
        let module = lowered_script_module(&i, source, &[]).expect("lowers");
        assert_eq!(
            call_effect_of(&module, &i, consumer).task,
            task,
            "iter::{consumer}: nothing in the pipeline suspends"
        );
        assert_eq!(module.main.task, task);
    }
}

// -- A body's task is the join over what it does ------------------------

/// RFC-0046's soundness claim: a call, a `Spawn` and an `Eval` are the
/// whole set, so every other instruction contributes the join's identity.
const NOTHING_ELSE_RAISES_IT: Task = Task::Sync;

/// The task of everything `body` does: each call's declared task, and
/// `Async` for every `Spawn` and `Eval`, which await (`spawn_split`).
fn join_of_what_it_does(body: &MirBody) -> Task {
    body.insts
        .iter()
        .fold(NOTHING_ELSE_RAISES_IT, |task, inst| {
            let of_inst = match &inst.kind {
                InstKind::FunctionCall { callee_ty, .. } => {
                    callee_ty
                        .effect()
                        .expect("a callee type is a function type")
                        .task
                }
                InstKind::Spawn { callee_ty, .. } => callee_ty
                    .effect()
                    .expect("a callee type is a function type")
                    .task
                    .join(Task::Async),
                InstKind::Eval { .. } => Task::Async,
                _ => NOTHING_ELSE_RAISES_IT,
            };
            task.join(of_inst)
        })
}

/// One program's module at one point of the pipeline.
struct Compiled {
    stage: &'static str,
    module: MirModule,
}

#[test]
fn a_bodys_task_is_the_join_over_its_calls_and_its_spawns() {
    for Pipeline {
        consumer, source, ..
    } in PIPELINES
    {
        let i = Interner::new();
        let compiled = [
            Compiled {
                stage: "lowered",
                module: lowered_script_module(&i, source, &[passed_through(&i)]).expect("lowers"),
            },
            Compiled {
                stage: "optimized",
                module: optimized_script_module(&i, source, &[passed_through(&i)])
                    .expect("optimizes"),
            },
        ];
        for Compiled { stage, module } in &compiled {
            assert_eq!(
                module.main.task,
                join_of_what_it_does(&module.main),
                "iter::{consumer} ({stage}): the entry body's task is the join over what it does"
            );
            for (label, body) in &module.closures {
                assert!(
                    body.task >= join_of_what_it_does(body),
                    "iter::{consumer} ({stage}), closure {label:?}: a closure's task is at least \
                     the join over what it does; above it is the demotion RFC-0046 joins in"
                );
            }
        }
    }
}

// -- The origin, without an iterator in it ------------------------------

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

fn effect_of(i: &Interner, functions: Vec<Function>, name: &str) -> Effect {
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(TypeRegistry::new()),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: None,
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext);
    assert!(!inf.has_errors(), "infer errors: {:?}", inf.errors());
    inf.outcomes
        .get(&QualifiedRef::root(i.intern(name)))
        .expect("an outcome for the target")
        .meta()
        .ty
        .effect()
        .expect("a function type")
}

fn step_running(i: &Interner, ns: &str, param: Ty, effect: Effect) -> Function {
    extern_at(
        i,
        QualifiedRef::qualified(i.intern(ns), i.intern("step")),
        param,
        effect,
    )
}

fn step(i: &Interner, ns: &str, param: Ty) -> Function {
    step_running(i, ns, param, Effect::OPAQUE.at_task(Task::Async))
}

/// The name is one declaration's alone: the call type carries the declared
/// effect itself, and the body has always read it.
#[test]
fn a_call_to_a_name_one_declaration_owns_takes_its_effect() {
    let i = Interner::new();
    let effect = effect_of(
        &i,
        vec![step(&i, "one", Ty::I64), local_fn(&i, "target", "step(1)")],
        "target",
    );
    assert_eq!(effect.task, Task::Async);
    assert!(!effect.is_pure());
}

/// Two declarations share the name, so the call goes through a signature
/// decision. Its effect is the one it settled on, not `Pure`.
#[test]
fn a_call_to_an_overloaded_name_takes_the_effect_it_settled_on() {
    let i = Interner::new();
    let effect = effect_of(
        &i,
        vec![
            step(&i, "one", Ty::I64),
            step(&i, "two", Ty::Bool),
            local_fn(&i, "target", "step(1)"),
        ],
        "target",
    );
    assert_eq!(effect.task, Task::Async);
    assert!(!effect.is_pure());
}

/// The structure under the two rows above, rather than their outcome: the
/// call's effect term is the settled instance's, so whatever that one
/// declaration carries the call carries, whole - contexts, purity and task.
/// The two declarations run different effects and neither is the bottom of
/// the lattice, so the assertion separates three answers a call could give:
/// the instance's, the other declaration's, and the `Pure/Sync` a term of
/// the call's own would freeze to. Only the first passes, and a name one
/// declaration owns is read by the same assertion.
#[test]
fn a_calls_effect_is_the_term_of_its_instance() {
    struct Declaration {
        namespace: &'static str,
        param: Ty,
        effect: Effect,
        argument: &'static str,
    }
    let declarations = [
        Declaration {
            namespace: "one",
            param: Ty::I64,
            effect: Effect::OPAQUE.at_task(Task::Heavy),
            argument: "1",
        },
        Declaration {
            namespace: "two",
            param: Ty::Bool,
            effect: Effect::IDEMPOTENT.at_task(Task::Async),
            argument: "true",
        },
    ];

    let i = Interner::new();
    let declared: Vec<Function> = declarations
        .iter()
        .map(|d| step_running(&i, d.namespace, d.param.clone(), d.effect.clone()))
        .collect();
    for (n, d) in declarations.iter().enumerate() {
        for externs in [&declared[n..=n], &declared[..]] {
            let module = lowered_script_module(&i, &format!("step({})", d.argument), externs)
                .expect("lowers");
            assert_eq!(
                call_effect_of(&module, &i, "step"),
                d.effect,
                "`step({})`, the name shared by {} declaration(s): the call's effect is the \
                 term of the instance it settled on",
                d.argument,
                externs.len()
            );
        }
    }
}

// -- The same rule over the RFC-0046 compiler tests' programs -----------

fn nullary(i: &Interner, name: &str, effect: Effect) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
            requires: vec![],
        },
        ty: fn_ty(i, &[], lift_to_poly(&Ty::I64), effect.into()),
    }
}

/// The programs `tests/task.rs` states RFC-0046's table with, read here as
/// modules rather than as function effects.
const TABLE_PROGRAMS: [&str; 6] = [
    "tick() + tick()",
    "tick() + fetch()",
    "tick() + hash()",
    "fetch() + hash()",
    "let f = |_x| -> fetch(); 1",
    "let f = |_x| -> fetch(); f(1)",
];

#[test]
fn the_rfcs_own_programs_join_the_same_way() {
    for source in TABLE_PROGRAMS {
        let i = Interner::new();
        let externs = vec![
            nullary(&i, "tick", Effect::PURE),
            nullary(&i, "fetch", Effect::OPAQUE.at_task(Task::Async)),
            nullary(&i, "hash", Effect::PURE.at_task(Task::Heavy)),
        ];
        let module = lowered_script_module(&i, source, &externs).expect("lowers");
        assert_eq!(
            module.main.task,
            join_of_what_it_does(&module.main),
            "`{source}`: the entry body's task is the join over what it does"
        );
    }
}
