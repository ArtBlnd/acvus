//! RFC-0046: a call's task is an effect.
//!
//! Every consumer of `Iterator` has two instances — a plain `fn` at
//! `Task::Sync` and the `async fn` above it — and the solver picks by the
//! pipeline's task. These tests read the answer, which must not depend on
//! which one ran, and the prepared shape, which must.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use acvus_extern::{Externs, Registry, extern_fn, extern_registry};
use acvus_interpreter::code::Code;
use acvus_interpreter::{AcvusRuntime, PrepareCtx, TokioExecutor, Value, prepare_module};
use acvus_interpreter_test::listing::{code_listing, family_of, ops_of_anywhere};
use acvus_interpreter_test::*;
use acvus_mir::graph::{ParsedAst, QualifiedRef};
use acvus_mir::ir::MirBody;
use acvus_mir::ty::{Task, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

// -- The externs the pipelines are built from ---------------------------

/// Not pure, so `optimize::spawn_split` turns each call into a `Spawn` and
/// an `Eval`: a closure that calls it suspends, and a pipeline whose stage
/// holds that closure is `Async`.
#[extern_fn(effect = opaque)]
fn passed_through(x: i64) -> i64 {
    x
}

/// A `heavy` extern: a Rust `fn` the runtime hands to the blocking pool
/// and awaits. It returns the thread it ran on so a test can compare that
/// with the caller's.
#[extern_fn(heavy, effect = pure)]
fn on_which_thread(spin: i64) -> i64 {
    let mut sink = 0i64;
    for k in 0..spin {
        sink = sink.wrapping_add(k);
    }
    std::hint::black_box(sink);
    thread_word(std::thread::current().id())
}

#[extern_fn(effect = pure)]
fn this_thread() -> i64 {
    thread_word(std::thread::current().id())
}

/// A `heavy` extern whose cost is wall time and nothing else: two calls of
/// it overlap or they do not, and no amount of machine speed hides the
/// difference.
#[extern_fn(heavy, effect = pure)]
fn rest(millis: i64) -> i64 {
    let wait = u64::try_from(millis).expect("rest takes a non-negative number of milliseconds");
    std::thread::sleep(Duration::from_millis(wait));
    millis
}

/// `ThreadId` exposes no integer. Its `Debug` is unique per live thread,
/// which is all an "is this the same thread" comparison needs; the fold is
/// a hash of it and wraps on purpose.
fn thread_word(id: ThreadId) -> i64 {
    format!("{id:?}").bytes().fold(0i64, |acc, byte| {
        acc.wrapping_mul(HASH_RADIX).wrapping_add(i64::from(byte))
    })
}

const HASH_RADIX: i64 = 131;

fn task_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "task",
        fns: [passed_through, on_which_thread, this_thread, rest],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(task_registry());
    regs
}

/// Every `expect` below is a fixture that either stands or the test does
/// not run: a failure is the test failing, which is the disposition a test
/// wants.
async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), registries(), ret)
        .await
        .value
}

async fn run_on_tokio(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    run_parsed_on(
        &i,
        ast,
        Context::default(),
        registries(),
        ret,
        |_| {},
        Arc::new(TokioExecutor),
    )
    .await
    .value
}

/// The entry body of `source`, prepared, with the closures it makes.
fn prepared_entry(source: &str, ret: Ty) -> (Code, MirBody) {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    let cr = compile_source_with_externs(&i, ast, &FxHashMap::default(), registries(), ret);
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    let ctx = PrepareCtx {
        interner: &i,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
        instances: &acvus_extern::NoInstances,
    };
    let prepared = prepare_module(module, &ctx);
    let main = Arc::try_unwrap(prepared.main).unwrap_or_else(|_| panic!("one reference to main"));
    (main, module.main.clone())
}

// -- The registry holds two instances per consumer ----------------------

/// One consumer of `Iterator`, and how many member types it is
/// monomorphized over: the registry holds one instance pair per member.
struct Consumer {
    name: &'static str,
    members: usize,
}

const fn consumer(name: &'static str, members: usize) -> Consumer {
    Consumer { name, members }
}

/// The nineteen consumers RFC-0046's second brief names.
const CONSUMERS: [Consumer; 19] = [
    consumer("all", 1),
    consumer("any", 1),
    consumer("collect", 1),
    consumer("contains", 5),
    consumer("count", 1),
    consumer("find", 1),
    consumer("fold", 1),
    consumer("join", 1),
    consumer("last", 1),
    consumer("max", 2),
    consumer("max_by_key", 1),
    consumer("min", 2),
    consumer("min_by_key", 1),
    consumer("next", 1),
    consumer("nth", 1),
    consumer("position", 1),
    consumer("product", 2),
    consumer("reduce", 1),
    consumer("sum", 2),
];

#[test]
fn every_consumer_has_a_sync_instance_beside_its_async_one() {
    let i = Interner::new();
    let externs = Externs::<AcvusRuntime>::combine(registries(), &i).expect("registries combine");
    let iter_ns = i.intern("iter");
    for Consumer { name, members } in CONSUMERS {
        let qref = QualifiedRef::qualified(iter_ns, i.intern(name));
        let handlers = externs
            .handlers
            .get(&qref)
            .unwrap_or_else(|| panic!("iter::{name} is declared"));
        assert_eq!(
            handlers.len(),
            2 * members,
            "iter::{name} has one synchronous and one asynchronous instance per member type"
        );
        let synchronous = handlers.iter().filter(|h| h.is_sync()).count();
        assert_eq!(
            synchronous, members,
            "iter::{name}: one of each pair reaches its result without suspending"
        );
    }
}

// -- One answer, whichever instance ran ---------------------------------

/// One consumer written twice: over a pipeline of synchronous stages, and
/// over the same pipeline with a suspending stage in it. The two answers
/// are the same answer; only the instance the solver chose differs.
struct BothWays {
    consumer: &'static str,
    over_sync: &'static str,
    over_async: &'static str,
    ret: Ty,
}

const fn both_ways(
    consumer: &'static str,
    over_sync: &'static str,
    over_async: &'static str,
    ret: Ty,
) -> BothWays {
    BothWays {
        consumer,
        over_sync,
        over_async,
        ret,
    }
}

/// `map(|x| -> passed_through(x))` is the suspending stage: the closure
/// calls a non-pure extern, which is spawned and evaluated.
const BOTH_WAYS: [BothWays; 17] = [
    both_ways(
        "sum",
        "range(0, 5) | sum",
        "range(0, 5) | map(|x| -> passed_through(x)) | sum",
        Ty::I64,
    ),
    both_ways(
        "product",
        "range(1, 5) | product",
        "range(1, 5) | map(|x| -> passed_through(x)) | product",
        Ty::I64,
    ),
    both_ways(
        "count",
        "range(0, 5) | count",
        "range(0, 5) | map(|x| -> passed_through(x)) | count",
        Ty::I64,
    ),
    both_ways(
        "min",
        "if let Some(v) = range(0, 5) | min { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | min { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "max",
        "if let Some(v) = range(0, 5) | max { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | max { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "fold",
        "range(0, 5) | fold(0, |a, b| -> a + b)",
        "range(0, 5) | map(|x| -> passed_through(x)) | fold(0, |a, b| -> a + b)",
        Ty::I64,
    ),
    both_ways(
        "reduce",
        "if let Some(v) = range(0, 5) | reduce(|a, b| -> a + b) { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | reduce(|a, b| -> a + b) { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "find",
        "if let Some(v) = range(0, 5) | find(|x| -> *x > 2) { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | find(|x| -> *x > 2) { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "next",
        "let a = range(0, 5); if let Some(v) = next(&mut a) { v } else { 0 - 1 }",
        "let b = range(0, 5) | map(|x| -> passed_through(x)); if let Some(v) = next(&mut b) { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "collect",
        "let c = range(0, 5) | collect; c.len()",
        "let d = range(0, 5) | map(|x| -> passed_through(x)) | collect; d.len()",
        Ty::U64,
    ),
    both_ways(
        "any",
        "if range(0, 5) | any(|x| -> *x == 3) { 1 } else { 0 }",
        "if range(0, 5) | map(|x| -> passed_through(x)) | any(|x| -> *x == 3) { 1 } else { 0 }",
        Ty::I64,
    ),
    both_ways(
        "all",
        "if range(0, 5) | all(|x| -> *x < 9) { 1 } else { 0 }",
        "if range(0, 5) | map(|x| -> passed_through(x)) | all(|x| -> *x < 9) { 1 } else { 0 }",
        Ty::I64,
    ),
    both_ways(
        "position",
        "if let Some(v) = range(0, 5) | position(|x| -> *x == 4) { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | position(|x| -> *x == 4) { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "last",
        "if let Some(v) = range(0, 5) | last { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | last { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "nth",
        "if let Some(v) = range(0, 5) | nth(2) { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | nth(2) { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "min_by_key",
        "if let Some(v) = range(0, 5) | min_by_key(|x| -> 0 - *x) { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | min_by_key(|x| -> 0 - *x) { v } else { 0 - 1 }",
        Ty::I64,
    ),
    both_ways(
        "max_by_key",
        "if let Some(v) = range(0, 5) | max_by_key(|x| -> 0 - *x) { v } else { 0 - 1 }",
        "if let Some(v) = range(0, 5) | map(|x| -> passed_through(x)) | max_by_key(|x| -> 0 - *x) { v } else { 0 - 1 }",
        Ty::I64,
    ),
];

#[tokio::test]
async fn a_consumer_answers_the_same_through_either_instance() {
    for BothWays {
        consumer,
        over_sync,
        over_async,
        ret,
    } in BOTH_WAYS
    {
        let from_sync = run(over_sync, ret.clone()).await.as_int();
        let from_async = run(over_async, ret).await.as_int();
        assert_eq!(
            from_sync, from_async,
            "iter::{consumer}: `{over_sync}` and `{over_async}` are the same answer"
        );
    }
}

/// `contains` and `join` take an element the other seventeen do not, so
/// they are written out rather than folded into the table.
#[tokio::test]
async fn contains_and_join_answer_the_same_through_either_instance() {
    let over_sync = run("if range(0, 5) | contains(3) { 1 } else { 0 }", Ty::I64)
        .await
        .as_int();
    let over_async = run(
        "if range(0, 5) | map(|x| -> passed_through(x)) | contains(3) { 1 } else { 0 }",
        Ty::I64,
    )
    .await
    .as_int();
    assert_eq!(over_sync, 1);
    assert_eq!(over_sync, over_async);

    let joined = run(
        r#"let j = into_iter(split_str("a,b,c", ",")) | join("-".to_string()); j.len()"#,
        Ty::U64,
    )
    .await
    .as_int();
    assert_eq!(joined, 5);
}

// -- Heavy --------------------------------------------------------------

/// `tokio::task::spawn_blocking` uses the blocking pool whatever the
/// runtime's flavor, so a current-thread test runtime shows the offload.
#[tokio::test]
async fn a_heavy_extern_runs_off_the_runtime_thread() {
    let here = run_on_tokio("this_thread()", Ty::I64).await.as_int();
    let there = run_on_tokio("on_which_thread(100000)", Ty::I64)
        .await
        .as_int();
    assert_ne!(
        here, there,
        "a heavy extern runs on the blocking pool, not on the runtime thread"
    );
}

/// `SequentialExecutor::spawn_blocking` has no pool: it defers the closure
/// and `eval` runs it on the awaiting thread. The await is real; the
/// offload is not.
#[tokio::test]
async fn a_heavy_extern_on_the_sequential_executor_runs_inline() {
    let here = thread_word(std::thread::current().id());
    let there = run("on_which_thread(1000)", Ty::I64).await.as_int();
    assert_eq!(here, there);
}

/// Compilation is outside the measurement: what is timed is the run.
async fn executed_in(source: &str, ret: Ty) -> (Duration, i64) {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    let cr = compile_source_with_externs(&i, ast, &FxHashMap::default(), registries(), ret);
    let (_shared, mut interp) = execute_compiled(&i, cr, HashMap::new(), Arc::new(TokioExecutor));
    let start = Instant::now();
    let value: Value = interp.execute().await;
    (start.elapsed(), value.as_int())
}

const REST_MILLIS: i64 = 200;

const TWO_CALLS_AT_MOST: f64 = 1.5;

#[tokio::test]
async fn two_independent_pure_heavy_calls_overlap() {
    let (one, once) = executed_in(&format!("rest({REST_MILLIS})"), Ty::I64).await;
    let (two, twice) = executed_in(
        &format!("let a = rest({REST_MILLIS}); let b = rest({REST_MILLIS}); a + b"),
        Ty::I64,
    )
    .await;
    assert_eq!(once, REST_MILLIS);
    assert_eq!(twice, 2 * REST_MILLIS);
    assert!(
        two < one.mul_f64(TWO_CALLS_AT_MOST),
        "two independent pure Heavy calls run at once: one call took {one:?}, two took \
         {two:?}, over the ceiling of {TWO_CALLS_AT_MOST} x one = {:?}. Serialized, two \
         would take about {:?}",
        one.mul_f64(TWO_CALLS_AT_MOST),
        2 * one
    );
}

#[tokio::test]
async fn a_pipeline_over_a_heavy_extern_is_asynchronous() {
    let source = "range(0, 4) | map(|x| -> on_which_thread(10)) | count";
    let (_, entry) = prepared_entry(source, Ty::I64);
    assert!(
        entry.task > Task::Sync,
        "the body joins the task of the closure it makes, and that closure calls a heavy extern"
    );
    assert_eq!(run(source, Ty::I64).await.as_int(), 4);
}

// -- The shape the type bought ------------------------------------------

fn loop_count(code: &Code) -> usize {
    ops_of_anywhere(&code_listing(code))
        .iter()
        .filter(|name| family_of(name) == "Loop")
        .count()
}

fn may_suspend(code: &Code) -> bool {
    code.may_suspend()
}

#[tokio::test]
async fn a_while_let_over_a_sync_iterator_is_one_loop_operation() {
    let source = "let v = range(0, 8) | collect; let it = as_iter(&v); let acc = 0; \
                  while let Some(x) = next(&mut it) { acc = acc + *x; } acc";
    let (main, _) = prepared_entry(source, Ty::I64);
    assert!(
        !may_suspend(&main),
        "every call in the body is synchronous, so the body is"
    );
    assert_eq!(
        loop_count(&main),
        1,
        "the `while let` head is one Loop operation"
    );
    assert_eq!(run(source, Ty::I64).await.as_int(), 28);
}

#[tokio::test]
async fn a_while_let_over_an_async_iterator_stays_asynchronous() {
    let source = "let it = range(0, 8) | map(|x| -> passed_through(x)); let acc = 0; \
                  while let Some(x) = next(&mut it) { acc = acc + x; } acc";
    let (main, _) = prepared_entry(source, Ty::I64);
    assert!(
        may_suspend(&main),
        "a stage of the pipeline suspends, so the head does, so the body does"
    );
    assert_eq!(
        loop_count(&main),
        0,
        "a head that may suspend is not a Loop operation"
    );
    assert_eq!(run(source, Ty::I64).await.as_int(), 28);
}
