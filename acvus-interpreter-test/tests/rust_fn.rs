//! RFC-0097 rule 2 at the host's contract: an extern returns a `RustFn`, and
//! the script calls it as any function value.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{Declared, Registry, Runtime, RustFn, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, HostError, HostGraph, SequentialExecutor, Source};
use acvus_interpreter_test::{Context, Helper, Refusal, check_graph, check_source, execute_compiled, run_script_with_externs};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{Effect, Flows, IntTy, ObjectTy, ParamTerm, Poly, PolyParam, PolyTy, Ty, TyTerm};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[extern_fn(effect = pure)]
fn adder<R>(k: i64) -> RustFn<(i64,), i64, R>
where
    R: Runtime,
{
    RustFn::new(move |_, args| args.with(0, |n: &i64| n + k).expect("the argument is an `i64`"))
}

static RELEASED: AtomicUsize = AtomicUsize::new(0);
static RELEASED_AT_EACH_CALL: Mutex<Vec<usize>> = Mutex::new(Vec::new());

struct CountsItsRelease(i64);

impl CountsItsRelease {
    fn add(&self, n: i64) -> i64 {
        n + self.0
    }
}

impl Drop for CountsItsRelease {
    fn drop(&mut self) {
        RELEASED.fetch_add(1, Ordering::SeqCst);
    }
}

#[extern_fn(effect = opaque)]
fn counted_adder<R>(k: i64) -> RustFn<(i64,), i64, R>
where
    R: Runtime,
{
    let state = CountsItsRelease(k);
    RustFn::new(move |_, args| {
        RELEASED_AT_EACH_CALL
            .lock()
            .expect("no test panicked holding the lock")
            .push(RELEASED.load(Ordering::SeqCst));
        args.with(0, |n: &i64| state.add(*n)).expect("the argument is an `i64`")
    })
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [adder, counted_adder],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(registry());
    registries
}

async fn run_i64(source: &str) -> i64 {
    let i = Interner::new();
    run_script_with_externs(&i, source, Context::default(), registries(), Ty::I64)
        .await
        .value
        .as_int()
}

#[tokio::test]
async fn a_rust_fn_is_called_twice_with_the_constant_it_captured() {
    assert_eq!(run_i64("let plus = adder(10);\nplus(1) * 100 + plus(2)").await, 1112);
}

/// The effect is the one `adder` declares, `pure`; change one and the other
/// must follow.
fn adder_result_ty(i: &Interner) -> PolyTy {
    TyTerm::Fn {
        params: vec![ParamTerm::<Poly>::new(i.intern("_0"), TyTerm::Int(IntTy::I64))],
        ret: Box::new(TyTerm::Int(IntTy::I64)),
        captures: vec![],
        effect: Effect::PURE.into(),
        flows: Flows::Every.into(),
    }
}

fn applies(i: &Interner) -> Vec<PolyParam> {
    vec![
        ParamTerm::<Poly>::new(i.intern("f"), adder_result_ty(i)),
        ParamTerm::<Poly>::new(i.intern("x"), TyTerm::Int(IntTy::I64)),
    ]
}

fn run_with_helpers(i: &Interner, helpers: &[Helper<'_>], main: &str, opt: Opt) -> i64 {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(i, parsed, helpers, &FxHashMap::default(), registries(), Ty::I64, opt, |_| {})
        .unwrap_or_else(|refused| panic!("{opt:?} refused:\n  {}", refused.messages.join("\n  ")));
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        std::sync::Arc::new(SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    runtime
        .block_on(interp.execute())
        .expect("the seeds hold every context the run fetches")
        .as_int()
}

#[test]
fn a_rust_fn_in_a_variable_is_passed_to_a_script_function_that_calls_it() {
    let i = Interner::new();
    let helpers = [Helper {
        name: "apply",
        source: "$f($x)\n",
        params: applies(&i),
    }];
    for opt in [Opt::None, Opt::Full] {
        assert_eq!(run_with_helpers(&i, &helpers, "let plus = adder(5);\napply(plus, 7)\n", opt), 12, "{opt:?}");
    }
}

#[tokio::test]
async fn the_state_is_released_once_after_the_last_call() {
    assert_eq!(run_i64("let plus = counted_adder(1);\nplus(1) + plus(2)").await, 5);
    assert_eq!(
        *RELEASED_AT_EACH_CALL.lock().expect("no test panicked holding the lock"),
        [0, 0]
    );
    assert_eq!(RELEASED.load(Ordering::SeqCst), 1);
}

fn refused(source: &str) -> Vec<String> {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the source parses"));
    match check_source(&i, parsed, &FxHashMap::default(), registries(), Ty::I64, Opt::Full, |_| {}) {
        Ok(_) => panic!("`{source}` compiled"),
        Err(Refusal { messages, .. }) => messages,
    }
}

#[test]
fn a_call_with_a_second_argument_is_a_compile_error() {
    let messages = refused("let plus = adder(1);\nplus(1, 2)");
    assert_eq!(messages, ["[main] this closure expects 1 arguments, got 2"]);
}

#[test]
fn a_call_with_a_string_argument_is_a_compile_error() {
    let messages = refused("let plus = adder(1);\nplus(\"one\".to_string())");
    assert_eq!(messages, ["[main] type mismatch: expected i64, got String"]);
}

// -- Nothing callable crosses a host boundary (RFC-0095 rule 4) --------------

pub struct TakesAnAdder;

impl Declared for TakesAnAdder {
    fn declared(interner: &Interner) -> PolyTy {
        TyTerm::Object(ObjectTy::written([(interner.intern("add"), adder_result_ty(interner))].into_iter().collect()))
    }
}

pub struct AnAdder;

impl Declared for AnAdder {
    fn declared(interner: &Interner) -> PolyTy {
        adder_result_ty(interner)
    }
}

fn exposing<I, R>(main: &'static str, exposed: &'static str) -> Result<HostGraph, HostError>
where
    I: Declared,
    R: Declared,
{
    HostGraph::new(registries())
        .host("a", move |host| Ok(host.entry::<(), i64>("main", Source::Script(main))))
        .and_then(|g| g.host("b", move |host| Ok(host.entry::<I, R>("e", Source::Script(exposed)))))
        .map(|g| g.expose("a", "e", "b", "e").entry("a", "main"))
}

fn crossing_refusals(graph: Result<HostGraph, HostError>) -> Vec<String> {
    let refusals = match graph.and_then(|graph| graph.compile(SequentialExecutor)) {
        Ok(_) => panic!("the graph compiled"),
        Err(HostError::Refused(refusals)) => refusals,
        Err(other) => panic!("a graph is refused with its refusals, not {other:?}"),
    };
    refusals
        .into_iter()
        .map(|refusal| refusal.message)
        .filter(|message| message.contains("nothing callable crosses"))
        .collect()
}

#[test]
fn a_rust_fn_in_an_exposed_input_is_refused() {
    let messages = crossing_refusals(exposing::<TakesAnAdder, i64>("e({ add: adder(1), })", "$add(1)"));
    assert_eq!(messages.len(), 1, "{messages:#?}");
    assert!(messages[0].contains("at `$add` of its input the function type"), "{messages:#?}");
}

#[test]
fn a_rust_fn_as_an_exposed_result_is_refused() {
    let messages = crossing_refusals(exposing::<(), AnAdder>("0", "adder(1)"));
    assert_eq!(messages.len(), 1, "{messages:#?}");
    assert!(messages[0].contains("at `result` of its result the function type"), "{messages:#?}");
}
