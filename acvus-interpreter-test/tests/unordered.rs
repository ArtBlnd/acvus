//! RFC-0075 rule 2: `unordered` takes a `map` stage and joins its calls.
//!
//! The first pull draws the whole input, calls the closure on every element
//! at once, and keeps the results; each pull hands out the next result in
//! input order. The closure is not asked to commute, and a pipeline that
//! cannot suspend is refused (RFC-0011 rule 5).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, TokioExecutor};
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// How long a probe stays in flight: a window in which the other calls of
/// the join can start, so the counter can see them together.
const PROBE_HOLD: Duration = Duration::from_millis(20);

/// The step between the holds of `probe_desc`: element `x` of five holds
/// `(5 - x)` steps, so the last element completes first.
const DESC_STEP: Duration = Duration::from_millis(15);

/// The calls a probe saw: how many there were, how many were in flight at
/// once, and the order in which they completed.
struct Probe {
    calls: AtomicUsize,
    in_flight: AtomicUsize,
    max: AtomicUsize,
    completed: Mutex<Vec<i64>>,
}

impl Probe {
    fn new() -> Arc<Self> {
        Arc::new(Probe {
            calls: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
            completed: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn max(&self) -> usize {
        self.max.load(Ordering::SeqCst)
    }

    fn completed(&self) -> Vec<i64> {
        self.completed.lock().expect("the completion log").clone()
    }

    async fn hold(&self, x: i64, hold: Duration) -> i64 {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(hold).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.completed.lock().expect("the completion log").push(x);
        x
    }
}

/// Declared as `http::get` is: idempotent, and not commutative.
#[extern_fn(effect = idempotent)]
async fn probe(#[state] p: &Arc<Probe>, x: i64) -> i64 {
    p.hold(x, PROBE_HOLD).await
}

#[extern_fn(effect = idempotent)]
async fn probe_desc(#[state] p: &Arc<Probe>, x: i64) -> i64 {
    let steps = u32::try_from(5 - x).expect("probe_desc takes an element of range(0, 5)");
    p.hold(x, DESC_STEP * steps).await
}

#[extern_fn(effect = opaque)]
async fn probe_opaque(#[state] p: &Arc<Probe>, x: i64) -> i64 {
    p.hold(x, PROBE_HOLD).await
}

#[extern_fn(heavy, effect = pure)]
fn tripled(x: i64) -> i64 {
    x * 3
}

fn registries(p: &Arc<Probe>) -> Vec<Registry<AcvusRuntime>> {
    let p = Arc::clone(p);
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "unordered",
        fns: [
            probe(Arc::clone(&p)),
            probe_desc(Arc::clone(&p)),
            probe_opaque(Arc::clone(&p)),
            tripled,
        ],
    });
    regs
}

async fn run_on_tokio(source: &str, probe: &Arc<Probe>) -> i64 {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    run_parsed_on(
        &i,
        ast,
        Context::default(),
        registries(probe),
        Ty::I64,
        |_| {},
        Arc::new(TokioExecutor),
    )
    .await
    .value
    .as_int()
}

/// The messages `source` is refused with; a source that compiles fails the
/// test.
fn refused(source: &str) -> Vec<String> {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    match check_source(
        &i,
        ast,
        &FxHashMap::default(),
        registries(&Probe::new()),
        Ty::I64,
        Opt::Full,
        |_| {},
    ) {
        Ok(_) => panic!("`{source}` compiled; it is refused"),
        Err(refusal) => refusal.messages,
    }
}

/// Folding the digits in order reads the results' order: `0, 1, 2, 3, 4`
/// folds to 1234 and any other order to another number.
const IN_ORDER: i64 = 1234;

#[tokio::test]
async fn the_calls_of_the_map_are_in_flight_together() {
    let p = Probe::new();
    let v = run_on_tokio(
        "range(0, 5) | map(|x| -> probe(x)) | unordered() | fold(0, |a, x| -> a * 10 + x)",
        &p,
    )
    .await;
    assert_eq!(v, IN_ORDER, "the results come out in input order");
    assert_eq!(p.calls(), 5);
    assert_eq!(p.max(), 5, "every call is in flight at once");
}

#[tokio::test]
async fn results_keep_input_order_when_completion_is_reversed() {
    let p = Probe::new();
    let v = run_on_tokio(
        "range(0, 5) | map(|x| -> probe_desc(x)) | unordered() | fold(0, |a, x| -> a * 10 + x)",
        &p,
    )
    .await;
    assert_eq!(
        p.completed(),
        vec![4, 3, 2, 1, 0],
        "the calls completed last first"
    );
    assert_eq!(v, IN_ORDER, "the results come out in input order");
}

#[tokio::test]
async fn the_first_pull_calls_the_closure_on_the_whole_input() {
    let p = Probe::new();
    let v = run_on_tokio(
        "range(0, 5) | map(|x| -> probe(x)) | unordered() | take(2) | count()",
        &p,
    )
    .await;
    assert_eq!(v, 2, "take hands out two results");
    assert_eq!(p.calls(), 5, "the closure ran on every element");
}

#[tokio::test]
async fn a_closure_calling_an_opaque_extern_is_admitted() {
    let p = Probe::new();
    let v = run_on_tokio(
        "range(0, 5) | map(|x| -> probe_opaque(x) + 1) | unordered() | fold(0, |a, x| -> a * 10 + x)",
        &p,
    )
    .await;
    assert_eq!(v, 12345);
    assert_eq!(p.max(), 5, "the opaque calls are joined as well");
}

#[tokio::test]
async fn a_closure_calling_a_heavy_extern_is_admitted() {
    let p = Probe::new();
    let v = run_on_tokio(
        "range(1, 4) | map(|x| -> tripled(x)) | unordered() | fold(0, |a, x| -> a * 100 + x)",
        &p,
    )
    .await;
    assert_eq!(v, 30609);
}

#[tokio::test]
async fn an_empty_input_makes_no_calls() {
    let p = Probe::new();
    let v = run_on_tokio(
        "range(0, 0) | map(|x| -> probe(x)) | unordered() | count()",
        &p,
    )
    .await;
    assert_eq!(v, 0);
    assert_eq!(p.calls(), 0);
}

#[test]
fn a_pipeline_that_cannot_suspend_is_refused() {
    let messages = refused("range(0, 5) | map(|x| -> x + 1) | unordered() | count()");
    assert!(
        messages.iter().any(|m| m.contains(
            "a function whose task is Sync was given where one that suspends is required"
        )),
        "refused for its task: {messages:?}"
    );
}

#[test]
fn a_stage_other_than_map_is_refused() {
    let messages = refused(
        "range(0, 5) | map(|x| -> probe(x)) | filter(|x| -> *x > 0) | unordered() | count()",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("type mismatch: expected Map<") && m.contains("got Filter<")),
        "refused for its stage: {messages:?}"
    );
}
