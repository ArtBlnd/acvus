//! `return e` leaves the enclosing body -- main or a closure -- with `e`,
//! from any depth: the value each shape produces at both optimization
//! levels, what the machine runs a loop holding one as, and what the return
//! edge releases.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::listing::ops_of_anywhere;
use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::listing::script_listing;
use acvus_interpreter_test::*;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// The value both levels produce, rendered as `acvus run` prints it. A shape
/// whose two levels disagree fails here before its value is read.
fn value(source: &str) -> String {
    let full = corpus::attempt(source, Opt::Full, Stage::Run);
    let none = corpus::attempt(source, Opt::None, Stage::Run);
    assert_eq!(full, none, "the two levels disagree on `{source}`");
    match full {
        Outcome::Value(v) => v,
        other => panic!("`{source}` produced no value: {other:?}"),
    }
}

// -- What each shape leaves with ---------------------------------------

#[test]
fn a_return_before_the_tail_leaves_with_its_value() {
    assert_eq!(value("let x = 1; return x + 1; 99"), "2");
}

#[test]
fn a_return_inside_an_if_leaves_and_the_path_around_it_reaches_the_tail() {
    assert_eq!(value("let n = 5; if n > 3 { return 10; }; 20"), "10");
    assert_eq!(value("let n = 1; if n > 3 { return 10; }; 20"), "20");
}

#[test]
fn a_return_inside_a_for_leaves_at_the_element_it_stands_on() {
    assert_eq!(
        value("let acc = 0; for i in 0..10 { acc = acc + i; if acc > 5 { return acc; }; } acc"),
        "6"
    );
}

#[test]
fn a_return_inside_a_while_leaves_the_body_and_not_the_loop() {
    assert_eq!(
        value("let i = 0; while i < 10 { i = i + 1; if i == 3 { return i * 100; }; } 0 - 1"),
        "300"
    );
}

#[test]
fn a_return_inside_a_while_let_leaves_the_body() {
    assert_eq!(
        value(
            "let d = deque(); d.push_back(1); d.push_back(2); let acc = 0; \
             while let Some(x) = d.pop_front() { acc = acc + x; if acc > 0 { return acc + 50; }; } acc"
        ),
        "51"
    );
}

#[test]
fn a_return_inside_a_nested_loop_leaves_both() {
    assert_eq!(
        value(
            "let acc = 0; for i in 0..3 { for j in 0..3 { acc = acc + 1; \
             if i + j == 2 { return acc; }; } } acc"
        ),
        "3"
    );
}

#[test]
fn a_return_inside_a_borrowing_traversal_leaves_with_what_it_read() {
    assert_eq!(
        value(
            "let v = vec([1, 2, 3]); let s = \"abc\".to_string(); \
             for x in &v { if *x == 2 { return len(&s) as i64 + *x; }; } 0"
        ),
        "5"
    );
}

/// The closure's own return, which is the `return_ty` the checker swaps in
/// for a lambda body: `f` leaves, the body that called it goes on.
#[test]
fn a_return_inside_a_closure_leaves_the_closure_and_the_caller_continues() {
    assert_eq!(
        value("let f = |n| -> { if n > 2 { return 100; }; n }; f(5) + f(1)"),
        "101"
    );
}

#[test]
fn a_return_of_a_result_stands_beside_a_question_mark_in_one_body() {
    assert_eq!(
        value("let r = Ok(7); let v = r?; if v > 5 { return Ok(v * 2); }; Ok(v + 1)"),
        "{\"Ok\":14}"
    );
    assert_eq!(
        value("let r = Ok(2); let v = r?; if v > 5 { return Ok(v * 2); }; Ok(v + 1)"),
        "{\"Ok\":3}"
    );
    assert_eq!(
        value("let r = Err(\"bad\".to_string()); let v = r?; Ok(v + 1)"),
        "{\"Err\":\"bad\"}"
    );
}

#[test]
fn a_return_in_one_branch_leaves_the_other_branch_s_value() {
    assert_eq!(value("let x = if true { return 1 } else { 2 }; x"), "1");
    assert_eq!(value("let x = if false { return 1 } else { 2 }; x"), "2");
}

#[tokio::test]
#[should_panic(expected = "[main] type mismatch: expected i64, got String")]
async fn a_return_of_another_type_than_the_host_declared_is_refused() {
    let i = Interner::new();
    let _ = run_script_mode(
        &i,
        "let x = 1; return \"no\".to_string(); x",
        Context::default(),
        Ty::I64,
    )
    .await;
}

// -- What the machine runs a loop holding a `return` as -----------------

const RETURN_IN_A_RANGE: &str =
    "let acc = 0; for i in 0..10 { acc = acc + i; if acc > 5 { return acc; }; } acc";

/// RFC-0057 rule 8 names `return` as the third exit edge a region's body
/// may hold, beside `break` and `continue`; change that decision and this
/// test moves with it.
#[test]
fn a_loop_a_return_leaves_is_one_region_that_hands_the_return_up() {
    let i = Interner::new();
    let blocks = script_listing(&i, RETURN_IN_A_RANGE, Context::default(), Ty::I64);
    let regions: Vec<String> = blocks
        .iter()
        .flat_map(|block| block.regions.iter())
        .map(|region| region.name.clone())
        .collect();
    assert_eq!(
        regions,
        ["For<Range<i64>, Escapes>"],
        "the loop is one region"
    );

    let ops = ops_of_anywhere(&blocks);
    assert!(
        ops.iter().any(|op| op.starts_with("Escape<")),
        "the `return` is a branch inside the body: {ops:?}"
    );
    assert!(
        !ops.iter().any(|op| op.starts_with("ForStart<")),
        "no preheader lays a counter the region keeps in a local: {ops:?}"
    );
    assert_eq!(value(RETURN_IN_A_RANGE), "6");
}

// -- What the return edge releases -------------------------------------

static RELEASES: AtomicUsize = AtomicUsize::new(0);

/// One counter, and the harness runs these tests on parallel threads.
static TRACKED_SCRIPTS: Mutex<()> = Mutex::new(());

struct Measured {
    at_start: usize,
    _scripts: MutexGuard<'static, ()>,
}

impl Measured {
    fn start() -> Self {
        let scripts = TRACKED_SCRIPTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Self {
            at_start: RELEASES.load(Ordering::SeqCst),
            _scripts: scripts,
        }
    }

    fn count(&self) -> usize {
        RELEASES.load(Ordering::SeqCst) - self.at_start
    }
}

struct Counted;

impl Drop for Counted {
    fn drop(&mut self) {
        RELEASES.fetch_add(1, Ordering::SeqCst);
    }
}

/// An extension type over a `Vec`, so the value crosses as a `Large` and its
/// release is what the counter sees.
#[derive(ExternType)]
#[repr(transparent)]
struct Tracked(Vec<Counted>);

#[extern_fn(effect = pure)]
fn tracked(n: i64) -> Tracked {
    Tracked((0..n).map(|_| Counted).collect())
}

#[extern_fn(effect = pure)]
fn rank(t: &Tracked) -> i64 {
    t.0.len() as i64
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        types: [Tracked],
        fns: [tracked, rank],
    });
    regs
}

async fn tracked_int(source: &str) -> i64 {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), regs(), Ty::I64)
        .await
        .value
        .as_int()
}

const OWNERS_ACROSS_A_RETURN: &str = "let v = vec([tracked(1), tracked(1), tracked(1)]); let s = tracked(1); \
     let n = rank(&v[0]) + rank(&s); if n == 2 { return n + 10; }; 0";

const OWNERS_ACROSS_THE_TAIL: &str = "let v = vec([tracked(1), tracked(1), tracked(1)]); let s = tracked(1); \
     let n = rank(&v[0]) + rank(&s); n + 10";

/// Four owners live where the `return` stands and four are released:
/// `optimize::drop_insertion` owns the return edge, as it owns the tail's,
/// and the lowering of `return` adds no scope-stack drop of its own.
#[tokio::test]
async fn a_vec_of_owners_and_an_owner_live_across_a_return_and_are_released_once_each() {
    let measured = Measured::start();
    assert_eq!(tracked_int(OWNERS_ACROSS_A_RETURN).await, 12);
    assert_eq!(measured.count(), 4);
}

/// The counterfactual, one variable apart: the same owners reaching the same
/// value through the tail rather than through a `return`.
#[tokio::test]
async fn the_same_owners_reaching_the_tail_are_released_once_each() {
    let measured = Measured::start();
    assert_eq!(tracked_int(OWNERS_ACROSS_THE_TAIL).await, 12);
    assert_eq!(measured.count(), 4);
}
