//! A closure made inside a closure body: three levels of `map`, and a
//! closure that leaves the body that made it. Each level's `MakeClosure`
//! runs while the level above is already executing as a closure, so the
//! closure bodies a running `FnValue` can reach are what these measure.

use std::time::Duration;

use acvus_interpreter::Value;
use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::*;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;

const LIMIT: Duration = Duration::from_secs(30);

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default(), ret).await
}

fn assert_close(v: &Value, expected: f64) {
    let actual = v.as_float();
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

#[tokio::test]
async fn three_nested_maps_sum_to_the_product_of_the_levels() {
    let v = run(
        "range(1, 4) \
         | map(|a| -> range(1, 4) \
             | map(|b| -> range(1, 4) | map(|c| -> c as f64) | sum) \
             | sum) \
         | sum",
        Ty::Float,
    )
    .await;
    assert_close(&v, 54.0);
}

#[tokio::test]
async fn a_word_reaches_the_innermost_of_three_closures_as_a_copy() {
    let v = run(
        "let k = 2.0; \
         range(0, 2) \
         | map(|a| -> range(0, 2) \
             | map(|b| -> range(0, 2) | map(|c| -> k) | sum) \
             | sum) \
         | sum",
        Ty::Float,
    )
    .await;
    assert_close(&v, 16.0);
}

/// A closure called in a handler's window calls a driving extern itself, so
/// the inner handler's window is the one above the closure's frame (RFC-0050
/// rule 6, RFC-0052 rule 7).
#[tokio::test]
async fn a_closure_in_the_window_drives_an_extern_of_its_own() {
    let v = run(
        "range(0, 4) | map(|x| -> range(0, x) | sum) | sum",
        Ty::Int(IntTy::I64),
    )
    .await;
    assert_eq!(v.as_int(), 4);
}

#[tokio::test]
async fn a_capture_crosses_two_windows_into_a_closure_a_handler_calls() {
    let v = run(
        "range(0, 3) | map(|x| -> range(0, 3) | map(|y| -> x * y) | sum) | sum",
        Ty::Int(IntTy::I64),
    )
    .await;
    assert_eq!(v.as_int(), 9);
}

/// Four windows stacked: the innermost callee does not fit the cells left
/// above the third, so it roots a frame of its own (`machine::run_rooted`).
#[tokio::test]
async fn a_pipeline_deeper_than_the_window_roots_a_frame_and_agrees() {
    let v = run(
        "range(1, 3) \
         | map(|a| -> range(1, 3) \
             | map(|b| -> range(1, 3) \
                 | map(|c| -> range(1, 3) | map(|d| -> d) | sum) \
                 | sum) \
             | sum) \
         | sum",
        Ty::Int(IntTy::I64),
    )
    .await;
    assert_eq!(v.as_int(), 24);
}

#[tokio::test]
async fn a_closure_returned_from_a_closure_is_called_at_the_top_level() {
    let v = run(
        "let mk = |s| -> |x| -> x * 3.0; let triple = mk(1.0); triple(4.0)",
        Ty::Float,
    )
    .await;
    assert_close(&v, 12.0);
}

/// A run that crashes the process is an outcome here, not the end of the
/// test binary, so each program runs in a process of its own.
fn outcome(source: &str, opt: Opt) -> Outcome {
    acvus_interpreter_test::attempt_within!(source, opt, Stage::Run, LIMIT)
        .unwrap_or_else(|lapse| panic!("at {opt:?}, {lapse:?}: {source}"))
}

fn runs_to(source: &str, value: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Value(got) => assert_eq!(got, value, "at {opt:?}: {source}"),
            other => panic!("at {opt:?}, expected {value}, got {other:?}: {source}"),
        }
    }
}

#[test]
fn corpus_child() {
    corpus::child();
}

/// A lambda that makes and calls a lambda of its own, called where the
/// caller's body already has blocks: splicing it there keeps the inner
/// `MakeClosure` naming the body the module holds, which is not one of the
/// caller's blocks.
const TWO_DEEP: &str = "let f = | | -> { let g = | | -> 2; g() };";
const THREE_DEEP: &str =
    "let n = 3; let f = | | -> { let g = | | -> { let h = | | -> n; h() }; g() };";

#[test]
fn a_lambda_making_a_lambda_runs_under_an_if() {
    runs_to(&format!("{TWO_DEEP} if true {{ f() }} else {{ 0 }}"), "2");
    runs_to(&format!("{THREE_DEEP} if true {{ f() }} else {{ 0 }}"), "3");
}

#[test]
fn a_lambda_making_a_lambda_runs_under_a_match() {
    runs_to(&format!("{TWO_DEEP} match f() {{ 2 => 7, _ => 0, }}"), "7");
    runs_to(
        &format!("{THREE_DEEP} match f() {{ 3 => 7, _ => 0, }}"),
        "7",
    );
}

#[test]
fn a_lambda_making_a_lambda_runs_in_a_loop() {
    runs_to(
        &format!("{TWO_DEEP} let s = 0; for i in 0..3 {{ s = s + f(); }} s"),
        "6",
    );
    runs_to(
        &format!("{THREE_DEEP} let s = 0; for i in 0..3 {{ s = s + f(); }} s"),
        "9",
    );
}
