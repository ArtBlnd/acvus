//! Integration tests for larger script-mode programs.
//!
//! Each test loads a `.acvus` script via `include_str!` and executes it
//! with different contexts. Run with `--nocapture` to see printed results.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::{LenTerm, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: Vec<(&str, TypedValue)>) -> Context {
    entries
        .into_iter()
        .map(|(name, val)| (i.intern(name), val))
        .collect()
}

// =======================================================================
//  Algorithm: Collatz sequence
// =======================================================================

const COLLATZ: &str = include_str!("scripts/collatz.acvus");

#[tokio::test]
async fn collatz_start_6() {
    let i = Interner::new();
    let c = ctx(&i, vec![("start", typed(Ty::Int, Value::int(6)))]);
    let result = run_script_mode(&i, COLLATZ, c).await;
    eprintln!("collatz(6) max_val = {result:?}");
    // 6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
    assert_eq!(result.as_int(), 16);
}

#[tokio::test]
async fn collatz_start_27() {
    let i = Interner::new();
    let c = ctx(&i, vec![("start", typed(Ty::Int, Value::int(27)))]);
    let result = run_script_mode(&i, COLLATZ, c).await;
    eprintln!("collatz(27) max_val = {result:?}");
    // Famous case: reaches 9232 before falling back to 1
    assert_eq!(result.as_int(), 9232);
}

#[tokio::test]
async fn collatz_start_1() {
    let i = Interner::new();
    let c = ctx(&i, vec![("start", typed(Ty::Int, Value::int(1)))]);
    let result = run_script_mode(&i, COLLATZ, c).await;
    eprintln!("collatz(1) max_val = {result:?}");
    // Already at 1 - while body never executes
    assert_eq!(result.as_int(), 1);
}

// =======================================================================
//  Control flow: Grade classifier
// =======================================================================

// =======================================================================
//  Program: grade classifier (restored from the `for` cut in 69eac8d)
// =======================================================================

const GRADE_CLASSIFIER: &str = include_str!("scripts/grade_classifier.acvus");

fn student(i: &Interner, name: &str, score: i64) -> Value {
    Value::object(FxHashMap::from_iter([
        (i.intern("name"), Value::string(name)),
        (i.intern("score"), Value::int(score)),
    ]))
}

fn student_ty(i: &Interner) -> Ty {
    Ty::Object(FxHashMap::from_iter([
        (i.intern("name"), Ty::String),
        (i.intern("score"), Ty::Int),
    ]))
}

fn students(i: &Interner, items: Vec<Value>) -> TypedValue {
    let len = items.len();
    typed(
        Ty::Array(Box::new(student_ty(i)), LenTerm::Known(len)),
        Value::array(items),
    )
}

#[tokio::test]
async fn grade_classifier_mixed() {
    let i = Interner::new();
    let students = students(
        &i,
        vec![
            student(&i, "alice", 95),
            student(&i, "bob", 72),
            student(&i, "charlie", 45),
            student(&i, "diana", 98),
            student(&i, "eve", 55),
        ],
    );
    let c = ctx(&i, vec![("students", students)]);
    let result = run_script_mode(&i, GRADE_CLASSIFIER, c).await;
    // honor: alice, diana; pass: bob; fail: charlie, eve.
    // passing_total = 95 + 72 + 98 = 265; fail > 0 -> 265
    assert_eq!(result.as_int(), 265);
}

#[tokio::test]
async fn grade_classifier_all_passing() {
    let i = Interner::new();
    let students = students(
        &i,
        vec![
            student(&i, "alice", 95),
            student(&i, "bob", 80),
            student(&i, "charlie", 70),
        ],
    );
    let c = ctx(&i, vec![("students", students)]);
    let result = run_script_mode(&i, GRADE_CLASSIFIER, c).await;
    // passing_total = 245; fail == 0 -> 245 + best(95) = 340
    assert_eq!(result.as_int(), 340);
}
