//! Integration tests for larger script-mode programs.
//!
//! Each test loads a `.acvus` script via `include_str!` and executes it
//! with different contexts. Run with `--nocapture` to see printed results.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Value)]) -> FxHashMap<acvus_utils::Astr, Value> {
    entries
        .iter()
        .map(|(name, val)| (i.intern(name), val.clone()))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  Algorithm: Collatz sequence
// ═══════════════════════════════════════════════════════════════════════

const COLLATZ: &str = include_str!("scripts/collatz.acvus");

#[tokio::test]
async fn collatz_start_6() {
    let i = Interner::new();
    let c = ctx(&i, &[("start", Value::Int(6))]);
    let result = run_script_mode(&i, COLLATZ, c).await;
    eprintln!("collatz(6) max_val = {result:?}");
    // 6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1
    assert_eq!(result, Value::Int(16));
}

#[tokio::test]
async fn collatz_start_27() {
    let i = Interner::new();
    let c = ctx(&i, &[("start", Value::Int(27))]);
    let result = run_script_mode(&i, COLLATZ, c).await;
    eprintln!("collatz(27) max_val = {result:?}");
    // Famous case: reaches 9232 before falling back to 1
    assert_eq!(result, Value::Int(9232));
}

#[tokio::test]
async fn collatz_start_1() {
    let i = Interner::new();
    let c = ctx(&i, &[("start", Value::Int(1))]);
    let result = run_script_mode(&i, COLLATZ, c).await;
    eprintln!("collatz(1) max_val = {result:?}");
    // Already at 1 — while body never executes
    assert_eq!(result, Value::Int(1));
}

// ═══════════════════════════════════════════════════════════════════════
//  Control flow: Grade classifier
// ═══════════════════════════════════════════════════════════════════════

const GRADE_CLASSIFIER: &str = include_str!("scripts/grade_classifier.acvus");

fn student(i: &Interner, name: &str, score: i64) -> Value {
    Value::object(FxHashMap::from_iter([
        (i.intern("name"), Value::string(name)),
        (i.intern("score"), Value::Int(score)),
    ]))
}

#[tokio::test]
async fn grade_classifier_mixed() {
    let i = Interner::new();
    let students = Value::list(vec![
        student(&i, "alice", 95),
        student(&i, "bob", 72),
        student(&i, "charlie", 45),
        student(&i, "diana", 98),
        student(&i, "eve", 55),
    ]);
    let c = ctx(&i, &[("students", students)]);
    let result = run_script_mode(&i, GRADE_CLASSIFIER, c).await;
    eprintln!("grade_classifier(mixed) = {result:?}");
    // honor: alice(95), diana(98)  → 2
    // pass:  bob(72)               → 1
    // fail:  charlie(45), eve(55)  → 2
    // passing_total: 95 + 72 + 98 = 265
    // fail > 0 → result = 265
    assert_eq!(result, Value::Int(265));
}

#[tokio::test]
async fn grade_classifier_all_passing() {
    let i = Interner::new();
    let students = Value::list(vec![
        student(&i, "alice", 95),
        student(&i, "bob", 80),
        student(&i, "charlie", 70),
    ]);
    let c = ctx(&i, &[("students", students)]);
    let result = run_script_mode(&i, GRADE_CLASSIFIER, c).await;
    eprintln!("grade_classifier(all_passing) = {result:?}");
    // honor: alice(95)             → 1
    // pass:  bob(80), charlie(70)  → 2
    // fail:  0
    // best: 95
    // passing_total: 95 + 80 + 70 = 245
    // fail == 0 → result = 245 + 95 = 340
    assert_eq!(result, Value::Int(340));
}
