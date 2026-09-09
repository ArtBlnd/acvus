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

// =======================================================================
//  Algorithm: Collatz sequence
// =======================================================================

const COLLATZ: &str = include_str!("scripts/collatz.acvus");

#[tokio::test]
async fn collatz_start_6() {
    let i = Interner::new();
    let c = ctx(&i, &[("start", Value::Int(6))]);
    let result = run_script_mode(&i, COLLATZ, c).await;
    eprintln!("collatz(6) max_val = {result:?}");
    // 6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
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
    // Already at 1 - while body never executes
    assert_eq!(result, Value::Int(1));
}

// =======================================================================
//  Control flow: Grade classifier
// =======================================================================
