//! Interpreter e2e tests for script-mode: let, for-loop, if-let, context writes.

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
//  Let binding
// =======================================================================

#[tokio::test]
async fn let_simple_bind() {
    let i = Interner::new();
    let c = ctx(&i, &[("x", Value::Int(10))]);
    let result = run_script(&i, "y = @x + 1; y", c).await;
    assert_eq!(result, Value::Int(11));
}

#[tokio::test]
async fn let_multiple_binds() {
    let i = Interner::new();
    let c = ctx(&i, &[("x", Value::Int(5))]);
    let result = run_script(&i, "a = @x; b = a + a; b", c).await;
    assert_eq!(result, Value::Int(10));
}

#[tokio::test]
async fn let_context_store_then_read() {
    let i = Interner::new();
    let c = ctx(&i, &[("x", Value::Int(0))]);
    let result = run_script(&i, "@x = 42; @x", c).await;
    assert_eq!(result, Value::Int(42));
}

// =======================================================================
//  If-let (match-bind)
// =======================================================================

#[tokio::test]
async fn if_let_irrefutable() {
    let i = Interner::new();
    let c = ctx(&i, &[("data", Value::Int(5)), ("out", Value::Int(0))]);
    let result = run_script(&i, "x = @data { @out = x * 2; }; @out", c).await;
    assert_eq!(result, Value::Int(10));
}

#[tokio::test]
async fn if_let_refutable_match() {
    let i = Interner::new();
    let c = ctx(&i, &[("val", Value::Int(42)), ("out", Value::Int(0))]);
    let result = run_script(&i, "42 = @val { @out = 1; }; @out", c).await;
    assert_eq!(result, Value::Int(1));
}

#[tokio::test]
async fn if_let_refutable_no_match() {
    let i = Interner::new();
    let c = ctx(&i, &[("val", Value::Int(99)), ("out", Value::Int(0))]);
    let result = run_script(&i, "42 = @val { @out = 1; }; @out", c).await;
    assert_eq!(result, Value::Int(0)); // body not executed
}
