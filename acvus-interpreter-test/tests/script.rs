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

// =======================================================================
//  Iteration: `while let Some(x) = next(&mut it)` (RFC-0015)
//  (restored from the `for` tests cut in 69eac8d)
// =======================================================================

fn ints(xs: &[i64]) -> Value {
    Value::array(xs.iter().map(|&x| Value::Int(x)).collect())
}

#[tokio::test]
async fn iter_sum() {
    let i = Interner::new();
    let c = ctx(&i, &[("items", ints(&[1, 2, 3])), ("sum", Value::Int(0))]);
    let result = run_script_mode(
        &i,
        "let it = iter(@items); while let Some(x) = next(&mut it) { @sum = @sum + x; } @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(6));
}

#[tokio::test]
async fn iter_count() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", ints(&[10, 20, 30])), ("count", Value::Int(0))],
    );
    let result = run_script_mode(
        &i,
        "let it = iter(@items); while let Some(x) = next(&mut it) { @count = @count + 1; } @count",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(3));
}

#[tokio::test]
async fn iter_nested() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("matrix", Value::array(vec![ints(&[1, 2]), ints(&[3, 4])])),
            ("sum", Value::Int(0)),
        ],
    );
    let result = run_script_mode(
        &i,
        "let rows = iter(@matrix); while let Some(row) = next(&mut rows) { let xs = iter(row); while let Some(x) = next(&mut xs) { @sum = @sum + x; } } @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(10));
}

#[tokio::test]
async fn iter_empty_list() {
    let i = Interner::new();
    let c = ctx(&i, &[("items", ints(&[])), ("sum", Value::Int(99))]);
    let result = run_script_mode(
        &i,
        "let it = iter(@items); while let Some(x) = next(&mut it) { @sum = @sum + x; } @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(99));
}

#[tokio::test]
async fn iter_sequential_loops() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("a", ints(&[1, 2])),
            ("b", ints(&[10, 20])),
            ("sum", Value::Int(0)),
        ],
    );
    let result = run_script_mode(
        &i,
        "let ia = iter(@a); while let Some(x) = next(&mut ia) { @sum = @sum + x; } let ib = iter(@b); while let Some(y) = next(&mut ib) { @sum = @sum + y; } @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(33));
}

#[tokio::test]
async fn iter_loop_with_conditional() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", ints(&[0, 1, 0, 2])), ("count", Value::Int(0))],
    );
    let result = run_script_mode(
        &i,
        "let it = iter(@items); while let Some(x) = next(&mut it) { if x == 0 { @count = @count + 1; }; } @count",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(2));
}

#[tokio::test]
async fn iter_accumulate_product() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("items", ints(&[2, 3, 4])),
            ("sum", Value::Int(0)),
            ("product", Value::Int(1)),
        ],
    );
    let result = run_script_mode(
        &i,
        "let it = iter(@items); while let Some(x) = next(&mut it) { @sum = @sum + x; @product = @product * x; } @sum + @product",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(33));
}

#[tokio::test]
async fn iter_field_then_loop() {
    let i = Interner::new();
    let obj = Value::object(FxHashMap::from_iter([(i.intern("items"), ints(&[10, 20]))]));
    let c = ctx(&i, &[("data", obj), ("sum", Value::Int(0))]);
    let result = run_script_mode(
        &i,
        "let it = iter(@data.items); while let Some(x) = next(&mut it) { @sum = @sum + x; } @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(30));
}

#[tokio::test]
async fn iter_with_to_string() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", ints(&[1, 2, 3])), ("out", Value::string(""))],
    );
    let result = run_script_mode(
        &i,
        "let it = iter(@items); while let Some(x) = next(&mut it) { @out = @out + to_string(x); } @out",
        c,
    )
    .await;
    assert_eq!(result, Value::string("123"));
}
