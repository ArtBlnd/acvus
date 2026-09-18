//! Interpreter e2e tests for script-mode: let, for-loop, if-let, context writes.

use acvus_extern::Owned;
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

fn int(n: i64) -> TypedValue {
    typed(Ty::I64, Value::int(n))
}

fn assert_str(v: &Value, expected: &str) {
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    assert_eq!(unsafe { v.as_str() }, expected);
}

// =======================================================================
//  Let binding
// =======================================================================

#[tokio::test]
async fn let_simple_bind() {
    let i = Interner::new();
    let c = ctx(&i, vec![("x", int(10))]);
    let result = run_script(&i, "let y = @x + 1; y", c).await;
    assert_eq!(result.as_int(), 11);
}

#[tokio::test]
async fn let_multiple_binds() {
    let i = Interner::new();
    let c = ctx(&i, vec![("x", int(5))]);
    let result = run_script(&i, "let a = @x; let b = a + a; b", c).await;
    assert_eq!(result.as_int(), 10);
}

#[tokio::test]
async fn let_context_store_then_read() {
    let i = Interner::new();
    let c = ctx(&i, vec![("x", int(0))]);
    let result = run_script(&i, "@x = 42; @x", c).await;
    assert_eq!(result.as_int(), 42);
}

// =======================================================================
//  If-let (match-bind)
// =======================================================================

#[tokio::test]
async fn if_let_irrefutable() {
    let i = Interner::new();
    let c = ctx(&i, vec![("data", int(5)), ("out", int(0))]);
    let result = run_script(&i, "if let x = @data { @out = x * 2; }; @out", c).await;
    assert_eq!(result.as_int(), 10);
}

#[tokio::test]
async fn if_let_refutable_match() {
    let i = Interner::new();
    let c = ctx(&i, vec![("val", int(42)), ("out", int(0))]);
    let result = run_script(&i, "if let 42 = @val { @out = 1; }; @out", c).await;
    assert_eq!(result.as_int(), 1);
}

#[tokio::test]
async fn if_let_refutable_no_match() {
    let i = Interner::new();
    let c = ctx(&i, vec![("val", int(99)), ("out", int(0))]);
    let result = run_script(&i, "if let 42 = @val { @out = 1; }; @out", c).await;
    assert_eq!(result.as_int(), 0);
}

// =======================================================================
//  Iteration: `while let Some(x) = next(&mut it)` (RFC-0015)
//  (restored from the `for` tests cut in 69eac8d)
// =======================================================================

fn ints_ty(len: usize) -> Ty {
    Ty::Array(Box::new(Ty::I64), LenTerm::Known(len))
}

fn ints_value(xs: &[i64]) -> Value {
    Value::array(
        xs.iter()
            .map(|&x| Owned::from_value(Value::int(x)))
            .collect(),
    )
}

fn ints(xs: &[i64]) -> TypedValue {
    typed(ints_ty(xs.len()), ints_value(xs))
}

#[tokio::test]
async fn iter_sum() {
    let i = Interner::new();
    let c = ctx(&i, vec![("items", ints(&[1, 2, 3])), ("sum", int(0))]);
    let result = run_script_mode(
        &i,
        "let it = as_iter(&@items); while let Some(x) = next(&mut it) { @sum = @sum + *x; } @sum",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 6);
}

#[tokio::test]
async fn iter_count() {
    let i = Interner::new();
    let c = ctx(&i, vec![("items", ints(&[10, 20, 30])), ("count", int(0))]);
    let result = run_script_mode(
        &i,
        "let it = as_iter(&@items); while let Some(x) = next(&mut it) { @count = @count + 1; } @count",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 3);
}

#[tokio::test]
async fn iter_nested() {
    let i = Interner::new();
    let matrix = typed(
        Ty::Array(Box::new(ints_ty(2)), LenTerm::Known(2)),
        Value::array(vec![
            Owned::from_value(ints_value(&[1, 2])),
            Owned::from_value(ints_value(&[3, 4])),
        ]),
    );
    let c = ctx(&i, vec![("matrix", matrix), ("sum", int(0))]);
    let result = run_script_mode(
        &i,
        "let rows = as_iter(&@matrix); while let Some(row) = next(&mut rows) { let xs = as_iter(row); while let Some(x) = next(&mut xs) { @sum = @sum + *x; } } @sum",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 10);
}

#[tokio::test]
async fn iter_empty_list() {
    let i = Interner::new();
    let c = ctx(&i, vec![("items", ints(&[])), ("sum", int(99))]);
    let result = run_script_mode(
        &i,
        "let it = as_iter(&@items); while let Some(x) = next(&mut it) { @sum = @sum + *x; } @sum",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 99);
}

#[tokio::test]
async fn iter_sequential_loops() {
    let i = Interner::new();
    let c = ctx(
        &i,
        vec![
            ("a", ints(&[1, 2])),
            ("b", ints(&[10, 20])),
            ("sum", int(0)),
        ],
    );
    let result = run_script_mode(
        &i,
        "let ia = as_iter(&@a); while let Some(x) = next(&mut ia) { @sum = @sum + *x; } let ib = as_iter(&@b); while let Some(y) = next(&mut ib) { @sum = @sum + *y; } @sum",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 33);
}

#[tokio::test]
async fn iter_loop_with_conditional() {
    let i = Interner::new();
    let c = ctx(&i, vec![("items", ints(&[0, 1, 0, 2])), ("count", int(0))]);
    let result = run_script_mode(
        &i,
        "let it = as_iter(&@items); while let Some(x) = next(&mut it) { if *x == 0 { @count = @count + 1; }; } @count",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 2);
}

#[tokio::test]
async fn iter_accumulate_product() {
    let i = Interner::new();
    let c = ctx(
        &i,
        vec![
            ("items", ints(&[2, 3, 4])),
            ("sum", int(0)),
            ("product", int(1)),
        ],
    );
    let result = run_script_mode(
        &i,
        "let it = as_iter(&@items); while let Some(x) = next(&mut it) { @sum = @sum + *x; @product = @product * *x; } @sum + @product",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 33);
}

#[tokio::test]
async fn iter_field_then_loop() {
    let i = Interner::new();
    let items = i.intern("items");
    let data = typed(
        Ty::Object(FxHashMap::from_iter([(items, ints_ty(2))])),
        Value::object(FxHashMap::from_iter([(
            items,
            Owned::from_value(ints_value(&[10, 20])),
        )])),
    );
    let c = ctx(&i, vec![("data", data), ("sum", int(0))]);
    let result = run_script_mode(
        &i,
        "let it = as_iter(&@data.items); while let Some(x) = next(&mut it) { @sum = @sum + *x; } @sum",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 30);
}

#[tokio::test]
async fn iter_with_to_string() {
    let i = Interner::new();
    let c = ctx(
        &i,
        vec![
            ("items", ints(&[1, 2, 3])),
            ("out", typed(Ty::String, Value::string(""))),
        ],
    );
    let result = run_script_mode(
        &i,
        "let it = as_iter(&@items); while let Some(x) = next(&mut it) { @out = @out + to_string(x); } @out",
        c,
    )
    .await;
    assert_str(&result, "123");
}

/// A closure reads a context as it is when the closure runs, not as it
/// was when the closure was made: the store before the call that runs
/// it is visible, as it is to any call (RFC-0014).
#[tokio::test]
async fn closure_reads_context_at_call() {
    let i = Interner::new();
    let c = ctx(&i, vec![("items", ints(&[1, 2])), ("x", int(1))]);
    let result = run_script_mode(
        &i,
        "@x = 5; let f = |v| -> *v + @x; @x = 9; as_iter(&@items) | map(f) | fold(0, |a, b| -> a + b)",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 21);
}

#[tokio::test]
async fn a_factor_the_nested_loops_never_assign_still_reaches_the_inner_body() {
    let i = Interner::new();
    let c = ctx(&i, vec![("n", int(3))]);
    let result = run_script(
        &i,
        "let d = @n; let k = d + 3; let total = 0; let t = 0; \
         while t < d { let i = 0; while i < d { total = total + k; i = i + 1; } t = t + 1; } \
         total",
        c,
    )
    .await;
    assert_eq!(result.as_int(), 54);
}
