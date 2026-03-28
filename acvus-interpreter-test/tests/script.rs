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

// ═══════════════════════════════════════════════════════════════════════
//  Let binding
// ═══════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════
//  For loop (iteration)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn for_sum() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "items",
                Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            ),
            ("sum", Value::Int(0)),
        ],
    );
    let result = run_script(&i, "x in @items { @sum = @sum + x; }; @sum", c).await;
    assert_eq!(result, Value::Int(6));
}

#[tokio::test]
async fn for_count() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "items",
                Value::list(vec![Value::Int(10), Value::Int(20), Value::Int(30)]),
            ),
            ("count", Value::Int(0)),
        ],
    );
    let result = run_script(&i, "x in @items { @count = @count + 1; }; @count", c).await;
    assert_eq!(result, Value::Int(3));
}

#[tokio::test]
async fn for_nested() {
    let i = Interner::new();
    let inner1 = Value::list(vec![Value::Int(1), Value::Int(2)]);
    let inner2 = Value::list(vec![Value::Int(3), Value::Int(4)]);
    let c = ctx(
        &i,
        &[
            ("matrix", Value::list(vec![inner1, inner2])),
            ("sum", Value::Int(0)),
        ],
    );
    let result = run_script(
        &i,
        "row in @matrix { x in row { @sum = @sum + x; }; }; @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(10));
}

#[tokio::test]
async fn for_empty_list() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", Value::list(vec![])), ("sum", Value::Int(99))],
    );
    let result = run_script(&i, "x in @items { @sum = @sum + x; }; @sum", c).await;
    assert_eq!(result, Value::Int(99)); // unchanged
}

#[tokio::test]
async fn for_sequential_loops() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("a", Value::list(vec![Value::Int(1), Value::Int(2)])),
            ("b", Value::list(vec![Value::Int(10), Value::Int(20)])),
            ("sum", Value::Int(0)),
        ],
    );
    let result = run_script(
        &i,
        "x in @a { @sum = @sum + x; }; y in @b { @sum = @sum + y; }; @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(33));
}

// ═══════════════════════════════════════════════════════════════════════
//  If-let (match-bind)
// ═══════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════
//  Combined
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn combined_loop_with_conditional() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "items",
                Value::list(vec![
                    Value::Int(0),
                    Value::Int(1),
                    Value::Int(0),
                    Value::Int(2),
                ]),
            ),
            ("count", Value::Int(0)),
        ],
    );
    // Count non-zero elements
    let result = run_script(
        &i,
        "x in @items { 0 = x { @count = @count + 1; }; }; @count",
        c,
    )
    .await;
    // 0 = x matches when x == 0, so body executes for 0s → count = 2
    assert_eq!(result, Value::Int(2));
}

#[tokio::test]
async fn combined_accumulate_product() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "items",
                Value::list(vec![Value::Int(2), Value::Int(3), Value::Int(4)]),
            ),
            ("sum", Value::Int(0)),
            ("product", Value::Int(1)),
        ],
    );
    let result = run_script(
        &i,
        "x in @items { @sum = @sum + x; @product = @product * x; }; @sum + @product",
        c,
    )
    .await;
    // sum = 2+3+4 = 9, product = 2*3*4 = 24, total = 33
    assert_eq!(result, Value::Int(33));
}

#[tokio::test]
async fn combined_bind_then_loop() {
    let i = Interner::new();
    let obj = Value::object(FxHashMap::from_iter([(
        i.intern("items"),
        Value::list(vec![Value::Int(10), Value::Int(20)]),
    )]));
    let c = ctx(&i, &[("data", obj), ("sum", Value::Int(0))]);
    let result = run_script(
        &i,
        "{ items, } = @data { x in items { @sum = @sum + x; }; }; @sum",
        c,
    )
    .await;
    assert_eq!(result, Value::Int(30));
}

#[tokio::test]
async fn for_with_builtin_to_string() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "items",
                Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            ),
            ("out", Value::string("")),
        ],
    );
    let result = run_script(
        &i,
        r#"x in @items { @out = @out + to_string(x); }; @out"#,
        c,
    )
    .await;
    assert_eq!(result, Value::String("123".into()));
}
