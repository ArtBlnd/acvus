use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

// ── Text only ────────────────────────────────────────────────────

#[tokio::test]
async fn text_only() {
    assert_eq!(run_simple("hello world").await, "hello world");
}

#[tokio::test]
async fn text_with_spaces() {
    assert_eq!(run_simple("  spaces  ").await, "  spaces  ");
}

// ── Inline expressions ──────────────────────────────────────────

#[tokio::test]
async fn inline_string() {
    assert_eq!(run_simple(r#"{{ "hello" }}"#).await, "hello");
}

#[tokio::test]
async fn inline_int_to_string() {
    // Template concat coerces via to_string? Or just raw?
    // For now, just test string expressions.
    assert_eq!(run_simple(r#"{{ "a" + "b" }}"#).await, "ab");
}

// ── Context read ─────────────────────────────────────────────────

#[tokio::test]
async fn context_read_string() {
    let i = Interner::new();
    let ctx = string_context(&i, "name", "alice");
    assert_eq!(run(&i, "{{ @name }}", ctx).await, "alice");
}

#[tokio::test]
async fn context_read_int() {
    let i = Interner::new();
    let ctx = int_context(&i, "x", 42);
    let result = run(&i, "{{ @x | to_string }}", ctx).await;
    assert_eq!(result, "42");
}

// ── String concat ────────────────────────────────────────────────

#[tokio::test]
async fn string_concat() {
    assert_eq!(
        run_simple(r#"{{ "hello" + " " + "world" }}"#).await,
        "hello world"
    );
}

// ── Template text + expression ───────────────────────────────────

#[tokio::test]
async fn text_and_expr() {
    let i = Interner::new();
    let ctx = string_context(&i, "name", "alice");
    assert_eq!(run(&i, "Hello, {{ @name }}!", ctx).await, "Hello, alice!");
}
