use std::collections::{BTreeMap, HashMap};

use acvus_mir::ty::Ty;
use acvus_mir_test::*;

// ── Text & literals ──────────────────────────────────────────────

#[test]
fn text_only() {
    let ir = compile_simple("hello world").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn string_emit() {
    let ir = compile_simple(r#"{{ "hello" }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn string_concat() {
    let ir = compile_simple(r#"{{ "hello" + " " + "world" }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn mixed_text_and_expr() {
    let storage = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir("Hello, {{ $name }}!", storage, HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Storage ──────────────────────────────────────────────────────

#[test]
fn storage_read() {
    let storage = HashMap::from([("count".into(), Ty::Int)]);
    let ir = compile_to_ir("{{ $count | to_string }}", storage, HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn storage_write() {
    let storage = HashMap::from([("count".into(), Ty::Int)]);
    let ir = compile_to_ir("{{ $count = 42 }}", storage, HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn storage_field_access() {
    let ir = compile_to_ir("{{ $user.name }}", user_storage(), HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Arithmetic ───────────────────────────────────────────────────

#[test]
fn arithmetic_to_string() {
    let storage = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir =
        compile_to_ir("{{ $a + $b | to_string }}", storage, HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Match blocks ─────────────────────────────────────────────────

#[test]
fn simple_match_binding() {
    let storage = HashMap::from([("name".into(), Ty::String)]);
    // Variable binding is body-less — defines x in current scope.
    let ir = compile_to_ir(
        r#"{{ x = $name }}{{ x }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn match_literal_filter() {
    let storage = HashMap::from([("role".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = $role }}admin page{{_}}guest page{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn multi_arm_match() {
    let storage = HashMap::from([("role".into(), Ty::String)]);
    // Trailing comma for continuation arms: {{ "user", }}
    let ir = compile_to_ir(
        r#"{{ "admin" = $role }}admin{{ "user" }}user{{_}}guest{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iteration_over_list() {
    // Use object destructuring to iterate and extract name.
    let ir = compile_to_ir(
        r#"{{ { name, } = $users }}{{ name }}{{/}}"#,
        users_list_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn nested_match() {
    let storage = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            (
                "posts".into(),
                Ty::List(Box::new(Ty::Object(BTreeMap::from([(
                    "title".into(),
                    Ty::String,
                )])))),
            ),
        ])))),
    )]);
    // Use object destructuring for both outer and inner iterations.
    let ir = compile_to_ir(
        r#"{{ { name, posts, } = $users }}{{ { title, } = posts }}{{ title }}{{/}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── List patterns ────────────────────────────────────────────────

#[test]
fn list_destructure_head() {
    let ir = compile_to_ir(
        r#"{{ [a, b, ..] = $items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Object patterns ──────────────────────────────────────────────

#[test]
fn object_pattern() {
    let ir = compile_to_ir(
        r#"{{ { name, age, } = $user }}{{ name }}{{/}}"#,
        user_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Range ────────────────────────────────────────────────────────

#[test]
fn range_iteration() {
    // Variable binding captures the range, then pipe to_string.
    let ir = compile_simple(
        r#"{{ x = 0..5 }}{{ x | to_string }}"#,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_pattern() {
    let storage = HashMap::from([("age".into(), Ty::Int)]);
    // Trailing comma for continuation arms.
    let ir = compile_to_ir(
        r#"{{ 0..10 = $age }}child{{ 10..=19 }}teen{{_}}adult{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Pipe & builtins ──────────────────────────────────────────────

#[test]
fn pipe_filter_map() {
    // Variable binding is body-less.
    let ir = compile_to_ir(
        r#"{{ x = $items | filter(x -> x != 0) | map(x -> x | to_string) }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn pipe_to_string() {
    let storage = HashMap::from([("n".into(), Ty::Int)]);
    let ir = compile_to_ir("{{ $n | to_string }}", storage, HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Lambda / closures ────────────────────────────────────────────

#[test]
fn lambda_in_filter() {
    // Variable binding is body-less.
    let ir = compile_to_ir(
        r#"{{ x = $items | filter(x -> x != 0) }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Extern functions ─────────────────────────────────────────────

#[test]
fn extern_async_call() {
    let externs = HashMap::from([("fetch_user".into(), (vec![Ty::Int], Ty::String))]);
    // Variable binding is body-less.
    let ir = compile_to_ir(
        r#"{{ user = fetch_user(1) }}{{ user }}"#,
        HashMap::new(),
        externs,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Error cases ──────────────────────────────────────────────────

#[test]
fn error_emit_non_string() {
    let result = compile_simple("{{ 42 }}");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_undefined_storage() {
    let result = compile_to_ir("{{ $unknown | to_string }}", HashMap::new(), HashMap::new());
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_undefined_variable() {
    let result = compile_to_ir(
        "{{ x = unknown }}{{_}}{{/}}",
        HashMap::new(),
        HashMap::new(),
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_type_mismatch() {
    let result = compile_simple(r#"{{ x = 1 + 2.0 }}{{_}}{{/}}"#);
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_range_float_bounds() {
    let result = compile_simple("{{ x = 1.0..2.0 }}{{_}}{{/}}");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}
