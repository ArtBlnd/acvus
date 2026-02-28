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
    let ir = compile_to_ir(
        r#"{{ "admin" = $role }}admin{{ "user" = }}user{{_}}guest{{/}}"#,
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
        r#"{{ { name, } in $users }}{{ name }}{{/}}"#,
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
        r#"{{ { name, posts, } in $users }}{{ { title, } in posts }}{{ title }}{{/}}{{/}}"#,
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
fn range_binding() {
    // Variable binding captures a range value.
    let ir = compile_simple(
        r#"{{ x = 0..5 }}{{ x | to_string }}"#,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_iteration() {
    // Explicit iteration with `in`.
    let ir = compile_simple(
        r#"{{ x in 0..3 }}{{ x | to_string }}{{/}}"#,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_pattern() {
    let storage = HashMap::from([("age".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ 0..10 = $age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
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

// ── Tuple ────────────────────────────────────────────────────────

#[test]
fn tuple_expression() {
    let storage = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ t = ($a, $b) }}{{ t | to_string }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_binding() {
    let storage = HashMap::from([
        ("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int])),
    ]);
    let ir = compile_to_ir(
        r#"{{ (name, age) = $pair }}{{ name }}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_wildcard() {
    let storage = HashMap::from([
        ("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int])),
    ]);
    let ir = compile_to_ir(
        r#"{{ (name, _) = $pair }}{{ name }}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_literal_match() {
    let storage = HashMap::from([
        ("a".into(), Ty::Int),
        ("b".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ (0, 1) = ($a, $b) }}zero-one{{ (1, _) = }}one-any{{_}}other{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_nested_destructure() {
    let storage = HashMap::from([(
        "data".into(),
        Ty::Tuple(vec![
            Ty::String,
            Ty::Object(BTreeMap::from([("x".into(), Ty::Int)])),
        ]),
    )]);
    let ir = compile_to_ir(
        r#"{{ (label, { x, }) = $data }}{{ label }}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_tuple_arity_mismatch() {
    let storage = HashMap::from([
        ("pair".into(), Ty::Tuple(vec![Ty::Int, Ty::Int])),
    ]);
    let result = compile_to_ir(
        r#"{{ (a, b, c) = $pair }}{{ a | to_string }}{{/}}"#,
        storage,
        HashMap::new(),
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
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

// ── Iteration (`in`) ────────────────────────────────────────────

#[test]
fn iter_list_binding() {
    let storage = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir("{{ x in $items }}{{ x | to_string }}{{/}}", storage, HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_object_destructure() {
    let ir = compile_to_ir(
        "{{ { name, } in $users }}{{ name }}{{/}}",
        users_list_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_tuple_destructure() {
    let storage = HashMap::from([(
        "pairs".into(),
        Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
    )]);
    let ir = compile_to_ir("{{ (a, _) in $pairs }}{{ a }}{{/}}", storage, HashMap::new()).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_with_catch_all() {
    let storage = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ x in $items }}{{ x | to_string }}{{_}}empty{{/}}",
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_iter_refutable_pattern() {
    let storage = HashMap::from([("roles".into(), Ty::List(Box::new(Ty::String)))]);
    let result = compile_to_ir(r#"{{ "admin" in $roles }}...{{/}}"#, storage, HashMap::new());
    assert!(result.is_err());
}

#[test]
fn error_iter_not_iterable() {
    let storage = HashMap::from([("name".into(), Ty::String)]);
    let result = compile_to_ir("{{ x in $name }}{{ x }}{{/}}", storage, HashMap::new());
    assert!(result.is_err());
}

// ── Edge case: new storage ref binding ──────────────────────────

#[test]
fn storage_new_ref_binding() {
    // $result is not in initial storage — dynamically created via $-binding.
    let storage = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ $result = $name }}{{ $result }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn storage_new_ref_in_match_arm() {
    // $selected is created inside a match arm, then read after the match.
    let storage = HashMap::from([("role".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = $role }}{{ $selected = "yes" }}{{_}}{{ $selected = "no" }}{{/}}{{ $selected }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested destructuring ─────────────────────────────

#[test]
fn list_of_tuples_destructure() {
    // Iterate over list of tuples, destructure each.
    let storage = HashMap::from([(
        "pairs".into(),
        Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
    )]);
    let ir = compile_to_ir(
        r#"{{ (name, age) in $pairs }}{{ name }}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn list_head_with_object_elements() {
    // List destructure where elements are objects.
    let storage = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
        ])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ [first, ..] = $users }}{{ first.name }}{{_}}empty{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_with_list_element() {
    // Tuple containing a list, extract and iterate.
    let storage = HashMap::from([(
        "data".into(),
        Ty::Tuple(vec![
            Ty::String,
            Ty::List(Box::new(Ty::Int)),
        ]),
    )]);
    let ir = compile_to_ir(
        r#"{{ (label, items) = $data }}{{ label }}{{ x in items }}{{ x | to_string }}{{/}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object expression & matching ─────────────────────

#[test]
fn object_literal_field_access() {
    let storage = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ o = { $name, } }}{{ o.name }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: comparison / boolean / unary ──────────────────────

#[test]
fn comparison_operators() {
    let storage = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ x = $a > $b }}{{ x | to_string }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn unary_negation() {
    let storage = HashMap::from([("n".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ x = -$n }}{{ x | to_string }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn boolean_not() {
    let storage = HashMap::from([("flag".into(), Ty::Bool)]);
    let ir = compile_to_ir(
        r#"{{ x = !$flag }}{{ x | to_string }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: to_float / to_int conversion ─────────────────────

#[test]
fn to_float_conversion() {
    let storage = HashMap::from([("n".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ x = $n | to_float }}{{ x | to_string }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn to_int_conversion() {
    let storage = HashMap::from([("f".into(), Ty::Float)]);
    let ir = compile_to_ir(
        r#"{{ x = $f | to_int }}{{ x | to_string }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: pmap builtin ─────────────────────────────────────

#[test]
fn pmap_builtin() {
    let ir = compile_to_ir(
        r#"{{ x = $items | pmap(i -> i | to_string) }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list tail destructure ────────────────────────────

#[test]
fn list_destructure_tail() {
    let ir = compile_to_ir(
        r#"{{ [.., a, b] = $items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple storage writes then read ────────────────

#[test]
fn storage_write_then_read() {
    let storage = HashMap::from([("x".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ $x = 42 }}{{ $x | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested iteration with binding ────────────────────

#[test]
fn nested_iteration_with_binding() {
    let storage = HashMap::from([(
        "matrix".into(),
        Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
    )]);
    let ir = compile_to_ir(
        r#"{{ row in $matrix }}{{ x in row }}{{ x | to_string }}{{/}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: range inclusive iteration ─────────────────────────

#[test]
fn range_inclusive_iteration() {
    let ir = compile_simple(
        r#"{{ x in 0..=3 }}{{ x | to_string }}{{/}}"#,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: deeply nested object ─────────────────────────────

#[test]
fn deeply_nested_object_access() {
    let storage = HashMap::from([(
        "data".into(),
        Ty::Object(BTreeMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([(
                "address".into(),
                Ty::Object(BTreeMap::from([("city".into(), Ty::String)])),
            )])),
        )])),
    )]);
    let ir = compile_to_ir(
        r#"{{ $data.user.address.city }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure capturing storage ref ────────────────────

#[test]
fn closure_capture_storage() {
    let storage = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("threshold".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ x = $items | filter(i -> i > $threshold) }}{{ x | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-arm with different pattern types ───────────

#[test]
fn multi_arm_range_and_literal() {
    let storage = HashMap::from([("score".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ 0 = $score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list literal ─────────────────────────────────────

#[test]
fn list_literal_expression() {
    let ir = compile_simple(
        r#"{{ x = [1, 2, 3] }}{{ x | to_string }}{{_}}{{/}}"#,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: lambda with arithmetic ───────────────────────────

#[test]
fn lambda_map_arithmetic() {
    // Lambda param type resolved via unification: map(List<Int>, x -> x + 1)
    let ir = compile_to_ir(
        r#"{{ x = $items | map(i -> i + 1) }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn lambda_filter_comparison() {
    // Lambda param type resolved via unification: filter(List<Int>, x -> x > 0)
    let ir = compile_to_ir(
        r#"{{ x = $items | filter(i -> i > 0) }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure with captured local var ──────────────────

#[test]
fn closure_capture_local() {
    // Closure captures local variable (not storage).
    let ir = compile_to_ir(
        r#"{{ threshold = 5 }}{{ x = $items | filter(i -> i > threshold) }}{{ x | to_string }}{{_}}{{/}}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list exact match (no rest) ───────────────────────

#[test]
fn list_exact_match() {
    // Exact list pattern: [a, b] without rest (..).
    let ir = compile_to_ir(
        r#"{{ [a, b] = $items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list rest in middle ──────────────────────────────

#[test]
fn list_destructure_head_and_tail() {
    // [a, .., z] pattern — head and tail extraction.
    let ir = compile_to_ir(
        r#"{{ [first, .., last] = $items }}{{ first | to_string }}{{_}}empty{{/}}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested tuple pattern ─────────────────────────────

#[test]
fn nested_tuple_pattern() {
    let storage = HashMap::from([(
        "data".into(),
        Ty::Tuple(vec![
            Ty::Tuple(vec![Ty::Int, Ty::Int]),
            Ty::String,
        ]),
    )]);
    let ir = compile_to_ir(
        r#"{{ ((a, b), label) = $data }}{{ label }}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: storage write of computed value ──────────────────

#[test]
fn storage_write_computed() {
    let storage = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ $a = $a + $b }}{{ $a | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match block with binding pattern + body ──────────

#[test]
fn match_binding_with_body() {
    // Object pattern with body (goes through normal match lowering).
    let ir = compile_to_ir(
        r#"{{ { name, } = $user }}{{ name }} is here{{_}}no user{{/}}"#,
        user_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable shadowing across scopes ─────────────────

#[test]
fn variable_shadowing() {
    let storage = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ x = "outer" }}{{ x = $name }}{{ x }}{{_}}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match inside match (nested match blocks) ─────────

#[test]
fn nested_match_blocks() {
    let storage = HashMap::from([
        ("role".into(), Ty::String),
        ("level".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ "admin" = $role }}{{ 1..10 = $level }}low{{_}}high{{/}}{{_}}guest{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: catch-all with nested binding ────────────────────

#[test]
fn catch_all_with_binding() {
    let storage = HashMap::from([("role".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = $role }}admin{{_}}{{ fallback = "guest" }}{{ fallback }}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple chained pipes ───────────────────────────

#[test]
fn triple_pipe_chain() {
    let ir = compile_to_ir(
        r#"{{ x = $items | filter(i -> i != 0) | map(i -> i + 1) | map(i -> i | to_string) }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: storage write in iteration body ──────────────────

#[test]
fn storage_write_in_iteration() {
    let storage = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("last".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ x in $items }}{{ $last = x }}{{/}}{{ $last | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: field access on destructured variable ────────────

#[test]
fn field_access_on_destructured() {
    let storage = HashMap::from([(
        "pair".into(),
        Ty::Tuple(vec![
            Ty::Object(BTreeMap::from([("name".into(), Ty::String)])),
            Ty::Int,
        ]),
    )]);
    let ir = compile_to_ir(
        r#"{{ (obj, _) = $pair }}{{ obj.name }}{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: boolean operators in match ────────────────────────

#[test]
fn equality_as_match_source() {
    let storage = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ true = $a == $b }}equal{{_}}not equal{{/}}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary negation on lambda param (Ty::Var) ─────────

#[test]
fn lambda_negate_param() {
    // Lambda param has Ty::Var initially; -i must resolve via unification.
    let ir = compile_to_ir(
        r#"{{ x = $items | map(i -> -i) }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary not on lambda param (Ty::Var) ──────────────

#[test]
fn lambda_not_param() {
    // Lambda param has Ty::Var initially; !i must resolve via unification.
    let storage = HashMap::from([("flags".into(), Ty::List(Box::new(Ty::Bool)))]);
    let ir = compile_to_ir(
        r#"{{ x = $flags | map(i -> !i) }}{{ x | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object pattern on match source (non-list) ────────

#[test]
fn object_destructure_match() {
    // Object pattern directly on Object source (not List<Object>).
    let ir = compile_to_ir(
        r#"{{ { name, age, } = $user }}{{ name }}{{ age | to_string }}{{_}}none{{/}}"#,
        user_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple closures sharing captured var ────────────

#[test]
fn multiple_closures_same_capture() {
    let storage = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("offset".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ x = $items | map(i -> i + $offset) | filter(i -> i > 0) }}{{ x | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: string comparison ─────────────────────────────────

#[test]
fn string_equality_in_filter() {
    let storage = HashMap::from([(
        "names".into(),
        Ty::List(Box::new(Ty::String)),
    )]);
    let ir = compile_to_ir(
        r#"{{ x = $names | filter(n -> n != "admin") }}{{ x | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested lambda (lambda returning lambda result) ────

#[test]
fn lambda_field_access() {
    // Lambda body accesses field on captured object.
    let storage = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
        ])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ x = $users | map(u -> u.name) }}{{ x | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: storage write with expression in iteration ────────

#[test]
fn storage_accumulate_in_loop() {
    // Write to storage on each iteration.
    let storage = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("sum".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ x in $items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}"#,
        storage,
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-level pipe with to_string in middle ────────

#[test]
fn pipe_map_to_string_then_filter() {
    let ir = compile_to_ir(
        r#"{{ x = $items | map(i -> i | to_string) | filter(s -> s != "0") }}{{ x | to_string }}"#,
        items_storage(),
        HashMap::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: extern function with object return ───────────────

#[test]
fn extern_fn_object_return() {
    let externs = HashMap::from([(
        "get_user".into(),
        (
            vec![Ty::Int],
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])),
        ),
    )]);
    let ir = compile_to_ir(
        r#"{{ u = get_user(1) }}{{ u.name }}"#,
        HashMap::new(),
        externs,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}
