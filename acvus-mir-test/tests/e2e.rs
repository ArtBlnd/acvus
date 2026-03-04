use std::collections::{BTreeMap, HashMap};

use acvus_mir::extern_module::ExternRegistry;
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
    let context = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir("Hello, {{ @name }}!", context, &ExternRegistry::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Context / Variables ──────────────────────────────────────────

#[test]
fn context_read() {
    let context = HashMap::from([("count".into(), Ty::Int)]);
    let ir = compile_to_ir("{{ @count | to_string }}", context, &ExternRegistry::new()).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_write() {
    let ir = compile_simple("{{ $count = 42 }}").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_field_access() {
    let ir = compile_to_ir("{{ @user.name }}", user_context(), &ExternRegistry::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Arithmetic ───────────────────────────────────────────────────

#[test]
fn arithmetic_to_string() {
    let context = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir("{{ @a + @b | to_string }}", context, &ExternRegistry::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Match blocks ─────────────────────────────────────────────────

#[test]
fn simple_match_binding() {
    let context = HashMap::from([("name".into(), Ty::String)]);
    // Variable binding is body-less — defines x in current scope.
    let ir = compile_to_ir(r#"{{ x = @name }}{{ x }}"#, context, &ExternRegistry::new()).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn match_literal_filter() {
    let context = HashMap::from([("role".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = @role }}admin page{{_}}guest page{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn multi_arm_match() {
    let context = HashMap::from([("role".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = @role }}admin{{ "user" = }}user{{_}}guest{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iteration_over_list() {
    // Use object destructuring to iterate and extract name.
    let ir = compile_to_ir(
        r#"{{ { name, } in @users }}{{ name }}{{/}}"#,
        users_list_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn nested_match() {
    let context = HashMap::from([(
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
        r#"{{ { name, posts, } in @users }}{{ { title, } in posts }}{{ title }}{{/}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── List patterns ────────────────────────────────────────────────

#[test]
fn list_destructure_head() {
    let ir = compile_to_ir(
        r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Object patterns ──────────────────────────────────────────────

#[test]
fn object_pattern() {
    let ir = compile_to_ir(
        r#"{{ { name, age, } = @user }}{{ name }}{{/}}"#,
        user_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Range ────────────────────────────────────────────────────────

#[test]
fn range_binding() {
    // Variable binding captures a range value; iterate to emit scalar elements.
    let ir = compile_simple(r#"{{ x in 0..5 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_iteration() {
    // Explicit iteration with `in`.
    let ir = compile_simple(r#"{{ x in 0..3 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_pattern() {
    let context = HashMap::from([("age".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ 0..10 = @age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Pipe & builtins ──────────────────────────────────────────────

#[test]
fn pipe_filter_map() {
    // Variable binding is body-less.
    let ir = compile_to_ir(
        r#"{{ x = @items | filter(x -> x != 0) | map(x -> x + 1) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn pipe_to_string() {
    let context = HashMap::from([("n".into(), Ty::Int)]);
    let ir = compile_to_ir("{{ @n | to_string }}", context, &ExternRegistry::new()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Lambda / closures ────────────────────────────────────────────

#[test]
fn lambda_in_filter() {
    // Variable binding is body-less.
    let ir = compile_to_ir(
        r#"{{ x = @items | filter(x -> x != 0) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Extern functions ─────────────────────────────────────────────

#[test]
fn extern_async_call() {
    use acvus_mir::extern_module::ExternModule;
    let mut ext = ExternModule::new("test");
    ext.add_fn("fetch_user", vec![Ty::Int], Ty::String, false);
    let mut registry = ExternRegistry::new();
    registry.register(&ext);
    // Variable binding is body-less.
    let ir = compile_to_ir(
        r#"{{ user = fetch_user(1) }}{{ user }}"#,
        HashMap::new(),
        &registry,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Tuple ────────────────────────────────────────────────────────

#[test]
fn tuple_expression() {
    let context = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ (a, b) = (@a, @b) }}{{ a | to_string }}{{ b }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_binding() {
    let context = HashMap::from([("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let ir = compile_to_ir(
        r#"{{ (name, age) = @pair }}{{ name }}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_wildcard() {
    let context = HashMap::from([("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let ir = compile_to_ir(
        r#"{{ (name, _) = @pair }}{{ name }}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_literal_match() {
    let context = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ (0, 1) = (@a, @b) }}zero-one{{ (1, _) = }}one-any{{_}}other{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_nested_destructure() {
    let context = HashMap::from([(
        "data".into(),
        Ty::Tuple(vec![
            Ty::String,
            Ty::Object(BTreeMap::from([("x".into(), Ty::Int)])),
        ]),
    )]);
    let ir = compile_to_ir(
        r#"{{ (label, { x, }) = @data }}{{ label }}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_tuple_arity_mismatch() {
    let context = HashMap::from([("pair".into(), Ty::Tuple(vec![Ty::Int, Ty::Int]))]);
    let result = compile_to_ir(
        r#"{{ (a, b, c) = @pair }}{{ a | to_string }}{{/}}"#,
        context,
        &ExternRegistry::new(),
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
fn error_undefined_context() {
    let result = compile_to_ir(
        "{{ @unknown | to_string }}",
        HashMap::new(),
        &ExternRegistry::new(),
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_undefined_variable() {
    let result = compile_to_ir(
        "{{ x = unknown }}{{_}}{{/}}",
        HashMap::new(),
        &ExternRegistry::new(),
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
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ x in @items }}{{ x | to_string }}{{/}}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_object_destructure() {
    let ir = compile_to_ir(
        "{{ { name, } in @users }}{{ name }}{{/}}",
        users_list_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_tuple_destructure() {
    let context = HashMap::from([(
        "pairs".into(),
        Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
    )]);
    let ir = compile_to_ir(
        "{{ (a, _) in @pairs }}{{ a }}{{/}}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_with_catch_all() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ x in @items }}{{ x | to_string }}{{_}}empty{{/}}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_iter_refutable_pattern() {
    let context = HashMap::from([("roles".into(), Ty::List(Box::new(Ty::String)))]);
    let result = compile_to_ir(
        r#"{{ "admin" in @roles }}...{{/}}"#,
        context,
        &ExternRegistry::new(),
    );
    assert!(result.is_err());
}

#[test]
fn error_iter_not_iterable() {
    let context = HashMap::from([("name".into(), Ty::String)]);
    let result = compile_to_ir(
        "{{ x in @name }}{{ x }}{{/}}",
        context,
        &ExternRegistry::new(),
    );
    assert!(result.is_err());
}

// ── Edge case: new variable ref binding ─────────────────────────

#[test]
fn variable_new_ref_binding() {
    // $result is not in initial context — dynamically created via $-binding.
    let context = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ $result = @name }}{{ $result }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_new_ref_in_match_arm() {
    // $selected is created inside a match arm, then read after the match.
    let context = HashMap::from([("role".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = @role }}{{ $selected = "yes" }}{{_}}{{ $selected = "no" }}{{/}}{{ $selected }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested destructuring ─────────────────────────────

#[test]
fn list_of_tuples_destructure() {
    // Iterate over list of tuples, destructure each.
    let context = HashMap::from([(
        "pairs".into(),
        Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
    )]);
    let ir = compile_to_ir(
        r#"{{ (name, age) in @pairs }}{{ name }}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn list_head_with_object_elements() {
    // List destructure where elements are objects.
    let context = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([(
            "name".into(),
            Ty::String,
        )])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ [first, ..] = @users }}{{ first.name }}{{_}}empty{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_with_list_element() {
    // Tuple containing a list, extract and iterate.
    let context = HashMap::from([(
        "data".into(),
        Ty::Tuple(vec![Ty::String, Ty::List(Box::new(Ty::Int))]),
    )]);
    let ir = compile_to_ir(
        r#"{{ (label, items) = @data }}{{ label }}{{ x in items }}{{ x | to_string }}{{/}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object expression & matching ─────────────────────

#[test]
fn object_literal_field_access() {
    let context = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ o = { @name, } }}{{ o.name }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: comparison / boolean / unary ──────────────────────

#[test]
fn comparison_operators() {
    let context = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ x = @a > @b }}{{ x | to_string }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn unary_negation() {
    let context = HashMap::from([("n".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ x = -@n }}{{ x | to_string }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn boolean_not() {
    let context = HashMap::from([("flag".into(), Ty::Bool)]);
    let ir = compile_to_ir(
        r#"{{ x = !@flag }}{{ x | to_string }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: to_float / to_int conversion ─────────────────────

#[test]
fn to_float_conversion() {
    let context = HashMap::from([("n".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ x = @n | to_float }}{{ x | to_string }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn to_int_conversion() {
    let context = HashMap::from([("f".into(), Ty::Float)]);
    let ir = compile_to_ir(
        r#"{{ x = @f | to_int }}{{ x | to_string }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: pmap builtin ─────────────────────────────────────

#[test]
fn pmap_builtin() {
    let ir = compile_to_ir(
        r#"{{ x = @items | pmap(i -> i + 1) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list tail destructure ────────────────────────────

#[test]
fn list_destructure_tail() {
    let ir = compile_to_ir(
        r#"{{ [.., a, b] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write then read ─────────────────────────

#[test]
fn variable_write_then_read() {
    let ir = compile_simple(r#"{{ $x = 42 }}{{ $x | to_string }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested iteration with binding ────────────────────

#[test]
fn nested_iteration_with_binding() {
    let context = HashMap::from([(
        "matrix".into(),
        Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
    )]);
    let ir = compile_to_ir(
        r#"{{ row in @matrix }}{{ x in row }}{{ x | to_string }}{{/}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: range inclusive iteration ─────────────────────────

#[test]
fn range_inclusive_iteration() {
    let ir = compile_simple(r#"{{ x in 0..=3 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: deeply nested object ─────────────────────────────

#[test]
fn deeply_nested_object_access() {
    let context = HashMap::from([(
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
        r#"{{ @data.user.address.city }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure capturing context ref ────────────────────

#[test]
fn closure_capture_context() {
    let context = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("threshold".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ x = @items | filter(i -> i > @threshold) }}{{ x | len | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-arm with different pattern types ───────────

#[test]
fn multi_arm_range_and_literal() {
    let context = HashMap::from([("score".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list literal ─────────────────────────────────────

#[test]
fn list_literal_expression() {
    let ir = compile_simple(r#"{{ x = [1, 2, 3] }}{{ x | len | to_string }}{{_}}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: lambda with arithmetic ───────────────────────────

#[test]
fn lambda_map_arithmetic() {
    // Lambda param type resolved via unification: map(List<Int>, x -> x + 1)
    let ir = compile_to_ir(
        r#"{{ x = @items | map(i -> i + 1) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn lambda_filter_comparison() {
    // Lambda param type resolved via unification: filter(List<Int>, x -> x > 0)
    let ir = compile_to_ir(
        r#"{{ x = @items | filter(i -> i > 0) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure with captured local var ──────────────────

#[test]
fn closure_capture_local() {
    // Closure captures local variable (not context).
    let ir = compile_to_ir(
        r#"{{ threshold = 5 }}{{ x = @items | filter(i -> i > threshold) }}{{ x | len | to_string }}{{_}}{{/}}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list exact match (no rest) ───────────────────────

#[test]
fn list_exact_match() {
    // Exact list pattern: [a, b] without rest (..).
    let ir = compile_to_ir(
        r#"{{ [a, b] = @items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list rest in middle ──────────────────────────────

#[test]
fn list_destructure_head_and_tail() {
    // [a, .., z] pattern — head and tail extraction.
    let ir = compile_to_ir(
        r#"{{ [first, .., last] = @items }}{{ first | to_string }}{{_}}empty{{/}}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested tuple pattern ─────────────────────────────

#[test]
fn nested_tuple_pattern() {
    let context = HashMap::from([(
        "data".into(),
        Ty::Tuple(vec![Ty::Tuple(vec![Ty::Int, Ty::Int]), Ty::String]),
    )]);
    let ir = compile_to_ir(
        r#"{{ ((a, b), label) = @data }}{{ label }}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write of computed value ─────────────────

#[test]
fn variable_write_computed() {
    let context = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ $result = @a + @b }}{{ $result | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match block with binding pattern + body ──────────

#[test]
fn match_binding_with_body() {
    // Object pattern with body (goes through normal match lowering).
    let ir = compile_to_ir(
        r#"{{ { name, } = @user }}{{ name }} is here{{_}}no user{{/}}"#,
        user_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable shadowing across scopes ─────────────────

#[test]
fn variable_shadowing() {
    let context = HashMap::from([("name".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ x = "outer" }}{{ x = @name }}{{ x }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match inside match (nested match blocks) ─────────

#[test]
fn nested_match_blocks() {
    let context = HashMap::from([("role".into(), Ty::String), ("level".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = @role }}{{ 1..10 = @level }}low{{_}}high{{/}}{{_}}guest{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: catch-all with nested binding ────────────────────

#[test]
fn catch_all_with_binding() {
    let context = HashMap::from([("role".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ "admin" = @role }}admin{{_}}{{ fallback = "guest" }}{{ fallback }}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple chained pipes ───────────────────────────

#[test]
fn triple_pipe_chain() {
    let ir = compile_to_ir(
        r#"{{ x = @items | filter(i -> i != 0) | map(i -> i + 1) | map(i -> i * 2) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write in iteration body ─────────────────

#[test]
fn variable_write_in_iteration() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        r#"{{ x in @items }}{{ $last = x }}{{/}}{{ $last | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: field access on destructured variable ────────────

#[test]
fn field_access_on_destructured() {
    let context = HashMap::from([(
        "pair".into(),
        Ty::Tuple(vec![
            Ty::Object(BTreeMap::from([("name".into(), Ty::String)])),
            Ty::Int,
        ]),
    )]);
    let ir = compile_to_ir(
        r#"{{ (obj, _) = @pair }}{{ obj.name }}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: boolean operators in match ────────────────────────

#[test]
fn equality_as_match_source() {
    let context = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let ir = compile_to_ir(
        r#"{{ true = @a == @b }}equal{{_}}not equal{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary negation on lambda param (Ty::Var) ─────────

#[test]
fn lambda_negate_param() {
    // Lambda param has Ty::Var initially; -i must resolve via unification.
    let ir = compile_to_ir(
        r#"{{ x = @items | map(i -> -i) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary not on lambda param (Ty::Var) ──────────────

#[test]
fn lambda_not_param() {
    // Lambda param has Ty::Var initially; !i must resolve via unification.
    let context = HashMap::from([("flags".into(), Ty::List(Box::new(Ty::Bool)))]);
    let ir = compile_to_ir(
        r#"{{ x = @flags | map(i -> !i) }}{{ x | len | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object pattern on match source (non-list) ────────

#[test]
fn object_destructure_match() {
    // Object pattern directly on Object source (not List<Object>).
    let ir = compile_to_ir(
        r#"{{ { name, age, } = @user }}{{ name }}{{ age | to_string }}{{_}}none{{/}}"#,
        user_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple closures sharing captured var ────────────

#[test]
fn multiple_closures_same_capture() {
    let context = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("offset".into(), Ty::Int),
    ]);
    let ir = compile_to_ir(
        r#"{{ x = @items | map(i -> i + @offset) | filter(i -> i > 0) }}{{ x | len | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: string comparison ─────────────────────────────────

#[test]
fn string_equality_in_filter() {
    let context = HashMap::from([("names".into(), Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        r#"{{ x = @names | filter(n -> n != "admin") }}{{ x | join(",") }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested lambda (lambda returning lambda result) ────

#[test]
fn lambda_field_access() {
    // Lambda body accesses field on captured object.
    let context = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
        ])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ x = @users | map(u -> u.name) }}{{ x | join(",") }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable accumulation in loop ─────────────────────

#[test]
fn variable_accumulate_in_loop() {
    // Write to variable on each iteration.
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        r#"{{ $sum = 0 }}{{ x in @items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-level pipe with to_string in middle ────────

#[test]
fn pipe_map_to_string_then_filter() {
    let ir = compile_to_ir(
        r#"{{ x = @items | map(i -> i + 1) | filter(i -> i != 0) }}{{ x | len | to_string }}"#,
        items_context(),
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: local $-var captured in lambda ───────────────────

#[test]
fn lambda_capture_local_var_ref() {
    // $offset is NOT in initial context — created as local $-var.
    // Lambda must capture it correctly (not fall through to StorageLoad).
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        r#"{{ $offset = 10 }}{{ x = @items | filter(i -> i > $offset) }}{{ x | len | to_string }}{{_}}{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple field accesses on same lambda param ─────

#[test]
fn lambda_multiple_field_access() {
    // Lambda body accesses two fields on the same param.
    let context = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
        ])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ x = @users | map(u -> (u.name, u.age)) }}{{ x | len | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: chained field access in lambda ───────────────────

#[test]
fn lambda_chained_field_access() {
    let context = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([(
            "address".into(),
            Ty::Object(BTreeMap::from([("city".into(), Ty::String)])),
        )])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ x = @users | map(u -> u.address.city) }}{{ x | join(",") }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: string concat in lambda ───────────────────────────

#[test]
fn lambda_string_concat() {
    // Lambda param is Ty::Var; string concat (+) must resolve via unification.
    let context = HashMap::from([("names".into(), Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        r#"{{ x = @names | map(n -> n + "!") }}{{ x | join(",") }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: filter then map with field access ────────────────

#[test]
fn pipe_filter_then_map_field() {
    let context = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
        ])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ x = @users | filter(u -> u.age > 18) | map(u -> u.name) }}{{ x | join(",") }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: error — field access on non-object ────────────────

#[test]
fn error_field_access_on_int() {
    let context = HashMap::from([("n".into(), Ty::Int)]);
    let result = compile_to_ir("{{ @n.foo | to_string }}", context, &ExternRegistry::new());
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Edge case: error — context write attempt ─────────────────────

#[test]
fn error_variable_write_type_mismatch() {
    // Attempting to write to a context key (read-only).
    let context = HashMap::from([("count".into(), Ty::Int)]);
    let result = compile_to_ir(r#"{{ @count = "hello" }}"#, context, &ExternRegistry::new());
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Edge case: float arithmetic in lambda ────────────────────────

#[test]
fn lambda_float_arithmetic() {
    let context = HashMap::from([("vals".into(), Ty::List(Box::new(Ty::Float)))]);
    let ir = compile_to_ir(
        r#"{{ x = @vals | map(v -> v * 2.0) }}{{ x | len | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: bool literal as match source ─────────────────────

#[test]
fn match_bool_literal() {
    let context = HashMap::from([("flag".into(), Ty::Bool)]);
    let ir = compile_to_ir(
        r#"{{ true = @flag }}on{{_}}off{{/}}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested pipe with filter on object field ──────────

#[test]
fn filter_object_field_equality() {
    let context = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("active".into(), Ty::Bool),
        ])))),
    )]);
    let ir = compile_to_ir(
        r#"{{ x = @users | filter(u -> u.active) }}{{ x | len | to_string }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: extern function with object return ───────────────

#[test]
fn extern_fn_object_return() {
    use acvus_mir::extern_module::ExternModule;
    let mut ext = ExternModule::new("test");
    ext.add_fn(
        "get_user",
        vec![Ty::Int],
        Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
        ])),
        false,
    );
    let mut registry = ExternRegistry::new();
    registry.register(&ext);
    let ir = compile_to_ir(
        r#"{{ u = get_user(1) }}{{ u.name }}"#,
        HashMap::new(),
        &registry,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── New builtins ─────────────────────────────────────────────────

#[test]
fn builtin_len() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ @items | len | to_string }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_reverse() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ x in @items | reverse }}{{ x | to_string }}{{/}}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_join() {
    let context = HashMap::from([("names".into(), Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        r#"{{ @names | join(", ") }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_contains() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ @items | contains(3) | to_string }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_find() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ @items | find(x -> x > 10) | to_string }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_reduce() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ @items | reduce((a, b) -> a + b) | to_string }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_fold() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ @items | fold(0, (acc, x) -> acc + x) | to_string }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_any() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ @items | any(x -> x > 10) | to_string }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_all() {
    let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        "{{ @items | all(x -> x > 0) | to_string }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Context Call ────────────────────────────────────────────────

#[test]
fn context_call_simple() {
    let context = HashMap::from([("node".into(), Ty::String)]);
    let ir = compile_to_ir(
        r#"{{ @node { key: "value", } }}"#,
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_call_shorthand() {
    let context = HashMap::from([("node".into(), Ty::String), ("other".into(), Ty::String)]);
    let ir = compile_to_ir(
        "{{ $var = 42 }}{{ @node { $var, @other, } }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_call_expr() {
    let context = HashMap::from([
        ("node".into(), Ty::String),
        ("items".into(), Ty::List(Box::new(Ty::Int))),
    ]);
    let ir = compile_to_ir(
        "{{ @node { count: @items | len, } }}",
        context,
        &ExternRegistry::new(),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}
