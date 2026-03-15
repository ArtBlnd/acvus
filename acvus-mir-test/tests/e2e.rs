use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::{Effect, FnKind, Ty};
use acvus_mir_test::*;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

/// Helper: build a context HashMap<Astr, Ty> from string pairs.
fn ctx(i: &Interner, pairs: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    pairs
        .iter()
        .map(|(k, v)| (i.intern(k), v.clone()))
        .collect()
}

/// Helper: build an Object type from string-keyed fields.
fn obj(i: &Interner, fields: &[(&str, Ty)]) -> Ty {
    Ty::Object(
        fields
            .iter()
            .map(|(k, v)| (i.intern(k), v.clone()))
            .collect(),
    )
}

// ── Text & literals ──────────────────────────────────────────────

#[test]
fn text_only() {
    let i = Interner::new();
    let ir = compile_simple(&i, "hello world").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn string_emit() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ "hello" }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn string_concat() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ "hello" + " " + "world" }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn mixed_text_and_expr() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(&i, "Hello, {{ @name }}!", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Context / Variables ──────────────────────────────────────────

#[test]
fn context_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        "{{ @count | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_write() {
    let i = Interner::new();
    let ir = compile_simple(&i, "{{ $count = 42 }}").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_field_access() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "{{ @user.name }}",
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Arithmetic ───────────────────────────────────────────────────

#[test]
fn arithmetic_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        "{{ @a + @b | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Match blocks ─────────────────────────────────────────────────

#[test]
fn simple_match_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    // Variable binding is body-less — defines x in current scope.
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @name }}{{ x }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn match_literal_filter() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}admin page{{_}}guest page{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn multi_arm_match() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}admin{{ "user" = }}user{{_}}guest{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iteration_over_list() {
    let i = Interner::new();
    // Use object destructuring to iterate and extract name.
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, } in @users }}{{ name }}{{/}}"#,
        &users_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn nested_match() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(
                &i,
                &[
                    ("name", Ty::String),
                    (
                        "posts",
                        Ty::List(Box::new(obj(&i, &[("title", Ty::String)]))),
                    ),
                ],
            ))),
        )],
    );
    // Use object destructuring for both outer and inner iterations.
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, posts, } in @users }}{{ { title, } in posts }}{{ title }}{{/}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── List patterns ────────────────────────────────────────────────

#[test]
fn list_destructure_head() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Object patterns ──────────────────────────────────────────────

#[test]
fn object_pattern() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, age, } = @user }}{{ name }}{{/}}"#,
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Range ────────────────────────────────────────────────────────

#[test]
fn range_binding() {
    // Variable binding captures a range value; iterate to emit scalar elements.
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ x in 0..5 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_iteration() {
    // Explicit iteration with `in`.
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ x in 0..3 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("age", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ 0..10 = @age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Pipe & builtins ──────────────────────────────────────────────

#[test]
fn pipe_filter_map() {
    let i = Interner::new();
    // Variable binding is body-less.
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(x -> x != 0) | map(x -> x + 1) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn pipe_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let ir = compile_to_ir(&i, "{{ @n | to_string }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Lambda / closures ────────────────────────────────────────────

#[test]
fn lambda_in_filter() {
    let i = Interner::new();
    // Variable binding is body-less.
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(x -> x != 0) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Extern functions ─────────────────────────────────────────────

#[test]
fn extern_async_call() {
    let i = Interner::new();
    let context = ctx(&i, &[("fetch_user", Ty::Fn {
        params: vec![Ty::Int],
        ret: Box::new(Ty::String),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    })]);
    // Variable binding is body-less.
    let ir = compile_to_ir(
        &i,
        r#"{{ user = @fetch_user(1) }}{{ user }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Tuple ────────────────────────────────────────────────────────

#[test]
fn tuple_expression() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ (a, b) = (@a, @b) }}{{ a | to_string }}{{ b }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ (name, age) = @pair }}{{ name }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_wildcard() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ (name, _) = @pair }}{{ name }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_literal_match() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ (0, 1) = (@a, @b) }}zero-one{{ (1, _) = }}one-any{{_}}other{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_nested_destructure() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::String, obj(&i, &[("x", Ty::Int)])]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ (label, { x, }) = @data }}{{ label }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_tuple_arity_mismatch() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::Int, Ty::Int]))]);
    let result = compile_to_ir(
        &i,
        r#"{{ (a, b, c) = @pair }}{{ a | to_string }}{{/}}"#,
        &context,
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Error cases ──────────────────────────────────────────────────

#[test]
fn error_emit_non_string() {
    let i = Interner::new();
    let result = compile_simple(&i, "{{ 42 }}");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_undefined_context() {
    let i = Interner::new();
    let result = compile_to_ir(
        &i,
        "{{ @unknown | to_string }}",
        &FxHashMap::default(),
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_undefined_variable() {
    let i = Interner::new();
    let result = compile_to_ir(
        &i,
        "{{ x = unknown }}{{_}}{{/}}",
        &FxHashMap::default(),
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_type_mismatch() {
    let i = Interner::new();
    let result = compile_simple(&i, r#"{{ x = 1 + 2.0 }}{{_}}{{/}}"#);
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_range_float_bounds() {
    let i = Interner::new();
    let result = compile_simple(&i, "{{ x = 1.0..2.0 }}{{_}}{{/}}");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Iteration (`in`) ────────────────────────────────────────────

#[test]
fn iter_list_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ x in @items }}{{ x | to_string }}{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_object_destructure() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "{{ { name, } in @users }}{{ name }}{{/}}",
        &users_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_tuple_destructure() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "pairs",
            Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        "{{ (a, _) in @pairs }}{{ a }}{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_with_catch_all() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ x in @items }}{{ x | to_string }}{{_}}empty{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_iter_refutable_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("roles", Ty::List(Box::new(Ty::String)))]);
    let result = compile_to_ir(
        &i,
        r#"{{ "admin" in @roles }}...{{/}}"#,
        &context,
    );
    assert!(result.is_err());
}

#[test]
fn error_iter_not_iterable() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let result = compile_to_ir(
        &i,
        "{{ x in @name }}{{ x }}{{/}}",
        &context,
    );
    assert!(result.is_err());
}

// ── Edge case: new variable ref binding ─────────────────────────

#[test]
fn variable_new_ref_binding() {
    let i = Interner::new();
    // $result is not in initial context — dynamically created via $-binding.
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ $result = @name }}{{ $result }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_new_ref_in_match_arm() {
    let i = Interner::new();
    // $selected is created inside a match arm, then read after the match.
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}{{ $selected = "yes" }}{{_}}{{ $selected = "no" }}{{/}}{{ $selected }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested destructuring ─────────────────────────────

#[test]
fn list_of_tuples_destructure() {
    let i = Interner::new();
    // Iterate over list of tuples, destructure each.
    let context = ctx(
        &i,
        &[(
            "pairs",
            Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ (name, age) in @pairs }}{{ name }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn list_head_with_object_elements() {
    let i = Interner::new();
    // List destructure where elements are objects.
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ [first, ..] = @users }}{{ first.name }}{{_}}empty{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_with_list_element() {
    let i = Interner::new();
    // Tuple containing a list, extract and iterate.
    let context = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::String, Ty::List(Box::new(Ty::Int))]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ (label, items) = @data }}{{ label }}{{ x in items }}{{ x | to_string }}{{/}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object expression & matching ─────────────────────

#[test]
fn object_literal_field_access() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ o = { @name, } }}{{ o.name }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: comparison / boolean / unary ──────────────────────

#[test]
fn comparison_operators() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @a > @b }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn unary_negation() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = -@n }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn boolean_not() {
    let i = Interner::new();
    let context = ctx(&i, &[("flag", Ty::Bool)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = !@flag }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: to_float / to_int conversion ─────────────────────

#[test]
fn to_float_conversion() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @n | to_float }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn to_int_conversion() {
    let i = Interner::new();
    let context = ctx(&i, &[("f", Ty::Float)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @f | to_int }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: pmap builtin ─────────────────────────────────────

#[test]
fn pmap_builtin() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | pmap(i -> i + 1) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list tail destructure ────────────────────────────

#[test]
fn list_destructure_tail() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ [.., a, b] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write then read ─────────────────────────

#[test]
fn variable_write_then_read() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ $x = 42 }}{{ $x | to_string }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested iteration with binding ────────────────────

#[test]
fn nested_iteration_with_binding() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))))],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ row in @matrix }}{{ x in row }}{{ x | to_string }}{{/}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: range inclusive iteration ─────────────────────────

#[test]
fn range_inclusive_iteration() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ x in 0..=3 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: deeply nested object ─────────────────────────────

#[test]
fn deeply_nested_object_access() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "data",
            obj(
                &i,
                &[(
                    "user",
                    obj(&i, &[("address", obj(&i, &[("city", Ty::String)]))]),
                )],
            ),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ @data.user.address.city }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure capturing context ref ────────────────────

#[test]
fn closure_capture_context() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("threshold", Ty::Int),
        ],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(i -> i > @threshold) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-arm with different pattern types ───────────

#[test]
fn multi_arm_range_and_literal() {
    let i = Interner::new();
    let context = ctx(&i, &[("score", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list literal ─────────────────────────────────────

#[test]
fn list_literal_expression() {
    let i = Interner::new();
    let ir = compile_simple(
        &i,
        r#"{{ x = [1, 2, 3] }}{{ x | len | to_string }}{{_}}{{/}}"#,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: lambda with arithmetic ───────────────────────────

#[test]
fn lambda_map_arithmetic() {
    let i = Interner::new();
    // Lambda param type resolved via unification: map(List<Int>, x -> x + 1)
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(i -> i + 1) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn lambda_filter_comparison() {
    let i = Interner::new();
    // Lambda param type resolved via unification: filter(List<Int>, x -> x > 0)
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(i -> i > 0) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure with captured local var ──────────────────

#[test]
fn closure_capture_local() {
    let i = Interner::new();
    // Closure captures local variable (not context).
    let ir = compile_to_ir(
        &i,
        r#"{{ threshold = 5 }}{{ x = @items | iter | filter(i -> i > threshold) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list exact match (no rest) ───────────────────────

#[test]
fn list_exact_match() {
    let i = Interner::new();
    // Exact list pattern: [a, b] without rest (..).
    let ir = compile_to_ir(
        &i,
        r#"{{ [a, b] = @items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list rest in middle ──────────────────────────────

#[test]
fn list_destructure_head_and_tail() {
    let i = Interner::new();
    // [a, .., z] pattern — head and tail extraction.
    let ir = compile_to_ir(
        &i,
        r#"{{ [first, .., last] = @items }}{{ first | to_string }}{{_}}empty{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested tuple pattern ─────────────────────────────

#[test]
fn nested_tuple_pattern() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::Tuple(vec![Ty::Int, Ty::Int]), Ty::String]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ ((a, b), label) = @data }}{{ label }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write of computed value ─────────────────

#[test]
fn variable_write_computed() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ $result = @a + @b }}{{ $result | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match block with binding pattern + body ──────────

#[test]
fn match_binding_with_body() {
    let i = Interner::new();
    // Object pattern with body (goes through normal match lowering).
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, } = @user }}{{ name }} is here{{_}}no user{{/}}"#,
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable shadowing across scopes ─────────────────

#[test]
fn variable_shadowing() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = "outer" }}{{ x = @name }}{{ x }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match inside match (nested match blocks) ─────────

#[test]
fn nested_match_blocks() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String), ("level", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}{{ 1..10 = @level }}low{{_}}high{{/}}{{_}}guest{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: catch-all with nested binding ────────────────────

#[test]
fn catch_all_with_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}admin{{_}}{{ fallback = "guest" }}{{ fallback }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple chained pipes ───────────────────────────

#[test]
fn triple_pipe_chain() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(i -> i != 0) | map(i -> i + 1) | map(i -> i * 2) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write in iteration body ─────────────────

#[test]
fn variable_write_in_iteration() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x in @items }}{{ $last = x }}{{/}}{{ $last | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: field access on destructured variable ────────────

#[test]
fn field_access_on_destructured() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "pair",
            Ty::Tuple(vec![obj(&i, &[("name", Ty::String)]), Ty::Int]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ (obj, _) = @pair }}{{ obj.name }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: boolean operators in match ────────────────────────

#[test]
fn equality_as_match_source() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ true = @a == @b }}equal{{_}}not equal{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary negation on lambda param (Ty::Var) ─────────

#[test]
fn lambda_negate_param() {
    let i = Interner::new();
    // Lambda param has Ty::Var initially; -i must resolve via unification.
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(i -> -i) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary not on lambda param (Ty::Var) ──────────────

#[test]
fn lambda_not_param() {
    let i = Interner::new();
    // Lambda param has Ty::Var initially; !i must resolve via unification.
    let context = ctx(&i, &[("flags", Ty::List(Box::new(Ty::Bool)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @flags | iter | map(i -> !i) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object pattern on match source (non-list) ────────

#[test]
fn object_destructure_match() {
    let i = Interner::new();
    // Object pattern directly on Object source (not List<Object>).
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, age, } = @user }}{{ name }}{{ age | to_string }}{{_}}none{{/}}"#,
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple closures sharing captured var ────────────

#[test]
fn multiple_closures_same_capture() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("offset", Ty::Int)],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(i -> i + @offset) | filter(i -> i > 0) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: string comparison ─────────────────────────────────

#[test]
fn string_equality_in_filter() {
    let i = Interner::new();
    let context = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @names | iter | filter(n -> n != "admin") }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested lambda (lambda returning lambda result) ────

#[test]
fn lambda_field_access() {
    let i = Interner::new();
    // Lambda body accesses field on captured object.
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | map(u -> u.name) }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable accumulation in loop ─────────────────────

#[test]
fn variable_accumulate_in_loop() {
    let i = Interner::new();
    // Write to variable on each iteration.
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ $sum = 0 }}{{ x in @items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-level pipe with to_string in middle ────────

#[test]
fn pipe_map_to_string_then_filter() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(i -> i + 1) | filter(i -> i != 0) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: local $-var captured in lambda ───────────────────

#[test]
fn lambda_capture_local_var_ref() {
    let i = Interner::new();
    // $offset is NOT in initial context — created as local $-var.
    // Lambda must capture it correctly (not fall through to StorageLoad).
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ $offset = 10 }}{{ x = @items | iter | filter(i -> i > $offset) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple field accesses on same lambda param ─────

#[test]
fn lambda_multiple_field_access() {
    let i = Interner::new();
    // Lambda body accesses two fields on the same param.
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | map(u -> (u.name, u.age)) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: chained field access in lambda ───────────────────

#[test]
fn lambda_chained_field_access() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(
                &i,
                &[("address", obj(&i, &[("city", Ty::String)]))],
            ))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | map(u -> u.address.city) }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: string concat in lambda ───────────────────────────

#[test]
fn lambda_string_concat() {
    let i = Interner::new();
    // Lambda param is Ty::Var; string concat (+) must resolve via unification.
    let context = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @names | iter | map(n -> n + "!") }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: filter then map with field access ────────────────

#[test]
fn pipe_filter_then_map_field() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | filter(u -> u.age > 18) | map(u -> u.name) }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: error — field access on non-object ────────────────

#[test]
fn error_field_access_on_int() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let result = compile_to_ir(
        &i,
        "{{ @n.foo | to_string }}",
        &context,
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Edge case: error — context write attempt ─────────────────────

#[test]
fn error_variable_write_type_mismatch() {
    let i = Interner::new();
    // Attempting to write to a context key (read-only).
    let context = ctx(&i, &[("count", Ty::Int)]);
    let result = compile_to_ir(
        &i,
        r#"{{ @count = "hello" }}"#,
        &context,
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Edge case: float arithmetic in lambda ────────────────────────

#[test]
fn lambda_float_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("vals", Ty::List(Box::new(Ty::Float)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @vals | iter | map(v -> v * 2.0) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: bool literal as match source ─────────────────────

#[test]
fn match_bool_literal() {
    let i = Interner::new();
    let context = ctx(&i, &[("flag", Ty::Bool)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ true = @flag }}on{{_}}off{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested pipe with filter on object field ──────────

#[test]
fn filter_object_field_equality() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(
                &i,
                &[("name", Ty::String), ("active", Ty::Bool)],
            ))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | filter(u -> u.active) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: extern function with object return ───────────────

#[test]
fn extern_fn_object_return() {
    let i = Interner::new();
    let context = ctx(&i, &[("get_user", Ty::Fn {
        params: vec![Ty::Int],
        ret: Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)])),
        kind: FnKind::Extern,
        captures: vec![],
        effect: Effect::Pure,
    })]);
    let ir = compile_to_ir(
        &i,
        r#"{{ u = @get_user(1) }}{{ u.name }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── New builtins ─────────────────────────────────────────────────

#[test]
fn builtin_len() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | len | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_reverse() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ x in @items | reverse }}{{ x | to_string }}{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_join() {
    let i = Interner::new();
    let context = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ @names | join(", ") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_contains() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | contains(3) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_find() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | find(x -> x > 10) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_reduce() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | reduce((a, b) -> a + b) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_fold() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | fold(0, (acc, x) -> acc + x) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_any() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | any(x -> x > 10) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_all() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | all(x -> x > 0) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Context Call ────────────────────────────────────────────────

// ── Variant (Option) ────────────────────────────────────────────

#[test]
fn variant_some_expr() {
    let i = Interner::new();
    let ir = compile_simple(&i, "{{ x = Some(42) }}{{_}}{{/}}").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_none_expr() {
    let i = Interner::new();
    let ir = compile_simple(&i, "{{ x = None }}{{_}}{{/}}").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_some_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ Some(v) = @opt }}{{ v | to_string }}{{_}}nope{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_none_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ None = @opt }}none{{_}}has value{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn structural_enum_variant_merge() {
    let i = Interner::new();
    let template = acvus_ast::parse(&i, "{{ A::B = @a }}hi{{/}}{{ A::C = @a }}bye{{/}}").unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i,
        &template,
        &ContextTypeRegistry::all_system(FxHashMap::default()),
    )
    .unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
    assert!(ir.contains("C"), "variant C missing from IR:\n{ir}");
}

// ── Structural enum tests ──────────────────────────────────────

#[test]
fn structural_enum_single_variant() {
    let i = Interner::new();
    let template = acvus_ast::parse(&i, "{{ A::B = @a }}yes{{_}}no{{/}}").unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
}

#[test]
fn structural_enum_three_variants_merge() {
    let i = Interner::new();
    let src = "{{ S::X = @v }}x{{/}}{{ S::Y = @v }}y{{/}}{{ S::Z = @v }}z{{/}}";
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
    assert!(ir.contains("Y"), "variant Y missing:\n{ir}");
    assert!(ir.contains("Z"), "variant Z missing:\n{ir}");
}

#[test]
fn structural_enum_with_payload() {
    let i = Interner::new();
    let src = r#"{{ R::Ok(v) = @r }}{{ v | to_string }}{{_}}err{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Ok"), "variant Ok missing:\n{ir}");
}

#[test]
fn structural_enum_mixed_payload_and_unit() {
    let i = Interner::new();
    let src = r#"{{ R::Ok(v) = @r }}{{ v | to_string }}{{ R::Err = }}fail{{_}}??{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Ok"), "variant Ok missing:\n{ir}");
    assert!(ir.contains("Err"), "variant Err missing:\n{ir}");
}

#[test]
fn structural_enum_same_var_different_blocks_merge() {
    // Key regression test: separate match blocks on the same context var must merge.
    let i = Interner::new();
    let src = "{{ A::B = @a }}b{{/}}{{ A::C = @a }}c{{/}}";
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
    assert!(ir.contains("C"), "variant C missing:\n{ir}");
}

#[test]
fn structural_enum_different_enums_different_vars() {
    let i = Interner::new();
    let src = "{{ X::A = @x }}xa{{/}}{{ Y::B = @y }}yb{{/}}";
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("A"), "variant A missing:\n{ir}");
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
}

#[test]
fn structural_enum_name_mismatch_is_error() {
    // Matching X::A and Y::B on the same var should fail (different enum names).
    let i = Interner::new();
    let src = "{{ X::A = @v }}a{{/}}{{ Y::B = @v }}b{{/}}";
    let template = acvus_ast::parse(&i, src).unwrap();
    let result = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    );
    assert!(result.is_err(), "should fail: different enum names on same var");
}

#[test]
fn structural_enum_payload_unifies_with_inner_match() {
    // Payload variable must unify with patterns inside the arm body.
    let i = Interner::new();
    let src = r#"{{ A::X(x) = @a }}{{ 0 = x }}zero{{_}}other{{/}}{{_}}none{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
}

#[test]
fn structural_enum_payload_unifies_with_emit() {
    // Payload bound by variant pattern can be used in expressions (emit).
    let i = Interner::new();
    let src = r#"{{ A::Val(v) = @a }}{{ v + 1 | to_string }}{{_}}n/a{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Val"), "variant Val missing:\n{ir}");
}

#[test]
fn structural_enum_payload_type_propagates_through_context() {
    // When context provides an enum type with payload, payload type should propagate.
    let i = Interner::new();
    let mut variants = FxHashMap::default();
    variants.insert(i.intern("Ok"), Some(Box::new(Ty::Int)));
    variants.insert(i.intern("Err"), None);
    let ctx_types = ctx(&i, &[("r", Ty::Enum { name: i.intern("R"), variants })]);
    let src = r#"{{ R::Ok(v) = @r }}{{ v + 1 | to_string }}{{ R::Err = }}err{{_}}??{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(ctx_types),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Ok"), "variant Ok missing:\n{ir}");
    assert!(ir.contains("Err"), "variant Err missing:\n{ir}");
}

// ── Variant unification inside Tuple/List patterns ─────────────
// Regression: nested Variant patterns inside Tuple/List must merge
// variant sets across match arms via the shared Var chain.

#[test]
fn variant_merge_inside_tuple_pattern() {
    // Two arms with different variants nested inside a tuple pattern.
    // Both A and B must appear in the final merged Enum type.
    let i = Interner::new();
    let src = r#"{{ (S::A, x) = @t }}{{ x }}{{ (S::B, y) = }}{{ y }}{{_}}??{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    eprintln!("=== TUPLE VARIANT IR ===\n{ir}\n=== END ===");
    assert!(ir.contains("A"), "variant A missing from IR:\n{ir}");
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
}

#[test]
fn variant_merge_inside_tuple_three_arms() {
    let i = Interner::new();
    let src = r#"{{ (S::X, _) = @t }}x{{ (S::Y, _) = }}y{{ (S::Z, _) = }}z{{_}}??{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
    assert!(ir.contains("Y"), "variant Y missing:\n{ir}");
    assert!(ir.contains("Z"), "variant Z missing:\n{ir}");
}

#[test]
fn variant_merge_inside_list_pattern() {
    // Variant inside list head pattern should merge across arms.
    let i = Interner::new();
    let src = r#"{{ [S::A, ..] = @lst }}a{{ [S::B, ..] = }}b{{_}}??{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("A"), "variant A missing:\n{ir}");
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
}

#[test]
fn infer_unifies_with_concrete_type() {
    // Ty::Infer in context types should not block unification.
    let i = Interner::new();
    let mut variants = FxHashMap::default();
    variants.insert(i.intern("X"), Some(Box::new(Ty::Infer)));
    let ctx_types = ctx(&i, &[("a", Ty::Enum { name: i.intern("A"), variants })]);
    let src = r#"{{ A::X(x) = @a }}{{ x + 1 | to_string }}{{_}}no{{/}}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    // Should compile without error — Infer payload unifies with Int from `x + 1`.
    acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(ctx_types),
    ).unwrap();
}

#[test]
fn pruned_context_keys_in_dead_catch_all() {
    // Outer match on @Impersonation with catch_all containing @Pov.
    // When Impersonation=NoPersona is known, catch_all is dead,
    // so @Pov should appear in partition.pruned.
    use acvus_mir_pass::AnalysisPass;
    use acvus_mir_pass::analysis::val_def::ValDefMapAnalysis;
    use acvus_mir_pass::analysis::reachable_context::{partition_context_keys, KnownValue};

    let i = Interner::new();
    let src = r#"{-{ Impersonation::NoPersona = @Impersonation }}{-{_}}{-{ Pov::User = @Pov }}user{-{ Pov::Char = }}char{-{_}}other{-{ / }}{-{ / }}"#;
    let template = acvus_ast::parse(&i, src).unwrap();
    let (module, _) = acvus_mir::compile_analysis(
        &i, &template, &ContextTypeRegistry::all_system(FxHashMap::default()),
    ).unwrap();

    let ir = acvus_mir::printer::dump_with(&i, &module);
    eprintln!("=== PRUNED TEST IR ===\n{ir}\n=== END ===");

    let val_def = ValDefMapAnalysis.run(&module, ());
    let mut known = FxHashMap::default();
    known.insert(
        i.intern("Impersonation"),
        KnownValue::Variant { tag: i.intern("NoPersona"), payload: None },
    );
    let partition = partition_context_keys(&module, &known, &val_def);

    eprintln!("eager: {:?}", partition.eager.iter().map(|a| i.resolve(*a)).collect::<Vec<_>>());
    eprintln!("lazy: {:?}", partition.lazy.iter().map(|a| i.resolve(*a)).collect::<Vec<_>>());
    eprintln!("reachable_known: {:?}", partition.reachable_known.iter().map(|a| i.resolve(*a)).collect::<Vec<_>>());
    eprintln!("pruned: {:?}", partition.pruned.iter().map(|a| i.resolve(*a)).collect::<Vec<_>>());

    assert!(
        partition.pruned.contains(&i.intern("Pov")),
        "Pov should be pruned when Impersonation=NoPersona, but got:\n  eager: {:?}\n  lazy: {:?}\n  pruned: {:?}",
        partition.eager.iter().map(|a| i.resolve(*a)).collect::<Vec<_>>(),
        partition.lazy.iter().map(|a| i.resolve(*a)).collect::<Vec<_>>(),
        partition.pruned.iter().map(|a| i.resolve(*a)).collect::<Vec<_>>(),
    );
}
