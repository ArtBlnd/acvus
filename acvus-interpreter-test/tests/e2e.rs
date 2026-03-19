use acvus_interpreter::{
    CollectionOp, Interpreter, LazyValue,
    RuntimeErrorKind, TypedValue, Value,
};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// Helper: build a typed context from (name, TypedValue) pairs.
fn typed_ctx(i: &Interner, pairs: &[(&str, TypedValue)]) -> FxHashMap<acvus_utils::Astr, TypedValue> {
    pairs
        .iter()
        .map(|(k, v)| (i.intern(k), v.clone()))
        .collect()
}

/// Helper: build a TypedValue from a Ty and a Value.
fn typed_val(ty: Ty, val: Value) -> TypedValue {
    TypedValue::new(val, ty)
}

/// Helper: build an Object type from string-keyed fields.
fn obj_ty(i: &Interner, fields: &[(&str, Ty)]) -> Ty {
    Ty::Object(
        fields
            .iter()
            .map(|(k, v)| (i.intern(k), v.clone()))
            .collect(),
    )
}

/// Helper: build an Object value from string-keyed fields.
fn obj_val(i: &Interner, fields: &[(&str, Value)]) -> Value {
    Value::object(
        fields
            .iter()
            .map(|(k, v)| (i.intern(k), v.clone()))
            .collect(),
    )
}

// ── Text & literals ──────────────────────────────────────────────

#[tokio::test]
async fn text_only() {
    assert_eq!(run_simple("hello world").await, "hello world");
}

#[tokio::test]
async fn string_emit() {
    assert_eq!(run_simple(r#"{{ "hello" }}"#).await, "hello");
}

#[tokio::test]
async fn string_concat() {
    assert_eq!(
        run_simple(r#"{{ "hello" + " " + "world" }}"#).await,
        "hello world"
    );
}

#[tokio::test]
async fn mixed_text_and_expr() {
    let i = Interner::new();
    let val = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(&i, "Hello, {{ @name }}!".into(), val).await,
        "Hello, alice!"
    );
}

// ── Context / Variables ─────────────────────────────────────────

#[tokio::test]
async fn context_read() {
    let i = Interner::new();
    let val = int_context(&i, "count", 42);
    assert_eq!(
        run_ctx(&i, "{{ @count | to_string }}".into(), val).await,
        "42"
    );
}

#[tokio::test]
async fn variable_write() {
    assert_eq!(run_simple("{{ $count = 42 }}").await, "");
}

#[tokio::test]
async fn variable_write_then_read() {
    assert_eq!(run_simple("{{ $x = 42 }}{{ $x | to_string }}").await, "42");
}

#[tokio::test]
async fn context_field_access() {
    let i = Interner::new();
    let val = user_context(&i);
    assert_eq!(
        run_ctx(&i, "{{ @user.name }}".into(), val).await,
        "alice"
    );
}

#[tokio::test]
async fn variable_write_computed() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::int(10)), ("b", TypedValue::int(32))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ $result = @a + @b }}{{ $result | to_string }}",
            ctx
        )
        .await,
        "42"
    );
}

// ── Arithmetic ───────────────────────────────────────────────────

#[tokio::test]
async fn arithmetic_to_string() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::int(3)), ("b", TypedValue::int(7))]);
    assert_eq!(
        run_ctx(&i, "{{ @a + @b | to_string }}".into(), ctx).await,
        "10"
    );
}

#[tokio::test]
async fn unary_negation() {
    let i = Interner::new();
    let val = int_context(&i, "n", 5);
    assert_eq!(
        run_ctx(&i, r#"{{ x = -@n }}{{ x | to_string }}{{_}}{{/}}"#, val).await,
        "-5"
    );
}

#[tokio::test]
async fn boolean_not() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("flag", TypedValue::bool_(true))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = !@flag }}{{ x | to_string }}{{_}}{{/}}"#,
            ctx
        )
        .await,
        "false"
    );
}

#[tokio::test]
async fn comparison_operators() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::int(10)), ("b", TypedValue::int(5))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @a > @b }}{{ x | to_string }}{{_}}{{/}}"#,
            ctx
        )
        .await,
        "true"
    );
}

// ── Match blocks ─────────────────────────────────────────────────

#[tokio::test]
async fn simple_match_binding() {
    let i = Interner::new();
    let val = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(&i, r#"{{ x = @name }}{{ x }}"#, val).await,
        "alice"
    );
}

#[tokio::test]
async fn match_literal_filter_hit() {
    let i = Interner::new();
    let val = string_context(&i, "role", "admin");
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ "admin" = @role }}admin page{{_}}guest page{{/}}"#,
            val
        )
        .await,
        "admin page"
    );
}

#[tokio::test]
async fn match_literal_filter_miss() {
    let i = Interner::new();
    let val = string_context(&i, "role", "user");
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ "admin" = @role }}admin page{{_}}guest page{{/}}"#,
            val
        )
        .await,
        "guest page"
    );
}

#[tokio::test]
async fn multi_arm_match() {
    let i = Interner::new();
    let val = string_context(&i, "role", "user");
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ "admin" = @role }}admin{{ "user" = }}user{{_}}guest{{/}}"#,
            val
        )
        .await,
        "user"
    );
}

#[tokio::test]
async fn match_bool_literal() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("flag", TypedValue::bool_(true))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @flag }}on{{_}}off{{/}}"#, ctx).await,
        "on"
    );
}

#[tokio::test]
async fn match_binding_with_body() {
    let i = Interner::new();
    let val = user_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ { name, } = @user }}{{ name }} is here{{_}}no user{{/}}"#,
            val
        )
        .await,
        "alice is here"
    );
}

#[tokio::test]
async fn variable_shadowing() {
    let i = Interner::new();
    let val = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = "outer" }}{{ x = @name }}{{ x }}{{_}}{{/}}"#,
            val
        )
        .await,
        "alice"
    );
}

#[tokio::test]
async fn catch_all_with_binding() {
    let i = Interner::new();
    let val = string_context(&i, "role", "viewer");
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ "admin" = @role }}admin{{_}}{{ fallback = "guest" }}{{ fallback }}{{/}}"#,
            val
        )
        .await,
        "guest"
    );
}

#[tokio::test]
async fn equality_as_match_source() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::int(5)), ("b", TypedValue::int(5))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ true = @a == @b }}equal{{_}}not equal{{/}}"#,
            ctx
        )
        .await,
        "equal"
    );
}

// ── Nested match blocks ──────────────────────────────────────────

#[tokio::test]
async fn nested_match_blocks() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("role", TypedValue::string("admin")), ("level", TypedValue::int(5))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ "admin" = @role }}{{ 1..10 = @level }}low{{_}}high{{/}}{{_}}guest{{/}}"#,
            ctx
        )
        .await,
        "low"
    );
}

// ── Variable ref in match arm ────────────────────────────────────

#[tokio::test]
async fn variable_new_ref_binding() {
    let i = Interner::new();
    let val = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(&i, r#"{{ $result = @name }}{{ $result }}"#, val).await,
        "alice"
    );
}

#[tokio::test]
async fn variable_new_ref_in_match_arm() {
    let i = Interner::new();
    let val = string_context(&i, "role", "admin");
    assert_eq!(
        run_ctx(&i,
            r#"{{ "admin" = @role }}{{ $selected = "yes" }}{{_}}{{ $selected = "no" }}{{/}}{{ $selected }}"#,
            val,
        )
        .await,
        "yes"
    );
}

// ── Range ────────────────────────────────────────────────────────

#[tokio::test]
async fn range_binding() {
    assert_eq!(
        run_simple(r#"{{ x in 0..5 }}{{ x | to_string }}{{/}}"#).await,
        "01234"
    );
}

#[tokio::test]
async fn range_iteration() {
    assert_eq!(
        run_simple(r#"{{ x in 0..3 }}{{ x | to_string }}{{/}}"#).await,
        "012"
    );
}

#[tokio::test]
async fn range_inclusive_iteration() {
    assert_eq!(
        run_simple(r#"{{ x in 0..=3 }}{{ x | to_string }}{{/}}"#).await,
        "0123"
    );
}

#[tokio::test]
async fn range_pattern_hit() {
    let i = Interner::new();
    let val = int_context(&i, "age", 5);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ 0..10 = @age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
            val
        )
        .await,
        "child"
    );
}

#[tokio::test]
async fn range_pattern_miss() {
    let i = Interner::new();
    let val = int_context(&i, "age", 25);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ 0..10 = @age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
            val
        )
        .await,
        "adult"
    );
}

// ── Iteration ────────────────────────────────────────────────────

#[tokio::test]
async fn iter_list_binding() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ x in @items }}{{ x | to_string }}{{/}}".into(),
            val
        )
        .await,
        "123"
    );
}

#[tokio::test]
async fn iter_object_destructure() {
    let i = Interner::new();
    let val = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            "{{ { name, } in @users }}{{ name }}{{/}}".into(),
            val
        )
        .await,
        "alicebob"
    );
}

#[tokio::test]
async fn iter_tuple_destructure() {
    let i = Interner::new();
    let val = typed_ctx(
        &i,
        &[(
            "pairs",
            typed_val(
                Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
                Value::list(vec![
                    Value::tuple(vec![Value::string("a"), Value::int(1)]),
                    Value::tuple(vec![Value::string("b"), Value::int(2)]),
                ]),
            ),
        )],
    );
    assert_eq!(
        run_ctx(&i, "{{ (a, _) in @pairs }}{{ a }}{{/}}".into(), val).await,
        "ab"
    );
}

#[tokio::test]
async fn nested_iteration() {
    let i = Interner::new();
    let val = typed_ctx(
        &i,
        &[(
            "matrix",
            typed_val(
                Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
                Value::list(vec![
                    Value::list(vec![Value::int(1), Value::int(2)]),
                    Value::list(vec![Value::int(3), Value::int(4)]),
                ]),
            ),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            "{{ row in @matrix }}{{ x in row }}{{ x | to_string }}{{/}}{{/}}",
            val
        )
        .await,
        "1234"
    );
}

#[tokio::test]
async fn variable_write_in_iteration() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ $last = 0 }}{{ x in @items }}{{ $last = x }}{{/}}{{ $last | to_string }}",
            val
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn variable_accumulate_in_loop() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ $sum = 0 }}{{ x in @items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}",
            val
        )
        .await,
        "6"
    );
}

// ── List patterns ────────────────────────────────────────────────

#[tokio::test]
async fn list_destructure_head() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
            val
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn list_destructure_tail() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ [.., a, b] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
            val
        )
        .await,
        "20"
    );
}

#[tokio::test]
async fn list_destructure_head_and_tail() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ [first, .., last] = @items }}{{ first | to_string }}-{{ last | to_string }}{{_}}empty{{/}}"#,
            val,
        )
        .await,
        "1-5"
    );
}

#[tokio::test]
async fn list_exact_match_hit() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ [a, b] = @items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
            val
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn list_exact_match_miss() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ [a, b] = @items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
            val
        )
        .await,
        "wrong length"
    );
}

// ── Object patterns ──────────────────────────────────────────────

#[tokio::test]
async fn object_pattern() {
    let i = Interner::new();
    let val = user_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ { name, age, } = @user }}{{ name }}:{{ age | to_string }}{{/}}"#,
            val
        )
        .await,
        "alice:30"
    );
}

#[tokio::test]
async fn deeply_nested_object_access() {
    let i = Interner::new();
    let data_ty = obj_ty(
        &i,
        &[(
            "user",
            obj_ty(&i, &[("address", obj_ty(&i, &[("city", Ty::String)]))]),
        )],
    );
    let data_val = obj_val(
        &i,
        &[(
            "user",
            obj_val(
                &i,
                &[(
                    "address",
                    obj_val(&i, &[("city", Value::string("Seoul"))]),
                )],
            ),
        )],
    );
    let val = typed_ctx(&i, &[("data", typed_val(data_ty, data_val))]);
    assert_eq!(
        run_ctx(&i, "{{ @data.user.address.city }}".into(), val).await,
        "Seoul"
    );
}

// ── Tuple ────────────────────────────────────────────────────────

#[tokio::test]
async fn tuple_expression() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::int(42)), ("b", TypedValue::string("hello"))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ (x, y) = (@a, @b) }}{{ x | to_string }}, {{ y }}{{/}}"#,
            ctx
        )
        .await,
        "42, hello"
    );
}

#[tokio::test]
async fn tuple_pattern_binding() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "pair",
        typed_val(
            Ty::Tuple(vec![Ty::String, Ty::Int]),
            Value::tuple(vec![Value::string("alice"), Value::int(30)]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ (name, age) = @pair }}{{ name }}{{/}}"#,
            ctx
        )
        .await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_wildcard() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "pair",
        typed_val(
            Ty::Tuple(vec![Ty::String, Ty::Int]),
            Value::tuple(vec![Value::string("alice"), Value::int(30)]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ (name, _) = @pair }}{{ name }}{{/}}"#,
            ctx
        )
        .await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_literal_match_hit() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::int(0)), ("b", TypedValue::int(1))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ (0, 1) = (@a, @b) }}zero-one{{ (1, _) = }}one-any{{_}}other{{/}}"#,
            ctx
        )
        .await,
        "zero-one"
    );
}

#[tokio::test]
async fn nested_tuple_pattern() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "data",
        typed_val(
            Ty::Tuple(vec![Ty::Tuple(vec![Ty::Int, Ty::Int]), Ty::String]),
            Value::tuple(vec![
                Value::tuple(vec![Value::int(1), Value::int(2)]),
                Value::string("hello"),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ ((a, b), label) = @data }}{{ label }}{{/}}"#,
            ctx
        )
        .await,
        "hello"
    );
}

// ── Pipe & builtins ──────────────────────────────────────────────

#[tokio::test]
async fn pipe_to_string() {
    let i = Interner::new();
    let val = int_context(&i, "n", 42);
    assert_eq!(
        run_ctx(&i, "{{ @n | to_string }}".into(), val).await,
        "42"
    );
}

#[tokio::test]
async fn to_float_conversion() {
    let i = Interner::new();
    let val = int_context(&i, "n", 5);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @n | to_float }}{{ x | to_string }}{{_}}{{/}}"#,
            val
        )
        .await,
        "5"
    );
}

#[tokio::test]
async fn to_int_conversion() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("f", TypedValue::float(3.7))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @f | to_int }}{{ x | to_string }}{{_}}{{/}}"#,
            ctx
        )
        .await,
        "3"
    );
}

// ── Lambda / closures ────────────────────────────────────────────

#[tokio::test]
async fn lambda_filter() {
    let i = Interner::new();
    let val = items_context(&i, vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(x -> x != 0) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            val
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn lambda_map() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @items | map(i -> i + 1) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            val
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn lambda_pmap() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @items | pmap(i -> (i | to_string)) | collect }}{{ x | join(", ") }}"#,
            val
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn pipe_filter_map() {
    let i = Interner::new();
    let val = items_context(&i, vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(x -> x != 0) | map(x -> (x | to_string)) | collect }}{{ x | join(", ") }}"#,
            val,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn triple_pipe_chain() {
    let i = Interner::new();
    let val = items_context(&i, vec![0, 1, 2, 3]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(i -> i != 0) | map(i -> i + 1) | map(i -> (i | to_string)) | collect }}{{ x | join(", ") }}"#,
            val,
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn closure_capture_local() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 3, 5, 7, 10]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ threshold = 5 }}{{ x = @items | filter(i -> i > threshold) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            val,
        )
        .await,
        "7, 10"
    );
}

#[tokio::test]
async fn closure_capture_context() {
    let i = Interner::new();
    let ctx = typed_ctx(
        &i,
        &[
            (
                "items",
                typed_val(
                    Ty::List(Box::new(Ty::Int)),
                    Value::list(vec![Value::int(1), Value::int(5), Value::int(10)]),
                ),
            ),
            ("threshold", TypedValue::int(3)),
        ],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(i -> i > @threshold) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            ctx
        )
        .await,
        "5, 10"
    );
}

#[tokio::test]
async fn lambda_field_access() {
    let i = Interner::new();
    let val = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @users | map(u -> u.name) | collect }}{{ x | join(", ") }}"#,
            val
        )
        .await,
        "alice, bob"
    );
}

#[tokio::test]
async fn lambda_negate_param() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @items | map(i -> -i) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            val
        )
        .await,
        "-1, -2, -3"
    );
}

#[tokio::test]
async fn lambda_not_param() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "flags",
        typed_val(
            Ty::List(Box::new(Ty::Bool)),
            Value::list(vec![
                Value::bool_(true),
                Value::bool_(false),
                Value::bool_(true),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @flags | map(i -> !i) | collect }}{{ x | map(b -> (b | to_string)) | collect | join(", ") }}"#,
            ctx
        )
        .await,
        "false, true, false"
    );
}

#[tokio::test]
async fn lambda_string_concat() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "names",
        typed_val(
            Ty::List(Box::new(Ty::String)),
            Value::list(vec![
                Value::string("alice"),
                Value::string("bob"),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @names | map(n -> n + "!") | collect }}{{ x | join(", ") }}"#,
            ctx
        )
        .await,
        "alice!, bob!"
    );
}

#[tokio::test]
async fn lambda_float_arithmetic() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "vals",
        typed_val(
            Ty::List(Box::new(Ty::Float)),
            Value::list(vec![Value::float(1.5), Value::float(2.5)]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @vals | map(v -> v * 2.0) | collect }}{{ x | map(v -> (v | to_string)) | collect | join(", ") }}"#,
            ctx
        )
        .await,
        "3, 5"
    );
}

#[tokio::test]
async fn filter_then_map_field() {
    let i = Interner::new();
    let val = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @users | filter(u -> u.age > 18) | map(u -> u.name) | collect }}{{ x | join(", ") }}"#,
            val,
        )
        .await,
        "alice, bob"
    );
}

#[tokio::test]
async fn multiple_closures_same_capture() {
    let i = Interner::new();
    let ctx = typed_ctx(
        &i,
        &[
            (
                "items",
                typed_val(
                    Ty::List(Box::new(Ty::Int)),
                    Value::list(vec![
                        Value::int(-1),
                        Value::int(0),
                        Value::int(1),
                        Value::int(2),
                    ]),
                ),
            ),
            ("offset", TypedValue::int(1)),
        ],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | map(i -> i + @offset) | filter(i -> i > 0) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "1, 2, 3"
    );
}


// ── Logical operators (&&, ||) ───────────────────────────────────

#[tokio::test]
async fn and_both_true() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::bool_(true)), ("b", TypedValue::bool_(true))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a && @b }}yes{{_}}no{{/}}"#, ctx).await,
        "yes"
    );
}

#[tokio::test]
async fn and_one_false() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::bool_(true)), ("b", TypedValue::bool_(false))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a && @b }}yes{{_}}no{{/}}"#, ctx).await,
        "no"
    );
}

#[tokio::test]
async fn or_one_true() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::bool_(false)), ("b", TypedValue::bool_(true))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a || @b }}yes{{_}}no{{/}}"#, ctx).await,
        "yes"
    );
}

#[tokio::test]
async fn or_both_false() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("a", TypedValue::bool_(false)), ("b", TypedValue::bool_(false))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a || @b }}yes{{_}}no{{/}}"#, ctx).await,
        "no"
    );
}

#[tokio::test]
async fn and_or_precedence() {
    let i = Interner::new();
    // a || b && c => a || (b && c) — && binds tighter
    let ctx = typed_ctx(&i, &[
        ("a", TypedValue::bool_(true)),
        ("b", TypedValue::bool_(false)),
        ("c", TypedValue::bool_(false)),
    ]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ true = @a || @b && @c }}yes{{_}}no{{/}}"#,
            ctx
        )
        .await,
        "yes"
    );
}

#[tokio::test]
async fn and_with_comparison() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("x", TypedValue::int(15))]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ true = @x > 10 && @x < 20 }}in range{{_}}out{{/}}"#,
            ctx
        )
        .await,
        "in range"
    );
}

#[tokio::test]
async fn logical_in_filter() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 5, 10, 15, 20, 25]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(i -> i > 5 && i < 20) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            val
        )
        .await,
        "10, 15"
    );
}

// ── Complex scenarios ───────────────────────────────────────────

#[tokio::test]
async fn nested_match_with_variable_write() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("role", TypedValue::string("admin")), ("level", TypedValue::int(5))]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ "admin" = @role }}{{ 0..10 = @level }}{{ $result = "low-admin" }}{{_}}{{ $result = "high-admin" }}{{/}}{{_}}{{ $result = "guest" }}{{/}}{{ $result }}"#,
            ctx
        )
        .await,
        "low-admin"
    );
}

#[tokio::test]
async fn filter_map_with_object_pattern() {
    let i = Interner::new();
    let products_ty = Ty::List(Box::new(obj_ty(
        &i,
        &[("name", Ty::String), ("price", Ty::Int)],
    )));
    let products_val = Value::list(vec![
        obj_val(
            &i,
            &[
                ("name", Value::string("apple")),
                ("price", Value::int(100)),
            ],
        ),
        obj_val(
            &i,
            &[
                ("name", Value::string("banana")),
                ("price", Value::int(50)),
            ],
        ),
        obj_val(
            &i,
            &[
                ("name", Value::string("cherry")),
                ("price", Value::int(200)),
            ],
        ),
    ]);
    let val = typed_ctx(&i, &[("products", typed_val(products_ty, products_val))]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @products | filter(p -> p.price >= 100) | map(p -> p.name) | collect }}{{ x | join(", ") }}"#,
            val,
        )
        .await,
        "apple, cherry"
    );
}

#[tokio::test]
async fn iteration_with_match_per_item() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x in @items }}{{ 1..=3 = x }}s{{_}}b{{/}}{{/}}"#,
            val
        )
        .await,
        "sssbb"
    );
}

#[tokio::test]
async fn multi_context_interaction() {
    let i = Interner::new();
    let ctx = typed_ctx(
        &i,
        &[
            (
                "items",
                typed_val(
                    Ty::List(Box::new(Ty::Int)),
                    Value::list(vec![
                        Value::int(3),
                        Value::int(7),
                        Value::int(1),
                        Value::int(9),
                        Value::int(4),
                    ]),
                ),
            ),
            ("min", TypedValue::int(2)),
            ("max", TypedValue::int(8)),
        ],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ filtered = @items | filter(i -> i >= @min && i <= @max) | collect }}{{ $count = 0 }}{{ x in filtered }}{{ $count = $count + 1 }}{{/}}{{ $count | to_string }}{{_}}{{/}}"#,
            ctx,
        )
        .await,
        "3"
    );
}

#[tokio::test]
async fn object_destructure_in_iteration_with_emit() {
    let i = Interner::new();
    let val = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ { name, age, } in @users }}{{ name }}({{ age | to_string }}) {{/}}"#,
            val
        )
        .await,
        "alice(30) bob(25) "
    );
}

#[tokio::test]
async fn chained_pipe_with_logical_filter() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "nums",
        typed_val(
            Ty::List(Box::new(Ty::Int)),
            Value::list(vec![
                Value::int(-5),
                Value::int(0),
                Value::int(3),
                Value::int(7),
                Value::int(12),
                Value::int(20),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @nums | filter(n -> n > 0 && n < 10) | map(n -> n * n) | collect }}{{ x | map(n -> (n | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "9, 49"
    );
}

#[tokio::test]
async fn nested_list_iteration_with_accumulator() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "matrix",
        typed_val(
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
            Value::list(vec![
                Value::list(vec![Value::int(1), Value::int(2)]),
                Value::list(vec![Value::int(3), Value::int(4)]),
                Value::list(vec![Value::int(5), Value::int(6)]),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(&i,
            "{{ $sum = 0 }}{{ row in @matrix }}{{ x in row }}{{ $sum = $sum + x }}{{/}}{{/}}{{ $sum | to_string }}",
            ctx
        )
        .await,
        "21"
    );
}

#[tokio::test]
async fn map_then_iterate_with_match() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ doubled = @items | map(i -> i * 2) | collect }}{{ x in doubled }}{{ true = x > 6 }}{{ x | to_string }} {{_}}{{/}}{{/}}"#,
            val,
        )
        .await,
        "8 10 "
    );
}


#[tokio::test]
async fn complex_object_filter_format() {
    let i = Interner::new();
    let users_ty = Ty::List(Box::new(obj_ty(
        &i,
        &[("name", Ty::String), ("age", Ty::Int), ("active", Ty::Bool)],
    )));
    let users_val = Value::list(vec![
        obj_val(
            &i,
            &[
                ("name", Value::string("alice")),
                ("age", Value::int(30)),
                ("active", Value::bool_(true)),
            ],
        ),
        obj_val(
            &i,
            &[
                ("name", Value::string("bob")),
                ("age", Value::int(17)),
                ("active", Value::bool_(true)),
            ],
        ),
        obj_val(
            &i,
            &[
                ("name", Value::string("carol")),
                ("age", Value::int(25)),
                ("active", Value::bool_(false)),
            ],
        ),
        obj_val(
            &i,
            &[
                ("name", Value::string("dave")),
                ("age", Value::int(40)),
                ("active", Value::bool_(true)),
            ],
        ),
    ]);
    let val = typed_ctx(&i, &[("users", typed_val(users_ty, users_val))]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ eligible = @users | filter(u -> u.active && u.age >= 18) | map(u -> u.name) | collect }}{{ name in eligible }}{{ name }} {{/}}"#,
            val,
        )
        .await,
        "alice dave "
    );
}

// ── List literal ─────────────────────────────────────────────────

#[tokio::test]
async fn list_literal_expression() {
    assert_eq!(
        run_simple(
            r#"{{ x = [1, 2, 3] }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}{{_}}{{/}}"#
        )
        .await,
        "1, 2, 3"
    );
}

// ── Multi-arm with range ─────────────────────────────────────────

#[tokio::test]
async fn multi_arm_range_and_literal() {
    let i = Interner::new();
    let val = int_context(&i, "score", 0);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            val
        )
        .await,
        "zero"
    );
}

#[tokio::test]
async fn multi_arm_range_and_literal_low() {
    let i = Interner::new();
    let val = int_context(&i, "score", 5);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            val
        )
        .await,
        "low"
    );
}

#[tokio::test]
async fn multi_arm_range_and_literal_high() {
    let i = Interner::new();
    let val = int_context(&i, "score", 50);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            val
        )
        .await,
        "high"
    );
}

// ── Variant (Option) ────────────────────────────────────────────

#[tokio::test]
async fn variant_some_extract_value() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "opt",
        typed_val(
            Ty::Option(Box::new(Ty::String)),
            Value::variant(i.intern("Some"), Some(Box::new(Value::string("hello")))),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ Some(value) = @opt }}{{ value }}{{_}}empty{{/}}",
            ctx
        )
        .await,
        "hello"
    );
}

#[tokio::test]
async fn variant_none_match() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "opt",
        typed_val(
            Ty::Option(Box::new(Ty::Int)),
            Value::variant(i.intern("None"), None),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @opt }}none{{_}}has value{{/}}".into(),
            ctx
        )
        .await,
        "none"
    );
}

#[tokio::test]
async fn variant_some_catch_all() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "opt",
        typed_val(
            Ty::Option(Box::new(Ty::Int)),
            Value::variant(i.intern("None"), None),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ Some(v) = @opt }}{{ v | to_string }}{{_}}no value{{/}}",
            ctx
        )
        .await,
        "no value"
    );
}

#[tokio::test]
async fn variant_some_with_literal_pattern() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "opt",
        typed_val(
            Ty::Option(Box::new(Ty::Int)),
            Value::variant(i.intern("Some"), Some(Box::new(Value::int(42)))),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ Some(42) = @opt }}matched{{_}}no{{/}}".into(),
            ctx
        )
        .await,
        "matched"
    );
}

#[tokio::test]
async fn variant_construct_some() {
    assert_eq!(
        run_simple("{{ x = Some(42) }}{{ Some(v) = x }}{{ v | to_string }}{{_}}{{/}}{{_}}{{/}}")
            .await,
        "42"
    );
}

#[tokio::test]
async fn variant_construct_none() {
    assert_eq!(
        run_simple("{{ x = None }}{{ None = x }}none{{_}}some{{/}}{{_}}{{/}}").await,
        "none"
    );
}

#[tokio::test]
async fn to_utf8_returns_option_and_unwrap() {
    // valid utf8: to_utf8 returns Some, unwrap extracts the string
    assert_eq!(
        run_simple(r#"{{ "hello" | to_bytes | to_utf8 | unwrap }}"#).await,
        "hello"
    );
}

#[tokio::test]
async fn to_utf8_none_on_invalid() {
    let i = Interner::new();
    // 0xFF is not valid utf8 -> to_utf8 returns None
    let ctx = typed_ctx(&i, &[(
        "data",
        typed_val(
            Ty::bytes(),
            Value::list(vec![Value::byte(0xFF), Value::byte(0xFE)]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @data | to_utf8 }}invalid{{_}}valid{{/}}",
            ctx,
        )
        .await,
        "invalid"
    );
}

// ── Error propagation ───────────────────────────────────────────

/// HOF find on empty list -> Stepped::Error (not panic).
#[tokio::test]
async fn error_find_empty_list() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("items", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![])))]);

    let err = run_expect_error(
        &i,
        r#"{{ x = @items | find(x -> x == 99) }}{{ x | to_string }}{{_}}{{/}}"#,
        ctx,
    )
    .await;

    assert!(
        matches!(err.kind, RuntimeErrorKind::EmptyCollection { op: CollectionOp::Find }),
        "expected EmptyCollection error, got: {err}",
    );
}

/// HOF reduce on empty list -> Stepped::Error (not panic).
#[tokio::test]
async fn error_reduce_empty_list() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("items", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![])))]);

    let err = run_expect_error(
        &i,
        r#"{{ x = @items | reduce((a, b) -> a + b) }}{{ x | to_string }}{{_}}{{/}}"#,
        ctx,
    )
    .await;

    assert!(
        matches!(err.kind, RuntimeErrorKind::EmptyCollection { op: CollectionOp::Reduce }),
        "expected EmptyCollection error, got: {err}",
    );
}

// ── Structural enum runtime tests ──────────────────────────────

#[tokio::test]
async fn structural_enum_match_unit_variant() {
    let i = Interner::new();
    let status_ty = Ty::Enum {
        name: i.intern("Status"),
        variants: FxHashMap::from_iter([
            (i.intern("Active"), None),
            (i.intern("Inactive"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[("s", typed_val(status_ty, Value::variant(i.intern("Active"), None)))]);
    assert_eq!(
        run_ctx(&i, "{{ Status::Active = @s }}yes{{_}}no{{/}}", ctx).await,
        "yes"
    );
}

#[tokio::test]
async fn structural_enum_match_fallthrough() {
    let i = Interner::new();
    let status_ty = Ty::Enum {
        name: i.intern("Status"),
        variants: FxHashMap::from_iter([
            (i.intern("Active"), None),
            (i.intern("Inactive"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[("s", typed_val(status_ty, Value::variant(i.intern("Inactive"), None)))]);
    assert_eq!(
        run_ctx(&i, "{{ Status::Active = @s }}yes{{_}}no{{/}}", ctx).await,
        "no"
    );
}

#[tokio::test]
async fn structural_enum_multi_arm_match() {
    let i = Interner::new();
    let color_ty = Ty::Enum {
        name: i.intern("Color"),
        variants: FxHashMap::from_iter([
            (i.intern("Red"), None),
            (i.intern("Green"), None),
            (i.intern("Blue"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[("c", typed_val(color_ty, Value::variant(i.intern("Green"), None)))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ Color::Red = @c }}r{{ Color::Green = }}g{{ Color::Blue = }}b{{/}}",
            ctx,
        ).await,
        "g"
    );
}

#[tokio::test]
async fn structural_enum_with_payload_match() {
    let i = Interner::new();
    let res_ty = Ty::Enum {
        name: i.intern("Res"),
        variants: FxHashMap::from_iter([
            (i.intern("Ok"), Some(Box::new(Ty::String))),
            (i.intern("Err"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[("r", typed_val(res_ty, Value::variant(i.intern("Ok"), Some(Box::new(Value::string("hello"))))))]);
    assert_eq!(
        run_ctx(&i, "{{ Res::Ok(v) = @r }}{{ v }}{{_}}err{{/}}", ctx).await,
        "hello"
    );
}

#[tokio::test]
async fn structural_enum_payload_fallthrough_to_unit() {
    let i = Interner::new();
    let res_ty = Ty::Enum {
        name: i.intern("Res"),
        variants: FxHashMap::from_iter([
            (i.intern("Ok"), Some(Box::new(Ty::String))),
            (i.intern("Err"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[("r", typed_val(res_ty, Value::variant(i.intern("Err"), None)))]);
    assert_eq!(
        run_ctx(&i, "{{ Res::Ok(v) = @r }}{{ v }}{{ Res::Err = }}fail{{/}}", ctx).await,
        "fail"
    );
}

#[tokio::test]
async fn structural_enum_separate_blocks_both_match() {
    // Two separate match blocks on same enum context — regression test.
    let i = Interner::new();
    let ab_ty = Ty::Enum {
        name: i.intern("AB"),
        variants: FxHashMap::from_iter([
            (i.intern("A"), None),
            (i.intern("B"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[("s", typed_val(ab_ty, Value::variant(i.intern("B"), None)))]);
    assert_eq!(
        run_ctx(&i, "{{ AB::A = @s }}a{{/}}{{ AB::B = @s }}b{{/}}", ctx).await,
        "b"
    );
}

// ── Script with output_ty hint (Val(N) not yet defined repro) ───

/// Helper: compile a script with output_ty hint and execute it.
async fn run_script_with_hint(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<acvus_utils::Astr, Ty>,
    hint: Option<&Ty>,
) -> Value {
    let script = acvus_ast::parse_script(interner, source).expect("parse failed");
    let (module, _hints, _tail_ty) =
        acvus_mir::compile_script_with_hint(interner, &script, &acvus_mir::context_registry::ContextTypeRegistry::all_system(context_types.clone()), hint)
            .expect("compile failed");

    let interp = Interpreter::new(interner, module);
    let emits = interp.execute_with_context(FxHashMap::default()).await;
    assert!(emits.len() <= 1, "script emitted {} values, expected at most 1", emits.len());
    emits.into_iter().next().map(|tv| tv.into_inner()).unwrap_or(Value::unit())
}

#[tokio::test]
async fn script_hint_enum_variant() {
    let i = Interner::new();
    let focus_ty = Ty::Enum {
        name: i.intern("Focus"),
        variants: FxHashMap::from_iter([
            (i.intern("User"), None),
            (i.intern("System"), None),
            (i.intern("Character"), None),
        ]),
    };
    let result = run_script_with_hint(&i, "Focus::User", &FxHashMap::default(), Some(&focus_ty)).await;
    match result {
        Value::Lazy(LazyValue::Variant { tag, .. }) => assert_eq!(i.resolve(tag), "User"),
        other => panic!("expected Variant, got {other:?}"),
    }
}

#[tokio::test]
async fn script_hint_bool() {
    let i = Interner::new();
    let result = run_script_with_hint(&i, "false", &FxHashMap::default(), Some(&Ty::Bool)).await;
    assert_eq!(result, Value::bool_(false));
}

#[tokio::test]
async fn script_hint_string() {
    let i = Interner::new();
    let result = run_script_with_hint(&i, "\"\"", &FxHashMap::default(), Some(&Ty::String)).await;
    assert_eq!(result, Value::string(""));
}

#[tokio::test]
async fn script_hint_context_object_with_empty_lists() {
    let i = Interner::new();
    let entry_ty = Ty::Object(FxHashMap::from_iter([
        (i.intern("name"), Ty::String),
        (i.intern("description"), Ty::String),
        (i.intern("content"), Ty::String),
        (i.intern("content_type"), Ty::String),
    ]));
    let context_ty = Ty::Object(FxHashMap::from_iter([
        (i.intern("system"), Ty::List(Box::new(entry_ty.clone()))),
        (i.intern("character"), Ty::List(Box::new(entry_ty.clone()))),
        (i.intern("world_info"), Ty::List(Box::new(entry_ty.clone()))),
        (i.intern("lorebook"), Ty::List(Box::new(entry_ty.clone()))),
        (i.intern("memory"), Ty::List(Box::new(entry_ty.clone()))),
        (i.intern("custom"), Ty::List(Box::new(entry_ty))),
    ]));
    let result = run_script_with_hint(
        &i,
        "{system: [], character: [], world_info: [], lorebook: [], memory: [], custom: [],}",
        &FxHashMap::default(),
        Some(&context_ty),
    ).await;
    match result {
        Value::Lazy(LazyValue::Object(fields)) => {
            assert_eq!(fields.len(), 6);
        }
        other => panic!("expected Object, got {other:?}"),
    }
}

#[tokio::test]
async fn script_hint_enum_with_payload() {
    let i = Interner::new();
    let length_ty = Ty::Enum {
        name: i.intern("Length"),
        variants: FxHashMap::from_iter([
            (i.intern("Dynamic"), None),
            (i.intern("Short"), None),
            (i.intern("Medium"), None),
            (i.intern("Long"), None),
            (i.intern("Custom"), Some(Box::new(Ty::Int))),
        ]),
    };
    let result = run_script_with_hint(&i, "Length::Dynamic", &FxHashMap::default(), Some(&length_ty)).await;
    match result {
        Value::Lazy(LazyValue::Variant { tag, .. }) => assert_eq!(i.resolve(tag), "Dynamic"),
        other => panic!("expected Variant, got {other:?}"),
    }
}

#[tokio::test]
async fn script_no_hint_enum_variant() {
    // Same as above but WITHOUT hint — should also work (CLI path)
    let i = Interner::new();
    let result = run_script_with_hint(&i, "Focus::User", &FxHashMap::default(), None).await;
    match result {
        Value::Lazy(LazyValue::Variant { tag, .. }) => assert_eq!(i.resolve(tag), "User"),
        other => panic!("expected Variant, got {other:?}"),
    }
}

#[tokio::test]
async fn template_enum_multi_arm_match_full_type() {
    // Reproduces frontend path: context has FULL enum type (all variants)
    // Template matches multiple arms against it
    let i = Interner::new();
    let focus_ty = Ty::Enum {
        name: i.intern("Focus"),
        variants: FxHashMap::from_iter([
            (i.intern("User"), None),
            (i.intern("Char"), None),
            (i.intern("System"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[("Focus", typed_val(focus_ty.clone(), Value::variant(i.intern("User"), None)))]);

    let output = run_ctx(
        &i,
        "{-{ Focus::User = @Focus }}user{-{ Focus::Char = }}char{-{ Focus::System = }}sys{-{ / }}",
        ctx,
    ).await;
    assert_eq!(output, "user");
}

#[tokio::test]
async fn template_enum_tuple_match_full_type() {
    // Reproduces: {-{ (true, Impersonation::Deny) = (@Attempt, @Impersonation) }}
    let i = Interner::new();
    let imp_ty = Ty::Enum {
        name: i.intern("Impersonation"),
        variants: FxHashMap::from_iter([
            (i.intern("Deny"), None),
            (i.intern("Allowed"), None),
            (i.intern("AllowActionOnly"), None),
            (i.intern("NoPersona"), None),
        ]),
    };
    let ctx = typed_ctx(&i, &[
        ("Attempt", TypedValue::bool_(true)),
        ("Impersonation", typed_val(imp_ty.clone(), Value::variant(i.intern("Deny"), None))),
    ]);

    let output = run_ctx(
        &i,
        "{-{ (true, Impersonation::Deny) = (@Attempt, @Impersonation) }}deny{-{ (true, Impersonation::Allowed) = }}allow{-{ (false, Impersonation::Deny) = }}fdenial{-{ / }}",
        ctx,
    ).await;
    assert_eq!(output, "deny");
}

#[tokio::test]
async fn template_enum_match_with_payload_full_type() {
    // Focus::Custom({ custom, }) pattern — enum with payload
    let i = Interner::new();
    let focus_ty = Ty::Enum {
        name: i.intern("Focus"),
        variants: FxHashMap::from_iter([
            (i.intern("User"), None),
            (i.intern("Char"), None),
            (i.intern("Custom"), Some(Box::new(Ty::Object(FxHashMap::from_iter([
                (i.intern("custom"), Ty::String),
            ]))))),
        ]),
    };

    // Test with Custom variant
    let ctx = typed_ctx(&i, &[("Focus", typed_val(focus_ty.clone(), Value::variant(i.intern("Custom"), Some(Box::new(Value::object(FxHashMap::from_iter([
            (i.intern("custom"), Value::string("hello")),
        ])))))))]);

    let output = run_ctx(
        &i,
        "{-{ Focus::User = @Focus }}user{-{ Focus::Char = }}char{-{ Focus::Custom({custom,}) = }}{{ custom }}{-{ / }}",
        ctx,
    ).await;
    assert_eq!(output, "hello");
}

#[tokio::test]
async fn template_var_scoped_inside_match_arm() {
    // Variable defined inside a match arm body is properly scoped.
    // It's usable INSIDE the arm, not outside.
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("cond", TypedValue::bool_(true))]);

    // x is defined and used INSIDE the same arm — this must work
    let output = run_ctx(
        &i,
        "before{-{ true = @cond }}{{ x = \"hello\" }}{{ x }}{-{ / }}after",
        ctx,
    ).await;
    assert_eq!(output, "beforehelloafter");
}

#[tokio::test]
async fn template_var_scoped_inside_match_arm_not_taken() {
    // When arm is not taken, variables inside it are never accessed
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("cond", TypedValue::bool_(false))]);

    let output = run_ctx(
        &i,
        "before{-{ true = @cond }}{{ x = \"hello\" }}{{ x }}{-{ / }}after",
        ctx,
    ).await;
    assert_eq!(output, "beforeafter");
}

#[tokio::test]
async fn template_enum_single_variant_context_type() {
    // CLI path: context has SINGLE-variant enum type (only the variant used in the expr)
    let i = Interner::new();
    let focus_ty = Ty::Enum {
        name: i.intern("Focus"),
        variants: FxHashMap::from_iter([
            (i.intern("User"), None),  // only one variant!
        ]),
    };
    let ctx = typed_ctx(&i, &[("Focus", typed_val(focus_ty, Value::variant(i.intern("User"), None)))]);

    // Template tries to match multiple variants — typechecker should handle this
    let output = run_ctx(
        &i,
        "{-{ Focus::User = @Focus }}user{-{ Focus::Char = }}char{-{ Focus::System = }}sys{-{ / }}",
        ctx,
    ).await;
    assert_eq!(output, "user");
}

#[tokio::test]
async fn script_hint_flatten_with_context() {
    // Simulates: @turn.history | map(v -> v.entrypoint) | collect | flatten | flatten
    let i = Interner::new();
    let entry_ty = Ty::Object(FxHashMap::from_iter([
        (i.intern("entrypoint"), Ty::List(Box::new(
            Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([
                (i.intern("role"), Ty::String),
                (i.intern("content"), Ty::String),
            ]))))
        ))),
    ]));
    let turn_ty = Ty::Object(FxHashMap::from_iter([
        (i.intern("index"), Ty::Int),
        (i.intern("history"), Ty::List(Box::new(entry_ty))),
    ]));
    let context_types = FxHashMap::from_iter([
        (i.intern("turn"), turn_ty),
    ]);
    let script = "@turn.history | map(v -> v.entrypoint) | collect | flatten | flatten";

    // Compile with no hint — should work (this is the CLI path)
    let ast = acvus_ast::parse_script(&i, script).expect("parse failed");
    let reg = acvus_mir::context_registry::ContextTypeRegistry::all_system(context_types);
    let result = acvus_mir::compile_script_with_hint(&i, &ast, &reg, None);
    assert!(result.is_ok(), "compile without hint failed: {:?}", result.err());
}

// ── Block expressions ───────────────────────────────────────────

#[tokio::test]
async fn block_single_expr() {
    assert_eq!(run_simple(r#"{{ { "hello" } }}"#).await, "hello");
}

#[tokio::test]
async fn block_bind_and_return() {
    assert_eq!(
        run_simple(r#"{{ { a = 1; a | to_string } }}"#).await,
        "1"
    );
}

#[tokio::test]
async fn block_chained_binds() {
    assert_eq!(
        run_simple(r#"{{ { a = 1; b = a + 1; b | to_string } }}"#).await,
        "2"
    );
}

#[tokio::test]
async fn block_nested() {
    assert_eq!(
        run_simple(r#"{{ { a = { b = 10; b + 5 }; a | to_string } }}"#).await,
        "15"
    );
}

#[tokio::test]
async fn block_scope_isolation() {
    // Inner block defines `a`, outer block defines a different `a`.
    assert_eq!(
        run_simple(r#"{{ { a = 100; x = { a = 1; a }; a + x | to_string } }}"#).await,
        "101"
    );
}

#[tokio::test]
async fn block_in_lambda() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | map(x -> { y = x * 10; y }) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "10, 20, 30"
    );
}

#[tokio::test]
async fn block_in_pipe() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | x -> { a = last(x); unwrap_or(a, 0) } | to_string }}"#,
            val,
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn block_with_field_access() {
    let i = Interner::new();
    let val = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ { a = last(@users); unwrap(a).name } }}"#,
            val,
        )
        .await,
        "bob"
    );
}

#[tokio::test]
async fn block_string_operations() {
    assert_eq!(
        run_simple(r#"{{ { a = "hello"; b = " world"; a + b | upper } }}"#).await,
        "HELLO WORLD"
    );
}

#[tokio::test]
async fn block_with_boolean_logic() {
    assert_eq!(
        run_simple(r#"{{ { a = 5; b = 10; a < b | to_string } }}"#).await,
        "true"
    );
}

#[tokio::test]
async fn block_discard_intermediate_exprs() {
    // Stmt::Expr results are discarded; only tail matters.
    assert_eq!(
        run_simple(r#"{{ { 1 + 2; 3 + 4; "result" } }}"#).await,
        "result"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Builtin E2E Tests
// ═══════════════════════════════════════════════════════════════════

// ── Type Conversions ────────────────────────────────────────────

#[tokio::test]
async fn builtin_to_string_int() {
    assert_eq!(run_simple("{{ 42 | to_string }}").await, "42");
}

#[tokio::test]
async fn builtin_to_string_float() {
    assert_eq!(run_simple("{{ 3.14 | to_string }}").await, "3.14");
}

#[tokio::test]
async fn builtin_to_string_bool() {
    assert_eq!(run_simple("{{ true | to_string }}").await, "true");
}

#[tokio::test]
async fn builtin_to_string_string() {
    assert_eq!(run_simple(r#"{{ "hello" | to_string }}"#).await, "hello");
}

#[tokio::test]
async fn builtin_to_int_from_float() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("f", TypedValue::float(9.99))]);
    assert_eq!(
        run_ctx(&i, "{{ @f | to_int | to_string }}", ctx).await,
        "9"
    );
}

#[tokio::test]
async fn builtin_to_int_from_byte() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("b", TypedValue::byte(65))]);
    assert_eq!(
        run_ctx(&i, "{{ @b | to_int | to_string }}", ctx).await,
        "65"
    );
}

#[tokio::test]
async fn builtin_to_float_from_int() {
    assert_eq!(
        run_simple(r#"{{ x = 5 | to_float }}{{ x | to_string }}{{_}}{{/}}"#).await,
        "5"
    );
}

#[tokio::test]
async fn builtin_char_to_int() {
    assert_eq!(
        run_simple(r#"{{ "A" | char_to_int | to_string }}"#).await,
        "65"
    );
}

#[tokio::test]
async fn builtin_int_to_char() {
    assert_eq!(run_simple("{{ 65 | int_to_char }}").await, "A");
}

#[tokio::test]
async fn builtin_char_to_int_roundtrip() {
    assert_eq!(
        run_simple(r#"{{ "Z" | char_to_int | int_to_char }}"#).await,
        "Z"
    );
}

// ── List Operations ─────────────────────────────────────────────

#[tokio::test]
async fn builtin_len_list() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(&i, "{{ @items | len | to_string }}", val).await,
        "3"
    );
}

#[tokio::test]
async fn builtin_len_empty_list() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("items", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![])))]);
    assert_eq!(
        run_ctx(&i, "{{ @items | len | to_string }}", ctx).await,
        "0"
    );
}

#[tokio::test]
async fn builtin_rev_iter_list() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | rev_iter | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "3, 2, 1"
    );
}

#[tokio::test]
async fn builtin_flatten_list() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "items",
        typed_val(
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
            Value::list(vec![
                Value::list(vec![Value::int(1), Value::int(2)]),
                Value::list(vec![Value::int(3)]),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flatten | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn builtin_join_list() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "words",
        typed_val(
            Ty::List(Box::new(Ty::String)),
            Value::list(vec![
                Value::string("a"),
                Value::string("b"),
                Value::string("c"),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(&i, r#"{{ @words | join("-") }}"#, ctx).await,
        "a-b-c"
    );
}

#[tokio::test]
async fn builtin_join_empty_list() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("words", typed_val(Ty::List(Box::new(Ty::String)), Value::list(vec![])))]);
    assert_eq!(
        run_ctx(&i, r#"{{ @words | join(", ") }}"#, ctx).await,
        ""
    );
}

#[tokio::test]
async fn builtin_contains_list_found() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, "{{ @items | contains(2) | to_string }}", val).await,
        "true"
    );
}

#[tokio::test]
async fn builtin_contains_list_not_found() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, "{{ @items | contains(99) | to_string }}", val).await,
        "false"
    );
}

#[tokio::test]
async fn builtin_first_list_some() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(&i, "{{ @items | first | unwrap | to_string }}", val).await,
        "10"
    );
}

#[tokio::test]
async fn builtin_first_list_none() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("items", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![])))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | first }}empty{{_}}has{{/}}",
            ctx,
        )
        .await,
        "empty"
    );
}

#[tokio::test]
async fn builtin_last_list_some() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(&i, "{{ @items | last | unwrap | to_string }}", val).await,
        "30"
    );
}

#[tokio::test]
async fn builtin_last_list_none() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("items", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![])))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | last }}empty{{_}}has{{/}}",
            ctx,
        )
        .await,
        "empty"
    );
}

// ── String Operations ───────────────────────────────────────────

#[tokio::test]
async fn builtin_contains_str_found() {
    assert_eq!(
        run_simple(r#"{{ "hello world" | contains_str("world") | to_string }}"#).await,
        "true"
    );
}

#[tokio::test]
async fn builtin_contains_str_not_found() {
    assert_eq!(
        run_simple(r#"{{ "hello world" | contains_str("xyz") | to_string }}"#).await,
        "false"
    );
}

#[tokio::test]
async fn builtin_substring() {
    assert_eq!(
        run_simple(r#"{{ "hello world" | substring(6, 5) }}"#).await,
        "world"
    );
}

#[tokio::test]
async fn builtin_substring_zero_len() {
    assert_eq!(
        run_simple(r#"{{ "hello" | substring(0, 0) }}"#).await,
        ""
    );
}

#[tokio::test]
async fn builtin_len_str() {
    assert_eq!(
        run_simple(r#"{{ "hello" | len_str | to_string }}"#).await,
        "5"
    );
}

#[tokio::test]
async fn builtin_len_str_unicode() {
    assert_eq!(
        run_simple(r#"{{ "한글" | len_str | to_string }}"#).await,
        "2"
    );
}

#[tokio::test]
async fn builtin_trim() {
    assert_eq!(
        run_simple(r#"{{ "  hello  " | trim }}"#).await,
        "hello"
    );
}

#[tokio::test]
async fn builtin_trim_start() {
    assert_eq!(
        run_simple(r#"{{ "  hello  " | trim_start }}"#).await,
        "hello  "
    );
}

#[tokio::test]
async fn builtin_trim_end() {
    assert_eq!(
        run_simple(r#"{{ "  hello  " | trim_end }}"#).await,
        "  hello"
    );
}

#[tokio::test]
async fn builtin_upper() {
    assert_eq!(
        run_simple(r#"{{ "hello" | upper }}"#).await,
        "HELLO"
    );
}

#[tokio::test]
async fn builtin_lower() {
    assert_eq!(
        run_simple(r#"{{ "HELLO" | lower }}"#).await,
        "hello"
    );
}

#[tokio::test]
async fn builtin_replace_str() {
    assert_eq!(
        run_simple(r#"{{ "hello world" | replace_str("world", "rust") }}"#).await,
        "hello rust"
    );
}

#[tokio::test]
async fn builtin_replace_str_multiple() {
    assert_eq!(
        run_simple(r#"{{ "aaa" | replace_str("a", "bb") }}"#).await,
        "bbbbbb"
    );
}

#[tokio::test]
async fn builtin_split_str() {
    assert_eq!(
        run_simple(r#"{{ "a,b,c" | split_str(",") | join(" ") }}"#).await,
        "a b c"
    );
}

#[tokio::test]
async fn builtin_starts_with_str_true() {
    assert_eq!(
        run_simple(r#"{{ "hello" | starts_with_str("hel") | to_string }}"#).await,
        "true"
    );
}

#[tokio::test]
async fn builtin_starts_with_str_false() {
    assert_eq!(
        run_simple(r#"{{ "hello" | starts_with_str("xyz") | to_string }}"#).await,
        "false"
    );
}

#[tokio::test]
async fn builtin_ends_with_str_true() {
    assert_eq!(
        run_simple(r#"{{ "hello" | ends_with_str("llo") | to_string }}"#).await,
        "true"
    );
}

#[tokio::test]
async fn builtin_ends_with_str_false() {
    assert_eq!(
        run_simple(r#"{{ "hello" | ends_with_str("xyz") | to_string }}"#).await,
        "false"
    );
}

#[tokio::test]
async fn builtin_repeat_str() {
    assert_eq!(
        run_simple(r#"{{ "ab" | repeat_str(3) }}"#).await,
        "ababab"
    );
}

#[tokio::test]
async fn builtin_repeat_str_zero() {
    assert_eq!(
        run_simple(r#"{{ "hello" | repeat_str(0) }}"#).await,
        ""
    );
}

// ── Byte Operations ─────────────────────────────────────────────

#[tokio::test]
async fn builtin_to_bytes_and_back() {
    assert_eq!(
        run_simple(r#"{{ "hello" | to_bytes | to_utf8 | unwrap }}"#).await,
        "hello"
    );
}

#[tokio::test]
async fn builtin_to_utf8_lossy() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "data",
        typed_val(
            Ty::bytes(),
            Value::list(vec![
                Value::byte(0x48), // H
                Value::byte(0x69), // i
                Value::byte(0xFF), // invalid
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(&i, "{{ @data | to_utf8_lossy }}", ctx).await,
        "Hi\u{FFFD}"
    );
}

// ── Option Operations ───────────────────────────────────────────

#[tokio::test]
async fn builtin_unwrap_some() {
    assert_eq!(
        run_simple("{{ Some(42) | unwrap | to_string }}").await,
        "42"
    );
}

#[tokio::test]
async fn builtin_unwrap_or_some() {
    assert_eq!(
        run_simple("{{ Some(42) | unwrap_or(0) | to_string }}").await,
        "42"
    );
}

#[tokio::test]
async fn builtin_unwrap_or_none() {
    assert_eq!(
        run_simple("{{ None | unwrap_or(99) | to_string }}").await,
        "99"
    );
}

// ── Iterator Constructors & Basic Ops ───────────────────────────

#[tokio::test]
async fn builtin_map_collect() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn builtin_rev_iter() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | rev_iter | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "3, 2, 1"
    );
}

#[tokio::test]
async fn builtin_take() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30, 40, 50]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | take(3) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "10, 20, 30"
    );
}

#[tokio::test]
async fn builtin_skip() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30, 40, 50]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | skip(2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "30, 40, 50"
    );
}

#[tokio::test]
async fn builtin_chain() {
    let i = Interner::new();
    let ctx = typed_ctx(
        &i,
        &[
            ("a", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![Value::int(1), Value::int(2)]))),
            ("b", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![Value::int(3), Value::int(4)]))),
        ],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @a | chain(@b) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "1, 2, 3, 4"
    );
}

// ── Iterator HOFs ───────────────────────────────────────────────

#[tokio::test]
async fn builtin_filter() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | filter(x -> x > 3) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "4, 5"
    );
}

#[tokio::test]
async fn builtin_map() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | map(x -> x * 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "2, 4, 6"
    );
}

#[tokio::test]
async fn builtin_find() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | find(x -> x > 3) | to_string }}"#,
            val,
        )
        .await,
        "4"
    );
}

#[tokio::test]
async fn builtin_reduce() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | reduce((a, b) -> a + b) | to_string }}"#,
            val,
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn builtin_fold() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | fold(100, (acc, x) -> acc + x) | to_string }}"#,
            val,
        )
        .await,
        "106"
    );
}

#[tokio::test]
async fn builtin_any_true() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | any(x -> x > 2) | to_string }}",
            val,
        )
        .await,
        "true"
    );
}

#[tokio::test]
async fn builtin_any_false() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | any(x -> x > 10) | to_string }}",
            val,
        )
        .await,
        "false"
    );
}

#[tokio::test]
async fn builtin_all_true() {
    let i = Interner::new();
    let val = items_context(&i, vec![2, 4, 6]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | all(x -> x > 0) | to_string }}",
            val,
        )
        .await,
        "true"
    );
}

#[tokio::test]
async fn builtin_all_false() {
    let i = Interner::new();
    let val = items_context(&i, vec![2, 4, 6]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | all(x -> x > 3) | to_string }}",
            val,
        )
        .await,
        "false"
    );
}

// ── Iterator Overloaded Builtins (new) ──────────────────────────

#[tokio::test]
async fn builtin_flatten_iter() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "items",
        typed_val(
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
            Value::list(vec![
                Value::list(vec![Value::int(1), Value::int(2)]),
                Value::list(vec![Value::int(3), Value::int(4)]),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flatten | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "1, 2, 3, 4"
    );
}

#[tokio::test]
async fn builtin_flat_map_iter() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flat_map(x -> [x, x * 10]) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "1, 10, 2, 20, 3, 30"
    );
}

#[tokio::test]
async fn builtin_join_iter() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "words",
        typed_val(
            Ty::List(Box::new(Ty::String)),
            Value::list(vec![
                Value::string("hello"),
                Value::string("world"),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @words | join(" ") }}"#,
            ctx,
        )
        .await,
        "hello world"
    );
}

#[tokio::test]
async fn builtin_contains_iter_found() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | contains(20) | to_string }}",
            val,
        )
        .await,
        "true"
    );
}

#[tokio::test]
async fn builtin_contains_iter_not_found() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | contains(99) | to_string }}",
            val,
        )
        .await,
        "false"
    );
}

#[tokio::test]
async fn builtin_first_iter_some() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | first | unwrap | to_string }}",
            val,
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn builtin_first_iter_none() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("items", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![])))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | first }}empty{{_}}has{{/}}",
            ctx,
        )
        .await,
        "empty"
    );
}

#[tokio::test]
async fn builtin_last_iter_some() {
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | last | unwrap | to_string }}",
            val,
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn builtin_last_iter_none() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[("items", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![])))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | last }}empty{{_}}has{{/}}",
            ctx,
        )
        .await,
        "empty"
    );
}

// ── Iterator Pipeline Combos ────────────────────────────────────

#[tokio::test]
async fn builtin_filter_map_collect() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | filter(x -> x > 2) | map(x -> x * 10) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "30, 40, 50, 60"
    );
}

#[tokio::test]
async fn builtin_take_skip_combo() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | skip(2) | take(3) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "3, 4, 5"
    );
}

#[tokio::test]
async fn builtin_chain_filter_map() {
    let i = Interner::new();
    let ctx = typed_ctx(
        &i,
        &[
            ("a", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![Value::int(1), Value::int(2), Value::int(3)]))),
            ("b", typed_val(Ty::List(Box::new(Ty::Int)), Value::list(vec![Value::int(4), Value::int(5), Value::int(6)]))),
        ],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @a | chain(@b) | filter(x -> x > 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "3, 4, 5, 6"
    );
}

#[tokio::test]
async fn builtin_rev_iter_filter() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | rev_iter | filter(x -> x > 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "5, 4, 3"
    );
}

#[tokio::test]
async fn builtin_flatten_iter_then_filter() {
    let i = Interner::new();
    let ctx = typed_ctx(&i, &[(
        "items",
        typed_val(
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
            Value::list(vec![
                Value::list(vec![Value::int(1), Value::int(2)]),
                Value::list(vec![Value::int(3), Value::int(4)]),
            ]),
        ),
    )]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flatten | filter(x -> x > 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "3, 4"
    );
}

// ── Pmap (parallel map) ─────────────────────────────────────────

#[tokio::test]
async fn builtin_pmap() {
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    // pmap should produce same results as map (order preserved)
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | pmap(x -> x * 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            val,
        )
        .await,
        "2, 4, 6"
    );
}

// ── Closure self-containment ─────────────────────────────────────

#[tokio::test]
async fn closure_basic_capture_local() {
    // A closure captures a local variable and uses it in its body.
    assert_eq!(
        run_simple(
            r#"{{ prefix = "hello" }}{{ f = x -> prefix + " " + x }}{{ f("world") }}{{_}}{{/}}"#,
        )
        .await,
        "hello world"
    );
}

#[tokio::test]
async fn closure_nested_make_closure() {
    // A closure creates another closure inside (MakeClosure within MakeClosure).
    // The outer closure captures `base`, the inner captures both `base` (transitively) and `scale`.
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ base = 100 }}{{ make_adder = scale -> (x -> x * scale + base) }}{{ adder = make_adder(10) }}{{ @items | map(adder) | map(x -> (x | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            val,
        )
        .await,
        "110, 120, 130"
    );
}

#[tokio::test]
async fn closure_passed_to_higher_order_function() {
    // A closure that captures a variable is passed to map as an argument.
    let i = Interner::new();
    let val = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ offset = 5 }}{{ add_offset = x -> x + offset }}{{ @items | map(add_offset) | map(x -> (x | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            val,
        )
        .await,
        "15, 25, 35"
    );
}

#[tokio::test]
async fn closure_returned_from_closure() {
    // An inner closure captures the outer closure's parameter.
    // make_scaler returns a closure that multiplies by the captured factor.
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ make_scaler = factor -> (x -> x * factor) }}{{ double = make_scaler(2) }}{{ @items | map(double) | map(x -> (x | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            val,
        )
        .await,
        "2, 4, 6"
    );
}

#[tokio::test]
async fn multiple_closures_sharing_capture_local() {
    // Two separate closures capture the same local variable.
    // Both should independently use the captured value.
    let i = Interner::new();
    let val = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ threshold = 3 }}{{ above = @items | filter(x -> x > threshold) | map(x -> (x | to_string)) | collect | join(", ") }}{{ below = @items | filter(x -> x <= threshold) | map(x -> (x | to_string)) | collect | join(", ") }}{{ above }}|{{ below }}"#,
            val,
        )
        .await,
        "4, 5|1, 2, 3"
    );
}

// ── Closure self-containment ────────────────────────────────────

#[tokio::test]
async fn closure_basic_map() {
    // Basic closure |x| x + 1 applied via map.
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3] | map(x -> x + 1) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn closure_with_capture() {
    // Closure captures local variable y.
    assert_eq!(
        run_simple(
            r#"{{ y = 10 }}{{ [1, 2, 3] | map(x -> x + y) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "11, 12, 13"
    );
}

#[tokio::test]
async fn closure_captures_two_locals() {
    // A single closure captures two local variables.
    assert_eq!(
        run_simple(
            r#"{{ lo = 2 }}{{ hi = 4 }}{{ [1, 2, 3, 4, 5] | filter(x -> x >= lo && x <= hi) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn closure_capture_used_in_separate_exprs() {
    // Same captured variable used by closures in separate template expressions.
    assert_eq!(
        run_simple(
            r#"{{ offset = 100 }}{{ a = [1, 2] | map(x -> x + offset) | map(x -> (x | to_string)) | collect | join(",") }}{{ b = [3, 4] | map(x -> x + offset) | map(x -> (x | to_string)) | collect | join(",") }}{{ a }};{{ b }}"#,
        )
        .await,
        "101,102;103,104"
    );
}

#[tokio::test]
async fn closure_multiple_sharing_capture() {
    // Two closures capture the same variable and produce different results.
    assert_eq!(
        run_simple(
            r#"{{ base = 10 }}{{ a = [1, 2] | map(x -> x + base) | map(x -> (x | to_string)) | collect | join(",") }}{{ b = [1, 2] | map(x -> x * base) | map(x -> (x | to_string)) | collect | join(",") }}{{ a }}|{{ b }}"#,
        )
        .await,
        "11,12|10,20"
    );
}

// ── Sequence builtins (Deque → Sequence lazy ops) ───────────────

#[tokio::test]
async fn seq_map_deque() {
    // [1,2,3] is Deque, map coerces it to Sequence, collect coerces to Iterator.
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3] | map(x -> x * 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "2, 4, 6"
    );
}

#[tokio::test]
async fn seq_take_deque() {
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3, 4, 5] | take(3) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn seq_skip_deque() {
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3, 4, 5] | skip(2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "3, 4, 5"
    );
}

#[tokio::test]
async fn seq_filter_deque() {
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3, 4, 5] | filter(x -> x > 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "3, 4, 5"
    );
}

#[tokio::test]
async fn seq_collect_after_map() {
    // Sequence from map, then collect (coerce Sequence → Iterator → List).
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3] | map(x -> x + 1) | collect | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn seq_chained_filter_map() {
    // filter then map on deque — both go through Sequence path.
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3, 4, 5] | filter(x -> x > 2) | map(x -> x * 10) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "30, 40, 50"
    );
}

#[tokio::test]
async fn seq_chain_same_origin() {
    // chain two slices of the same deque (take + skip) — same origin, ChainSeq applies.
    assert_eq!(
        run_simple(
            r#"{{ d = [10, 20, 30, 40, 50] }}{{ a = d | take(2) }}{{ b = d | skip(3) }}{{ a | chain(b) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
        )
        .await,
        "10, 20, 40, 50"
    );
}

// ── next builtin ────────────────────────────────────────────────

#[tokio::test]
async fn next_basic() {
    // $iter is mutable, rebind inside each match block
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2, 3] | iter }}{{ $out = "" }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + (x | to_string) + ", " }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + (x | to_string) + ", " }}{{ Some((x, _)) = $iter | next }}{{ $out = $out + (x | to_string) }}{{/}}{{/}}{{/}}{{ $out }}"#,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn next_empty() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [] | iter }}{{ None = $iter | next }}empty{{/}}"#,
        )
        .await,
        "empty"
    );
}

#[tokio::test]
async fn next_exhaustion() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1] | iter }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ x | to_string }}{{ None = $iter | next }}-done{{/}}{{/}}"#,
        )
        .await,
        "1-done"
    );
}

#[tokio::test]
async fn next_with_skip() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [10, 20, 30, 40] | iter | skip(2) }}{{ Some((x, _)) = $iter | next }}{{ x | to_string }}{{/}}"#,
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn next_with_map() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2, 3] | iter | map(x -> x * 10) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ x | to_string }}, {{ Some((x, _)) = $iter | next }}{{ x | to_string }}{{/}}{{/}}"#,
        )
        .await,
        "10, 20"
    );
}

#[tokio::test]
async fn next_with_filter() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2, 3, 4, 5] | iter | filter(x -> x > 3) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ x | to_string }}, {{ Some((x, _)) = $iter | next }}{{ x | to_string }}{{/}}{{/}}"#,
        )
        .await,
        "4, 5"
    );
}

#[tokio::test]
async fn next_with_take() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2, 3] | iter | take(2) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ x | to_string }}, {{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ x | to_string }}{{ None = $iter | next }}-end{{/}}{{/}}{{/}}"#,
        )
        .await,
        "1, 2-end"
    );
}

#[tokio::test]
async fn next_with_chain() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2] | iter | chain([3, 4] | iter) }}{{ $out = "" }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + (x | to_string) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + ", " + (x | to_string) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + ", " + (x | to_string) }}{{ Some((x, _)) = $iter | next }}{{ $out = $out + ", " + (x | to_string) }}{{/}}{{/}}{{/}}{{/}}{{ $out }}"#,
        )
        .await,
        "1, 2, 3, 4"
    );
}

#[tokio::test]
async fn next_with_flat_map() {
    // flat_map: each element expands to a list, next pulls one at a time
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2] | iter | flat_map(x -> [x, x * 10]) }}{{ $out = "" }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + (x | to_string) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + ", " + (x | to_string) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + ", " + (x | to_string) }}{{ Some((x, _)) = $iter | next }}{{ $out = $out + ", " + (x | to_string) }}{{/}}{{/}}{{/}}{{/}}{{ $out }}"#,
        )
        .await,
        "1, 10, 2, 20"
    );
}

#[tokio::test]
async fn next_flat_map_empty_result() {
    // flat_map where some elements map to empty list (via take(0))
    // [1, 2, 3] | flat_map(x -> [x, x+10]) | take(3) — just verify flat_map + next works with take
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2, 3] | iter | flat_map(x -> [x, x + 10]) | take(3) }}{{ $out = "" }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + (x | to_string) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ $out = $out + ", " + (x | to_string) }}{{ Some((x, _)) = $iter | next }}{{ $out = $out + ", " + (x | to_string) }}{{/}}{{/}}{{/}}{{ $out }}"#,
        )
        .await,
        "1, 11, 2"
    );
}

#[tokio::test]
async fn next_map_filter_combo() {
    assert_eq!(
        run_simple(
            r#"{{ $iter = [1, 2, 3, 4] | iter | map(x -> x * 10) | filter(x -> x > 20) }}{{ Some((x, rest)) = $iter | next }}{{ $iter = rest }}{{ x | to_string }}, {{ Some((x, _)) = $iter | next }}{{ x | to_string }}{{/}}{{/}}"#,
        )
        .await,
        "30, 40"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Cast soundness & completeness tests
//
// Verify that the type system correctly inserts Cast instructions and
// that the interpreter executes them soundly (no implicit coercions).
// ═══════════════════════════════════════════════════════════════════

// -- A: Cast insertion correctness (verify pass catches mismatches) --
// These tests exercise paths that require Cast.  If Cast insertion is
// broken, the verify pass panics before execution begins.

// A1: Deque piped into List parameter (len)
#[tokio::test]
async fn cast_deque_to_list_via_len() {
    assert_eq!(run_simple(r#"{{ [1, 2, 3] | len | to_string }}"#).await, "3");
}

// A2: Deque piped into List parameter (reverse)
#[tokio::test]
async fn cast_deque_to_list_via_reverse() {
    assert_eq!(
        run_simple(r#"{{ [3, 2, 1] | reverse | first | unwrap | to_string }}"#).await,
        "1"
    );
}

// A3: Deque piped into Iterator parameter (filter)
#[tokio::test]
async fn cast_deque_to_iter_via_filter() {
    assert_eq!(
        run_simple(r#"{{ [1, 2, 3, 4] | filter(x -> x > 2) | collect | len | to_string }}"#).await,
        "2"
    );
}

// A4: Deque in iteration block
#[tokio::test]
async fn cast_deque_to_iter_loop() {
    assert_eq!(
        run_simple(r#"{{ x in [10, 20, 30] }}{{ x | to_string }} {{/}}"#).await,
        "10 20 30 "
    );
}

// A5: List (from context) in iteration block
#[tokio::test]
async fn cast_list_to_iter_loop() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, r#"{{ x in @items }}{{ x | to_string }}{{/}}"#, ctx).await,
        "123"
    );
}

// A6: Range in iteration block
#[tokio::test]
async fn cast_range_to_iter_loop() {
    assert_eq!(
        run_simple(r#"{{ x in 1..4 }}{{ x | to_string }}{{/}}"#).await,
        "123"
    );
}

// A9: Exactly one Cast for Deque → List (no double cast)
#[tokio::test]
async fn cast_deque_to_list_single_step() {
    // [1,2] | contains(1) requires Deque→List cast, then contains operates on List
    assert_eq!(
        run_simple(r#"{{ [1, 2, 3] | contains(2) | to_string }}"#).await,
        "true"
    );
}

// -- B: Soundness (structural_eq respects variant boundaries) --

// B4: Two Deque literals have different Origins — comparing them is a type error (soundness)
#[test]
fn soundness_deque_different_origin_eq_rejected() {
    let i = Interner::new();
    let template = acvus_ast::parse(&i, r#"{{ [1, 2] == [1, 2] | to_string }}"#).unwrap();
    let reg = acvus_mir::context_registry::ContextTypeRegistry::all_system(FxHashMap::default());
    let result = acvus_mir::compile(&i, &template, &reg);
    assert!(result.is_err(), "different-origin Deque == should be rejected by typechecker");
}

// B4b: Same-origin comparison works (via context List)
#[tokio::test]
async fn soundness_list_eq_same_type() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, r#"{{ @items == @items | to_string }}"#, ctx).await,
        "true"
    );
}

// -- D: Completeness (end-to-end correct output) --

// D1: Deque | len
#[tokio::test]
async fn completeness_deque_len() {
    assert_eq!(run_simple(r#"{{ [1, 2, 3] | len | to_string }}"#).await, "3");
}

// D2: Deque | reverse
#[tokio::test]
async fn completeness_deque_reverse() {
    assert_eq!(
        run_simple(r#"{{ [1, 2, 3] | reverse | len | to_string }}"#).await,
        "3"
    );
}

// D3: iteration over Deque
#[tokio::test]
async fn completeness_iter_deque() {
    assert_eq!(
        run_simple(r#"{{ x in [1, 2, 3] }}{{ x | to_string }}{{/}}"#).await,
        "123"
    );
}

// D4: iteration over Range
#[tokio::test]
async fn completeness_iter_range() {
    assert_eq!(
        run_simple(r#"{{ x in 1..4 }}{{ x | to_string }}{{/}}"#).await,
        "123"
    );
}

// D5: Deque | map
#[tokio::test]
async fn completeness_deque_map() {
    assert_eq!(
        run_simple(r#"{{ [1, 2, 3] | map(x -> x * 2) | collect | len | to_string }}"#).await,
        "3"
    );
}

// D6: Deque | filter
#[tokio::test]
async fn completeness_deque_filter() {
    assert_eq!(
        run_simple(r#"{{ [1, 2, 3, 4] | filter(x -> x > 2) | collect | len | to_string }}"#).await,
        "2"
    );
}

// D7: Deque | fold
#[tokio::test]
async fn completeness_deque_fold() {
    assert_eq!(
        run_simple(r#"{{ [1, 2, 3] | fold(0, (acc, x) -> acc + x) | to_string }}"#).await,
        "6"
    );
}

// D8: Deque | find
#[tokio::test]
async fn completeness_deque_find() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, r#"{{ @items | find(x -> x == 2) | to_string }}"#, ctx).await,
        "2"
    );
}

// D9: nested list flatten via iter path
#[tokio::test]
async fn completeness_nested_flatten() {
    let i = Interner::new();
    let ty = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
    let val = Value::list(vec![
        Value::list(vec![Value::int(1), Value::int(2)]),
        Value::list(vec![Value::int(3)]),
    ]);
    let ctx = typed_ctx(&i, &[("nested", typed_val(ty, val))]);
    // Use iter path: List → Iterator via iter builtin, then flatten_iter
    assert_eq!(
        run_ctx(&i, r#"{{ @nested | iter | flatten | collect | len | to_string }}"#, ctx).await,
        "3"
    );
}

// D10: chained pipe operations
#[tokio::test]
async fn completeness_chained_pipe() {
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3, 4, 5] | filter(x -> x > 2) | map(x -> x * 10) | collect | len | to_string }}"#,
        )
        .await,
        "3"
    );
}

// D12: equality on same-type lists (via context)
#[tokio::test]
async fn completeness_list_equality() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, r#"{{ @items == @items | to_string }}"#, ctx).await,
        "true"
    );
}

// -- E: Edge cases --

// E1: empty deque | len
#[tokio::test]
async fn edge_empty_deque_len() {
    assert_eq!(run_simple(r#"{{ [] | len | to_string }}"#).await, "0");
}

// E2: empty range iteration
#[tokio::test]
async fn edge_empty_range_iter() {
    assert_eq!(
        run_simple(r#"before{{ x in 0..0 }}x{{/}}after"#).await,
        "beforeafter"
    );
}

// E3: single element reverse
#[tokio::test]
async fn edge_single_element_reverse() {
    assert_eq!(
        run_simple(r#"{{ [42] | reverse | first | unwrap | to_string }}"#).await,
        "42"
    );
}

// E4: multi-stage pipe chain
#[tokio::test]
async fn edge_multi_stage_pipe() {
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3, 4, 5, 6] | filter(x -> x > 1) | map(x -> x * 2) | take(3) | collect | len | to_string }}"#,
        )
        .await,
        "3"
    );
}

// E5: Deque inside closure body (closure captures Deque, uses it with List builtin)
#[tokio::test]
async fn edge_closure_with_cast() {
    assert_eq!(
        run_simple(
            r#"{{ [1, 2, 3] | map(x -> x + 1) | collect | len | to_string }}"#,
        )
        .await,
        "3"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Lambda return coercion tests
//
// When a lambda body returns Deque but the signature expects Iterator,
// the typechecker detects the Fn ret coercion and the lowerer inserts
// a Cast at the lambda return site.
// ═══════════════════════════════════════════════════════════════════

// Lambda returns Deque, flat_map expects Fn(T) → Iterator<U>
#[tokio::test]
async fn lambda_ret_coercion_flat_map() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flat_map(x -> [x, x * 10]) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "1, 10, 2, 20, 3, 30"
    );
}

// Nested lambda: outer flat_map returns Deque (coercion), then map on result
#[tokio::test]
async fn lambda_ret_coercion_nested() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flat_map(x -> [x, x + 10]) | map(x -> x * 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ctx,
        )
        .await,
        "2, 22, 4, 24, 6, 26"
    );
}

// Lambda ret coercion with Deque → List (used with flatten)
#[tokio::test]
async fn lambda_ret_coercion_flat_map_then_flatten() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flat_map(x -> [x * 10, x * 100]) | collect | len | to_string }}"#,
            ctx,
        )
        .await,
        "4"
    );
}

// Empty result from lambda in flat_map
#[tokio::test]
async fn lambda_ret_coercion_flat_map_empty() {
    let i = Interner::new();
    let ctx = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flat_map(x -> []) | collect | len | to_string }}"#,
            ctx,
        )
        .await,
        "0"
    );
}
