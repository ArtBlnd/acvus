use std::collections::{BTreeMap, HashMap};

use acvus_interpreter::{ExternFn, ExternFnBody, ExternFnRegistry, ExternFnSig, Value};
use acvus_interpreter_test::*;
#[allow(unused_imports)]
use acvus_interpreter_test::{run_capturing_context_calls, run_obfuscated, run_simple_obfuscated};
use acvus_mir::ty::Ty;

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
    let (ty, val) = string_context("name", "alice");
    assert_eq!(
        run_with_context("Hello, {{ @name }}!", ty, val).await,
        "Hello, alice!"
    );
}

// ── Context / Variables ─────────────────────────────────────────

#[tokio::test]
async fn context_read() {
    let (ty, val) = int_context("count", 42);
    assert_eq!(
        run_with_context("{{ @count | to_string }}", ty, val).await,
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
    let (ty, val) = user_context();
    assert_eq!(run_with_context("{{ @user.name }}", ty, val).await, "alice");
}

#[tokio::test]
async fn variable_write_computed() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([("a".into(), Value::Int(10)), ("b".into(), Value::Int(32))]);
    assert_eq!(
        run_with_context(
            "{{ $result = @a + @b }}{{ $result | to_string }}",
            types,
            values
        )
        .await,
        "42"
    );
}

// ── Arithmetic ───────────────────────────────────────────────────

#[tokio::test]
async fn arithmetic_to_string() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([("a".into(), Value::Int(3)), ("b".into(), Value::Int(7))]);
    assert_eq!(
        run_with_context("{{ @a + @b | to_string }}", types, values).await,
        "10"
    );
}

#[tokio::test]
async fn unary_negation() {
    let (ty, val) = int_context("n", 5);
    assert_eq!(
        run_with_context(r#"{{ x = -@n }}{{ x | to_string }}{{_}}{{/}}"#, ty, val).await,
        "-5"
    );
}

#[tokio::test]
async fn boolean_not() {
    let types = HashMap::from([("flag".into(), Ty::Bool)]);
    let values = HashMap::from([("flag".into(), Value::Bool(true))]);
    assert_eq!(
        run_with_context(
            r#"{{ x = !@flag }}{{ x | to_string }}{{_}}{{/}}"#,
            types,
            values
        )
        .await,
        "false"
    );
}

#[tokio::test]
async fn comparison_operators() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([("a".into(), Value::Int(10)), ("b".into(), Value::Int(5))]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @a > @b }}{{ x | to_string }}{{_}}{{/}}"#,
            types,
            values
        )
        .await,
        "true"
    );
}

// ── Match blocks ─────────────────────────────────────────────────

#[tokio::test]
async fn simple_match_binding() {
    let (ty, val) = string_context("name", "alice");
    assert_eq!(
        run_with_context(r#"{{ x = @name }}{{ x }}"#, ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn match_literal_filter_hit() {
    let (ty, val) = string_context("role", "admin");
    assert_eq!(
        run_with_context(
            r#"{{ "admin" = @role }}admin page{{_}}guest page{{/}}"#,
            ty,
            val
        )
        .await,
        "admin page"
    );
}

#[tokio::test]
async fn match_literal_filter_miss() {
    let (ty, val) = string_context("role", "user");
    assert_eq!(
        run_with_context(
            r#"{{ "admin" = @role }}admin page{{_}}guest page{{/}}"#,
            ty,
            val
        )
        .await,
        "guest page"
    );
}

#[tokio::test]
async fn multi_arm_match() {
    let (ty, val) = string_context("role", "user");
    assert_eq!(
        run_with_context(
            r#"{{ "admin" = @role }}admin{{ "user" = }}user{{_}}guest{{/}}"#,
            ty,
            val
        )
        .await,
        "user"
    );
}

#[tokio::test]
async fn match_bool_literal() {
    let types = HashMap::from([("flag".into(), Ty::Bool)]);
    let values = HashMap::from([("flag".into(), Value::Bool(true))]);
    assert_eq!(
        run_with_context(r#"{{ true = @flag }}on{{_}}off{{/}}"#, types, values).await,
        "on"
    );
}

#[tokio::test]
async fn match_binding_with_body() {
    let (ty, val) = user_context();
    assert_eq!(
        run_with_context(
            r#"{{ { name, } = @user }}{{ name }} is here{{_}}no user{{/}}"#,
            ty,
            val
        )
        .await,
        "alice is here"
    );
}

#[tokio::test]
async fn variable_shadowing() {
    let (ty, val) = string_context("name", "alice");
    assert_eq!(
        run_with_context(
            r#"{{ x = "outer" }}{{ x = @name }}{{ x }}{{_}}{{/}}"#,
            ty,
            val
        )
        .await,
        "alice"
    );
}

#[tokio::test]
async fn catch_all_with_binding() {
    let (ty, val) = string_context("role", "viewer");
    assert_eq!(
        run_with_context(
            r#"{{ "admin" = @role }}admin{{_}}{{ fallback = "guest" }}{{ fallback }}{{/}}"#,
            ty,
            val
        )
        .await,
        "guest"
    );
}

#[tokio::test]
async fn equality_as_match_source() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([("a".into(), Value::Int(5)), ("b".into(), Value::Int(5))]);
    assert_eq!(
        run_with_context(
            r#"{{ true = @a == @b }}equal{{_}}not equal{{/}}"#,
            types,
            values
        )
        .await,
        "equal"
    );
}

// ── Nested match blocks ──────────────────────────────────────────

#[tokio::test]
async fn nested_match_blocks() {
    let types = HashMap::from([("role".into(), Ty::String), ("level".into(), Ty::Int)]);
    let values = HashMap::from([
        ("role".into(), Value::String("admin".into())),
        ("level".into(), Value::Int(5)),
    ]);
    assert_eq!(
        run_with_context(
            r#"{{ "admin" = @role }}{{ 1..10 = @level }}low{{_}}high{{/}}{{_}}guest{{/}}"#,
            types,
            values
        )
        .await,
        "low"
    );
}

// ── Variable ref in match arm ────────────────────────────────────

#[tokio::test]
async fn variable_new_ref_binding() {
    let (ty, val) = string_context("name", "alice");
    assert_eq!(
        run_with_context(r#"{{ $result = @name }}{{ $result }}"#, ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn variable_new_ref_in_match_arm() {
    let (ty, val) = string_context("role", "admin");
    assert_eq!(
        run_with_context(
            r#"{{ "admin" = @role }}{{ $selected = "yes" }}{{_}}{{ $selected = "no" }}{{/}}{{ $selected }}"#,
            ty,
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
    let (ty, val) = int_context("age", 5);
    assert_eq!(
        run_with_context(
            r#"{{ 0..10 = @age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
            ty,
            val
        )
        .await,
        "child"
    );
}

#[tokio::test]
async fn range_pattern_miss() {
    let (ty, val) = int_context("age", 25);
    assert_eq!(
        run_with_context(
            r#"{{ 0..10 = @age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
            ty,
            val
        )
        .await,
        "adult"
    );
}

// ── Iteration ────────────────────────────────────────────────────

#[tokio::test]
async fn iter_list_binding() {
    let (ty, val) = items_context(vec![1, 2, 3]);
    assert_eq!(
        run_with_context("{{ x in @items }}{{ x | to_string }}{{/}}", ty, val).await,
        "123"
    );
}

#[tokio::test]
async fn iter_object_destructure() {
    let (ty, val) = users_list_context();
    assert_eq!(
        run_with_context("{{ { name, } in @users }}{{ name }}{{/}}", ty, val).await,
        "alicebob"
    );
}

#[tokio::test]
async fn iter_tuple_destructure() {
    let ty = HashMap::from([(
        "pairs".into(),
        Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
    )]);
    let val = HashMap::from([(
        "pairs".into(),
        Value::List(vec![
            Value::Tuple(vec![Value::String("a".into()), Value::Int(1)]),
            Value::Tuple(vec![Value::String("b".into()), Value::Int(2)]),
        ]),
    )]);
    assert_eq!(
        run_with_context("{{ (a, _) in @pairs }}{{ a }}{{/}}", ty, val).await,
        "ab"
    );
}

#[tokio::test]
async fn nested_iteration() {
    let ty = HashMap::from([(
        "matrix".into(),
        Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
    )]);
    let val = HashMap::from([(
        "matrix".into(),
        Value::List(vec![
            Value::List(vec![Value::Int(1), Value::Int(2)]),
            Value::List(vec![Value::Int(3), Value::Int(4)]),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            "{{ row in @matrix }}{{ x in row }}{{ x | to_string }}{{/}}{{/}}",
            ty,
            val
        )
        .await,
        "1234"
    );
}

#[tokio::test]
async fn variable_write_in_iteration() {
    let (types, values) = items_context(vec![10, 20, 30]);
    assert_eq!(
        run_with_context(
            "{{ $last = 0 }}{{ x in @items }}{{ $last = x }}{{/}}{{ $last | to_string }}",
            types,
            values
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn variable_accumulate_in_loop() {
    let (types, values) = items_context(vec![1, 2, 3]);
    assert_eq!(
        run_with_context(
            "{{ $sum = 0 }}{{ x in @items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}",
            types,
            values
        )
        .await,
        "6"
    );
}

// ── List patterns ────────────────────────────────────────────────

#[tokio::test]
async fn list_destructure_head() {
    let (ty, val) = items_context(vec![10, 20, 30]);
    assert_eq!(
        run_with_context(
            r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
            ty,
            val
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn list_destructure_tail() {
    let (ty, val) = items_context(vec![10, 20, 30]);
    assert_eq!(
        run_with_context(
            r#"{{ [.., a, b] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
            ty,
            val
        )
        .await,
        "20"
    );
}

#[tokio::test]
async fn list_destructure_head_and_tail() {
    let (ty, val) = items_context(vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_with_context(
            r#"{{ [first, .., last] = @items }}{{ first | to_string }}-{{ last | to_string }}{{_}}empty{{/}}"#,
            ty,
            val,
        )
        .await,
        "1-5"
    );
}

#[tokio::test]
async fn list_exact_match_hit() {
    let (ty, val) = items_context(vec![10, 20]);
    assert_eq!(
        run_with_context(
            r#"{{ [a, b] = @items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
            ty,
            val
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn list_exact_match_miss() {
    let (ty, val) = items_context(vec![10, 20, 30]);
    assert_eq!(
        run_with_context(
            r#"{{ [a, b] = @items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
            ty,
            val
        )
        .await,
        "wrong length"
    );
}

// ── Object patterns ──────────────────────────────────────────────

#[tokio::test]
async fn object_pattern() {
    let (ty, val) = user_context();
    assert_eq!(
        run_with_context(
            r#"{{ { name, age, } = @user }}{{ name }}:{{ age | to_string }}{{/}}"#,
            ty,
            val
        )
        .await,
        "alice:30"
    );
}

#[tokio::test]
async fn deeply_nested_object_access() {
    let ty = HashMap::from([(
        "data".into(),
        Ty::Object(BTreeMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([(
                "address".into(),
                Ty::Object(BTreeMap::from([("city".into(), Ty::String)])),
            )])),
        )])),
    )]);
    let val = HashMap::from([(
        "data".into(),
        Value::Object(BTreeMap::from([(
            "user".into(),
            Value::Object(BTreeMap::from([(
                "address".into(),
                Value::Object(BTreeMap::from([(
                    "city".into(),
                    Value::String("Seoul".into()),
                )])),
            )])),
        )])),
    )]);
    assert_eq!(
        run_with_context("{{ @data.user.address.city }}", ty, val).await,
        "Seoul"
    );
}

// ── Tuple ────────────────────────────────────────────────────────

#[tokio::test]
async fn tuple_expression() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::String)]);
    let values = HashMap::from([
        ("a".into(), Value::Int(42)),
        ("b".into(), Value::String("hello".into())),
    ]);
    assert_eq!(
        run_with_context(
            r#"{{ (x, y) = (@a, @b) }}{{ x | to_string }}, {{ y }}{{/}}"#,
            types,
            values
        )
        .await,
        "42, hello"
    );
}

#[tokio::test]
async fn tuple_pattern_binding() {
    let types = HashMap::from([("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let values = HashMap::from([(
        "pair".into(),
        Value::Tuple(vec![Value::String("alice".into()), Value::Int(30)]),
    )]);
    assert_eq!(
        run_with_context(r#"{{ (name, age) = @pair }}{{ name }}{{/}}"#, types, values).await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_wildcard() {
    let types = HashMap::from([("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let values = HashMap::from([(
        "pair".into(),
        Value::Tuple(vec![Value::String("alice".into()), Value::Int(30)]),
    )]);
    assert_eq!(
        run_with_context(r#"{{ (name, _) = @pair }}{{ name }}{{/}}"#, types, values).await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_literal_match_hit() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([("a".into(), Value::Int(0)), ("b".into(), Value::Int(1))]);
    assert_eq!(
        run_with_context(
            r#"{{ (0, 1) = (@a, @b) }}zero-one{{ (1, _) = }}one-any{{_}}other{{/}}"#,
            types,
            values
        )
        .await,
        "zero-one"
    );
}

#[tokio::test]
async fn nested_tuple_pattern() {
    let types = HashMap::from([(
        "data".into(),
        Ty::Tuple(vec![Ty::Tuple(vec![Ty::Int, Ty::Int]), Ty::String]),
    )]);
    let values = HashMap::from([(
        "data".into(),
        Value::Tuple(vec![
            Value::Tuple(vec![Value::Int(1), Value::Int(2)]),
            Value::String("hello".into()),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            r#"{{ ((a, b), label) = @data }}{{ label }}{{/}}"#,
            types,
            values
        )
        .await,
        "hello"
    );
}

// ── Pipe & builtins ──────────────────────────────────────────────

#[tokio::test]
async fn pipe_to_string() {
    let (ty, val) = int_context("n", 42);
    assert_eq!(
        run_with_context("{{ @n | to_string }}", ty, val).await,
        "42"
    );
}

#[tokio::test]
async fn to_float_conversion() {
    let (ty, val) = int_context("n", 5);
    assert_eq!(
        run_with_context(
            r#"{{ x = @n | to_float }}{{ x | to_string }}{{_}}{{/}}"#,
            ty,
            val
        )
        .await,
        "5"
    );
}

#[tokio::test]
async fn to_int_conversion() {
    let types = HashMap::from([("f".into(), Ty::Float)]);
    let values = HashMap::from([("f".into(), Value::Float(3.7))]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @f | to_int }}{{ x | to_string }}{{_}}{{/}}"#,
            types,
            values
        )
        .await,
        "3"
    );
}

// ── Lambda / closures ────────────────────────────────────────────

#[tokio::test]
async fn lambda_filter() {
    let (ty, val) = items_context(vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | filter(x -> x != 0) }}{{ x | map(i -> i | to_string) | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn lambda_map() {
    let (ty, val) = items_context(vec![1, 2, 3]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | map(i -> i + 1) }}{{ x | map(i -> i | to_string) | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn lambda_pmap() {
    let (ty, val) = items_context(vec![1, 2, 3]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | pmap(i -> i | to_string) }}{{ x | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn pipe_filter_map() {
    let (ty, val) = items_context(vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | filter(x -> x != 0) | map(x -> x | to_string) }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn triple_pipe_chain() {
    let (ty, val) = items_context(vec![0, 1, 2, 3]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | filter(i -> i != 0) | map(i -> i + 1) | map(i -> i | to_string) }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn closure_capture_local() {
    let (ty, val) = items_context(vec![1, 3, 5, 7, 10]);
    assert_eq!(
        run_with_context(
            r#"{{ threshold = 5 }}{{ x = @items | filter(i -> i > threshold) }}{{ x | map(i -> i | to_string) | join(", ") }}{{_}}{{/}}"#,
            ty,
            val,
        )
        .await,
        "7, 10"
    );
}

#[tokio::test]
async fn closure_capture_context() {
    let types = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("threshold".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "items".into(),
            Value::List(vec![Value::Int(1), Value::Int(5), Value::Int(10)]),
        ),
        ("threshold".into(), Value::Int(3)),
    ]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | filter(i -> i > @threshold) }}{{ x | map(i -> i | to_string) | join(", ") }}"#,
            types,
            values
        )
        .await,
        "5, 10"
    );
}

#[tokio::test]
async fn lambda_field_access() {
    let (ty, val) = users_list_context();
    assert_eq!(
        run_with_context(
            r#"{{ x = @users | map(u -> u.name) }}{{ x | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "alice, bob"
    );
}

#[tokio::test]
async fn lambda_negate_param() {
    let (ty, val) = items_context(vec![1, 2, 3]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | map(i -> -i) }}{{ x | map(i -> i | to_string) | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "-1, -2, -3"
    );
}

#[tokio::test]
async fn lambda_not_param() {
    let types = HashMap::from([("flags".into(), Ty::List(Box::new(Ty::Bool)))]);
    let values = HashMap::from([(
        "flags".into(),
        Value::List(vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(true),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @flags | map(i -> !i) }}{{ x | map(b -> b | to_string) | join(", ") }}"#,
            types,
            values
        )
        .await,
        "false, true, false"
    );
}

#[tokio::test]
async fn lambda_string_concat() {
    let types = HashMap::from([("names".into(), Ty::List(Box::new(Ty::String)))]);
    let values = HashMap::from([(
        "names".into(),
        Value::List(vec![
            Value::String("alice".into()),
            Value::String("bob".into()),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @names | map(n -> n + "!") }}{{ x | join(", ") }}"#,
            types,
            values
        )
        .await,
        "alice!, bob!"
    );
}

#[tokio::test]
async fn lambda_float_arithmetic() {
    let types = HashMap::from([("vals".into(), Ty::List(Box::new(Ty::Float)))]);
    let values = HashMap::from([(
        "vals".into(),
        Value::List(vec![Value::Float(1.5), Value::Float(2.5)]),
    )]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @vals | map(v -> v * 2.0) }}{{ x | map(v -> v | to_string) | join(", ") }}"#,
            types,
            values
        )
        .await,
        "3, 5"
    );
}

#[tokio::test]
async fn filter_then_map_field() {
    let (ty, val) = users_list_context();
    assert_eq!(
        run_with_context(
            r#"{{ x = @users | filter(u -> u.age > 18) | map(u -> u.name) }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "alice, bob"
    );
}

#[tokio::test]
async fn multiple_closures_same_capture() {
    let types = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("offset".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "items".into(),
            Value::List(vec![
                Value::Int(-1),
                Value::Int(0),
                Value::Int(1),
                Value::Int(2),
            ]),
        ),
        ("offset".into(), Value::Int(1)),
    ]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | map(i -> i + @offset) | filter(i -> i > 0) }}{{ x | map(i -> i | to_string) | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "1, 2, 3"
    );
}

// ── Extern functions ─────────────────────────────────────────────

struct DoubleIt;

impl ExternFn for DoubleIt {
    fn name(&self) -> &str {
        "double"
    }
    fn sig(&self) -> ExternFnSig {
        ExternFnSig {
            params: vec![Ty::Int],
            ret: Ty::Int,
            effectful: false,
        }
    }
    fn into_body(self) -> ExternFnBody {
        ExternFnBody::new(|args| async move {
            match &args[0] {
                Value::Int(n) => Value::Int(n * 2),
                _ => panic!("expected Int"),
            }
        })
    }
}

#[tokio::test]
async fn extern_fn_call() {
    let mut extern_fns = ExternFnRegistry::new();
    extern_fns.register(DoubleIt);
    let (ty, val) = int_context("n", 21);
    let output = run(
        r#"{{ x = double(@n) }}{{ x | to_string }}{{_}}{{/}}"#,
        ty,
        val,
        extern_fns,
    )
    .await;
    assert_eq!(output, "42");
}

// ── Logical operators (&&, ||) ───────────────────────────────────

#[tokio::test]
async fn and_both_true() {
    let types = HashMap::from([("a".into(), Ty::Bool), ("b".into(), Ty::Bool)]);
    let values = HashMap::from([
        ("a".into(), Value::Bool(true)),
        ("b".into(), Value::Bool(true)),
    ]);
    assert_eq!(
        run_with_context(r#"{{ true = @a && @b }}yes{{_}}no{{/}}"#, types, values).await,
        "yes"
    );
}

#[tokio::test]
async fn and_one_false() {
    let types = HashMap::from([("a".into(), Ty::Bool), ("b".into(), Ty::Bool)]);
    let values = HashMap::from([
        ("a".into(), Value::Bool(true)),
        ("b".into(), Value::Bool(false)),
    ]);
    assert_eq!(
        run_with_context(r#"{{ true = @a && @b }}yes{{_}}no{{/}}"#, types, values).await,
        "no"
    );
}

#[tokio::test]
async fn or_one_true() {
    let types = HashMap::from([("a".into(), Ty::Bool), ("b".into(), Ty::Bool)]);
    let values = HashMap::from([
        ("a".into(), Value::Bool(false)),
        ("b".into(), Value::Bool(true)),
    ]);
    assert_eq!(
        run_with_context(r#"{{ true = @a || @b }}yes{{_}}no{{/}}"#, types, values).await,
        "yes"
    );
}

#[tokio::test]
async fn or_both_false() {
    let types = HashMap::from([("a".into(), Ty::Bool), ("b".into(), Ty::Bool)]);
    let values = HashMap::from([
        ("a".into(), Value::Bool(false)),
        ("b".into(), Value::Bool(false)),
    ]);
    assert_eq!(
        run_with_context(r#"{{ true = @a || @b }}yes{{_}}no{{/}}"#, types, values).await,
        "no"
    );
}

#[tokio::test]
async fn and_or_precedence() {
    // a || b && c => a || (b && c) — && binds tighter
    let types = HashMap::from([
        ("a".into(), Ty::Bool),
        ("b".into(), Ty::Bool),
        ("c".into(), Ty::Bool),
    ]);
    let values = HashMap::from([
        ("a".into(), Value::Bool(true)),
        ("b".into(), Value::Bool(false)),
        ("c".into(), Value::Bool(false)),
    ]);
    assert_eq!(
        run_with_context(
            r#"{{ true = @a || @b && @c }}yes{{_}}no{{/}}"#,
            types,
            values
        )
        .await,
        "yes"
    );
}

#[tokio::test]
async fn and_with_comparison() {
    let types = HashMap::from([("x".into(), Ty::Int)]);
    let values = HashMap::from([("x".into(), Value::Int(15))]);
    assert_eq!(
        run_with_context(
            r#"{{ true = @x > 10 && @x < 20 }}in range{{_}}out{{/}}"#,
            types,
            values
        )
        .await,
        "in range"
    );
}

#[tokio::test]
async fn logical_in_filter() {
    let (ty, val) = items_context(vec![1, 5, 10, 15, 20, 25]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @items | filter(i -> i > 5 && i < 20) }}{{ x | map(i -> i | to_string) | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "10, 15"
    );
}

// ── Complex scenarios ───────────────────────────────────────────

#[tokio::test]
async fn nested_match_with_variable_write() {
    let types = HashMap::from([("role".into(), Ty::String), ("level".into(), Ty::Int)]);
    let values = HashMap::from([
        ("role".into(), Value::String("admin".into())),
        ("level".into(), Value::Int(5)),
    ]);
    assert_eq!(
        run_with_context(
            r#"{{ "admin" = @role }}{{ 0..10 = @level }}{{ $result = "low-admin" }}{{_}}{{ $result = "high-admin" }}{{/}}{{_}}{{ $result = "guest" }}{{/}}{{ $result }}"#,
            types,
            values
        )
        .await,
        "low-admin"
    );
}

#[tokio::test]
async fn filter_map_with_object_pattern() {
    let ty = HashMap::from([(
        "products".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("price".into(), Ty::Int),
        ])))),
    )]);
    let val = HashMap::from([(
        "products".into(),
        Value::List(vec![
            Value::Object(BTreeMap::from([
                ("name".into(), Value::String("apple".into())),
                ("price".into(), Value::Int(100)),
            ])),
            Value::Object(BTreeMap::from([
                ("name".into(), Value::String("banana".into())),
                ("price".into(), Value::Int(50)),
            ])),
            Value::Object(BTreeMap::from([
                ("name".into(), Value::String("cherry".into())),
                ("price".into(), Value::Int(200)),
            ])),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @products | filter(p -> p.price >= 100) | map(p -> p.name) }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "apple, cherry"
    );
}

#[tokio::test]
async fn iteration_with_match_per_item() {
    let (ty, val) = items_context(vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_with_context(
            r#"{{ x in @items }}{{ 1..=3 = x }}s{{_}}b{{/}}{{/}}"#,
            ty,
            val
        )
        .await,
        "sssbb"
    );
}

#[tokio::test]
async fn multi_context_interaction() {
    let types = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("min".into(), Ty::Int),
        ("max".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "items".into(),
            Value::List(vec![
                Value::Int(3),
                Value::Int(7),
                Value::Int(1),
                Value::Int(9),
                Value::Int(4),
            ]),
        ),
        ("min".into(), Value::Int(2)),
        ("max".into(), Value::Int(8)),
    ]);
    assert_eq!(
        run_with_context(
            r#"{{ filtered = @items | filter(i -> i >= @min && i <= @max) }}{{ $count = 0 }}{{ x in filtered }}{{ $count = $count + 1 }}{{/}}{{ $count | to_string }}{{_}}{{/}}"#,
            types,
            values,
        )
        .await,
        "3"
    );
}

#[tokio::test]
async fn object_destructure_in_iteration_with_emit() {
    let (ty, val) = users_list_context();
    assert_eq!(
        run_with_context(
            r#"{{ { name, age, } in @users }}{{ name }}({{ age | to_string }}) {{/}}"#,
            ty,
            val
        )
        .await,
        "alice(30) bob(25) "
    );
}

#[tokio::test]
async fn chained_pipe_with_logical_filter() {
    let types = HashMap::from([("nums".into(), Ty::List(Box::new(Ty::Int)))]);
    let values = HashMap::from([(
        "nums".into(),
        Value::List(vec![
            Value::Int(-5),
            Value::Int(0),
            Value::Int(3),
            Value::Int(7),
            Value::Int(12),
            Value::Int(20),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            r#"{{ x = @nums | filter(n -> n > 0 && n < 10) | map(n -> n * n) }}{{ x | map(n -> n | to_string) | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "9, 49"
    );
}

#[tokio::test]
async fn nested_list_iteration_with_accumulator() {
    let types = HashMap::from([(
        "matrix".into(),
        Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
    )]);
    let values = HashMap::from([(
        "matrix".into(),
        Value::List(vec![
            Value::List(vec![Value::Int(1), Value::Int(2)]),
            Value::List(vec![Value::Int(3), Value::Int(4)]),
            Value::List(vec![Value::Int(5), Value::Int(6)]),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            "{{ $sum = 0 }}{{ row in @matrix }}{{ x in row }}{{ $sum = $sum + x }}{{/}}{{/}}{{ $sum | to_string }}",
            types,
            values
        )
        .await,
        "21"
    );
}

#[tokio::test]
async fn map_then_iterate_with_match() {
    let (ty, val) = items_context(vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_with_context(
            r#"{{ doubled = @items | map(i -> i * 2) }}{{ x in doubled }}{{ true = x > 6 }}{{ x | to_string }} {{_}}{{/}}{{/}}"#,
            ty,
            val,
        )
        .await,
        "8 10 "
    );
}

#[tokio::test]
async fn extern_fn_in_pipe_chain() {
    let mut extern_fns = ExternFnRegistry::new();
    extern_fns.register(DoubleIt);
    let (ty, val) = items_context(vec![1, 2, 3]);
    let output = run(
        r#"{{ x = @items | map(i -> double(i)) }}{{ x | map(i -> i | to_string) | join(", ") }}"#,
        ty,
        val,
        extern_fns,
    )
    .await;
    assert_eq!(output, "2, 4, 6");
}

#[tokio::test]
async fn complex_object_filter_format() {
    let ty = HashMap::from([(
        "users".into(),
        Ty::List(Box::new(Ty::Object(BTreeMap::from([
            ("name".into(), Ty::String),
            ("age".into(), Ty::Int),
            ("active".into(), Ty::Bool),
        ])))),
    )]);
    let val = HashMap::from([(
        "users".into(),
        Value::List(vec![
            Value::Object(BTreeMap::from([
                ("name".into(), Value::String("alice".into())),
                ("age".into(), Value::Int(30)),
                ("active".into(), Value::Bool(true)),
            ])),
            Value::Object(BTreeMap::from([
                ("name".into(), Value::String("bob".into())),
                ("age".into(), Value::Int(17)),
                ("active".into(), Value::Bool(true)),
            ])),
            Value::Object(BTreeMap::from([
                ("name".into(), Value::String("carol".into())),
                ("age".into(), Value::Int(25)),
                ("active".into(), Value::Bool(false)),
            ])),
            Value::Object(BTreeMap::from([
                ("name".into(), Value::String("dave".into())),
                ("age".into(), Value::Int(40)),
                ("active".into(), Value::Bool(true)),
            ])),
        ]),
    )]);
    assert_eq!(
        run_with_context(
            r#"{{ eligible = @users | filter(u -> u.active && u.age >= 18) | map(u -> u.name) }}{{ name in eligible }}{{ name }} {{/}}"#,
            ty,
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
            r#"{{ x = [1, 2, 3] }}{{ x | map(i -> i | to_string) | join(", ") }}{{_}}{{/}}"#
        )
        .await,
        "1, 2, 3"
    );
}

// ── Multi-arm with range ─────────────────────────────────────────

#[tokio::test]
async fn multi_arm_range_and_literal() {
    let (ty, val) = int_context("score", 0);
    assert_eq!(
        run_with_context(
            r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            ty,
            val
        )
        .await,
        "zero"
    );
}

#[tokio::test]
async fn multi_arm_range_and_literal_low() {
    let (ty, val) = int_context("score", 5);
    assert_eq!(
        run_with_context(
            r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            ty,
            val
        )
        .await,
        "low"
    );
}

#[tokio::test]
async fn multi_arm_range_and_literal_high() {
    let (ty, val) = int_context("score", 50);
    assert_eq!(
        run_with_context(
            r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            ty,
            val
        )
        .await,
        "high"
    );
}

// ── Obfuscation equivalence ────────────────────────────────────

#[tokio::test]
async fn obf_text_only() {
    assert_eq!(run_simple_obfuscated("hello world").await, "hello world");
}

#[tokio::test]
async fn obf_string_emit() {
    assert_eq!(run_simple_obfuscated(r#"{{ "hello" }}"#).await, "hello");
}

#[tokio::test]
async fn obf_string_concat() {
    assert_eq!(
        run_simple_obfuscated(r#"{{ "hello" + " " + "world" }}"#).await,
        "hello world"
    );
}

#[tokio::test]
async fn obf_mixed_text_and_expr() {
    let (ty, val) = string_context("name", "alice");
    assert_eq!(
        run_obfuscated("Hello, {{ @name }}!", ty, val, ExternFnRegistry::new()).await,
        "Hello, alice!"
    );
}

#[tokio::test]
async fn obf_int_arithmetic() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([("a".into(), Value::Int(3)), ("b".into(), Value::Int(7))]);
    assert_eq!(
        run_obfuscated(
            "{{ @a + @b | to_string }}",
            types,
            values,
            ExternFnRegistry::new()
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn obf_match_literal() {
    let (ty, val) = int_context("n", 42);
    assert_eq!(
        run_obfuscated(
            r#"{{ 42 = @n }}yes{{_}}no{{/}}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "yes"
    );
}

#[tokio::test]
async fn obf_match_string_literal() {
    let (ty, val) = string_context("name", "alice");
    assert_eq!(
        run_obfuscated(
            r#"{{ "alice" = @name }}found{{_}}nope{{/}}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "found"
    );
}

#[tokio::test]
async fn obf_variable_write_read() {
    assert_eq!(
        run_simple_obfuscated("{{ $x = 42 }}{{ $x | to_string }}").await,
        "42"
    );
}

#[tokio::test]
async fn obf_iteration() {
    let (ty, val) = items_context(vec![1, 2, 3]);
    assert_eq!(
        run_obfuscated(
            r#"{{ x in @items }}{{ x | to_string }} {{/}}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "1 2 3 "
    );
}

#[tokio::test]
async fn obf_nested_match_with_variable() {
    let types = HashMap::from([("role".into(), Ty::String), ("level".into(), Ty::Int)]);
    let values = HashMap::from([
        ("role".into(), Value::String("admin".into())),
        ("level".into(), Value::Int(5)),
    ]);
    assert_eq!(
        run_obfuscated(
            r#"{{ "admin" = @role }}{{ 0..10 = @level }}{{ $result = "low-admin" }}{{_}}{{ $result = "high-admin" }}{{/}}{{_}}{{ $result = "guest" }}{{/}}{{ $result }}"#,
            types,
            values,
            ExternFnRegistry::new(),
        )
        .await,
        "low-admin"
    );
}

#[tokio::test]
async fn obf_lambda_filter_map() {
    let (ty, val) = items_context(vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_obfuscated(
            r#"{{ x = @items | filter(x -> x != 0) | map(x -> x | to_string) }}{{ x | join(", ") }}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "1, 2, 3"
    );
}

// ── Complex obfuscation tests ─────────────────────────────────

#[tokio::test]
async fn obf_variable_accumulate_in_loop() {
    let (ty, val) = items_context(vec![10, 20, 30]);
    assert_eq!(
        run_obfuscated(
            r#"{{ $sum = 0 }}{{ x in @items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "60"
    );
}

#[tokio::test]
async fn obf_nested_iteration_with_match() {
    let ty = HashMap::from([(
        "rows".into(),
        Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
    )]);
    let val = HashMap::from([(
        "rows".into(),
        Value::List(vec![
            Value::List(vec![Value::Int(1), Value::Int(2)]),
            Value::List(vec![Value::Int(3), Value::Int(4)]),
        ]),
    )]);
    assert_eq!(
        run_obfuscated(
            r#"{{ row in @rows }}[{{ x in row }}{{ 3 = x }}three{{_}}{{ x | to_string }}{{/}} {{/}}]{{/}}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "[1 2 ][three 4 ]"
    );
}

#[tokio::test]
async fn obf_object_destructure_and_format() {
    let (ty, val) = users_list_context();
    assert_eq!(
        run_obfuscated(
            r#"{{ { name, age, } in @users }}{{ name }}({{ age | to_string }}) {{/}}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "alice(30) bob(25) "
    );
}

#[tokio::test]
async fn obf_multi_variable_interaction() {
    assert_eq!(
        run_simple_obfuscated(
            r#"{{ $a = 10 }}{{ $b = 20 }}{{ $c = $a + $b }}{{ $a = $c * 2 }}{{ $a | to_string }}-{{ $b | to_string }}-{{ $c | to_string }}"#,
        )
        .await,
        "60-20-30"
    );
}

#[tokio::test]
async fn obf_string_match_multi_arm() {
    let (ty, val) = string_context("lang", "rust");
    assert_eq!(
        run_obfuscated(
            r#"{{ "go" = @lang }}Go{{ "rust" = }}Rust{{ "python" = }}Python{{_}}Other{{/}}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "Rust"
    );
}

#[tokio::test]
async fn obf_range_pattern_with_variable() {
    let types = HashMap::from([("score".into(), Ty::Int)]);
    let values = HashMap::from([("score".into(), Value::Int(85))]);
    assert_eq!(
        run_obfuscated(
            r#"{{ 90..=100 = @score }}{{ $grade = "A" }}{{ 80..90 = }}{{ $grade = "B" }}{{ 70..80 = }}{{ $grade = "C" }}{{_}}{{ $grade = "F" }}{{/}}{{ $grade }}"#,
            types,
            values,
            ExternFnRegistry::new(),
        )
        .await,
        "B"
    );
}

#[tokio::test]
async fn obf_filter_accumulate_complex() {
    let (ty, val) = items_context(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(
        run_obfuscated(
            r#"{{ $sum = 0 }}{{ x in @items | filter(x -> x > 5) }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}"#,
            ty,
            val,
            ExternFnRegistry::new(),
        )
        .await,
        "40"
    );
}

#[tokio::test]
async fn obf_pipe_chain_with_context() {
    let types = HashMap::from([("names".into(), Ty::List(Box::new(Ty::String)))]);
    let values = HashMap::from([(
        "names".into(),
        Value::List(vec![
            Value::String("alice".into()),
            Value::String("bob".into()),
            Value::String("charlie".into()),
        ]),
    )]);
    assert_eq!(
        run_obfuscated(
            r#"{{ @names | join(", ") }}"#,
            types,
            values,
            ExternFnRegistry::new(),
        )
        .await,
        "alice, bob, charlie"
    );
}

#[tokio::test]
async fn obf_boolean_logic_in_match() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([("a".into(), Value::Int(5)), ("b".into(), Value::Int(10))]);
    assert_eq!(
        run_obfuscated(
            r#"{{ $result = "none" }}{{ 1..10 = @a }}{{ 5..15 = @b }}{{ $result = "both" }}{{_}}{{ $result = "a-only" }}{{/}}{{_}}{{ $result = "other" }}{{/}}{{ $result }}"#,
            types,
            values,
            ExternFnRegistry::new(),
        )
        .await,
        "both"
    );
}

// ── Context Call ────────────────────────────────────────────────

#[tokio::test]
async fn context_call_bindings_carried() {
    let types = HashMap::from([
        ("node".into(), Ty::String),
        ("items".into(), Ty::List(Box::new(Ty::Int))),
    ]);
    let values = HashMap::from([
        ("node".into(), Value::String("resolved".into())),
        (
            "items".into(),
            Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        ),
    ]);
    let result =
        run_capturing_context_calls("{{ @node { count: @items | len, } }}", types, values).await;
    assert_eq!(result.output, "resolved");
    assert_eq!(result.calls.len(), 1);
    assert_eq!(result.calls[0].0, "node");
    assert!(matches!(
        result.calls[0].1.get("count"),
        Some(Value::Int(3))
    ));
}

#[tokio::test]
async fn context_call_multiple_bindings() {
    let types = HashMap::from([("target".into(), Ty::String), ("name".into(), Ty::String)]);
    let values = HashMap::from([
        ("target".into(), Value::String("done".into())),
        ("name".into(), Value::String("alice".into())),
    ]);
    let result = run_capturing_context_calls(
        r#"{{ @target { greeting: "hello", @name, } }}"#,
        types,
        values,
    )
    .await;
    assert_eq!(result.output, "done");
    assert_eq!(result.calls.len(), 1);
    let bindings = &result.calls[0].1;
    assert!(matches!(bindings.get("greeting"), Some(Value::String(s)) if s == "hello"));
    assert!(matches!(bindings.get("name"), Some(Value::String(s)) if s == "alice"));
}

#[tokio::test]
async fn context_call_variable_shorthand() {
    let types = HashMap::from([("node".into(), Ty::String)]);
    let values = HashMap::from([("node".into(), Value::String("ok".into()))]);
    let result =
        run_capturing_context_calls("{{ $x = 42 }}{{ @node { $x, } }}", types, values).await;
    assert_eq!(result.output, "ok");
    assert_eq!(result.calls.len(), 1);
    assert!(matches!(result.calls[0].1.get("x"), Some(Value::Int(42))));
}

#[tokio::test]
async fn context_call_no_bindings_not_captured() {
    let types = HashMap::from([("data".into(), Ty::String)]);
    let values = HashMap::from([("data".into(), Value::String("hi".into()))]);
    let result = run_capturing_context_calls("{{ @data }}", types, values).await;
    assert_eq!(result.output, "hi");
    assert!(result.calls.is_empty());
}

// ── Variant (Option) ────────────────────────────────────────────

#[tokio::test]
async fn variant_some_extract_value() {
    let types = HashMap::from([("opt".into(), Ty::Option(Box::new(Ty::String)))]);
    let values = HashMap::from([(
        "opt".into(),
        Value::Variant {
            tag: "Some".into(),
            payload: Some(Box::new(Value::String("hello".into()))),
        },
    )]);
    assert_eq!(
        run_with_context(
            "{{ Some(value) = @opt }}{{ value }}{{_}}empty{{/}}",
            types,
            values
        )
        .await,
        "hello"
    );
}

#[tokio::test]
async fn variant_none_match() {
    let types = HashMap::from([("opt".into(), Ty::Option(Box::new(Ty::Int)))]);
    let values = HashMap::from([(
        "opt".into(),
        Value::Variant {
            tag: "None".into(),
            payload: None,
        },
    )]);
    assert_eq!(
        run_with_context("{{ None = @opt }}none{{_}}has value{{/}}", types, values).await,
        "none"
    );
}

#[tokio::test]
async fn variant_some_catch_all() {
    let types = HashMap::from([("opt".into(), Ty::Option(Box::new(Ty::Int)))]);
    let values = HashMap::from([(
        "opt".into(),
        Value::Variant {
            tag: "None".into(),
            payload: None,
        },
    )]);
    assert_eq!(
        run_with_context(
            "{{ Some(v) = @opt }}{{ v | to_string }}{{_}}no value{{/}}",
            types,
            values
        )
        .await,
        "no value"
    );
}

#[tokio::test]
async fn variant_some_with_literal_pattern() {
    let types = HashMap::from([("opt".into(), Ty::Option(Box::new(Ty::Int)))]);
    let values = HashMap::from([(
        "opt".into(),
        Value::Variant {
            tag: "Some".into(),
            payload: Some(Box::new(Value::Int(42))),
        },
    )]);
    assert_eq!(
        run_with_context("{{ Some(42) = @opt }}matched{{_}}no{{/}}", types, values).await,
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
    // 0xFF is not valid utf8 → to_utf8 returns None
    let types = HashMap::from([("data".into(), Ty::List(Box::new(Ty::Byte)))]);
    let values = HashMap::from([(
        "data".into(),
        Value::List(vec![Value::Byte(0xFF), Value::Byte(0xFE)]),
    )]);
    assert_eq!(
        run_with_context(
            "{{ None = @data | to_utf8 }}invalid{{_}}valid{{/}}",
            types,
            values,
        )
        .await,
        "invalid"
    );
}
