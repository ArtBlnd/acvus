use acvus_interpreter::{
    Interpreter,
    RuntimeErrorKind, Value,
};
use acvus_interpreter_test::*;
#[allow(unused_imports)]
use acvus_interpreter_test::{run_obfuscated, run_simple_obfuscated};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// Helper: build a context HashMap<Astr, Ty> from string pairs.
fn ctx(i: &Interner, pairs: &[(&str, Ty)]) -> FxHashMap<acvus_utils::Astr, Ty> {
    pairs
        .iter()
        .map(|(k, v)| (i.intern(k), v.clone()))
        .collect()
}

/// Helper: build a value HashMap<Astr, Value> from string pairs.
fn vals(i: &Interner, pairs: &[(&str, Value)]) -> FxHashMap<acvus_utils::Astr, Value> {
    pairs
        .iter()
        .map(|(k, v)| (i.intern(k), v.clone()))
        .collect()
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
    Value::Object(
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
    let (ty, val) = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(&i, "Hello, {{ @name }}!".into(), ty, val).await,
        "Hello, alice!"
    );
}

// ── Context / Variables ─────────────────────────────────────────

#[tokio::test]
async fn context_read() {
    let i = Interner::new();
    let (ty, val) = int_context(&i, "count", 42);
    assert_eq!(
        run_ctx(&i, "{{ @count | to_string }}".into(), ty, val).await,
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
    let (ty, val) = user_context(&i);
    assert_eq!(
        run_ctx(&i, "{{ @user.name }}".into(), ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn variable_write_computed() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let values = vals(&i, &[("a", Value::Int(10)), ("b", Value::Int(32))]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let values = vals(&i, &[("a", Value::Int(3)), ("b", Value::Int(7))]);
    assert_eq!(
        run_ctx(&i, "{{ @a + @b | to_string }}".into(), types, values).await,
        "10"
    );
}

#[tokio::test]
async fn unary_negation() {
    let i = Interner::new();
    let (ty, val) = int_context(&i, "n", 5);
    assert_eq!(
        run_ctx(&i, r#"{{ x = -@n }}{{ x | to_string }}{{_}}{{/}}"#, ty, val).await,
        "-5"
    );
}

#[tokio::test]
async fn boolean_not() {
    let i = Interner::new();
    let types = ctx(&i, &[("flag", Ty::Bool)]);
    let values = vals(&i, &[("flag", Value::Bool(true))]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let values = vals(&i, &[("a", Value::Int(10)), ("b", Value::Int(5))]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(&i, r#"{{ x = @name }}{{ x }}"#, ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn match_literal_filter_hit() {
    let i = Interner::new();
    let (ty, val) = string_context(&i, "role", "admin");
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "role", "user");
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "role", "user");
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("flag", Ty::Bool)]);
    let values = vals(&i, &[("flag", Value::Bool(true))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @flag }}on{{_}}off{{/}}"#, types, values).await,
        "on"
    );
}

#[tokio::test]
async fn match_binding_with_body() {
    let i = Interner::new();
    let (ty, val) = user_context(&i);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "role", "viewer");
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let values = vals(&i, &[("a", Value::Int(5)), ("b", Value::Int(5))]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("role", Ty::String), ("level", Ty::Int)]);
    let values = vals(
        &i,
        &[
            ("role", Value::String("admin".into())),
            ("level", Value::Int(5)),
        ],
    );
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "name", "alice");
    assert_eq!(
        run_ctx(&i, r#"{{ $result = @name }}{{ $result }}"#, ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn variable_new_ref_in_match_arm() {
    let i = Interner::new();
    let (ty, val) = string_context(&i, "role", "admin");
    assert_eq!(
        run_ctx(&i,
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
    let i = Interner::new();
    let (ty, val) = int_context(&i, "age", 5);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = int_context(&i, "age", 25);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ x in @items }}{{ x | to_string }}{{/}}".into(),
            ty,
            val
        )
        .await,
        "123"
    );
}

#[tokio::test]
async fn iter_object_destructure() {
    let i = Interner::new();
    let (ty, val) = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            "{{ { name, } in @users }}{{ name }}{{/}}".into(),
            ty,
            val
        )
        .await,
        "alicebob"
    );
}

#[tokio::test]
async fn iter_tuple_destructure() {
    let i = Interner::new();
    let ty = ctx(
        &i,
        &[(
            "pairs",
            Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
        )],
    );
    let val = vals(
        &i,
        &[(
            "pairs",
            Value::List(vec![
                Value::Tuple(vec![Value::String("a".into()), Value::Int(1)]),
                Value::Tuple(vec![Value::String("b".into()), Value::Int(2)]),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(&i, "{{ (a, _) in @pairs }}{{ a }}{{/}}".into(), ty, val).await,
        "ab"
    );
}

#[tokio::test]
async fn nested_iteration() {
    let i = Interner::new();
    let ty = ctx(
        &i,
        &[("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))))],
    );
    let val = vals(
        &i,
        &[(
            "matrix",
            Value::List(vec![
                Value::List(vec![Value::Int(1), Value::Int(2)]),
                Value::List(vec![Value::Int(3), Value::Int(4)]),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (types, values) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (types, values) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(&i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = user_context(&i);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let ty = ctx(
        &i,
        &[(
            "data",
            obj_ty(
                &i,
                &[(
                    "user",
                    obj_ty(&i, &[("address", obj_ty(&i, &[("city", Ty::String)]))]),
                )],
            ),
        )],
    );
    let val = vals(
        &i,
        &[(
            "data",
            obj_val(
                &i,
                &[(
                    "user",
                    obj_val(
                        &i,
                        &[(
                            "address",
                            obj_val(&i, &[("city", Value::String("Seoul".into()))]),
                        )],
                    ),
                )],
            ),
        )],
    );
    assert_eq!(
        run_ctx(&i, "{{ @data.user.address.city }}".into(), ty, val).await,
        "Seoul"
    );
}

// ── Tuple ────────────────────────────────────────────────────────

#[tokio::test]
async fn tuple_expression() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::String)]);
    let values = vals(
        &i,
        &[("a", Value::Int(42)), ("b", Value::String("hello".into()))],
    );
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let values = vals(
        &i,
        &[(
            "pair",
            Value::Tuple(vec![Value::String("alice".into()), Value::Int(30)]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ (name, age) = @pair }}{{ name }}{{/}}"#,
            types,
            values
        )
        .await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_wildcard() {
    let i = Interner::new();
    let types = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let values = vals(
        &i,
        &[(
            "pair",
            Value::Tuple(vec![Value::String("alice".into()), Value::Int(30)]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ (name, _) = @pair }}{{ name }}{{/}}"#,
            types,
            values
        )
        .await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_literal_match_hit() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let values = vals(&i, &[("a", Value::Int(0)), ("b", Value::Int(1))]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::Tuple(vec![Ty::Int, Ty::Int]), Ty::String]),
        )],
    );
    let values = vals(
        &i,
        &[(
            "data",
            Value::Tuple(vec![
                Value::Tuple(vec![Value::Int(1), Value::Int(2)]),
                Value::String("hello".into()),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = int_context(&i, "n", 42);
    assert_eq!(
        run_ctx(&i, "{{ @n | to_string }}".into(), ty, val).await,
        "42"
    );
}

#[tokio::test]
async fn to_float_conversion() {
    let i = Interner::new();
    let (ty, val) = int_context(&i, "n", 5);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("f", Ty::Float)]);
    let values = vals(&i, &[("f", Value::Float(3.7))]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(x -> x != 0) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn lambda_map() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @items | map(i -> i + 1) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn lambda_pmap() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @items | pmap(i -> (i | to_string)) | collect }}{{ x | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn pipe_filter_map() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(x -> x != 0) | map(x -> (x | to_string)) | collect }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn triple_pipe_chain() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![0, 1, 2, 3]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(i -> i != 0) | map(i -> i + 1) | map(i -> (i | to_string)) | collect }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "2, 3, 4"
    );
}

#[tokio::test]
async fn closure_capture_local() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 3, 5, 7, 10]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ threshold = 5 }}{{ x = @items | filter(i -> i > threshold) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            ty,
            val,
        )
        .await,
        "7, 10"
    );
}

#[tokio::test]
async fn closure_capture_context() {
    let i = Interner::new();
    let types = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("threshold", Ty::Int),
        ],
    );
    let values = vals(
        &i,
        &[
            (
                "items",
                Value::List(vec![Value::Int(1), Value::Int(5), Value::Int(10)]),
            ),
            ("threshold", Value::Int(3)),
        ],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(i -> i > @threshold) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            types,
            values
        )
        .await,
        "5, 10"
    );
}

#[tokio::test]
async fn lambda_field_access() {
    let i = Interner::new();
    let (ty, val) = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @users | map(u -> u.name) | collect }}{{ x | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "alice, bob"
    );
}

#[tokio::test]
async fn lambda_negate_param() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @items | map(i -> -i) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            ty,
            val
        )
        .await,
        "-1, -2, -3"
    );
}

#[tokio::test]
async fn lambda_not_param() {
    let i = Interner::new();
    let types = ctx(&i, &[("flags", Ty::List(Box::new(Ty::Bool)))]);
    let values = vals(
        &i,
        &[(
            "flags",
            Value::List(vec![
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(true),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @flags | map(i -> !i) | collect }}{{ x | map(b -> (b | to_string)) | collect | join(", ") }}"#,
            types,
            values
        )
        .await,
        "false, true, false"
    );
}

#[tokio::test]
async fn lambda_string_concat() {
    let i = Interner::new();
    let types = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let values = vals(
        &i,
        &[(
            "names",
            Value::List(vec![
                Value::String("alice".into()),
                Value::String("bob".into()),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @names | map(n -> n + "!") | collect }}{{ x | join(", ") }}"#,
            types,
            values
        )
        .await,
        "alice!, bob!"
    );
}

#[tokio::test]
async fn lambda_float_arithmetic() {
    let i = Interner::new();
    let types = ctx(&i, &[("vals", Ty::List(Box::new(Ty::Float)))]);
    let values = vals(
        &i,
        &[(
            "vals",
            Value::List(vec![Value::Float(1.5), Value::Float(2.5)]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @vals | map(v -> v * 2.0) | collect }}{{ x | map(v -> (v | to_string)) | collect | join(", ") }}"#,
            types,
            values
        )
        .await,
        "3, 5"
    );
}

#[tokio::test]
async fn filter_then_map_field() {
    let i = Interner::new();
    let (ty, val) = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ x = @users | filter(u -> u.age > 18) | map(u -> u.name) | collect }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "alice, bob"
    );
}

#[tokio::test]
async fn multiple_closures_same_capture() {
    let i = Interner::new();
    let types = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("offset", Ty::Int)],
    );
    let values = vals(
        &i,
        &[
            (
                "items",
                Value::List(vec![
                    Value::Int(-1),
                    Value::Int(0),
                    Value::Int(1),
                    Value::Int(2),
                ]),
            ),
            ("offset", Value::Int(1)),
        ],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | map(i -> i + @offset) | filter(i -> i > 0) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "1, 2, 3"
    );
}


// ── Logical operators (&&, ||) ───────────────────────────────────

#[tokio::test]
async fn and_both_true() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Bool), ("b", Ty::Bool)]);
    let values = vals(&i, &[("a", Value::Bool(true)), ("b", Value::Bool(true))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a && @b }}yes{{_}}no{{/}}"#, types, values).await,
        "yes"
    );
}

#[tokio::test]
async fn and_one_false() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Bool), ("b", Ty::Bool)]);
    let values = vals(&i, &[("a", Value::Bool(true)), ("b", Value::Bool(false))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a && @b }}yes{{_}}no{{/}}"#, types, values).await,
        "no"
    );
}

#[tokio::test]
async fn or_one_true() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Bool), ("b", Ty::Bool)]);
    let values = vals(&i, &[("a", Value::Bool(false)), ("b", Value::Bool(true))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a || @b }}yes{{_}}no{{/}}"#, types, values).await,
        "yes"
    );
}

#[tokio::test]
async fn or_both_false() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Bool), ("b", Ty::Bool)]);
    let values = vals(&i, &[("a", Value::Bool(false)), ("b", Value::Bool(false))]);
    assert_eq!(
        run_ctx(&i, r#"{{ true = @a || @b }}yes{{_}}no{{/}}"#, types, values).await,
        "no"
    );
}

#[tokio::test]
async fn and_or_precedence() {
    let i = Interner::new();
    // a || b && c => a || (b && c) — && binds tighter
    let types = ctx(&i, &[("a", Ty::Bool), ("b", Ty::Bool), ("c", Ty::Bool)]);
    let values = vals(
        &i,
        &[
            ("a", Value::Bool(true)),
            ("b", Value::Bool(false)),
            ("c", Value::Bool(false)),
        ],
    );
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("x", Ty::Int)]);
    let values = vals(&i, &[("x", Value::Int(15))]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 5, 10, 15, 20, 25]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @items | filter(i -> i > 5 && i < 20) | collect }}{{ x | map(i -> (i | to_string)) | collect | join(", ") }}"#,
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
    let i = Interner::new();
    let types = ctx(&i, &[("role", Ty::String), ("level", Ty::Int)]);
    let values = vals(
        &i,
        &[
            ("role", Value::String("admin".into())),
            ("level", Value::Int(5)),
        ],
    );
    assert_eq!(
        run_ctx(&i,
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
    let i = Interner::new();
    let ty = ctx(
        &i,
        &[(
            "products",
            Ty::List(Box::new(obj_ty(
                &i,
                &[("name", Ty::String), ("price", Ty::Int)],
            ))),
        )],
    );
    let val = vals(
        &i,
        &[(
            "products",
            Value::List(vec![
                obj_val(
                    &i,
                    &[
                        ("name", Value::String("apple".into())),
                        ("price", Value::Int(100)),
                    ],
                ),
                obj_val(
                    &i,
                    &[
                        ("name", Value::String("banana".into())),
                        ("price", Value::Int(50)),
                    ],
                ),
                obj_val(
                    &i,
                    &[
                        ("name", Value::String("cherry".into())),
                        ("price", Value::Int(200)),
                    ],
                ),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @products | filter(p -> p.price >= 100) | map(p -> p.name) | collect }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "apple, cherry"
    );
}

#[tokio::test]
async fn iteration_with_match_per_item() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("min", Ty::Int),
            ("max", Ty::Int),
        ],
    );
    let values = vals(
        &i,
        &[
            (
                "items",
                Value::List(vec![
                    Value::Int(3),
                    Value::Int(7),
                    Value::Int(1),
                    Value::Int(9),
                    Value::Int(4),
                ]),
            ),
            ("min", Value::Int(2)),
            ("max", Value::Int(8)),
        ],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ filtered = @items | filter(i -> i >= @min && i <= @max) | collect }}{{ $count = 0 }}{{ x in filtered }}{{ $count = $count + 1 }}{{/}}{{ $count | to_string }}{{_}}{{/}}"#,
            types,
            values,
        )
        .await,
        "3"
    );
}

#[tokio::test]
async fn object_destructure_in_iteration_with_emit() {
    let i = Interner::new();
    let (ty, val) = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("nums", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(
        &i,
        &[(
            "nums",
            Value::List(vec![
                Value::Int(-5),
                Value::Int(0),
                Value::Int(3),
                Value::Int(7),
                Value::Int(12),
                Value::Int(20),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ x = @nums | filter(n -> n > 0 && n < 10) | map(n -> n * n) | collect }}{{ x | map(n -> (n | to_string)) | collect | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "9, 49"
    );
}

#[tokio::test]
async fn nested_list_iteration_with_accumulator() {
    let i = Interner::new();
    let types = ctx(
        &i,
        &[("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))))],
    );
    let values = vals(
        &i,
        &[(
            "matrix",
            Value::List(vec![
                Value::List(vec![Value::Int(1), Value::Int(2)]),
                Value::List(vec![Value::Int(3), Value::Int(4)]),
                Value::List(vec![Value::Int(5), Value::Int(6)]),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(&i,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(&i,
            r#"{{ doubled = @items | map(i -> i * 2) | collect }}{{ x in doubled }}{{ true = x > 6 }}{{ x | to_string }} {{_}}{{/}}{{/}}"#,
            ty,
            val,
        )
        .await,
        "8 10 "
    );
}


#[tokio::test]
async fn complex_object_filter_format() {
    let i = Interner::new();
    let ty = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj_ty(
                &i,
                &[("name", Ty::String), ("age", Ty::Int), ("active", Ty::Bool)],
            ))),
        )],
    );
    let val = vals(
        &i,
        &[(
            "users",
            Value::List(vec![
                obj_val(
                    &i,
                    &[
                        ("name", Value::String("alice".into())),
                        ("age", Value::Int(30)),
                        ("active", Value::Bool(true)),
                    ],
                ),
                obj_val(
                    &i,
                    &[
                        ("name", Value::String("bob".into())),
                        ("age", Value::Int(17)),
                        ("active", Value::Bool(true)),
                    ],
                ),
                obj_val(
                    &i,
                    &[
                        ("name", Value::String("carol".into())),
                        ("age", Value::Int(25)),
                        ("active", Value::Bool(false)),
                    ],
                ),
                obj_val(
                    &i,
                    &[
                        ("name", Value::String("dave".into())),
                        ("age", Value::Int(40)),
                        ("active", Value::Bool(true)),
                    ],
                ),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(&i,
            r#"{{ eligible = @users | filter(u -> u.active && u.age >= 18) | map(u -> u.name) | collect }}{{ name in eligible }}{{ name }} {{/}}"#,
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
    let (ty, val) = int_context(&i, "score", 0);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = int_context(&i, "score", 5);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = int_context(&i, "score", 50);
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "name", "alice");
    assert_eq!(
        run_obfuscated(
            &i,
            "Hello, {{ @name }}!",
            ty,
            val,
        )
        .await,
        "Hello, alice!"
    );
}

#[tokio::test]
async fn obf_int_arithmetic() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let values = vals(&i, &[("a", Value::Int(3)), ("b", Value::Int(7))]);
    assert_eq!(
        run_obfuscated(
            &i,
            "{{ @a + @b | to_string }}",
            types,
            values,
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn obf_match_literal() {
    let i = Interner::new();
    let (ty, val) = int_context(&i, "n", 42);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ 42 = @n }}yes{{_}}no{{/}}"#,
            ty,
            val,
        )
        .await,
        "yes"
    );
}

#[tokio::test]
async fn obf_match_string_literal() {
    let i = Interner::new();
    let (ty, val) = string_context(&i, "name", "alice");
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ "alice" = @name }}found{{_}}nope{{/}}"#,
            ty,
            val,
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
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ x in @items }}{{ x | to_string }} {{/}}"#,
            ty,
            val,
        )
        .await,
        "1 2 3 "
    );
}

#[tokio::test]
async fn obf_nested_match_with_variable() {
    let i = Interner::new();
    let types = ctx(&i, &[("role", Ty::String), ("level", Ty::Int)]);
    let values = vals(
        &i,
        &[
            ("role", Value::String("admin".into())),
            ("level", Value::Int(5)),
        ],
    );
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ "admin" = @role }}{{ 0..10 = @level }}{{ $result = "low-admin" }}{{_}}{{ $result = "high-admin" }}{{/}}{{_}}{{ $result = "guest" }}{{/}}{{ $result }}"#,
            types,
            values,
        )
        .await,
        "low-admin"
    );
}

#[tokio::test]
async fn obf_lambda_filter_map() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ x = @items | filter(x -> x != 0) | map(x -> (x | to_string)) | collect }}{{ x | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "1, 2, 3"
    );
}

// ── Complex obfuscation tests ─────────────────────────────────

#[tokio::test]
async fn obf_variable_accumulate_in_loop() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ $sum = 0 }}{{ x in @items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}"#,
            ty,
            val,
        )
        .await,
        "60"
    );
}

#[tokio::test]
async fn obf_nested_iteration_with_match() {
    let i = Interner::new();
    let ty = ctx(
        &i,
        &[("rows", Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))))],
    );
    let val = vals(
        &i,
        &[(
            "rows",
            Value::List(vec![
                Value::List(vec![Value::Int(1), Value::Int(2)]),
                Value::List(vec![Value::Int(3), Value::Int(4)]),
            ]),
        )],
    );
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ row in @rows }}[{{ x in row }}{{ 3 = x }}three{{_}}{{ x | to_string }}{{/}} {{/}}]{{/}}"#,
            ty,
            val,
        )
        .await,
        "[1 2 ][three 4 ]"
    );
}

#[tokio::test]
async fn obf_object_destructure_and_format() {
    let i = Interner::new();
    let (ty, val) = users_list_context(&i);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ { name, age, } in @users }}{{ name }}({{ age | to_string }}) {{/}}"#,
            ty,
            val,
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
    let i = Interner::new();
    let (ty, val) = string_context(&i, "lang", "rust");
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ "go" = @lang }}Go{{ "rust" = }}Rust{{ "python" = }}Python{{_}}Other{{/}}"#,
            ty,
            val,
        )
        .await,
        "Rust"
    );
}

#[tokio::test]
async fn obf_range_pattern_with_variable() {
    let i = Interner::new();
    let types = ctx(&i, &[("score", Ty::Int)]);
    let values = vals(&i, &[("score", Value::Int(85))]);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ 90..=100 = @score }}{{ $grade = "A" }}{{ 80..90 = }}{{ $grade = "B" }}{{ 70..80 = }}{{ $grade = "C" }}{{_}}{{ $grade = "F" }}{{/}}{{ $grade }}"#,
            types,
            values,
        )
        .await,
        "B"
    );
}

#[tokio::test]
async fn obf_filter_accumulate_complex() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ $sum = 0 }}{{ x in (@items | filter(x -> x > 5) | collect) }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}"#,
            ty,
            val,
        )
        .await,
        "40"
    );
}

#[tokio::test]
async fn obf_pipe_chain_with_context() {
    let i = Interner::new();
    let types = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let values = vals(
        &i,
        &[(
            "names",
            Value::List(vec![
                Value::String("alice".into()),
                Value::String("bob".into()),
                Value::String("charlie".into()),
            ]),
        )],
    );
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ @names | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "alice, bob, charlie"
    );
}

#[tokio::test]
async fn obf_boolean_logic_in_match() {
    let i = Interner::new();
    let types = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let values = vals(&i, &[("a", Value::Int(5)), ("b", Value::Int(10))]);
    assert_eq!(
        run_obfuscated(
            &i,
            r#"{{ $result = "none" }}{{ 1..10 = @a }}{{ 5..15 = @b }}{{ $result = "both" }}{{_}}{{ $result = "a-only" }}{{/}}{{_}}{{ $result = "other" }}{{/}}{{ $result }}"#,
            types,
            values,
        )
        .await,
        "both"
    );
}

// ── Variant (Option) ────────────────────────────────────────────

#[tokio::test]
async fn variant_some_extract_value() {
    let i = Interner::new();
    let types = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::String)))]);
    let values = vals(
        &i,
        &[(
            "opt",
            Value::Variant {
                tag: i.intern("Some"),
                payload: Some(Box::new(Value::String("hello".into()))),
            },
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    let values = vals(
        &i,
        &[(
            "opt",
            Value::Variant {
                tag: i.intern("None"),
                payload: None,
            },
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @opt }}none{{_}}has value{{/}}".into(),
            types,
            values
        )
        .await,
        "none"
    );
}

#[tokio::test]
async fn variant_some_catch_all() {
    let i = Interner::new();
    let types = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    let values = vals(
        &i,
        &[(
            "opt",
            Value::Variant {
                tag: i.intern("None"),
                payload: None,
            },
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
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
    let i = Interner::new();
    let types = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    let values = vals(
        &i,
        &[(
            "opt",
            Value::Variant {
                tag: i.intern("Some"),
                payload: Some(Box::new(Value::Int(42))),
            },
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            "{{ Some(42) = @opt }}matched{{_}}no{{/}}".into(),
            types,
            values
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
    let types = ctx(&i, &[("data", Ty::bytes())]);
    let values = vals(
        &i,
        &[(
            "data",
            Value::List(vec![Value::Byte(0xFF), Value::Byte(0xFE)]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @data | to_utf8 }}invalid{{_}}valid{{/}}",
            types,
            values,
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
    let types = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(&i, &[("items", Value::List(vec![]))]);

    let err = run_expect_error(
        &i,
        r#"{{ x = @items | find(x -> x == 99) }}{{ x | to_string }}{{_}}{{/}}"#,
        types,
        values,
    )
    .await;

    assert!(
        matches!(err.kind, RuntimeErrorKind::EmptyCollection { ref operation } if operation == "find"),
        "expected EmptyCollection error, got: {err}",
    );
}

/// HOF reduce on empty list -> Stepped::Error (not panic).
#[tokio::test]
async fn error_reduce_empty_list() {
    let i = Interner::new();
    let types = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(&i, &[("items", Value::List(vec![]))]);

    let err = run_expect_error(
        &i,
        r#"{{ x = @items | reduce((a, b) -> a + b) }}{{ x | to_string }}{{_}}{{/}}"#,
        types,
        values,
    )
    .await;

    assert!(
        matches!(err.kind, RuntimeErrorKind::EmptyCollection { ref operation } if operation == "reduce"),
        "expected EmptyCollection error, got: {err}",
    );
}

// ── Structural enum runtime tests ──────────────────────────────

#[tokio::test]
async fn structural_enum_match_unit_variant() {
    let i = Interner::new();
    let types = ctx(&i, &[("s", Ty::Enum {
        name: i.intern("Status"),
        variants: FxHashMap::from_iter([
            (i.intern("Active"), None),
            (i.intern("Inactive"), None),
        ]),
    })]);
    let values = vals(&i, &[("s", Value::Variant {
        tag: i.intern("Active"),
        payload: None,
    })]);
    assert_eq!(
        run_ctx(&i, "{{ Status::Active = @s }}yes{{_}}no{{/}}", types, values).await,
        "yes"
    );
}

#[tokio::test]
async fn structural_enum_match_fallthrough() {
    let i = Interner::new();
    let types = ctx(&i, &[("s", Ty::Enum {
        name: i.intern("Status"),
        variants: FxHashMap::from_iter([
            (i.intern("Active"), None),
            (i.intern("Inactive"), None),
        ]),
    })]);
    let values = vals(&i, &[("s", Value::Variant {
        tag: i.intern("Inactive"),
        payload: None,
    })]);
    assert_eq!(
        run_ctx(&i, "{{ Status::Active = @s }}yes{{_}}no{{/}}", types, values).await,
        "no"
    );
}

#[tokio::test]
async fn structural_enum_multi_arm_match() {
    let i = Interner::new();
    let types = ctx(&i, &[("c", Ty::Enum {
        name: i.intern("Color"),
        variants: FxHashMap::from_iter([
            (i.intern("Red"), None),
            (i.intern("Green"), None),
            (i.intern("Blue"), None),
        ]),
    })]);
    let values = vals(&i, &[("c", Value::Variant {
        tag: i.intern("Green"),
        payload: None,
    })]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ Color::Red = @c }}r{{ Color::Green = }}g{{ Color::Blue = }}b{{/}}",
            types, values,
        ).await,
        "g"
    );
}

#[tokio::test]
async fn structural_enum_with_payload_match() {
    let i = Interner::new();
    let types = ctx(&i, &[("r", Ty::Enum {
        name: i.intern("Res"),
        variants: FxHashMap::from_iter([
            (i.intern("Ok"), Some(Box::new(Ty::String))),
            (i.intern("Err"), None),
        ]),
    })]);
    let values = vals(&i, &[("r", Value::Variant {
        tag: i.intern("Ok"),
        payload: Some(Box::new(Value::String("hello".into()))),
    })]);
    assert_eq!(
        run_ctx(&i, "{{ Res::Ok(v) = @r }}{{ v }}{{_}}err{{/}}", types, values).await,
        "hello"
    );
}

#[tokio::test]
async fn structural_enum_payload_fallthrough_to_unit() {
    let i = Interner::new();
    let types = ctx(&i, &[("r", Ty::Enum {
        name: i.intern("Res"),
        variants: FxHashMap::from_iter([
            (i.intern("Ok"), Some(Box::new(Ty::String))),
            (i.intern("Err"), None),
        ]),
    })]);
    let values = vals(&i, &[("r", Value::Variant {
        tag: i.intern("Err"),
        payload: None,
    })]);
    assert_eq!(
        run_ctx(&i, "{{ Res::Ok(v) = @r }}{{ v }}{{ Res::Err = }}fail{{/}}", types, values).await,
        "fail"
    );
}

#[tokio::test]
async fn structural_enum_separate_blocks_both_match() {
    // Two separate match blocks on same enum context — regression test.
    let i = Interner::new();
    let types = ctx(&i, &[("s", Ty::Enum {
        name: i.intern("AB"),
        variants: FxHashMap::from_iter([
            (i.intern("A"), None),
            (i.intern("B"), None),
        ]),
    })]);
    let values = vals(&i, &[("s", Value::Variant {
        tag: i.intern("B"),
        payload: None,
    })]);
    assert_eq!(
        run_ctx(&i, "{{ AB::A = @s }}a{{/}}{{ AB::B = @s }}b{{/}}", types, values).await,
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
    let mut emits = interp.execute_with_context(FxHashMap::default()).await;
    assert!(emits.len() <= 1, "script emitted {} values, expected at most 1", emits.len());
    emits.pop().unwrap_or(Value::Unit)
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
        Value::Variant { tag, .. } => assert_eq!(i.resolve(tag), "User"),
        other => panic!("expected Variant, got {other:?}"),
    }
}

#[tokio::test]
async fn script_hint_bool() {
    let i = Interner::new();
    let result = run_script_with_hint(&i, "false", &FxHashMap::default(), Some(&Ty::Bool)).await;
    assert_eq!(result, Value::Bool(false));
}

#[tokio::test]
async fn script_hint_string() {
    let i = Interner::new();
    let result = run_script_with_hint(&i, "\"\"", &FxHashMap::default(), Some(&Ty::String)).await;
    assert_eq!(result, Value::String("".into()));
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
        Value::Object(fields) => {
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
        Value::Variant { tag, .. } => assert_eq!(i.resolve(tag), "Dynamic"),
        other => panic!("expected Variant, got {other:?}"),
    }
}

#[tokio::test]
async fn script_no_hint_enum_variant() {
    // Same as above but WITHOUT hint — should also work (CLI path)
    let i = Interner::new();
    let result = run_script_with_hint(&i, "Focus::User", &FxHashMap::default(), None).await;
    match result {
        Value::Variant { tag, .. } => assert_eq!(i.resolve(tag), "User"),
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
    let types = ctx(&i, &[("Focus", focus_ty.clone())]);
    let values = vals(&i, &[("Focus", Value::Variant { tag: i.intern("User"), payload: None })]);

    let output = run_ctx(
        &i,
        "{-{ Focus::User = @Focus }}user{-{ Focus::Char = }}char{-{ Focus::System = }}sys{-{ / }}",
        types,
        values,
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
    let types = ctx(&i, &[
        ("Attempt", Ty::Bool),
        ("Impersonation", imp_ty.clone()),
    ]);
    let values = vals(&i, &[
        ("Attempt", Value::Bool(true)),
        ("Impersonation", Value::Variant { tag: i.intern("Deny"), payload: None }),
    ]);

    let output = run_ctx(
        &i,
        "{-{ (true, Impersonation::Deny) = (@Attempt, @Impersonation) }}deny{-{ (true, Impersonation::Allowed) = }}allow{-{ (false, Impersonation::Deny) = }}fdenial{-{ / }}",
        types,
        values,
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
    let types = ctx(&i, &[("Focus", focus_ty.clone())]);

    // Test with Custom variant
    let values = vals(&i, &[("Focus", Value::Variant {
        tag: i.intern("Custom"),
        payload: Some(Box::new(Value::Object(FxHashMap::from_iter([
            (i.intern("custom"), Value::String("hello".into())),
        ])))),
    })]);

    let output = run_ctx(
        &i,
        "{-{ Focus::User = @Focus }}user{-{ Focus::Char = }}char{-{ Focus::Custom({custom,}) = }}{{ custom }}{-{ / }}",
        types,
        values,
    ).await;
    assert_eq!(output, "hello");
}

#[tokio::test]
async fn template_var_scoped_inside_match_arm() {
    // Variable defined inside a match arm body is properly scoped.
    // It's usable INSIDE the arm, not outside.
    let i = Interner::new();
    let types = ctx(&i, &[("cond", Ty::Bool)]);
    let values = vals(&i, &[("cond", Value::Bool(true))]);

    // x is defined and used INSIDE the same arm — this must work
    let output = run_ctx(
        &i,
        "before{-{ true = @cond }}{{ x = \"hello\" }}{{ x }}{-{ / }}after",
        types,
        values,
    ).await;
    assert_eq!(output, "beforehelloafter");
}

#[tokio::test]
async fn template_var_scoped_inside_match_arm_not_taken() {
    // When arm is not taken, variables inside it are never accessed
    let i = Interner::new();
    let types = ctx(&i, &[("cond", Ty::Bool)]);
    let values = vals(&i, &[("cond", Value::Bool(false))]);

    let output = run_ctx(
        &i,
        "before{-{ true = @cond }}{{ x = \"hello\" }}{{ x }}{-{ / }}after",
        types,
        values,
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
    let types = ctx(&i, &[("Focus", focus_ty)]);
    let values = vals(&i, &[("Focus", Value::Variant { tag: i.intern("User"), payload: None })]);

    // Template tries to match multiple variants — typechecker should handle this
    let output = run_ctx(
        &i,
        "{-{ Focus::User = @Focus }}user{-{ Focus::Char = }}char{-{ Focus::System = }}sys{-{ / }}",
        types,
        values,
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
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | map(x -> { y = x * 10; y }) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "10, 20, 30"
    );
}

#[tokio::test]
async fn block_in_pipe() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | x -> { a = last(x); unwrap_or(a, 0) } | to_string }}"#,
            ty,
            val,
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn block_with_field_access() {
    let i = Interner::new();
    let (ty, val) = users_list_context(&i);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ { a = last(@users); unwrap(a).name } }}"#,
            ty,
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
    let types = ctx(&i, &[("f", Ty::Float)]);
    let values = vals(&i, &[("f", Value::Float(9.99))]);
    assert_eq!(
        run_ctx(&i, "{{ @f | to_int | to_string }}", types, values).await,
        "9"
    );
}

#[tokio::test]
async fn builtin_to_int_from_byte() {
    let i = Interner::new();
    let types = ctx(&i, &[("b", Ty::Byte)]);
    let values = vals(&i, &[("b", Value::Byte(65))]);
    assert_eq!(
        run_ctx(&i, "{{ @b | to_int | to_string }}", types, values).await,
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
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(&i, "{{ @items | len | to_string }}", ty, val).await,
        "3"
    );
}

#[tokio::test]
async fn builtin_len_empty_list() {
    let i = Interner::new();
    let types = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(&i, &[("items", Value::List(vec![]))]);
    assert_eq!(
        run_ctx(&i, "{{ @items | len | to_string }}", types, values).await,
        "0"
    );
}

#[tokio::test]
async fn builtin_rev_iter_list() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | rev_iter | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "3, 2, 1"
    );
}

#[tokio::test]
async fn builtin_flatten_list() {
    let i = Interner::new();
    let types = ctx(
        &i,
        &[(
            "items",
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
        )],
    );
    let values = vals(
        &i,
        &[(
            "items",
            Value::List(vec![
                Value::List(vec![Value::Int(1), Value::Int(2)]),
                Value::List(vec![Value::Int(3)]),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flatten | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn builtin_join_list() {
    let i = Interner::new();
    let types = ctx(&i, &[("words", Ty::List(Box::new(Ty::String)))]);
    let values = vals(
        &i,
        &[(
            "words",
            Value::List(vec![
                Value::String("a".into()),
                Value::String("b".into()),
                Value::String("c".into()),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(&i, r#"{{ @words | join("-") }}"#, types, values).await,
        "a-b-c"
    );
}

#[tokio::test]
async fn builtin_join_empty_list() {
    let i = Interner::new();
    let types = ctx(&i, &[("words", Ty::List(Box::new(Ty::String)))]);
    let values = vals(&i, &[("words", Value::List(vec![]))]);
    assert_eq!(
        run_ctx(&i, r#"{{ @words | join(", ") }}"#, types, values).await,
        ""
    );
}

#[tokio::test]
async fn builtin_contains_list_found() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, "{{ @items | contains(2) | to_string }}", ty, val).await,
        "true"
    );
}

#[tokio::test]
async fn builtin_contains_list_not_found() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(&i, "{{ @items | contains(99) | to_string }}", ty, val).await,
        "false"
    );
}

#[tokio::test]
async fn builtin_first_list_some() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(&i, "{{ @items | first | unwrap | to_string }}", ty, val).await,
        "10"
    );
}

#[tokio::test]
async fn builtin_first_list_none() {
    let i = Interner::new();
    let types = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(&i, &[("items", Value::List(vec![]))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | first }}empty{{_}}has{{/}}",
            types,
            values,
        )
        .await,
        "empty"
    );
}

#[tokio::test]
async fn builtin_last_list_some() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(&i, "{{ @items | last | unwrap | to_string }}", ty, val).await,
        "30"
    );
}

#[tokio::test]
async fn builtin_last_list_none() {
    let i = Interner::new();
    let types = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(&i, &[("items", Value::List(vec![]))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | last }}empty{{_}}has{{/}}",
            types,
            values,
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
    let types = ctx(&i, &[("data", Ty::bytes())]);
    let values = vals(
        &i,
        &[(
            "data",
            Value::List(vec![
                Value::Byte(0x48), // H
                Value::Byte(0x69), // i
                Value::Byte(0xFF), // invalid
            ]),
        )],
    );
    assert_eq!(
        run_ctx(&i, "{{ @data | to_utf8_lossy }}", types, values).await,
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
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "1, 2, 3"
    );
}

#[tokio::test]
async fn builtin_rev_iter() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | rev_iter | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "3, 2, 1"
    );
}

#[tokio::test]
async fn builtin_take() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30, 40, 50]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | take(3) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "10, 20, 30"
    );
}

#[tokio::test]
async fn builtin_skip() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30, 40, 50]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | skip(2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "30, 40, 50"
    );
}

#[tokio::test]
async fn builtin_chain() {
    let i = Interner::new();
    let types = ctx(
        &i,
        &[
            ("a", Ty::List(Box::new(Ty::Int))),
            ("b", Ty::List(Box::new(Ty::Int))),
        ],
    );
    let values = vals(
        &i,
        &[
            ("a", Value::List(vec![Value::Int(1), Value::Int(2)])),
            ("b", Value::List(vec![Value::Int(3), Value::Int(4)])),
        ],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @a | chain(@b) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "1, 2, 3, 4"
    );
}

// ── Iterator HOFs ───────────────────────────────────────────────

#[tokio::test]
async fn builtin_filter() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | filter(x -> x > 3) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "4, 5"
    );
}

#[tokio::test]
async fn builtin_map() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | map(x -> x * 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "2, 4, 6"
    );
}

#[tokio::test]
async fn builtin_find() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | find(x -> x > 3) | to_string }}"#,
            ty,
            val,
        )
        .await,
        "4"
    );
}

#[tokio::test]
async fn builtin_reduce() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | reduce((a, b) -> a + b) | to_string }}"#,
            ty,
            val,
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn builtin_fold() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | fold(100, (acc, x) -> acc + x) | to_string }}"#,
            ty,
            val,
        )
        .await,
        "106"
    );
}

#[tokio::test]
async fn builtin_any_true() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | any(x -> x > 2) | to_string }}",
            ty,
            val,
        )
        .await,
        "true"
    );
}

#[tokio::test]
async fn builtin_any_false() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | any(x -> x > 10) | to_string }}",
            ty,
            val,
        )
        .await,
        "false"
    );
}

#[tokio::test]
async fn builtin_all_true() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![2, 4, 6]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | all(x -> x > 0) | to_string }}",
            ty,
            val,
        )
        .await,
        "true"
    );
}

#[tokio::test]
async fn builtin_all_false() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![2, 4, 6]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | all(x -> x > 3) | to_string }}",
            ty,
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
    let types = ctx(
        &i,
        &[(
            "items",
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
        )],
    );
    let values = vals(
        &i,
        &[(
            "items",
            Value::List(vec![
                Value::List(vec![Value::Int(1), Value::Int(2)]),
                Value::List(vec![Value::Int(3), Value::Int(4)]),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flatten | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "1, 2, 3, 4"
    );
}

#[tokio::test]
async fn builtin_flat_map_iter() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flat_map(x -> [x, x * 10]) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "1, 10, 2, 20, 3, 30"
    );
}

#[tokio::test]
async fn builtin_join_iter() {
    let i = Interner::new();
    let types = ctx(&i, &[("words", Ty::List(Box::new(Ty::String)))]);
    let values = vals(
        &i,
        &[(
            "words",
            Value::List(vec![
                Value::String("hello".into()),
                Value::String("world".into()),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @words | join(" ") }}"#,
            types,
            values,
        )
        .await,
        "hello world"
    );
}

#[tokio::test]
async fn builtin_contains_iter_found() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | contains(20) | to_string }}",
            ty,
            val,
        )
        .await,
        "true"
    );
}

#[tokio::test]
async fn builtin_contains_iter_not_found() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | contains(99) | to_string }}",
            ty,
            val,
        )
        .await,
        "false"
    );
}

#[tokio::test]
async fn builtin_first_iter_some() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | first | unwrap | to_string }}",
            ty,
            val,
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn builtin_first_iter_none() {
    let i = Interner::new();
    let types = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(&i, &[("items", Value::List(vec![]))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | first }}empty{{_}}has{{/}}",
            types,
            values,
        )
        .await,
        "empty"
    );
}

#[tokio::test]
async fn builtin_last_iter_some() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ @items | last | unwrap | to_string }}",
            ty,
            val,
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn builtin_last_iter_none() {
    let i = Interner::new();
    let types = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let values = vals(&i, &[("items", Value::List(vec![]))]);
    assert_eq!(
        run_ctx(
            &i,
            "{{ None = @items | last }}empty{{_}}has{{/}}",
            types,
            values,
        )
        .await,
        "empty"
    );
}

// ── Iterator Pipeline Combos ────────────────────────────────────

#[tokio::test]
async fn builtin_filter_map_collect() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | filter(x -> x > 2) | map(x -> x * 10) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "30, 40, 50, 60"
    );
}

#[tokio::test]
async fn builtin_take_skip_combo() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | skip(2) | take(3) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "3, 4, 5"
    );
}

#[tokio::test]
async fn builtin_chain_filter_map() {
    let i = Interner::new();
    let types = ctx(
        &i,
        &[
            ("a", Ty::List(Box::new(Ty::Int))),
            ("b", Ty::List(Box::new(Ty::Int))),
        ],
    );
    let values = vals(
        &i,
        &[
            ("a", Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)])),
            ("b", Value::List(vec![Value::Int(4), Value::Int(5), Value::Int(6)])),
        ],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @a | chain(@b) | filter(x -> x > 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "3, 4, 5, 6"
    );
}

#[tokio::test]
async fn builtin_rev_iter_filter() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | rev_iter | filter(x -> x > 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
            val,
        )
        .await,
        "5, 4, 3"
    );
}

#[tokio::test]
async fn builtin_flatten_iter_then_filter() {
    let i = Interner::new();
    let types = ctx(
        &i,
        &[(
            "items",
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
        )],
    );
    let values = vals(
        &i,
        &[(
            "items",
            Value::List(vec![
                Value::List(vec![Value::Int(1), Value::Int(2)]),
                Value::List(vec![Value::Int(3), Value::Int(4)]),
            ]),
        )],
    );
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | flatten | filter(x -> x > 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            types,
            values,
        )
        .await,
        "3, 4"
    );
}

// ── Pmap (parallel map) ─────────────────────────────────────────

#[tokio::test]
async fn builtin_pmap() {
    let i = Interner::new();
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    // pmap should produce same results as map (order preserved)
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ @items | pmap(x -> x * 2) | map(x -> (x | to_string)) | collect | join(", ") }}"#,
            ty,
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
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ base = 100 }}{{ make_adder = scale -> (x -> x * scale + base) }}{{ adder = make_adder(10) }}{{ @items | map(adder) | map(x -> (x | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            ty,
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
    let (ty, val) = items_context(&i, vec![10, 20, 30]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ offset = 5 }}{{ add_offset = x -> x + offset }}{{ @items | map(add_offset) | map(x -> (x | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            ty,
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
    let (ty, val) = items_context(&i, vec![1, 2, 3]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ make_scaler = factor -> (x -> x * factor) }}{{ double = make_scaler(2) }}{{ @items | map(double) | map(x -> (x | to_string)) | collect | join(", ") }}{{_}}{{/}}"#,
            ty,
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
    let (ty, val) = items_context(&i, vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_ctx(
            &i,
            r#"{{ threshold = 3 }}{{ above = @items | filter(x -> x > threshold) | map(x -> (x | to_string)) | collect | join(", ") }}{{ below = @items | filter(x -> x <= threshold) | map(x -> (x | to_string)) | collect | join(", ") }}{{ above }}|{{ below }}"#,
            ty,
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
