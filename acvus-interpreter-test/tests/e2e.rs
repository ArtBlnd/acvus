use std::collections::{BTreeMap, HashMap};

use acvus_interpreter::{ExternFn, ExternFnBody, ExternFnRegistry, ExternFnSig, PureValue, Value};
use acvus_interpreter_test::*;
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
    let (ty, val) = string_storage("name", "alice");
    assert_eq!(
        run_with_storage("Hello, {{ $name }}!", ty, val).await,
        "Hello, alice!"
    );
}

// ── Storage ──────────────────────────────────────────────────────

#[tokio::test]
async fn storage_read() {
    let (ty, val) = int_storage("count", 42);
    assert_eq!(
        run_with_storage("{{ $count | to_string }}", ty, val).await,
        "42"
    );
}

#[tokio::test]
async fn storage_write() {
    let (ty, val) = int_storage("count", 0);
    assert_eq!(run_with_storage("{{ $count = 42 }}", ty, val).await, "");
}

#[tokio::test]
async fn storage_write_then_read() {
    let (ty, val) = int_storage("x", 0);
    assert_eq!(
        run_with_storage("{{ $x = 42 }}{{ $x | to_string }}", ty, val).await,
        "42"
    );
}

#[tokio::test]
async fn storage_field_access() {
    let (ty, val) = user_storage();
    assert_eq!(
        run_with_storage("{{ $user.name }}", ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn storage_write_computed() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([
        ("a".into(), PureValue::Int(10)),
        ("b".into(), PureValue::Int(32)),
    ]);
    assert_eq!(
        run_with_storage("{{ $a = $a + $b }}{{ $a | to_string }}", types, values).await,
        "42"
    );
}

// ── Arithmetic ───────────────────────────────────────────────────

#[tokio::test]
async fn arithmetic_to_string() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([
        ("a".into(), PureValue::Int(3)),
        ("b".into(), PureValue::Int(7)),
    ]);
    assert_eq!(
        run_with_storage("{{ $a + $b | to_string }}", types, values).await,
        "10"
    );
}

#[tokio::test]
async fn unary_negation() {
    let (ty, val) = int_storage("n", 5);
    assert_eq!(
        run_with_storage(r#"{{ x = -$n }}{{ x | to_string }}{{_}}{{/}}"#, ty, val).await,
        "-5"
    );
}

#[tokio::test]
async fn boolean_not() {
    let types = HashMap::from([("flag".into(), Ty::Bool)]);
    let values = HashMap::from([("flag".into(), PureValue::Bool(true))]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = !$flag }}{{ x | to_string }}{{_}}{{/}}"#,
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
    let values = HashMap::from([
        ("a".into(), PureValue::Int(10)),
        ("b".into(), PureValue::Int(5)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $a > $b }}{{ x | to_string }}{{_}}{{/}}"#,
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
    let (ty, val) = string_storage("name", "alice");
    assert_eq!(
        run_with_storage(r#"{{ x = $name }}{{ x }}"#, ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn match_literal_filter_hit() {
    let (ty, val) = string_storage("role", "admin");
    assert_eq!(
        run_with_storage(
            r#"{{ "admin" = $role }}admin page{{_}}guest page{{/}}"#,
            ty,
            val
        )
        .await,
        "admin page"
    );
}

#[tokio::test]
async fn match_literal_filter_miss() {
    let (ty, val) = string_storage("role", "user");
    assert_eq!(
        run_with_storage(
            r#"{{ "admin" = $role }}admin page{{_}}guest page{{/}}"#,
            ty,
            val
        )
        .await,
        "guest page"
    );
}

#[tokio::test]
async fn multi_arm_match() {
    let (ty, val) = string_storage("role", "user");
    assert_eq!(
        run_with_storage(
            r#"{{ "admin" = $role }}admin{{ "user" = }}user{{_}}guest{{/}}"#,
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
    let values = HashMap::from([("flag".into(), PureValue::Bool(true))]);
    assert_eq!(
        run_with_storage(r#"{{ true = $flag }}on{{_}}off{{/}}"#, types, values).await,
        "on"
    );
}

#[tokio::test]
async fn match_binding_with_body() {
    let (ty, val) = user_storage();
    assert_eq!(
        run_with_storage(
            r#"{{ { name, } = $user }}{{ name }} is here{{_}}no user{{/}}"#,
            ty,
            val
        )
        .await,
        "alice is here"
    );
}

#[tokio::test]
async fn variable_shadowing() {
    let (ty, val) = string_storage("name", "alice");
    assert_eq!(
        run_with_storage(
            r#"{{ x = "outer" }}{{ x = $name }}{{ x }}{{_}}{{/}}"#,
            ty,
            val
        )
        .await,
        "alice"
    );
}

#[tokio::test]
async fn catch_all_with_binding() {
    let (ty, val) = string_storage("role", "viewer");
    assert_eq!(
        run_with_storage(
            r#"{{ "admin" = $role }}admin{{_}}{{ fallback = "guest" }}{{ fallback }}{{/}}"#,
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
    let values = HashMap::from([
        ("a".into(), PureValue::Int(5)),
        ("b".into(), PureValue::Int(5)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ true = $a == $b }}equal{{_}}not equal{{/}}"#,
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
        ("role".into(), PureValue::String("admin".into())),
        ("level".into(), PureValue::Int(5)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ "admin" = $role }}{{ 1..10 = $level }}low{{_}}high{{/}}{{_}}guest{{/}}"#,
            types,
            values
        )
        .await,
        "low"
    );
}

// ── Storage ref in match arm ─────────────────────────────────────

#[tokio::test]
async fn storage_new_ref_binding() {
    let (ty, val) = string_storage("name", "alice");
    assert_eq!(
        run_with_storage(r#"{{ $result = $name }}{{ $result }}"#, ty, val).await,
        "alice"
    );
}

#[tokio::test]
async fn storage_new_ref_in_match_arm() {
    let (ty, val) = string_storage("role", "admin");
    assert_eq!(
        run_with_storage(
            r#"{{ "admin" = $role }}{{ $selected = "yes" }}{{_}}{{ $selected = "no" }}{{/}}{{ $selected }}"#,
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
        run_simple(r#"{{ x = 0..5 }}{{ x | to_string }}"#).await,
        "0..5"
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
    let (ty, val) = int_storage("age", 5);
    assert_eq!(
        run_with_storage(
            r#"{{ 0..10 = $age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
            ty,
            val
        )
        .await,
        "child"
    );
}

#[tokio::test]
async fn range_pattern_miss() {
    let (ty, val) = int_storage("age", 25);
    assert_eq!(
        run_with_storage(
            r#"{{ 0..10 = $age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
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
    let (ty, val) = items_storage(vec![1, 2, 3]);
    assert_eq!(
        run_with_storage("{{ x in $items }}{{ x | to_string }}{{/}}", ty, val).await,
        "123"
    );
}

#[tokio::test]
async fn iter_object_destructure() {
    let (ty, val) = users_list_storage();
    assert_eq!(
        run_with_storage("{{ { name, } in $users }}{{ name }}{{/}}", ty, val).await,
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
        PureValue::List(vec![
            PureValue::Tuple(vec![
                PureValue::String("a".into()),
                PureValue::Int(1),
            ]),
            PureValue::Tuple(vec![
                PureValue::String("b".into()),
                PureValue::Int(2),
            ]),
        ]),
    )]);
    assert_eq!(
        run_with_storage("{{ (a, _) in $pairs }}{{ a }}{{/}}", ty, val).await,
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
        PureValue::List(vec![
            PureValue::List(vec![PureValue::Int(1), PureValue::Int(2)]),
            PureValue::List(vec![PureValue::Int(3), PureValue::Int(4)]),
        ]),
    )]);
    assert_eq!(
        run_with_storage(
            "{{ row in $matrix }}{{ x in row }}{{ x | to_string }}{{/}}{{/}}",
            ty,
            val
        )
        .await,
        "1234"
    );
}

#[tokio::test]
async fn storage_write_in_iteration() {
    let types = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("last".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "items".into(),
            PureValue::List(vec![
                PureValue::Int(10),
                PureValue::Int(20),
                PureValue::Int(30),
            ]),
        ),
        ("last".into(), PureValue::Int(0)),
    ]);
    assert_eq!(
        run_with_storage(
            "{{ x in $items }}{{ $last = x }}{{/}}{{ $last | to_string }}",
            types,
            values
        )
        .await,
        "30"
    );
}

#[tokio::test]
async fn storage_accumulate_in_loop() {
    let types = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("sum".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "items".into(),
            PureValue::List(vec![
                PureValue::Int(1),
                PureValue::Int(2),
                PureValue::Int(3),
            ]),
        ),
        ("sum".into(), PureValue::Int(0)),
    ]);
    assert_eq!(
        run_with_storage(
            "{{ x in $items }}{{ $sum = $sum + x }}{{/}}{{ $sum | to_string }}",
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
    let (ty, val) = items_storage(vec![10, 20, 30]);
    assert_eq!(
        run_with_storage(
            r#"{{ [a, b, ..] = $items }}{{ a | to_string }}{{_}}empty{{/}}"#,
            ty,
            val
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn list_destructure_tail() {
    let (ty, val) = items_storage(vec![10, 20, 30]);
    assert_eq!(
        run_with_storage(
            r#"{{ [.., a, b] = $items }}{{ a | to_string }}{{_}}empty{{/}}"#,
            ty,
            val
        )
        .await,
        "20"
    );
}

#[tokio::test]
async fn list_destructure_head_and_tail() {
    let (ty, val) = items_storage(vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_with_storage(
            r#"{{ [first, .., last] = $items }}{{ first | to_string }}-{{ last | to_string }}{{_}}empty{{/}}"#,
            ty,
            val,
        )
        .await,
        "1-5"
    );
}

#[tokio::test]
async fn list_exact_match_hit() {
    let (ty, val) = items_storage(vec![10, 20]);
    assert_eq!(
        run_with_storage(
            r#"{{ [a, b] = $items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
            ty,
            val
        )
        .await,
        "10"
    );
}

#[tokio::test]
async fn list_exact_match_miss() {
    let (ty, val) = items_storage(vec![10, 20, 30]);
    assert_eq!(
        run_with_storage(
            r#"{{ [a, b] = $items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
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
    let (ty, val) = user_storage();
    assert_eq!(
        run_with_storage(
            r#"{{ { name, age, } = $user }}{{ name }}:{{ age | to_string }}{{/}}"#,
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
        PureValue::Object(BTreeMap::from([(
            "user".into(),
            PureValue::Object(BTreeMap::from([(
                "address".into(),
                PureValue::Object(BTreeMap::from([(
                    "city".into(),
                    PureValue::String("Seoul".into()),
                )])),
            )])),
        )])),
    )]);
    assert_eq!(
        run_with_storage("{{ $data.user.address.city }}", ty, val).await,
        "Seoul"
    );
}

// ── Tuple ────────────────────────────────────────────────────────

#[tokio::test]
async fn tuple_expression() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::String)]);
    let values = HashMap::from([
        ("a".into(), PureValue::Int(42)),
        ("b".into(), PureValue::String("hello".into())),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ t = ($a, $b) }}{{ t | to_string }}{{_}}{{/}}"#,
            types,
            values
        )
        .await,
        "(42, hello)"
    );
}

#[tokio::test]
async fn tuple_pattern_binding() {
    let types = HashMap::from([("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let values = HashMap::from([(
        "pair".into(),
        PureValue::Tuple(vec![
            PureValue::String("alice".into()),
            PureValue::Int(30),
        ]),
    )]);
    assert_eq!(
        run_with_storage(r#"{{ (name, age) = $pair }}{{ name }}{{/}}"#, types, values).await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_wildcard() {
    let types = HashMap::from([("pair".into(), Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let values = HashMap::from([(
        "pair".into(),
        PureValue::Tuple(vec![
            PureValue::String("alice".into()),
            PureValue::Int(30),
        ]),
    )]);
    assert_eq!(
        run_with_storage(r#"{{ (name, _) = $pair }}{{ name }}{{/}}"#, types, values).await,
        "alice"
    );
}

#[tokio::test]
async fn tuple_pattern_literal_match_hit() {
    let types = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
    let values = HashMap::from([
        ("a".into(), PureValue::Int(0)),
        ("b".into(), PureValue::Int(1)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ (0, 1) = ($a, $b) }}zero-one{{ (1, _) = }}one-any{{_}}other{{/}}"#,
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
        Ty::Tuple(vec![
            Ty::Tuple(vec![Ty::Int, Ty::Int]),
            Ty::String,
        ]),
    )]);
    let values = HashMap::from([(
        "data".into(),
        PureValue::Tuple(vec![
            PureValue::Tuple(vec![PureValue::Int(1), PureValue::Int(2)]),
            PureValue::String("hello".into()),
        ]),
    )]);
    assert_eq!(
        run_with_storage(r#"{{ ((a, b), label) = $data }}{{ label }}{{/}}"#, types, values).await,
        "hello"
    );
}

// ── Pipe & builtins ──────────────────────────────────────────────

#[tokio::test]
async fn pipe_to_string() {
    let (ty, val) = int_storage("n", 42);
    assert_eq!(
        run_with_storage("{{ $n | to_string }}", ty, val).await,
        "42"
    );
}

#[tokio::test]
async fn to_float_conversion() {
    let (ty, val) = int_storage("n", 5);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $n | to_float }}{{ x | to_string }}{{_}}{{/}}"#,
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
    let values = HashMap::from([("f".into(), PureValue::Float(3.7))]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $f | to_int }}{{ x | to_string }}{{_}}{{/}}"#,
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
    let (ty, val) = items_storage(vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | filter(x -> x != 0) }}{{ x | to_string }}"#,
            ty,
            val
        )
        .await,
        "[1, 2, 3]"
    );
}

#[tokio::test]
async fn lambda_map() {
    let (ty, val) = items_storage(vec![1, 2, 3]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | map(i -> i + 1) }}{{ x | to_string }}"#,
            ty,
            val
        )
        .await,
        "[2, 3, 4]"
    );
}

#[tokio::test]
async fn lambda_pmap() {
    let (ty, val) = items_storage(vec![1, 2, 3]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | pmap(i -> i | to_string) }}{{ x | to_string }}"#,
            ty,
            val
        )
        .await,
        "[1, 2, 3]"
    );
}

#[tokio::test]
async fn pipe_filter_map() {
    let (ty, val) = items_storage(vec![0, 1, 2, 0, 3]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | filter(x -> x != 0) | map(x -> x | to_string) }}{{ x | to_string }}"#,
            ty,
            val,
        )
        .await,
        "[1, 2, 3]"
    );
}

#[tokio::test]
async fn triple_pipe_chain() {
    let (ty, val) = items_storage(vec![0, 1, 2, 3]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | filter(i -> i != 0) | map(i -> i + 1) | map(i -> i | to_string) }}{{ x | to_string }}"#,
            ty,
            val,
        )
        .await,
        "[2, 3, 4]"
    );
}

#[tokio::test]
async fn closure_capture_local() {
    let (ty, val) = items_storage(vec![1, 3, 5, 7, 10]);
    assert_eq!(
        run_with_storage(
            r#"{{ threshold = 5 }}{{ x = $items | filter(i -> i > threshold) }}{{ x | to_string }}{{_}}{{/}}"#,
            ty,
            val,
        )
        .await,
        "[7, 10]"
    );
}

#[tokio::test]
async fn closure_capture_storage() {
    let types = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("threshold".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "items".into(),
            PureValue::List(vec![
                PureValue::Int(1),
                PureValue::Int(5),
                PureValue::Int(10),
            ]),
        ),
        ("threshold".into(), PureValue::Int(3)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | filter(i -> i > $threshold) }}{{ x | to_string }}"#,
            types,
            values
        )
        .await,
        "[5, 10]"
    );
}

#[tokio::test]
async fn lambda_field_access() {
    let (ty, val) = users_list_storage();
    assert_eq!(
        run_with_storage(
            r#"{{ x = $users | map(u -> u.name) }}{{ x | to_string }}"#,
            ty,
            val
        )
        .await,
        "[alice, bob]"
    );
}

#[tokio::test]
async fn lambda_negate_param() {
    let (ty, val) = items_storage(vec![1, 2, 3]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | map(i -> -i) }}{{ x | to_string }}"#,
            ty,
            val
        )
        .await,
        "[-1, -2, -3]"
    );
}

#[tokio::test]
async fn lambda_not_param() {
    let types = HashMap::from([("flags".into(), Ty::List(Box::new(Ty::Bool)))]);
    let values = HashMap::from([(
        "flags".into(),
        PureValue::List(vec![
            PureValue::Bool(true),
            PureValue::Bool(false),
            PureValue::Bool(true),
        ]),
    )]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $flags | map(i -> !i) }}{{ x | to_string }}"#,
            types,
            values
        )
        .await,
        "[false, true, false]"
    );
}

#[tokio::test]
async fn lambda_string_concat() {
    let types = HashMap::from([("names".into(), Ty::List(Box::new(Ty::String)))]);
    let values = HashMap::from([(
        "names".into(),
        PureValue::List(vec![
            PureValue::String("alice".into()),
            PureValue::String("bob".into()),
        ]),
    )]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $names | map(n -> n + "!") }}{{ x | to_string }}"#,
            types,
            values
        )
        .await,
        "[alice!, bob!]"
    );
}

#[tokio::test]
async fn lambda_float_arithmetic() {
    let types = HashMap::from([("vals".into(), Ty::List(Box::new(Ty::Float)))]);
    let values = HashMap::from([(
        "vals".into(),
        PureValue::List(vec![
            PureValue::Float(1.5),
            PureValue::Float(2.5),
        ]),
    )]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $vals | map(v -> v * 2.0) }}{{ x | to_string }}"#,
            types,
            values
        )
        .await,
        "[3, 5]"
    );
}

#[tokio::test]
async fn filter_then_map_field() {
    let (ty, val) = users_list_storage();
    assert_eq!(
        run_with_storage(
            r#"{{ x = $users | filter(u -> u.age > 18) | map(u -> u.name) }}{{ x | to_string }}"#,
            ty,
            val,
        )
        .await,
        "[alice, bob]"
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
            PureValue::List(vec![
                PureValue::Int(-1),
                PureValue::Int(0),
                PureValue::Int(1),
                PureValue::Int(2),
            ]),
        ),
        ("offset".into(), PureValue::Int(1)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | map(i -> i + $offset) | filter(i -> i > 0) }}{{ x | to_string }}"#,
            types,
            values,
        )
        .await,
        "[1, 2, 3]"
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
    let (ty, val) = int_storage("n", 21);
    let output = run(
        r#"{{ x = double($n) }}{{ x | to_string }}{{_}}{{/}}"#,
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
        ("a".into(), PureValue::Bool(true)),
        ("b".into(), PureValue::Bool(true)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ true = $a && $b }}yes{{_}}no{{/}}"#,
            types,
            values
        )
        .await,
        "yes"
    );
}

#[tokio::test]
async fn and_one_false() {
    let types = HashMap::from([("a".into(), Ty::Bool), ("b".into(), Ty::Bool)]);
    let values = HashMap::from([
        ("a".into(), PureValue::Bool(true)),
        ("b".into(), PureValue::Bool(false)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ true = $a && $b }}yes{{_}}no{{/}}"#,
            types,
            values
        )
        .await,
        "no"
    );
}

#[tokio::test]
async fn or_one_true() {
    let types = HashMap::from([("a".into(), Ty::Bool), ("b".into(), Ty::Bool)]);
    let values = HashMap::from([
        ("a".into(), PureValue::Bool(false)),
        ("b".into(), PureValue::Bool(true)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ true = $a || $b }}yes{{_}}no{{/}}"#,
            types,
            values
        )
        .await,
        "yes"
    );
}

#[tokio::test]
async fn or_both_false() {
    let types = HashMap::from([("a".into(), Ty::Bool), ("b".into(), Ty::Bool)]);
    let values = HashMap::from([
        ("a".into(), PureValue::Bool(false)),
        ("b".into(), PureValue::Bool(false)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ true = $a || $b }}yes{{_}}no{{/}}"#,
            types,
            values
        )
        .await,
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
        ("a".into(), PureValue::Bool(true)),
        ("b".into(), PureValue::Bool(false)),
        ("c".into(), PureValue::Bool(false)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ true = $a || $b && $c }}yes{{_}}no{{/}}"#,
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
    let values = HashMap::from([("x".into(), PureValue::Int(15))]);
    assert_eq!(
        run_with_storage(
            r#"{{ true = $x > 10 && $x < 20 }}in range{{_}}out{{/}}"#,
            types,
            values
        )
        .await,
        "in range"
    );
}

#[tokio::test]
async fn logical_in_filter() {
    let (ty, val) = items_storage(vec![1, 5, 10, 15, 20, 25]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $items | filter(i -> i > 5 && i < 20) }}{{ x | to_string }}"#,
            ty,
            val
        )
        .await,
        "[10, 15]"
    );
}

// ── Complex scenarios ───────────────────────────────────────────

#[tokio::test]
async fn nested_match_with_storage_write() {
    let types = HashMap::from([
        ("role".into(), Ty::String),
        ("level".into(), Ty::Int),
        ("result".into(), Ty::String),
    ]);
    let values = HashMap::from([
        ("role".into(), PureValue::String("admin".into())),
        ("level".into(), PureValue::Int(5)),
        ("result".into(), PureValue::String("".into())),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ "admin" = $role }}{{ 0..10 = $level }}{{ $result = "low-admin" }}{{_}}{{ $result = "high-admin" }}{{/}}{{_}}{{ $result = "guest" }}{{/}}{{ $result }}"#,
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
        PureValue::List(vec![
            PureValue::Object(BTreeMap::from([
                ("name".into(), PureValue::String("apple".into())),
                ("price".into(), PureValue::Int(100)),
            ])),
            PureValue::Object(BTreeMap::from([
                ("name".into(), PureValue::String("banana".into())),
                ("price".into(), PureValue::Int(50)),
            ])),
            PureValue::Object(BTreeMap::from([
                ("name".into(), PureValue::String("cherry".into())),
                ("price".into(), PureValue::Int(200)),
            ])),
        ]),
    )]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $products | filter(p -> p.price >= 100) | map(p -> p.name) }}{{ x | to_string }}"#,
            ty,
            val,
        )
        .await,
        "[apple, cherry]"
    );
}

#[tokio::test]
async fn iteration_with_match_per_item() {
    let (ty, val) = items_storage(vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_with_storage(
            r#"{{ x in $items }}{{ 1..=3 = x }}s{{_}}b{{/}}{{/}}"#,
            ty,
            val
        )
        .await,
        "sssbb"
    );
}

#[tokio::test]
async fn multi_storage_interaction() {
    let types = HashMap::from([
        ("items".into(), Ty::List(Box::new(Ty::Int))),
        ("min".into(), Ty::Int),
        ("max".into(), Ty::Int),
        ("count".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "items".into(),
            PureValue::List(vec![
                PureValue::Int(3),
                PureValue::Int(7),
                PureValue::Int(1),
                PureValue::Int(9),
                PureValue::Int(4),
            ]),
        ),
        ("min".into(), PureValue::Int(2)),
        ("max".into(), PureValue::Int(8)),
        ("count".into(), PureValue::Int(0)),
    ]);
    assert_eq!(
        run_with_storage(
            r#"{{ filtered = $items | filter(i -> i >= $min && i <= $max) }}{{ x in filtered }}{{ $count = $count + 1 }}{{/}}{{ $count | to_string }}{{_}}{{/}}"#,
            types,
            values,
        )
        .await,
        "3"
    );
}

#[tokio::test]
async fn object_destructure_in_iteration_with_emit() {
    let (ty, val) = users_list_storage();
    assert_eq!(
        run_with_storage(
            r#"{{ { name, age, } in $users }}{{ name }}({{ age | to_string }}) {{/}}"#,
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
        PureValue::List(vec![
            PureValue::Int(-5),
            PureValue::Int(0),
            PureValue::Int(3),
            PureValue::Int(7),
            PureValue::Int(12),
            PureValue::Int(20),
        ]),
    )]);
    assert_eq!(
        run_with_storage(
            r#"{{ x = $nums | filter(n -> n > 0 && n < 10) | map(n -> n * n) }}{{ x | to_string }}"#,
            types,
            values,
        )
        .await,
        "[9, 49]"
    );
}

#[tokio::test]
async fn nested_list_iteration_with_accumulator() {
    let types = HashMap::from([
        (
            "matrix".into(),
            Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))),
        ),
        ("sum".into(), Ty::Int),
    ]);
    let values = HashMap::from([
        (
            "matrix".into(),
            PureValue::List(vec![
                PureValue::List(vec![PureValue::Int(1), PureValue::Int(2)]),
                PureValue::List(vec![PureValue::Int(3), PureValue::Int(4)]),
                PureValue::List(vec![PureValue::Int(5), PureValue::Int(6)]),
            ]),
        ),
        ("sum".into(), PureValue::Int(0)),
    ]);
    assert_eq!(
        run_with_storage(
            "{{ row in $matrix }}{{ x in row }}{{ $sum = $sum + x }}{{/}}{{/}}{{ $sum | to_string }}",
            types,
            values
        )
        .await,
        "21"
    );
}

#[tokio::test]
async fn map_then_iterate_with_match() {
    let (ty, val) = items_storage(vec![1, 2, 3, 4, 5]);
    assert_eq!(
        run_with_storage(
            r#"{{ doubled = $items | map(i -> i * 2) }}{{ x in doubled }}{{ true = x > 6 }}{{ x | to_string }} {{_}}{{/}}{{/}}"#,
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
    let (ty, val) = items_storage(vec![1, 2, 3]);
    let output = run(
        r#"{{ x = $items | map(i -> double(i)) }}{{ x | to_string }}"#,
        ty,
        val,
        extern_fns,
    )
    .await;
    assert_eq!(output, "[2, 4, 6]");
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
        PureValue::List(vec![
            PureValue::Object(BTreeMap::from([
                ("name".into(), PureValue::String("alice".into())),
                ("age".into(), PureValue::Int(30)),
                ("active".into(), PureValue::Bool(true)),
            ])),
            PureValue::Object(BTreeMap::from([
                ("name".into(), PureValue::String("bob".into())),
                ("age".into(), PureValue::Int(17)),
                ("active".into(), PureValue::Bool(true)),
            ])),
            PureValue::Object(BTreeMap::from([
                ("name".into(), PureValue::String("carol".into())),
                ("age".into(), PureValue::Int(25)),
                ("active".into(), PureValue::Bool(false)),
            ])),
            PureValue::Object(BTreeMap::from([
                ("name".into(), PureValue::String("dave".into())),
                ("age".into(), PureValue::Int(40)),
                ("active".into(), PureValue::Bool(true)),
            ])),
        ]),
    )]);
    assert_eq!(
        run_with_storage(
            r#"{{ eligible = $users | filter(u -> u.active && u.age >= 18) | map(u -> u.name) }}{{ name in eligible }}{{ name }} {{/}}"#,
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
        run_simple(r#"{{ x = [1, 2, 3] }}{{ x | to_string }}{{_}}{{/}}"#).await,
        "[1, 2, 3]"
    );
}

// ── Multi-arm with range ─────────────────────────────────────────

#[tokio::test]
async fn multi_arm_range_and_literal() {
    let (ty, val) = int_storage("score", 0);
    assert_eq!(
        run_with_storage(
            r#"{{ 0 = $score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            ty,
            val
        )
        .await,
        "zero"
    );
}

#[tokio::test]
async fn multi_arm_range_and_literal_low() {
    let (ty, val) = int_storage("score", 5);
    assert_eq!(
        run_with_storage(
            r#"{{ 0 = $score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            ty,
            val
        )
        .await,
        "low"
    );
}

#[tokio::test]
async fn multi_arm_range_and_literal_high() {
    let (ty, val) = int_storage("score", 50);
    assert_eq!(
        run_with_storage(
            r#"{{ 0 = $score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
            ty,
            val
        )
        .await,
        "high"
    );
}
