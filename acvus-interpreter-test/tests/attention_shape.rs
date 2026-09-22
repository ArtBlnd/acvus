//! A dot-product attention written in script mode, and the findings met on
//! the way to it, each stated at the script's contract: its value against
//! the context `query`, `keys`, `values`. A test that fails is a finding,
//! kept as it fails. One more finding, a compiler stack overflow, aborts the
//! test process and therefore lives alone in `attention_shape_overflow.rs`.

use acvus_interpreter::Value;
use acvus_interpreter_test::scripts::ATTENTION;
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn context(i: &Interner) -> Context {
    let data = serde_json::json!({
        "query": [1.0, 0.0],
        "keys": [[1.0, 0.0], [0.0, 1.0]],
        "values": [[1.0, 2.0], [3.0, 4.0]],
    });
    data.as_object()
        .expect("an object")
        .iter()
        .map(|(k, v)| (i.intern(k), value_from_json(i, v)))
        .collect()
}

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script_mode(&i, source, context(&i), ret).await
}

fn compile(source: &str, ret: Ty) {
    let i = Interner::new();
    let context_types: FxHashMap<Astr, Ty> = context(&i)
        .iter()
        .map(|(name, typed)| (*name, typed.ty.clone()))
        .collect();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    compile_source_with_externs(
        &i,
        ast,
        &context_types,
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        ret,
    );
}

fn assert_close(v: &Value, expected: f64) {
    let actual = v.as_float();
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

const ATTENTION_AS_FUNCTION_CALLS: &str = "
let d = len(&@query);
let n = len(&@keys);
let scale = 1.0 / sqrt(d as f64);

let scores = deque();
let t = 0;
while t < n {
    let s = 0.0;
    let i = 0;
    while i < d {
        let key = &@keys[t];
        s = s + @query[i] * key[i];
        i = i + 1;
    }
    push_back(&mut scores, s * scale);
    t = t + 1;
}

let m = if let Some(m) = as_iter(&scores) | map(|s| -> *s) | max { m } else { 0.0 };
let weights = as_iter(&scores) | map(|s| -> exp(*s - m)) | collect;
let z = as_iter(&weights) | map(|w| -> *w) | sum;

let out = deque();
let j = 0;
while j < d {
    let acc = 0.0;
    let t = 0;
    while t < n {
        let value = &@values[t];
        acc = acc + weights[t] / z * value[j];
        t = t + 1;
    }
    push_back(&mut out, acc);
    j = j + 1;
}
";

fn expected_attention_of_e1_over_the_standard_basis() -> [f64; 2] {
    let scaled_score_of_e1 = 0.5f64.sqrt();
    let weight_of_row_0 = scaled_score_of_e1.exp() / (scaled_score_of_e1.exp() + 1.0);
    let weight_of_row_1 = 1.0 - weight_of_row_0;
    [
        weight_of_row_0 * 1.0 + weight_of_row_1 * 3.0,
        weight_of_row_0 * 2.0 + weight_of_row_1 * 4.0,
    ]
}

#[tokio::test]
async fn attention_of_e1_over_the_standard_basis_is_the_softmax_weighted_sum_of_the_rows() {
    let expected = expected_attention_of_e1_over_the_standard_basis();
    let first = run(&format!("{ATTENTION} *out.get(0)"), Ty::Float).await;
    let second = run(&format!("{ATTENTION} *out.get(1)"), Ty::Float).await;
    assert_close(&first, expected[0]);
    assert_close(&second, expected[1]);
}

const ATTENTION_AS_CHAINS: &str = "
let scale = 1.0 / (@query.len() as f64).sqrt();

let dot = |k| -> as_iter(k)
    .fold({ i: 0, s: 0.0, }, |acc, x| -> { i: acc.i + 1, s: acc.s + @query[acc.i] * *x, })
    .s;

let scores = @keys.as_iter().map(|k| -> dot(k) * scale).collect();
let peak = scores.as_iter().map(|s| -> *s).max().unwrap();
let weights = scores.as_iter().map(|s| -> (*s - peak).exp()).collect();
let z = weights.as_iter().map(|w| -> *w).sum();

let column_0 = weights.as_iter()
    .fold({ t: 0, s: 0.0, }, |acc, w| -> { t: acc.t + 1, s: acc.s + *w / z * @values[acc.t][0], })
    .s;
let column_1 = weights.as_iter()
    .fold({ t: 0, s: 0.0, }, |acc, w| -> { t: acc.t + 1, s: acc.s + *w / z * @values[acc.t][1], })
    .s;
";

#[tokio::test]
async fn attention_written_as_chains_is_the_same_value() {
    let expected = expected_attention_of_e1_over_the_standard_basis();
    let first = run(&format!("{ATTENTION_AS_CHAINS} column_0"), Ty::Float).await;
    let second = run(&format!("{ATTENTION_AS_CHAINS} column_1"), Ty::Float).await;
    assert_close(&first, expected[0]);
    assert_close(&second, expected[1]);
}

/// RFC-0064 Decision 2 admits the capture the inner lambda makes of `b`, a
/// reference parameter of the outer one, and the product it reads is the
/// dot product of `@query` with itself.
#[tokio::test]
async fn a_reference_parameter_captured_by_an_inner_lambda_is_read_at_the_call() {
    let v = run(
        "let dot = |a, b| -> as_iter(a).map(|x| -> *x * b[0]).sum(); dot(&@query, &@query)",
        Ty::Float,
    )
    .await;
    assert_close(&v, 1.0);
}

#[tokio::test]
async fn a_method_call_on_a_reference_parameter_lends_it_once() {
    let v = run("let dot = |k| -> as_iter(k).map(|x| -> *x).sum(); let dot_m = |k| -> k.as_iter().map(|x| -> *x).sum(); dot(&@query) + dot_m(&@query)", Ty::Float)
        .await;
    assert_close(&v, 2.0);
}

#[tokio::test]
async fn attention_written_as_function_calls_is_the_same_value() {
    let expected = expected_attention_of_e1_over_the_standard_basis();
    let first = run(
        &format!("{ATTENTION_AS_FUNCTION_CALLS} *out.get(0)"),
        Ty::Float,
    )
    .await;
    let second = run(
        &format!("{ATTENTION_AS_FUNCTION_CALLS} *out.get(1)"),
        Ty::Float,
    )
    .await;
    assert_close(&first, expected[0]);
    assert_close(&second, expected[1]);
}

const LAMBDA_INSIDE_LAMBDA: &str = "let xs = as_iter(&@keys) | map(|k| -> as_iter(k) | map(|x| -> *x * 2.0) | sum) | collect; \
     xs[0] + xs[1] * 10.0";

#[test]
fn a_lambda_inside_a_lambda_compiles() {
    compile(LAMBDA_INSIDE_LAMBDA, Ty::Float);
}

#[tokio::test]
async fn a_lambda_inside_a_lambda_runs() {
    let v = run(LAMBDA_INSIDE_LAMBDA, Ty::Float).await;
    assert_close(&v, 22.0);
}

/// A closure owns what it captures, so a lambda that names the enclosing
/// lambda's capture moves it out of a closure `map` calls again: a word
/// copies and anything else is refused (RFC-0018). The owner settled that
/// on 2026-09-17, after this test met it as a validation failure.
#[tokio::test]
async fn a_word_captured_through_two_lambdas_is_read_by_the_inner_one() {
    let v = run(
        "let half = 0.5; \
         as_iter(&@values) | map(|row| -> as_iter(row) | map(|x| -> half * *x) | sum) | sum",
        Ty::Float,
    )
    .await;
    assert_close(&v, 5.0);
}

#[test]
#[should_panic(expected = "cannot move `w` out of a closure's capture (type Array<Float, 2>)")]
fn an_array_captured_through_two_lambdas_is_refused() {
    compile(
        "let w = [0.5, 0.5]; \
         as_iter(&@values) | map(|row| -> as_iter(row) | map(|x| -> w[0] * *x) | sum) | sum",
        Ty::Never,
    );
}

#[tokio::test]
async fn a_local_captured_through_one_lambda_is_read() {
    let v = run(
        "let w = [0.5, 0.5]; as_iter(&@keys) | map(|k| -> w[0]) | sum",
        Ty::Float,
    )
    .await;
    assert_close(&v, 1.0);
}

#[tokio::test]
async fn a_let_bound_lambda_is_callable_from_inside_another_lambda() {
    let v = run(
        "let dot = |k| -> k[0]; as_iter(&@keys) | map(|k| -> dot(k)) | sum",
        Ty::Float,
    )
    .await;
    assert_close(&v, 1.0);
}

#[tokio::test]
async fn a_let_bound_lambda_is_callable_at_the_top_level() {
    let v = run("let dot = |k| -> k[0]; dot(&@query)", Ty::Float).await;
    assert_close(&v, 1.0);
}

#[tokio::test]
#[should_panic(expected = "compile failed")]
async fn a_let_bound_lambda_whose_signature_overlaps_an_extern_of_the_same_name_is_ambiguous() {
    run("let len = |k| -> 7.0; len(&@query)", Ty::Never).await;
}

#[tokio::test]
async fn a_let_bound_lambda_named_like_an_extern_is_the_callee_where_the_extern_has_no_instance() {
    let v = run("let len = |k| -> k + 7; len(1)", Ty::I64).await;
    assert_eq!(v.as_int(), 8);
}

#[tokio::test]
async fn a_let_bound_lambda_named_like_an_extern_of_another_shape_is_the_callee() {
    let v = run("let count = |k| -> 7.0; count(&@query)", Ty::Float).await;
    assert_close(&v, 7.0);
}

#[tokio::test]
#[should_panic(
    expected = "context @query is moved out here and not assigned again before the run ends"
)]
async fn a_context_taken_into_a_local_and_not_written_back_is_refused() {
    run("let q = @query; let dot = |k| -> k[0]; dot(&q)", Ty::Never).await;
}

#[tokio::test]
async fn a_context_taken_into_a_local_and_written_back_is_read_in_between() {
    let v = run("let q = @query; let r = q[0]; @query = q; r", Ty::Float).await;
    assert_close(&v, 1.0);
}

#[tokio::test]
#[should_panic(expected = "compile failed")]
async fn a_closure_parameter_whose_signature_overlaps_an_extern_of_the_same_name_is_ambiguous() {
    run("let f = |len| -> len(&@query); f(|k| -> 7.0)", Ty::Never).await;
}

#[tokio::test]
async fn a_closure_parameter_named_like_an_extern_is_the_callee_where_the_extern_has_no_instance() {
    let v = run("let f = |len| -> len(1); f(|k| -> k + 7)", Ty::I64).await;
    assert_eq!(v.as_int(), 8);
}

#[tokio::test]
async fn a_closure_parameter_named_like_an_extern_of_another_shape_is_the_callee() {
    let v = run(
        "let f = |count| -> count(&@query); f(|k| -> 7.0)",
        Ty::Float,
    )
    .await;
    assert_close(&v, 7.0);
}

#[tokio::test]
async fn a_closure_parameter_named_unlike_any_extern_is_the_callee() {
    let v = run("let f = |g| -> g(&@query); f(|k| -> 7.0)", Ty::Float).await;
    assert_close(&v, 7.0);
}

#[tokio::test]
async fn a_method_call_of_an_extern_inside_a_lambda_is_unaffected_by_a_binding_of_the_same_name() {
    let v = run(
        "let len = |k| -> k + 7; @keys.as_iter().map(|k| -> k.len() as f64).sum()",
        Ty::Float,
    )
    .await;
    assert_close(&v, 4.0);
}

#[tokio::test]
async fn a_qualified_call_inside_a_lambda_is_unaffected_by_a_binding_of_the_same_name() {
    let v = run(
        "let len = |k| -> k + 7; @keys.as_iter().map(|k| -> array::len(k) as f64).sum()",
        Ty::Float,
    )
    .await;
    assert_close(&v, 4.0);
}

#[tokio::test]
#[should_panic(expected = "`len` is declared by\n  array::len\n  the binding `len`")]
async fn a_method_receiver_a_binding_and_an_extern_take_in_different_modes_is_ambiguous() {
    run("let len = |k| -> k + 7; len(@query.len())", Ty::Never).await;
}

#[tokio::test]
async fn a_closure_parameter_captured_by_an_inner_lambda_is_callable() {
    let v = run(
        "let f = |h| -> |x| -> h(x); let g = f(|k| -> k + 7); g(1)",
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 8);
}
