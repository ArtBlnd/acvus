//! A dot-product attention written in script mode, and the findings met on
//! the way to it, each stated at the script's contract: its value against
//! the context `query`, `keys`, `values`. A test that fails is a finding,
//! kept as it fails. One more finding, a compiler stack overflow, aborts the
//! test process and therefore lives alone in `attention_shape_overflow.rs`.

use acvus_interpreter::Value;
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

async fn run(source: &str) -> Value {
    let i = Interner::new();
    run_script_mode(&i, source, context(&i)).await
}

fn compile(source: &str) {
    let i = Interner::new();
    let context_types: FxHashMap<Astr, Ty> = context(&i)
        .iter()
        .map(|(name, typed)| (*name, typed.ty.clone()))
        .collect();
    let ast = ParsedAst::Script(acvus_ast::parse_script_mode(&i, source).expect("parse error"));
    compile_source_with_externs(
        &i,
        ast,
        &context_types,
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
    );
}

fn assert_close(v: &Value, expected: f64) {
    let actual = v.as_float();
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

const ATTENTION: &str = "
let d = len(&@query);
let n = len(&@keys);
let scale = 1.0 / sqrt(to_float(d));

let scores = deque();
let t = 0;
while t < n {
    let s = 0.0;
    let i = 0;
    while i < d {
        s = s + *get(&@query, i) * *get(get(&@keys, t), i);
        i = i + 1;
    }
    push_back(&mut scores, s * scale);
    t = t + 1;
}

let m = if let Some(m) = as_iter(&scores) | map(|s| -> *s) | max { m } else { 0.0 };
let weights = as_iter(&scores) | map(|s| -> exp(*s - *m)) | collect;
let z = as_iter(&weights) | map(|w| -> *w) | sum;

let out = deque();
let j = 0;
while j < d {
    let acc = 0.0;
    let t = 0;
    while t < n {
        acc = acc + *get(&weights, t) / z * *get(get(&@values, t), j);
        t = t + 1;
    }
    push_back(&mut out, acc);
    j = j + 1;
}
";

const ATTENTION_AS_METHOD_CALLS: &str = "
let d = @query.len();
let n = @keys.len();
let scale = 1.0 / d.to_float().sqrt();

let scores = deque();
let t = 0;
while t < n {
    let s = 0.0;
    let i = 0;
    while i < d {
        s = s + *@query.get(i) * *@keys.get(t).get(i);
        i = i + 1;
    }
    scores.push_back(s * scale);
    t = t + 1;
}

let m = if let Some(m) = scores.as_iter().map(|s| -> *s).max() { m } else { 0.0 };
let weights = scores.as_iter().map(|s| -> (*s - *m).exp()).collect();
let z = weights.as_iter().map(|w| -> *w).sum();

let out = deque();
let j = 0;
while j < d {
    let acc = 0.0;
    let t = 0;
    while t < n {
        acc = acc + *weights.get(t) / z * *@values.get(t).get(j);
        t = t + 1;
    }
    out.push_back(acc);
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
    let first = run(&format!("{ATTENTION} *get(&out, 0)")).await;
    let second = run(&format!("{ATTENTION} *get(&out, 1)")).await;
    assert_close(&first, expected[0]);
    assert_close(&second, expected[1]);
}

#[tokio::test]
async fn attention_written_as_method_calls_is_the_same_value() {
    let expected = expected_attention_of_e1_over_the_standard_basis();
    let first = run(&format!("{ATTENTION_AS_METHOD_CALLS} *out.get(0)")).await;
    let second = run(&format!("{ATTENTION_AS_METHOD_CALLS} *out.get(1)")).await;
    assert_close(&first, expected[0]);
    assert_close(&second, expected[1]);
}

const LAMBDA_INSIDE_LAMBDA: &str = "let xs = as_iter(&@keys) | map(|k| -> as_iter(k) | map(|x| -> *x * 2.0) | sum) | collect; \
     *get(&xs, 0) + *get(&xs, 1) * 10.0";

#[test]
fn a_lambda_inside_a_lambda_compiles() {
    compile(LAMBDA_INSIDE_LAMBDA);
}

#[tokio::test]
async fn a_lambda_inside_a_lambda_runs() {
    let v = run(LAMBDA_INSIDE_LAMBDA).await;
    assert_close(&v, 22.0);
}

#[tokio::test]
async fn a_local_captured_through_two_lambdas_is_read_by_the_inner_one() {
    let v = run(
        "let w = [0.5, 0.5]; \
         let o = range(0, 2) | map(|j| -> range(0, 2) | map(|t| -> *get(w, t) * *get(get(&@values, t), *j)) | sum) | collect; \
         *get(&o, 0) + *get(&o, 1) * 10.0",
    )
    .await;
    assert_close(&v, 32.0);
}

#[tokio::test]
async fn a_local_captured_through_one_lambda_is_read() {
    let v = run("let w = [0.5, 0.5]; range(0, 2) | map(|t| -> *get(w, t)) | sum").await;
    assert_close(&v, 1.0);
}

#[tokio::test]
async fn a_let_bound_lambda_is_callable_from_inside_another_lambda() {
    let v = run("let dot = |k| -> *get(k, 0); as_iter(&@keys) | map(|k| -> dot(k)) | sum").await;
    assert_close(&v, 1.0);
}

#[tokio::test]
async fn a_let_bound_lambda_is_callable_at_the_top_level() {
    let v = run("let dot = |k| -> *get(k, 0); dot(&@query)").await;
    assert_close(&v, 1.0);
}
