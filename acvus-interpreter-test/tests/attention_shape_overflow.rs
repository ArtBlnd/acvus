//! One finding from `attention_shape.rs`, alone in its own test binary
//! because it overflows the compiler's stack and a stack overflow aborts
//! the whole process rather than failing one test. It is kept as it fails.
//! The overflow needs both halves of the script: a let-bound lambda called
//! from inside another lambda, whose call resolves to no function, and an
//! `if let` over a `max` of the values that call produced. With the
//! `if let` line removed the script reports the unresolved call and stops.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_utils::Interner;

fn context(i: &Interner) -> Context {
    let data = serde_json::json!({ "keys": [[1.0, 0.0], [0.0, 1.0]] });
    data.as_object()
        .expect("an object")
        .iter()
        .map(|(k, v)| (i.intern(k), value_from_json(i, v)))
        .collect()
}

#[tokio::test]
async fn an_unresolved_call_inside_a_lambda_followed_by_if_let_over_max_is_reported_not_overflowed() {
    let i = Interner::new();
    let v: Value = run_script_mode(
        &i,
        "let dot = |k| -> 1.0; \
         let scores = as_iter(&@keys) | map(|k| -> dot(k)) | collect; \
         let m = if let Some(m) = as_iter(&scores) | max { m } else { 0.0 }; \
         let weights = as_iter(&scores) | map(|s| -> exp(*s - m)) | collect; \
         weights.len()",
        context(&i),
    )
    .await;
    assert_eq!(v.as_int(), 2);
}
