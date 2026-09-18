//! A closure made inside a closure body: three levels of `map`, and a
//! closure that leaves the body that made it. Each level's `MakeClosure`
//! runs while the level above is already executing as a closure, so the
//! closure bodies a running `FnValue` can reach are what these measure.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default(), ret).await
}

fn assert_close(v: &Value, expected: f64) {
    let actual = v.as_float();
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

#[tokio::test]
async fn three_nested_maps_sum_to_the_product_of_the_levels() {
    let v = run(
        "range(1, 4) \
         | map(|a| -> range(1, 4) \
             | map(|b| -> range(1, 4) | map(|c| -> to_float(c)) | sum) \
             | sum) \
         | sum",
        Ty::Float,
    )
    .await;
    assert_close(&v, 54.0);
}

#[tokio::test]
async fn a_word_reaches_the_innermost_of_three_closures_as_a_copy() {
    let v = run(
        "let k = 2.0; \
         range(0, 2) \
         | map(|a| -> range(0, 2) \
             | map(|b| -> range(0, 2) | map(|c| -> *k) | sum) \
             | sum) \
         | sum",
        Ty::Float,
    )
    .await;
    assert_close(&v, 16.0);
}

#[tokio::test]
async fn a_closure_returned_from_a_closure_is_called_at_the_top_level() {
    let v = run(
        "let mk = |s| -> |x| -> x * 3.0; let triple = mk(1.0); triple(4.0)",
        Ty::Float,
    )
    .await;
    assert_close(&v, 12.0);
}
