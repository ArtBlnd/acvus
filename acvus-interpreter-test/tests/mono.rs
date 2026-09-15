//! A monomorphized ExternFn end to end: the script picks the instance by
//! the argument's type, and the runtime runs that instance.

use acvus_interpreter::Value;
use acvus_interpreter_test::run_script;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn assert_str(v: &Value, expected: &str) {
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    assert_eq!(unsafe { v.as_str() }, expected);
}

#[tokio::test]
async fn to_string_runs_the_instance_for_the_argument_type() {
    let i = Interner::new();
    assert_str(
        &run_script(&i, "to_string(1.5)", FxHashMap::default()).await,
        "1.5",
    );
    assert_str(
        &run_script(&i, "to_string(42)", FxHashMap::default()).await,
        "42",
    );
    assert_str(
        &run_script(&i, "to_string(true)", FxHashMap::default()).await,
        "true",
    );
}

#[tokio::test]
async fn to_int_reads_a_string_through_its_own_instance() {
    let i = Interner::new();
    assert_eq!(
        run_script(&i, "to_int(\"42\") + to_int(1.9)", FxHashMap::default())
            .await
            .as_int(),
        43
    );
}
