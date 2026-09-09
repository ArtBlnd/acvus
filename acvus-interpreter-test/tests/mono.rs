//! A monomorphized ExternFn end to end: the script picks the instance by
//! the argument's type, and the runtime runs that instance.

use acvus_interpreter::Value;
use acvus_interpreter_test::run_script;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[tokio::test]
async fn to_string_runs_the_instance_for_the_argument_type() {
    let i = Interner::new();
    assert_eq!(
        run_script(&i, "to_string(1.5)", FxHashMap::default()).await,
        Value::string("1.5")
    );
    assert_eq!(
        run_script(&i, "to_string(42)", FxHashMap::default()).await,
        Value::string("42")
    );
    assert_eq!(
        run_script(&i, "to_string(true)", FxHashMap::default()).await,
        Value::string("true")
    );
}

#[tokio::test]
async fn to_int_reads_a_string_through_its_own_instance() {
    let i = Interner::new();
    assert_eq!(
        run_script(&i, "to_int(\"42\") + to_int(1.9)", FxHashMap::default()).await,
        Value::Int(43)
    );
}
