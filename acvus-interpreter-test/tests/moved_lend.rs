//! A reference to a moved storage is refused at compile time, so the three
//! scripts that used to reach the interpreter and read an emptied register
//! never run. A storage assigned back is alive again and runs.

use acvus_interpreter_test::*;
use acvus_utils::Interner;

async fn run(source: &str) -> acvus_interpreter::Value {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default()).await
}

#[tokio::test]
#[should_panic(expected = "use of `a` after it was moved")]
async fn a_method_receiver_lent_from_a_moved_storage_does_not_run() {
    run("let a = [1, 2]; let b = a; a.len()").await;
}

#[tokio::test]
#[should_panic(expected = "use of `a` after it was moved")]
async fn a_reference_to_a_moved_storage_does_not_run() {
    run("let a = [1, 2]; let b = a; len(&a)").await;
}

#[tokio::test]
#[should_panic(expected = "use of `a` after it was moved")]
async fn a_reference_bound_from_a_moved_storage_does_not_run() {
    run("let a = [1, 2]; let b = a; let r = &a; r[0]").await;
}

#[tokio::test]
async fn a_storage_assigned_back_is_lent_again() {
    let v = run("let a = [1, 2]; let b = a; a = [3, 4]; a.len()").await;
    assert_eq!(v.as_int(), 2);
}
