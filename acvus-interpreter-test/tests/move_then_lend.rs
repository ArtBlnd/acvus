//! A place moved out of and then lent. A bare read of the moved place is
//! refused at validation; a lend of it passes validation and the
//! interpreter panics at the read instead. A test that fails is a finding,
//! kept as it fails.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_utils::Interner;

async fn run(source: &str) -> Value {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default()).await
}

#[tokio::test]
#[should_panic(expected = "validation failed")]
async fn a_bare_read_of_a_moved_array_is_refused() {
    run("let a = [1, 2]; let b = a; a").await;
}

#[tokio::test]
#[should_panic(expected = "validation failed")]
async fn a_lend_of_an_array_moved_into_a_binding_is_refused() {
    run("let a = [1, 2]; let b = a; a.len()").await;
}

#[tokio::test]
#[should_panic(expected = "validation failed")]
async fn a_lend_of_a_vec_moved_into_a_binding_is_refused() {
    run("let a = vec([1, 2]); let b = a; a.len()").await;
}

#[tokio::test]
#[should_panic(expected = "validation failed")]
async fn a_field_read_of_an_object_moved_into_a_binding_is_refused() {
    run("let a = { x: 1, }; let b = a; a.x").await;
}

#[tokio::test]
#[should_panic(expected = "validation failed")]
async fn a_lend_of_an_array_moved_into_a_call_is_refused() {
    run("let a = [1, 2]; let f = |v| -> 1; f(a); a.len()").await;
}

#[tokio::test]
#[should_panic(expected = "validation failed")]
async fn a_lend_of_an_array_moved_into_an_object_field_is_refused() {
    run("let a = [1, 2]; let o = { v: a, }; a.len()").await;
}

#[tokio::test]
async fn the_binding_an_array_moved_into_is_read() {
    let v = run("let a = [1, 2]; let b = a; b.len()").await;
    assert_eq!(v.as_int(), 2);
}
