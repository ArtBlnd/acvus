//! A storage is read — taken, lent, stored into at a part — only while
//! the part read is alive (RFC-0018): a move of the whole or of a part
//! refuses every later read that overlaps it, and a store into a part
//! revives that part.

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

#[tokio::test]
async fn a_part_moved_out_and_stored_back_is_lent() {
    let v = run("let o = { v: vec([1, 2]), w: 3, }; let x = o.v; o.v = vec([4]); o.v.len()").await;
    assert_eq!(v.as_int(), 1);
}

#[tokio::test]
async fn a_word_part_beside_a_moved_part_is_read() {
    let v = run("let o = { v: vec([1, 2]), w: 3, }; let x = o.v; o.w").await;
    assert_eq!(v.as_int(), 3);
}

#[tokio::test]
#[should_panic(expected = "use of `o` after it was moved")]
async fn a_lend_of_a_moved_part_is_refused() {
    run("let o = { v: vec([1, 2]), }; let x = o.v; o.v.len()").await;
}

#[tokio::test]
#[should_panic(expected = "use of `o` after it was moved")]
async fn a_reference_to_a_partly_moved_storage_is_refused() {
    run("let o = { v: vec([1, 2]), }; let x = o.v; let r = &o; r").await;
}
