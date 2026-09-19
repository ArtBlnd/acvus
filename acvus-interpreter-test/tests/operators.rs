//! An operator on a captured name, run. A lambda sees a captured name as
//! `&T` (RFC-0018), and these programs measure what the operator makes of
//! that at the interpreter's contract: the value it returns.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default(), ret).await
}

#[tokio::test]
async fn a_captured_word_multiplies() {
    let v = run("let k = 1; let f = |y| -> k * y; f(2)", Ty::I64).await;
    assert_eq!(v.as_int(), 2);
}

#[tokio::test]
async fn a_captured_word_adds() {
    let v = run("let k = 1; let f = |y| -> k + y; f(2)", Ty::I64).await;
    assert_eq!(v.as_int(), 3);
}

#[tokio::test]
async fn a_captured_word_subtracts() {
    let v = run("let k = 1; let f = |y| -> k - y; f(2)", Ty::I64).await;
    assert_eq!(v.as_int(), -1);
}

#[tokio::test]
async fn a_captured_word_divides() {
    let v = run("let k = 4; let f = |y| -> k / y; f(2)", Ty::I64).await;
    assert_eq!(v.as_int(), 2);
}

#[tokio::test]
async fn a_captured_word_compares() {
    let v = run("let k = 1; let f = |y| -> k < y; f(2)", Ty::Bool).await;
    assert!(v.as_bool());
}

#[tokio::test]
async fn a_captured_word_on_the_right_of_the_operator_reads_through_too() {
    let v = run("let k = 1; let f = |y| -> y - k; f(2)", Ty::I64).await;
    assert_eq!(v.as_int(), 1);
}

#[tokio::test]
async fn a_captured_string_concatenates() {
    let v = run(
        "let s = \"a\".to_string(); let f = |t| -> s + t; f(\"b\".to_string())",
        Ty::String,
    )
    .await;
    // SAFETY: the script's type is `String`.
    assert_eq!(unsafe { v.as_str() }, "ab");
}
