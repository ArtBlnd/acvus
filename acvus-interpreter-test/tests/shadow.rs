//! A `let` binding of a function is one more signature of its bare name
//! (RFC-0043), run: the call that settles on the binding reaches the
//! binding's body, and the call two signatures both take does not compile.

use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

async fn run(source: &str, ret: Ty) -> acvus_interpreter::Value {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default(), ret).await
}

#[tokio::test]
async fn a_binding_alone_takes_a_call_no_declared_len_takes() {
    let v = run("let len = |k| -> k + 7; len(1)", Ty::I64).await;
    assert_eq!(v.as_int(), 8);
}

#[tokio::test]
#[should_panic(expected = "`len` is declared by array::len and the binding `len`")]
async fn a_method_receiver_two_candidates_take_in_different_modes_is_ambiguous() {
    run(
        "let q = [1.0, 2.0]; let len = |k| -> 7.0; q.len()",
        Ty::Never,
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "`len` is declared by array::len and the binding `len`")]
async fn a_binding_and_an_extern_that_both_take_the_call_are_ambiguous() {
    run(
        "let q = [1.0, 2.0]; let len = |k| -> 7.0; len(&q)",
        Ty::Never,
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "`count` is declared by iter::count and the binding `count`")]
async fn a_binding_and_a_signature_that_converts_the_argument_are_ambiguous() {
    run(
        "let q = [1.0, 2.0]; let count = |k| -> 7.0; count(q)",
        Ty::Never,
    )
    .await;
}
