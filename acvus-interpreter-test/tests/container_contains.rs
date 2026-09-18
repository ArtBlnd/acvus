//! `contains` and `find` at the value level: one bare name over `string`
//! and `iter`, settled by the first argument (RFC-0043), and a piped `Vec`
//! reaching `iter::contains` through `into_iter`.

use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    run_script_mode_with_externs(&i, source, Context::default(), registries, ret)
        .await
        .value
}

#[tokio::test]
async fn contains_of_a_lent_string_finds_a_substring() {
    assert!(
        run("let s = \"ab\"; contains(&s, \"b\")", Ty::Bool)
            .await
            .as_bool()
    );
    assert!(
        !run("let s = \"ab\"; contains(&s, \"c\")", Ty::Bool)
            .await
            .as_bool()
    );
    assert!(
        run("let s = \"ab\"; string::contains(&s, \"ab\")", Ty::Bool)
            .await
            .as_bool()
    );
}

#[tokio::test]
async fn find_of_a_string_is_the_first_character_index() {
    assert_eq!(
        run(
            "let s = \"héllo\"; find(&s, \"l\") | unwrap_or(-1)",
            Ty::I64
        )
        .await
        .as_int(),
        2
    );
    assert_eq!(
        run("let s = \"abc\"; find(&s, \"z\") | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        -1
    );
}

#[tokio::test]
async fn contains_of_an_iterator_consumes_it() {
    assert!(
        run("into_iter([1, 2]) | contains(1)", Ty::Bool)
            .await
            .as_bool()
    );
    assert!(
        !run("into_iter([1, 2]) | contains(3)", Ty::Bool)
            .await
            .as_bool()
    );
}

#[tokio::test]
async fn a_piped_vec_reaches_the_iterator_contains_through_into_iter() {
    assert!(
        run("let v = vec([1, 2, 3]); v | contains(3)", Ty::Bool)
            .await
            .as_bool()
    );
    assert!(
        !run("let v = vec([1, 2, 3]); v | contains(4)", Ty::Bool)
            .await
            .as_bool()
    );
}

#[tokio::test]
async fn a_lambda_calling_contains_settles_when_applied_to_a_lent_string() {
    assert!(
        run(
            "let f = |c, y| -> c.contains(y); let s = \"ab\"; f(&s, \"b\")",
            Ty::Bool
        )
        .await
        .as_bool()
    );
}
