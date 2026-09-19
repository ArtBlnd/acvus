//! Each call form reaches its declaration's Rust body and brings back what
//! that body returned (RFC-0059 rule 3 amended).
//!
//! These run the language rather than calling a handler directly: the
//! handler is the operation's type parameter, so the form is chosen where
//! the operation is built, and only a prepared body exercises that choice.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::{Context, run_script_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn form0() -> i64 {
    7
}

#[extern_fn(effect = pure)]
fn form1(a: i64) -> i64 {
    a * 10
}

#[extern_fn(effect = pure)]
fn form2(a: i64, b: i64) -> i64 {
    a * 10 + b
}

#[extern_fn(effect = pure)]
fn form3(a: i64, b: i64, c: i64) -> i64 {
    (a * 10 + b) * 10 + c
}

#[extern_fn(effect = pure)]
fn form_window(a: i64, b: i64, c: i64, d: i64) -> i64 {
    ((a * 10 + b) * 10 + c) * 10 + d
}

#[extern_fn(effect = pure)]
fn form_string(s: String) -> String {
    s.to_uppercase()
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [form0, form1, form2, form3, form_window, form_string],
    }
}

async fn answer(source: &str) -> i64 {
    let interner = Interner::new();
    run_script_with_externs(
        &interner,
        source,
        Context::default(),
        vec![registry()],
        Ty::I64,
    )
    .await
    .value
    .as_int()
}

#[tokio::test]
async fn every_register_form_returns_what_its_body_returned() {
    assert_eq!(answer("form0()").await, 7);
    assert_eq!(answer("form1(4)").await, 40);
    assert_eq!(answer("form2(4, 2)").await, 42);
    assert_eq!(answer("form3(4, 2, 1)").await, 421);
}

#[tokio::test]
async fn the_window_form_returns_what_its_body_returned() {
    assert_eq!(answer("form_window(4, 2, 1, 3)").await, 4213);
}

#[tokio::test]
async fn the_slice_form_reaches_the_elements() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "let a = vec([1, 2, 3]); let i = 0; let n = len(&a); let total = 0; \
         while i < n { total = total + a[i]; i = i + n / n; } total",
        Context::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::I64,
    )
    .await;
    assert_eq!(ran.value.as_int(), 6);
}

#[tokio::test]
async fn a_large_value_crosses_the_one_argument_form() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "form_string(\"hello\")",
        Context::default(),
        vec![registry()],
        Ty::String,
    )
    .await;
    assert!(
        ran.value.is_string(),
        "expected a String, got {:?}",
        ran.value
    );
    // SAFETY: the assertion above is the witness.
    assert_eq!(unsafe { ran.value.as_str() }, "HELLO");
}

/// The fusion rule matches this run, and each of its nodes holds a handler
/// of its own type (RFC-0044 stage 6).
#[tokio::test]
async fn a_fused_run_returns_what_its_last_call_returned() {
    assert_eq!(answer("form1(form2(form1(1), 2))").await, 1020);
}

/// `count` is `acvus-ext`'s `async fn` declaration: the caller suspends and
/// the driver resumes it with the future's value.
#[tokio::test]
async fn an_awaited_declaration_resumes_with_its_value() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "let a = vec([1, 2, 3, 4]); a | filter(|x| -> x > 1) | count",
        Context::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::I64,
    )
    .await;
    assert_eq!(ran.value.as_int(), 3);
}

#[tokio::test]
async fn the_forms_compose_in_one_body() {
    assert_eq!(
        answer("form_window(form0() - 6, form1(0), form2(0, 1), form3(0, 0, 2))").await,
        1012
    );
}
