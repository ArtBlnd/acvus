//! An operator on a captured name, run. A lambda sees a captured name as
//! `&T` (RFC-0018), and these programs measure what the operator makes of
//! that at the interpreter's contract: the value it returns. An operator
//! reads each operand at what it names, one operand a value and the other
//! a reference alike, whether the operand's type is known at the operator
//! or settles later (RFC-0020).

use std::time::Duration;

use acvus_interpreter::Value;
use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::*;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

const LIMIT: Duration = Duration::from_secs(30);

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default(), ret).await
}

/// A run that crashes the process is an outcome here, not the end of the
/// test binary, so each program runs in a process of its own.
fn outcome(source: &str, opt: Opt) -> Outcome {
    acvus_interpreter_test::attempt_within!(source, opt, Stage::Run, LIMIT)
        .unwrap_or_else(|lapse| panic!("at {opt:?}, {lapse:?}: {source}"))
}

fn runs_to(source: &str, value: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Value(got) => assert_eq!(got, value, "at {opt:?}: {source}"),
            other => panic!("at {opt:?}, expected {value}, got {other:?}: {source}"),
        }
    }
}

fn refused_with(source: &str, reason: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Refused(why) => {
                assert!(
                    !why.contains("[validate:"),
                    "at {opt:?}, the MIR validator refused what the checker admitted: {why}"
                );
                assert!(why.contains(reason), "at {opt:?}: {why}");
            }
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {source}"),
        }
    }
}

#[test]
fn corpus_child() {
    corpus::child();
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

/// RFC-0018 rule 10: a captured `String` is seen as `&String`; RFC-0020:
/// `==` reads text operands and references to them alike. The comparison
/// runs when `k` is a `let`.
#[test]
fn a_comparison_with_a_captured_string() {
    runs_to(
        "let mk = |k| -> |x| -> x == k; let g = mk(\"q\".to_string()); g(\"q\".to_string())",
        "true",
    );
}

/// RFC-0020: `==` on text reads a `String` and a `&String` alike, with
/// the reference on either side and known at the operator or settled
/// later: by a lambda's parameters, or by a capture.
#[test]
fn text_and_a_reference_to_text_compare() {
    runs_to(
        "let a = \"q\".to_string(); let b = \"q\".to_string(); let r = &a; r == b",
        "true",
    );
    runs_to(
        "let a = \"q\".to_string(); let b = \"q\".to_string(); let r = &a; b != r",
        "false",
    );
    runs_to(
        "let b = \"q\".to_string(); let f = |x, y| -> x == y; f(\"q\".to_string(), &b)",
        "true",
    );
    runs_to(
        "let b = \"q\".to_string(); let f = |x, y| -> x == y; f(&b, \"q\".to_string())",
        "true",
    );
    runs_to(
        "let mk = |k| -> |x| -> k != x; let g = mk(\"q\".to_string()); g(\"r\".to_string())",
        "true",
    );
}

/// RFC-0020: a `&word` operand is read through the reference, under `==`
/// and under `<` alike, where its type is known and where it settles
/// later.
#[test]
fn a_word_and_a_reference_to_a_word_compare_and_order() {
    runs_to("let a = 1; let b = 1; let r = &a; r == b", "true");
    runs_to("let a = 1; let b = 2; let r = &a; r < b", "true");
    runs_to("let a = 1; let b = 2; let r = &b; a < r", "true");
    runs_to("let b = 2; let f = |x, y| -> x < y; f(1, &b)", "true");
    runs_to("let b = 2; let f = |x, y| -> x == y; f(&b, 2)", "true");
    runs_to("let b = 2; let f = |x, y| -> x > y; f(&b, 1)", "true");
}

/// RFC-0020: `+` on text concatenates, reading what a reference names at
/// any size, and `+` on a word reads a `&word` through.
#[test]
fn a_mixed_addition_reads_the_reference_through() {
    runs_to(
        "let a = \"a\".to_string(); let b = \"b\".to_string(); let r = &a; r + b",
        "\"ab\"",
    );
    runs_to(
        "let b = \"b\".to_string(); let f = |x, y| -> x + y; f(\"a\".to_string(), &b)",
        "\"ab\"",
    );
    runs_to(
        "let mk = |k| -> |x| -> x + k; let g = mk(\"b\".to_string()); g(\"a\".to_string())",
        "\"ab\"",
    );
    runs_to(
        "let a = \"a\".to_string(); let f = |x| -> x + \"b\"; f(&a)",
        "\"ab\"",
    );
    runs_to("let b = 2; let f = |x, y| -> x + y; f(1, &b)", "3");
}

/// RFC-0029 rules 3 and 5: a `&mut T` operand is read as the shared
/// reborrow of what it names, and a reference to a reference is the
/// reference it reborrows.
#[test]
fn a_mutable_or_reborrowed_operand_is_read_at_its_referent() {
    runs_to(
        "let a = \"q\".to_string(); let b = \"q\".to_string(); let r = &mut a; r == b",
        "true",
    );
    runs_to(
        "let a = \"a\".to_string(); let r = &mut a; r + \"b\"",
        "\"ab\"",
    );
    runs_to(
        "let v = vec([1]); let w = vec([1]); let r = &mut v; r == w",
        "true",
    );
    runs_to(
        "let a = \"q\".to_string(); let f = |x, y| -> x == y; f(&mut a, \"q\".to_string())",
        "true",
    );
    runs_to("let a = 1; let f = |x, y| -> x == y; f(&mut a, 1)", "true");
    runs_to(
        "let v = vec([1]); let f = |x, y| -> x == y; f(&mut v, vec([1]))",
        "true",
    );
    runs_to(
        "let a = \"q\".to_string(); let r = &a; let f = |x, y| -> x == y; f(&r, \"q\".to_string())",
        "true",
    );
}

/// RFC-0020: `&&` reads a `&Bool` operand through, where its type is
/// known and where it settles later.
#[test]
fn a_connective_reads_a_reference_to_a_word_through() {
    runs_to("let b = true; let r = &b; r && true", "true");
    runs_to(
        "let b = true; let f = |x, y| -> x && y; f(true, &b)",
        "true",
    );
}

/// The two operands meet at what they name, so operands naming two types
/// are the operator's mismatch, stated at their referents, where the
/// types are known and where they settle later.
#[test]
fn operands_naming_two_types_are_the_operators_mismatch() {
    refused_with(
        "let a = \"q\".to_string(); let r = &a; r == 1",
        "type mismatch in `==`: String vs i64",
    );
    refused_with(
        "let a = \"q\".to_string(); let f = |x, y| -> x == y; f(&a, 1)",
        "type mismatch in `==`: String vs i64",
    );
}
