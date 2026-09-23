//! `+`, `-`, `*`, `/`, `%` and unary `-` are calls of `core::add`,
//! `core::sub`, `core::mul`, `core::div`, `core::rem` and `core::neg`
//! (RFC-0020): at a word the language's instance is the instruction, at an
//! extension type the registry's instance runs, and a type with no instance
//! is refused. Each program runs at both optimization levels.

use std::time::Duration;

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

const LIMIT: Duration = Duration::from_secs(30);

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

fn fails_at_run_with(source: &str, reason: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::RunPanicked(why) => assert!(why.contains(reason), "at {opt:?}: {why}"),
            other => panic!("at {opt:?}, expected a run failure, got {other:?}: {source}"),
        }
    }
}

const X: &str = "let x = decimal(\"1.5\".to_string()).unwrap(); \
                 let y = decimal(\"2.25\".to_string()).unwrap(); ";

fn with_decimals(body: &str) -> String {
    format!("{X}{body}")
}

#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn decimal_arithmetic_on_known_operands_runs_the_registry_instances() {
    runs_to(&with_decimals("(x + y).to_string()"), "\"3.75\"");
    runs_to(&with_decimals("(x - y).to_string()"), "\"-0.75\"");
    runs_to(&with_decimals("(x * y).to_string()"), "\"3.375\"");
    runs_to(
        &with_decimals(
            "x / decimal(\"0.75\".to_string()).unwrap() == decimal(\"2\".to_string()).unwrap()",
        ),
        "true",
    );
    runs_to(
        &with_decimals(
            "decimal(\"7.5\".to_string()).unwrap() % decimal(\"2\".to_string()).unwrap() == x",
        ),
        "true",
    );
    runs_to(&with_decimals("(-x).to_string()"), "\"-1.5\"");
}

/// The call lends its operands, so the places stay readable after it.
#[test]
fn decimal_operands_are_lent_not_moved() {
    runs_to(
        &with_decimals("let a = x + y; let b = x - y; let c = -x; (a + b + c).to_string()"),
        "\"1.50\"",
    );
}

#[test]
fn decimal_arithmetic_on_lambda_parameters_settled_by_value() {
    runs_to(
        &with_decimals("let f = |a, b| -> a + b; f(x, y).to_string()"),
        "\"3.75\"",
    );
    runs_to(
        &with_decimals("let n = |a| -> -a; n(y).to_string()"),
        "\"-2.25\"",
    );
}

#[test]
fn decimal_arithmetic_on_lambda_parameters_settled_by_reference() {
    runs_to(
        &with_decimals("let f = |a, b| -> a * b; let r = f(&x, &y); (r + x).to_string()"),
        "\"4.875\"",
    );
    runs_to(
        &with_decimals("let n = |a| -> -a; let r = n(&x); (r + x).to_string()"),
        "\"0.0\"",
    );
}

/// RFC-0037 rule 2 at an extension type: the instance's failure is the
/// program's, not a value.
#[test]
fn decimal_division_by_zero_fails_the_run() {
    fails_at_run_with(
        &with_decimals("(x / decimal(\"0\".to_string()).unwrap()).to_string()"),
        "Division by zero",
    );
}

/// RFC-0037 rule 4: the literal's integer variable and the open operand
/// meet, and the instance settles at the width the call gives.
#[test]
fn an_integer_literal_meets_an_open_operand() {
    runs_to("let f = |k| -> k + 7; f(3)", "10");
    runs_to("let f = |k| -> k + 7; f(250u8)", "1");
    runs_to("let f = |k| -> k * 2; let n = f(3); n - 1", "5");
}

#[test]
fn word_arithmetic_on_lambda_parameters_settled_by_reference() {
    runs_to(
        "let f = |a, b| -> a / b; let p = 1.0; let q = 4.0; f(&p, &q)",
        "0.25",
    );
    runs_to("let n = 5; let f = |a| -> -a; f(&n)", "-5");
    runs_to(
        "let n = 7; let m = 2; let f = |a, b| -> a % b; f(&n, &m)",
        "1",
    );
}

/// RFC-0020: a `&word` operand is read through the reference, and an open
/// operand is read through what the call settles it to.
#[test]
fn an_open_operand_bound_is_on_what_a_reference_names() {
    runs_to(
        "let x = 1; let y = 2; let f = |a, b| -> a + b; f(&x, &y)",
        "3",
    );
}

/// RFC-0062 rule 3: an operand of `+` still open beside text is bound to
/// text, and the concatenation runs once the call settles it to `String`.
#[test]
fn a_concatenation_on_a_lambda_parameter() {
    runs_to("let f = |x| -> x + \"b\"; f(\"a\".to_string())", "\"ab\"");
}

/// RFC-0037 rule 2: nothing widens by itself.
/// RFC-0020: `core::add` has the language's instances at `String` and
/// `str`, so two open operands that settle to `String` concatenate.
#[test]
fn a_concatenation_of_two_lambda_parameters() {
    runs_to(
        "let f = |a, b| -> a + b; f(\"a\".to_string(), \"b\".to_string())",
        "\"ab\"",
    );
}

#[test]
fn operands_of_two_widths_are_refused() {
    refused_with("1u8 + 1u16", "type mismatch in `+`");
    refused_with("let f = |a, b| -> a + b; f(1u8, 1u16)", "");
    refused_with(&with_decimals("x + 1"), "type mismatch in `+`");
}

/// RFC-0020: `Bool` and `char` have no arithmetic instance.
#[test]
fn bool_and_char_are_no_arithmetic_operands() {
    refused_with("true + 1", "type mismatch in `+`");
    refused_with("true + true", "`+` has no instance of core::add");
    refused_with("-true", "`-` has no instance of core::neg");
    refused_with("'a' - 'b'", "`-` has no instance of core::sub");
    refused_with("-'a'", "`-` has no instance of core::neg");
    refused_with(
        "let f = |c| -> c - 'b'; f('a')",
        "`-` has no instance of core::sub",
    );
    refused_with("let f = |a, b| -> a - b; f('a', 'b')", "");
}

/// RFC-0019: `Object` has no instance of any `core` signature, and
/// `Vec<T>` none of `core::add`.
#[test]
fn a_vec_or_an_object_has_no_add() {
    refused_with("vec([1]) + vec([2])", "`+` has no instance of core::add");
    refused_with("{ a: 1, } + { a: 1, }", "`+` has no instance of core::add");
}
