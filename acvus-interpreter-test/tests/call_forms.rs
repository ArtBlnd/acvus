//! Each form a call is written in, pinned to the value it runs to or the
//! refusal the checker gives it.

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

#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn a_free_function_of_one_signature() {
    runs_to("abs(0 - 3)", "3");
}

#[test]
fn a_free_function_of_several_signatures() {
    runs_to("max(2, 5)", "5");
}

#[test]
fn a_method_of_one_signature() {
    runs_to("(0 - 3).abs()", "3");
    runs_to("let v = vec([1]); v.push(2); v.len()", "2");
}

#[test]
fn a_method_of_several_signatures() {
    runs_to("let v = vec([1, 2, 3]); v.len()", "3");
    runs_to("let v = [4, 5]; v.max()", "5");
}

#[test]
fn a_local_closure_is_called_directly_and_as_a_method() {
    runs_to("let v = [1, 2]; let f = |x| -> x.len(); f(&v) + v.f()", "4");
}

#[test]
fn a_pipe_stage_is_a_call_with_the_piped_value_first() {
    runs_to("(0 - 3) | abs", "3");
    runs_to("3 | max(5)", "5");
    runs_to("let v = vec([1, 2, 3]); &v | len", "3");
    runs_to("let f = |a, b| -> a + b; 1 | f(2)", "3");
    runs_to("2 | (|y| -> y * 3)", "6");
}

#[test]
fn an_index_is_the_container_s_own_as_slice() {
    runs_to("let v = vec([1, 2, 3]); v[1u64]", "2");
    runs_to("let a = [1, 2, 3]; a[2u64]", "3");
}

#[test]
fn a_for_over_a_borrow_is_the_container_s_own_as_slice() {
    runs_to(
        "let v = vec([1, 2, 3]); let t = 0; for x in &v { t = t + *x; } t",
        "6",
    );
    runs_to(
        "let a = [1, 2, 3]; let t = 0; for x in &a { t = t + *x; } t",
        "6",
    );
}

#[test]
fn an_operator_on_an_extension_value_is_a_call_of_its_shared_signature() {
    runs_to("let a = vec([1, 2]); a == a", "true");
    runs_to("let a = vec([1, 2]); a != a", "false");
    runs_to(
        "let a = decimal(\"1.5\".to_string()).unwrap(); \
         let b = decimal(\"2.5\".to_string()).unwrap(); a < b",
        "true",
    );
}

#[test]
fn a_call_of_the_wrong_arity_is_refused_in_every_form() {
    refused_with("abs(1, 2)", "function `abs` expects 1 arguments, got 2");
    refused_with("max(1, 2, 3)", "no `max` takes a call of type");
    refused_with("(0 - 1).abs(2)", "function `abs` expects 1 arguments, got 2");
    refused_with("[1].len(2)", "no `len` takes a call of type");
    refused_with("1 | abs(2)", "function `abs` expects 1 arguments, got 2");
    refused_with(
        "let f = |x| -> x; f(1, 2)",
        "this closure expects 1 arguments, got 2",
    );
    refused_with(
        "let f = |x| -> x; 1.f(2)",
        "this closure expects 1 arguments, got 2",
    );
    refused_with(
        "(|x| -> x)(1, 2)",
        "this closure expects 1 arguments, got 2",
    );
}

#[test]
fn a_lambda_s_parameter_is_the_type_an_earlier_argument_fixed() {
    runs_to("vec([1, 2]) | into_iter | map(|x| -> x * 10) | sum", "30");
    runs_to(
        "let s = vec([\"ab\".to_string(), \"c\".to_string()]); \
         let n = map(into_iter(s), |t| -> t.len()) | collect; n[0u64] + n[1u64]",
        "3",
    );
}

/// A receiver every candidate takes by value is read as a value, so an
/// element that moves is not taken out of its index; checked as a borrow,
/// it was, and the run read freed storage.
#[test]
fn a_receiver_every_signature_takes_by_value_is_not_moved_out_of_an_index() {
    refused_with(
        "let a = [Some(\"a\".to_string())]; a[0u64].unwrap()",
        "cannot move out of index",
    );
    runs_to("let a = [Some(1)]; a[0u64].unwrap()", "1");
}
