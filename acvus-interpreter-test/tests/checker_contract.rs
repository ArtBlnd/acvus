//! Programs whose checked type and lowered code once disagreed: each is
//! either refused by the checker or runs to the value its type says, at both
//! optimization levels.

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
fn a_call_under_a_borrow_moves_its_argument_out_of_an_index_only_where_it_may() {
    refused_with(
        "let a = [\"a\".to_string(), \"b\".to_string()]; let f = |s| -> s; let r = &f(a[0]); r.len()",
        "cannot move out of index",
    );
}

#[test]
fn an_if_without_else_is_unit() {
    runs_to("let n = 1; let v = if true { n + 1 }; v", "null");
    refused_with(
        "let v = if true { 1 }; v + 1",
        "type mismatch in `+`: Unit vs i64",
    );
}

#[test]
fn a_piped_value_is_an_argument_of_a_structural_variant() {
    refused_with(
        "let x = 5; let r = x | Shape::Circle(1); match r { Shape::Circle(n) => n, _ => 0, }",
        "Shape::Circle",
    );
}

#[test]
fn a_container_that_is_a_value_is_lent_from_a_temporary() {
    runs_to("let x = [1, 2, 3][1]; x", "2");
}

#[test]
fn a_local_closure_called_as_a_method_takes_its_receiver_as_its_parameter_does() {
    runs_to("let v = [1, 2]; let f = |x| -> x.len(); f(&v) + v.f()", "4");
}

/// A `String` copies (RFC-0018): the copy is the language's at every level.
#[test]
fn a_string_used_twice_is_copied_at_every_level() {
    runs_to("let s = \"a\".to_string(); let t = s; let u = s; t + &u", "\"aa\"");
    runs_to(
        "let s = \"abc\".to_string(); let f = |x| -> x; let a = f(s); let b = f(s); a + &b",
        "\"abcabc\"",
    );
}

#[test]
fn a_unary_operator_refuses_an_operand_it_does_not_take() {
    refused_with("let x = !5; x", "type mismatch in `!`");
    refused_with("let x = *5; x", "`*` needs a reference");
}
