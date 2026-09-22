//! Programs whose checked type and lowered code once disagreed: each is
//! either refused by the checker or runs to the value its type says, at both
//! optimization levels.

use acvus_interpreter_test::corpus::{Outcome, Stage, attempt};
use acvus_mir::graph::optimize::Opt;

fn runs_to(source: &str, value: &str) {
    for opt in [Opt::None, Opt::Full] {
        match attempt(source, opt, Stage::Run) {
            Outcome::Value(got) => assert_eq!(got, value, "at {opt:?}: {source}"),
            other => panic!("at {opt:?}, expected {value}, got {other:?}: {source}"),
        }
    }
}

fn refused_with(source: &str, reason: &str) {
    for opt in [Opt::None, Opt::Full] {
        match attempt(source, opt, Stage::Run) {
            Outcome::Refused(why) => assert!(why.contains(reason), "at {opt:?}: {why}"),
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {source}"),
        }
    }
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
