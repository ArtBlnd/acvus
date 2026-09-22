//! The run-level half of RFC-0042 rule 1 and RFC-0050 rule 8.

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
fn a_store_grows_the_bound_value() {
    runs_to("let x = { a: 1, }; x.b = 2; x.a + x.b", "3");
}

#[test]
fn a_store_grows_a_nested_value() {
    runs_to("let x = { a: { p: 1, }, }; x.a.q = 2; x.a.p + x.a.q", "3");
}

#[test]
fn a_store_through_a_second_name_grows_the_value() {
    runs_to("let x = { a: 1, }; let y = x; y.b = 2; y.a + y.b", "3");
}

#[test]
fn a_grown_value_leaves_whole() {
    runs_to("let x = { a: 1, }; x.b = 2; x", r#"{"a":1,"b":2}"#);
}

#[test]
fn a_value_assigned_to_an_uninitialized_name_grows() {
    runs_to("let x; x = { a: 1, }; x.b = 2; x.a + x.b", "3");
}

#[test]
fn a_value_grown_before_it_is_wrapped_is_bound_whole() {
    runs_to(
        "let x = { a: 1, }; x.b = 2; let o = Some(x); if let Some(y) = o { y.a + y.b } else { 0 }",
        "3",
    );
}

/// A payload is whole: the type of `y` is the payload's type, so the value
/// wrapped in `Some` would lack the field the arm stores.
#[test]
fn a_payload_grown_after_it_was_wrapped_is_refused() {
    refused_with(
        "let o = Some({ a: 1, }); if let Some(y) = o { y.b = 2; y.a + y.b } else { 0 }",
        "has no `b` stored on every path",
    );
    refused_with(
        "let o = Some({ a: 1, }); match o { Some(y) => { y.b = 2; y.a + y.b }, None => 0, }",
        "has no `b` stored on every path",
    );
}

#[test]
fn two_constructions_joined_by_a_branch_grow_together() {
    runs_to(
        "let x = if true { { a: 1, } } else { { b: 2, } }; x.c = 3; x.c",
        "3",
    );
}

#[test]
fn a_field_read_before_its_store_is_refused() {
    refused_with(
        "let x = { a: 1, }; let n = x.b; x.b = 2; n",
        "has no `b` stored on every path",
    );
}

#[test]
fn a_list_of_constructions_missing_a_field_is_refused() {
    refused_with(
        "let l = [{ a: 1, }, { b: 2, }]; l[0].b",
        "has no `b` stored on every path",
    );
}

#[test]
fn a_nested_field_read_before_its_store_is_refused() {
    refused_with(
        "let x = { a: { p: 1, }, }; let n = x.a.q; x.a.q = 2; n",
        "has no `a.q` stored on every path",
    );
}

#[test]
fn a_value_moved_to_a_second_name_keeps_its_missing_field() {
    refused_with(
        "let x = { a: 1, }; let y = x; let n = y.b; y.b = 2; n",
        "has no `b` stored on every path",
    );
}

#[test]
fn a_value_missing_a_field_cannot_leave_the_body() {
    refused_with(
        "let x = { a: 1, }; let n = 0; if n == 1 { x.b = 2; }; x",
        "has no `b` stored on every path",
    );
}

#[test]
fn a_field_read_before_its_store_in_a_lambda_is_refused() {
    refused_with(
        "let f = |n| -> { let x = { a: n, }; x.b }; f(1)",
        "has no `b` stored on every path",
    );
}
