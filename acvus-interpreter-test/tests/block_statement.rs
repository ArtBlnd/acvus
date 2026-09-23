//! An `if` or a `match` that begins a statement ends at its `}`: followed
//! by more of the body it is a statement whose value is dropped, and at the
//! end of the body it is the tail. The value each shape produces at both
//! optimization levels.

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

/// The value both levels produce, rendered as `acvus run` prints it. A shape
/// whose two levels disagree fails here before its value is read.
fn value(source: &str) -> String {
    let full = corpus::attempt(source, Opt::Full, Stage::Run);
    let none = corpus::attempt(source, Opt::None, Stage::Run);
    assert_eq!(full, none, "the two levels disagree on `{source}`");
    match full {
        Outcome::Value(v) => v,
        other => panic!("`{source}` produced no value: {other:?}"),
    }
}

#[test]
fn an_if_without_else_is_a_statement() {
    assert_eq!(value("let a = 1; if a == 1 { a = 2; } a"), "2");
    assert_eq!(value("let o = Some(4); let a = 0; if let Some(x) = o { a = x; } a"), "4");
}

#[test]
fn an_if_with_else_is_a_statement_and_its_value_is_dropped() {
    assert_eq!(value("let a = 1; if a == 2 { a = 5; } else { a = 7; } a"), "7");
    assert_eq!(value("let a = 1; if a == 1 { 10 } else { 20 } a"), "1");
}

#[test]
fn a_match_is_a_statement() {
    assert_eq!(value("let a = 0; match 2 { 1 => { a = 1; }, _ => { a = 3; } } a"), "3");
}

#[test]
fn a_block_that_begins_a_statement_reads_on() {
    assert_eq!(value("{ [5, 6] }[1]"), "6");
}

#[test]
fn last_in_the_body_it_is_the_tail() {
    assert_eq!(value("let a = 1; if a == 1 { 10 } else { 20 }"), "10");
    assert_eq!(value("let a = 1; match a { 1 => 10, _ => 20 }"), "10");
}

#[test]
fn a_semicolon_after_the_last_one_makes_it_a_statement() {
    assert_eq!(
        value("let a = 1; if a == 1 { 10 } else { 20 };"),
        value("let a = 1; a = 2;")
    );
}

#[test]
fn inside_a_for_it_is_a_statement() {
    assert_eq!(
        value("let v = vec([{ a: 1, }, { a: 3, }]); for x in &mut v { if x.a == 1 { x.a = 2; } } v.len()"),
        "2"
    );
    assert_eq!(
        value(
            "let v = vec([{ a: 1, }, { a: 3, }]); \
             for x in &mut v { if x.a == 1 { x.a = 2; } } \
             let s = 0; for x in &v { s = s + x.a; } s"
        ),
        "5"
    );
}
