//! A loop a `break`, a `continue` or a `?` leaves is still one region
//! (RFC-0057 amended).
//!
//! The contract is the value the script returns, at both optimization levels,
//! and then the shape the machine runs it as: one `For` or `Loop` operation
//! holding the body, with the branch that leaves it an `Escape` inside that
//! body rather than a block of its own.

use acvus_interpreter_test::Context;
use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::listing::{
    BlockListing, family_of, ops_of_anywhere, regions_named, script_listing,
};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[track_caller]
fn answers(source: &str, expected: &str) {
    for opt in [Opt::None, Opt::Full] {
        let outcome = corpus::attempt(source, opt, Stage::Run);
        let Outcome::Value(got) = &outcome else {
            panic!("{opt:?} did not run {source}: {outcome:?}");
        };
        assert_eq!(got, expected, "{opt:?}");
    }
}

fn blocks(source: &str, ret: Ty) -> Vec<BlockListing> {
    let interner = Interner::new();
    script_listing(&interner, source, Context::default(), ret)
}

fn families(found: &[BlockListing]) -> Vec<String> {
    ops_of_anywhere(found)
        .iter()
        .map(|name| family_of(name).to_string())
        .collect()
}

#[track_caller]
fn body_of_the_one_loop(source: &str, ret: Ty, named: &str) -> Vec<String> {
    let found = blocks(source, ret);
    let held = families(&found);
    assert_eq!(
        held.iter().filter(|f| *f == "For" || *f == "Loop").count(),
        1,
        "{source} prepares to one loop region: {held:?}"
    );
    let named_regions = regions_named(&found, named);
    let [region] = named_regions.as_slice() else {
        panic!("{source} prepares to one {named}: {held:?}")
    };
    region
        .part("body")
        .unwrap_or_else(|| panic!("a {named} owns a body chain"))
        .ops
        .iter()
        .map(|name| family_of(name).to_string())
        .collect()
}

// -- `break` ------------------------------------------------------------

#[test]
fn a_break_over_a_range_is_a_region() {
    let source = "let acc = 0; for i in 0..10 { if i > 5 { break; }; acc = acc + i; } acc";
    answers(source, "15");
    assert_eq!(
        body_of_the_one_loop(source, Ty::I64, "For"),
        ["Gt", "Escape", "Add"]
    );
}

#[test]
fn a_break_over_an_array_of_words_carries_the_value() {
    answers(
        "let a = [1, 2, 3, 4]; let acc = 0; \
         for x in a { if x > 2 { break; }; acc = acc + x; } acc",
        "3",
    );
}

#[test]
fn a_break_over_a_slice_carries_the_value() {
    answers(
        "let v = [1, 2, 3, 4]; let acc = 0; \
         for x in &v { if *x > 2 { break; }; acc = acc + *x; } acc",
        "3",
    );
}

#[test]
fn a_break_out_of_a_while_is_a_region() {
    let source = "let i = 0; let acc = 0; \
                  while i < 10 { i = i + 1; if i > 5 { break; }; acc = acc + i; } acc";
    answers(source, "15");
    assert_eq!(
        body_of_the_one_loop(source, Ty::I64, "Loop"),
        ["Add", "Gt", "Escape", "Add"]
    );
}

// -- `continue` ---------------------------------------------------------

#[test]
fn a_continue_over_a_range_is_a_region() {
    let source = "let acc = 0; for i in 0..6 { if i % 2 == 1 { continue; }; acc = acc + i; } acc";
    answers(source, "6");
    assert!(
        body_of_the_one_loop(source, Ty::I64, "For").contains(&"Escape".to_string()),
        "{source}"
    );
}

#[test]
fn a_continue_over_an_array_is_a_region() {
    let source = "let a = [1, 2, 3, 4]; let acc = 0; \
                  for x in a { if x > 2 { continue; }; acc = acc + x; } acc";
    answers(source, "3");
    assert_eq!(
        body_of_the_one_loop(source, Ty::I64, "For"),
        ["Gt", "Escape", "Add"]
    );
}

#[test]
fn a_continue_over_a_slice_carries_the_value() {
    answers(
        "let v = [1, 2, 3, 4]; let acc = 0; \
         for x in &v { if *x > 2 { continue; }; acc = acc + *x; } acc",
        "3",
    );
}

#[test]
fn a_continue_in_a_while_skips_the_rest_of_the_body() {
    answers(
        "let i = 0; let acc = 0; \
         while i < 6 { i = i + 1; if i % 2 == 1 { continue; }; acc = acc + i; } acc",
        "12",
    );
}

// -- `?` ----------------------------------------------------------------

#[test]
fn a_try_inside_a_while_returns_at_the_bad_element_out_of_the_region() {
    let source = "let i = 0; let acc = 0; \
                  while i < 4 { let step = if i == 3 { None } else { Some(i) }; \
                                acc = acc + step?; i = i + 1; } \
                  Some(acc)";
    answers(source, "\"None\"");
    let found = blocks(source, Ty::Option(Box::new(Ty::I64)));
    let held = families(&found);
    assert_eq!(regions_named(&found, "Loop").len(), 1, "{held:?}");
    assert!(held.iter().any(|family| family == "Escape"), "{held:?}");
}

#[test]
fn a_try_inside_a_while_that_never_fails_runs_to_the_end() {
    answers(
        "let i = 0; let acc = 0; \
         while i < 4 { let step = Some(i); acc = acc + step?; i = i + 1; } \
         Some(acc)",
        "6",
    );
}

#[test]
fn a_try_inside_a_for_returns_at_the_bad_element() {
    answers(
        "let acc = 0; \
         for i in 0..4 { let step = if i == 2 { None } else { Some(i) }; acc = acc + step?; } \
         Some(acc)",
        "\"None\"",
    );
}

// -- The escaping side laid second --------------------------------------

#[test]
fn an_else_that_breaks_is_a_region() {
    let source = "let i = 0; let acc = 0; \
                  while true { if i < 5 { acc = acc + i; i = i + 1; } else { break; }; } acc";
    answers(source, "10");
    assert!(
        body_of_the_one_loop(source, Ty::I64, "Loop").contains(&"Escape".to_string()),
        "{source}"
    );
}

#[test]
fn an_else_that_continues_skips_the_rest_of_the_body() {
    answers(
        "let i = 0; let acc = 0; \
         while i < 6 { i = i + 1; if i % 2 == 0 { acc = acc + i; } else { continue; }; } acc",
        "12",
    );
}

#[test]
fn an_else_that_returns_leaves_the_function() {
    answers(
        "let acc = 0; \
         for i in 0..10 { if i < 4 { acc = acc + i; } else { return acc; }; } acc",
        "6",
    );
}

// -- Nesting ------------------------------------------------------------

#[test]
fn a_break_under_a_nested_if_leaves_the_loop() {
    answers(
        "let acc = 0; \
         for i in 0..10 { if i > 2 { if i > 4 { break; }; acc = acc + 100; }; acc = acc + 1; } acc",
        "205",
    );
}

#[test]
fn two_exits_in_one_body() {
    answers(
        "let acc = 0; \
         for i in 0..10 { if i % 2 == 1 { continue; }; if i > 5 { break; }; acc = acc + i; } acc",
        "6",
    );
}

#[test]
fn a_break_in_an_inner_loop_leaves_the_inner_loop_alone() {
    answers(
        "let acc = 0; \
         for i in 0..3 { for j in 0..5 { if j > 1 { break; }; acc = acc + 1; } } acc",
        "6",
    );
}

#[test]
fn a_break_inside_a_match_arm_carries_the_value() {
    answers(
        "let acc = 0; \
         for i in 0..6 { match i { 4 => { break; }, _ => { acc = acc + i; } }; } acc",
        "6",
    );
}

#[test]
fn a_while_let_takes_a_break() {
    answers(
        "let n = 0; let acc = 0; \
         while let Some(x) = (if n < 5 { Some(n) } else { None }) { \
             if x > 2 { break; }; acc = acc + x; n = n + 1; } acc",
        "3",
    );
}

// -- What does not change -----------------------------------------------

#[test]
fn a_loop_with_no_exit_holds_no_verdict() {
    let found = blocks(
        "let acc = 0; for i in 0..10 { acc = acc + i; } acc",
        Ty::I64,
    );
    let ops = ops_of_anywhere(&found);
    assert!(
        ops.iter().any(|name| name == "For<Range<i64>, Rejoins>"),
        "{ops:?}"
    );
    assert!(
        !ops.iter().any(|name| family_of(name) == "Escape"),
        "{ops:?}"
    );
}
