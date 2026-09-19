//! A diamond of one pure node and a pass-through arm as one `Select`
//! (RFC-0052 §"a diamond of two pure arms is a select").
//!
//! A `Select` evaluates its node whichever way the condition goes, so every
//! test here that ends in a `Diamond` is a soundness test: the shapes below
//! are the ones the recognizer must refuse, and each refused one runs a value
//! the speculating form would have got wrong or raised on.

use acvus_interpreter_test::listing::{family_of, ops_of_anywhere, regions_named, script_listing};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

fn named_on_page(source: &str, family: &str, page: fn(&Interner) -> Context) -> Vec<String> {
    let interner = Interner::new();
    let page = page(&interner);
    ops_of_anywhere(&script_listing(&interner, source, page, Ty::I64))
        .into_iter()
        .filter(|name| family_of(name) == family)
        .collect()
}

fn named(source: &str, family: &str) -> Vec<String> {
    named_on_page(source, family, |_| Context::default())
}

fn count_of(source: &str, family: &str) -> usize {
    named(source, family).len()
}

const ADDS_WHEN_EVEN: &str =
    "let acc = 0; let i = 0; while i < 6 { if i % 2 == 0 { acc = acc + i; }; i = i + 1; } acc";

#[tokio::test]
async fn one_node_against_a_pass_through_arm_is_a_select() {
    assert_eq!(count_of(ADDS_WHEN_EVEN, "Diamond"), 0);
    assert_eq!(
        named(ADDS_WHEN_EVEN, "Select"),
        vec!["Select<i64, Slot, Slot, 1, true>"],
        "the node's operator is in the type and the `then` side is the one that computes"
    );
}

#[tokio::test]
async fn a_select_carries_the_word_of_the_side_the_condition_picked() {
    let i = Interner::new();
    let v = run_script(&i, ADDS_WHEN_EVEN, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 0 + 2 + 4);
}

/// The sabotage: were `/` admitted, the arm would run on the path the program
/// does not take and divide by zero (RFC-0037).
#[tokio::test]
async fn a_dividing_arm_stays_a_diamond() {
    let source = "let acc = 7; let z = 0; if z != 0 { acc = acc / z; }; acc";
    assert_eq!(count_of(source, "Select"), 0);
    assert_eq!(count_of(source, "Diamond"), 1);

    let i = Interner::new();
    let v = run_script(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 7);
}

#[tokio::test]
async fn a_remainder_arm_stays_a_diamond() {
    let source = "let acc = 7; let z = 0; if z != 0 { acc = acc % z; }; acc";
    assert_eq!(count_of(source, "Select"), 0);
    assert_eq!(count_of(source, "Diamond"), 1);

    let i = Interner::new();
    let v = run_script(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 7);
}

/// The cost rule: one operator node. A second one would put an intermediate
/// value in a register on a path the program does not take.
#[tokio::test]
async fn an_arm_of_two_operations_stays_a_diamond() {
    let source = "let acc = 0; let i = 0; while i < 6 { if i % 2 == 0 { acc = acc + i * 3; }; i = i + 1; } \
         acc";
    assert_eq!(count_of(source, "Select"), 0);
    assert_eq!(count_of(source, "Diamond"), 1);

    let i = Interner::new();
    let v = run_script(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), (0 + 2 + 4) * 3);
}

/// Both sides computing is refused for a reason outside this operation:
/// `assign_slots` coalesced the two arms' words into the join's one register
/// before the recognizer ran, so a select of both has one register for two
/// live words.
#[tokio::test]
async fn two_computing_arms_stay_a_diamond() {
    let source = "let i = 5; let d = if i % 2 == 0 { i + 1 } else { i + 2 }; d";
    assert_eq!(count_of(source, "Select"), 0);
    assert_eq!(count_of(source, "Diamond"), 1);

    let i = Interner::new();
    let v = run_script(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 7);
}

#[tokio::test]
async fn a_select_in_a_body_leaves_the_while_one_region() {
    let interner = Interner::new();
    let listing = script_listing(&interner, ADDS_WHEN_EVEN, Context::default(), Ty::I64);
    let loops = regions_named(&listing, "Loop");
    assert_eq!(loops.len(), 1);
    let body = loops[0].part("body").expect("a Loop holds a body");
    assert!(
        body.regions.is_empty(),
        "the branch in the body is a Select, which holds no part, so the `while` \
         is the only region left and the only chains are the head's and the body's"
    );
}

#[tokio::test]
async fn a_comparing_arm_is_a_select_whose_word_rides() {
    let source = "let f = false; if @n > 0 { f = @n > 3; }; if f { 1 } else { 0 }";
    assert_eq!(
        named_on_page(source, "Select", |i| int_context(i, "n", 5)),
        vec!["Select<i64, R0, R0, 0, true>"]
    );

    let i = Interner::new();
    let v = run_script(&i, source, int_context(&i, "n", 5), Ty::I64).await;
    assert_eq!(v.as_int(), 1);
}

#[tokio::test]
async fn a_comparing_arm_the_condition_skips_keeps_the_incoming_word() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let f = false; if @n > 0 { f = @n > 3; }; if f { 1 } else { 0 }",
        int_context(&i, "n", 0),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 0);
}
