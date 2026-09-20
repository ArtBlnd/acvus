//! A `match` on literals arrives at the value its arm names (RFC-0051,
//! extended to literal keys).
//!
//! The contract is the value the script returns, so each case runs every arm
//! of one `match` over the same source and asks for that arm's own answer,
//! at both optimization levels: the level chooses how hard the compiler
//! works, never which arm holds.
//!
//! Obligation across artifacts: one case per operation `prepare::switch_op`
//! can choose for a literal key -- `switch::SwitchWord` over an integer and
//! over a char, `control::JumpIf` over a `Bool`, and `string::SwitchStr`
//! over each of the three shapes `prepare::text_at` names: a `String` the
//! register holds, a `&String`, and the pair of a `&str`.

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::int_context;
use acvus_interpreter_test::listing::{BlockListing, ops_of_anywhere, script_listing_with_externs};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// The value the source comes to, the same at both levels.
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

#[test]
fn every_arm_of_an_integer_match_is_reached() {
    for (n, expected) in [(0, "0"), (1, "10"), (2, "20"), (3, "30"), (9, "99")] {
        answers(
            &format!("let n = {n}; match n {{ 0 => 0, 1 => 10, 2 => 20, 3 => 30, _ => 99 }}"),
            expected,
        );
    }
}

/// A narrow width is the register's word read at that width, which is what
/// `switch::SwitchWord<T>` normalizes and `prepare::word_key` matches.
#[test]
fn an_integer_match_at_a_narrow_width_reaches_the_arm() {
    for (n, expected) in [(0, "0"), (200, "2"), (255, "3")] {
        answers(
            &format!("let n = {n}u8; match n {{ 200u8 => 2, 255u8 => 3, _ => 0 }}"),
            expected,
        );
    }
}

#[test]
fn every_arm_of_a_char_match_is_reached() {
    for (c, expected) in [("a", "1"), ("b", "2"), ("z", "0")] {
        answers(
            &format!("let c = '{c}'; match c {{ 'a' => 1, 'b' => 2, _ => 0 }}"),
            expected,
        );
    }
}

/// Two arms and no catch-all: `Bool` is the one literal space a set of arms
/// closes, and the machine runs it as its own two-way branch.
#[test]
fn both_arms_of_a_bool_match_are_reached() {
    answers("let b = 1 == 1; match b { true => 7, false => 8 }", "7");
    answers("let b = 1 == 2; match b { true => 7, false => 8 }", "8");
}

#[test]
fn a_bool_match_written_with_a_catch_all_reaches_it() {
    answers("let b = 1 == 2; match b { true => 7, _ => 8 }", "8");
    answers("let b = 1 == 1; match b { false => 7, _ => 8 }", "8");
}

/// `let s = "put"` gives a `&str`, whose text is the pair of registers
/// `prepare::text_at` reads as `LentText::Pair`.
#[test]
fn every_arm_of_a_match_on_a_str_is_reached() {
    for (s, expected) in [("get", "1"), ("put", "2"), ("del", "0")] {
        answers(
            &format!("let s = \"{s}\"; match s {{ \"get\" => 1, \"put\" => 2, _ => 0 }}"),
            expected,
        );
    }
}

/// `to_string` hands back a `String` the register holds itself, which is
/// `LentText::Own`.
#[test]
fn a_match_on_an_owned_string_reaches_the_arm() {
    for (n, expected) in [(7, "1"), (8, "2"), (9, "0")] {
        answers(
            &format!(
                "let n = {n}; let s = &n | to_string; match s {{ \"7\" => 1, \"8\" => 2, _ => 0 }}"
            ),
            expected,
        );
    }
}

/// A `&String` is read through its reference, which is `LentText::Through`.
///
/// This is also the case that names the loan a dispatch holds: the scrutinee
/// is a reference to a local `String`, and a `drop` of that local placed
/// before the dispatch reads through it is a use after free.
#[test]
fn a_match_through_a_reference_to_a_string_reaches_the_arm() {
    for (n, expected) in [(7, "1"), (9, "0")] {
        answers(
            &format!(
                "let n = {n}; let s = &n | to_string; let r = &s; \
                 match r {{ \"7\" => 1, _ => 0 }}"
            ),
            expected,
        );
    }
}

/// A literal `match` whose arms all rejoin, inside a loop: the region form
/// of the dispatch, which `recognize_switch` builds from the same
/// `InstKind::Switch` a tag dispatch is recognized from.
const LITERAL_MATCH_IN_A_LOOP: &str = "\
let acc = 0; let i = 0; \
while i < @n { \
let picked = match i % 3 { 0 => 1, 1 => 20, _ => 300 }; \
acc = acc + picked; i = i + 1; } acc";

#[test]
fn a_literal_match_inside_a_loop_reaches_every_arm_in_turn() {
    // 1, 20, 300 by i % 3: three iterations sum to 321, five to 342.
    answers(&LITERAL_MATCH_IN_A_LOOP.replace("@n", "3"), "321");
    answers(&LITERAL_MATCH_IN_A_LOOP.replace("@n", "5"), "342");
}

fn ops(source: &str) -> Vec<String> {
    let interner = Interner::new();
    let blocks: Vec<BlockListing> = script_listing_with_externs(
        &interner,
        source,
        int_context(&interner, "n", 3),
        acvus_ext::std_registries(),
        Ty::I64,
    );
    let ends: Vec<String> = blocks.iter().map(|block| block.end.clone()).collect();
    ops_of_anywhere(&blocks).into_iter().chain(ends).collect()
}

#[test]
fn a_rejoining_literal_match_is_one_region_operation_and_no_chain_of_tests() {
    let ops = ops(LITERAL_MATCH_IN_A_LOOP);
    assert_eq!(
        ops.iter()
            .filter(|op| op.starts_with("SwitchWordRegion"))
            .count(),
        1,
        "the rejoining `match` is one region dispatch: {ops:?}"
    );
    assert!(
        !ops.iter().any(|op| op.starts_with("TestInt")),
        "no test of the chain is left: {ops:?}"
    );
}

/// A float arm is no dispatch key, so this `match` keeps the chain of
/// `TestFloat` and branches, and still arrives at the arm it names.
#[test]
fn a_match_on_float_literals_still_reaches_the_arm() {
    for (x, expected) in [("1.5", "1"), ("2.5", "2"), ("3.5", "0")] {
        answers(
            &format!("let x = {x}; match x {{ 1.5 => 1, 2.5 => 2, _ => 0 }}"),
            expected,
        );
    }
}
