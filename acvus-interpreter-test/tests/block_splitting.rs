//! Where a body's blocks begin and end, and what a region holds (RFC-0052
//! §1, §3).
//!
//! The contract under test, as `prepare::Split` states it: a block begins at
//! the entry, at every label a jump names, and after every terminator; it
//! ends at its terminator, or falls through to the next block with a `Goto`.
//! A recognized `while` or `if` is an **operation** of the block it sits in,
//! holding operation lists of its own — so it splits no block, and the
//! operations after it carry on in the same one.
//!
//! What follows an `if` is an operation of the enclosing list either way.
//! `optimize::code_motion` moves a tail that reads neither arm above the
//! branch, and the two sources below differ in exactly that: the first
//! loop's `n = n + 1` sits before the `Diamond`, the second's `acc = acc * 2`
//! reads what both arms wrote and sits after it.
//!
//! What this file reads is the shape `Op::owns` exposes — the parts and the
//! operations in them — and the value the script computes.

use acvus_interpreter_test::listing::{
    BlockListing, PartListing, RegionListing, ops_of_anywhere, regions_named, script_listing,
};
use acvus_interpreter_test::*;
use acvus_utils::Interner;

/// A `while` whose body holds an `if` and nothing after it that reads an arm.
const TAIL_ABOVE_THE_BRANCH: &str =
    "let acc = 0; let n = 0; while n < 6 { if n % 2 == 0 { acc = acc + n; }; n = n + 1; } acc";

/// The same loop with a tail that reads what both arms wrote.
const TAIL_BELOW_THE_JOIN: &str = "let acc = 0; let n = 0; while n < 6 { \
     if n % 2 == 0 { acc = acc + n; } else { acc = acc + 1; }; acc = acc * 2; n = n + 1; } acc";

fn ends(blocks: &[BlockListing]) -> Vec<&str> {
    blocks.iter().map(|b| b.end.as_str()).collect()
}

fn part_of<'r>(region: &'r RegionListing, part: &str) -> &'r PartListing {
    region
        .part(part)
        .unwrap_or_else(|| panic!("the region holds no {part} part"))
}

fn loop_of(source: &str) -> Vec<BlockListing> {
    let interner = Interner::new();
    let blocks = script_listing(&interner, source, Context::default());
    assert_eq!(
        ends(&blocks),
        ["Goto", "Return<true>"],
        "the whole `while` is an operation of the entry block; the block after it \
         is where the join of the loop's exit edge lands, and it returns"
    );
    blocks
}

fn one_loop(blocks: &[BlockListing]) -> &RegionListing {
    let found = regions_named(blocks, "Loop");
    assert_eq!(found.len(), 1, "the source holds one `while`");
    found[0]
}

#[tokio::test]
async fn a_regions_head_is_one_operation_list() {
    let blocks = loop_of(TAIL_ABOVE_THE_BRANCH);
    assert_eq!(
        part_of(one_loop(&blocks), "head").ops,
        ["Lt<i64>"],
        "the head is the condition alone: no terminator, because the head's \
         `JumpIf` is the `cond` the Loop reads itself"
    );
}

#[tokio::test]
async fn a_tail_that_reads_neither_arm_sits_before_the_diamond() {
    let blocks = loop_of(TAIL_ABOVE_THE_BRANCH);
    let body = part_of(one_loop(&blocks), "body");
    assert_eq!(
        body.ops,
        [
            "Chain2<i64, 0, 0>",
            "Add<i64>",
            "Diamond",
            "Mov<false, true>"
        ],
        "`n = n + 1` reads neither arm, so code_motion put it above the branch; \
         the diamond follows in the same list, and the loop's back edge is the \
         `Mov` after it"
    );
    let diamond = &body.regions[0];
    assert_eq!(part_of(diamond, "on_true").ops, ["Add<i64>"]);
    assert_eq!(part_of(diamond, "on_false").ops, Vec::<String>::new());
}

#[tokio::test]
async fn a_tail_that_reads_an_arm_sits_after_the_diamond() {
    let blocks = loop_of(TAIL_BELOW_THE_JOIN);
    let body = part_of(one_loop(&blocks), "body");
    assert_eq!(
        body.ops,
        [
            "Chain2<i64, 0, 0>",
            "Add<i64>",
            "Diamond",
            "Mul<i64>",
            "Mov<false, true>"
        ],
        "`acc = acc * 2` reads what both arms wrote, so it follows the Diamond \
         in the same list, and the loop's back edge is the `Mov` after it"
    );
    assert_eq!(
        body.leaves_with, 1,
        "the back edge carries one move, as an operation of the body"
    );
}

/// The decision `ops/control.rs` records: there is no flag a region tests for
/// a `return` inside it, because the recognizer never admits one.
#[tokio::test]
async fn a_while_that_returns_is_not_a_region() {
    let interner = Interner::new();
    let source = r#"let i = 0; let acc = 0;
                    while i < 4 { let step = if i == 3 { None } else { Some(i) };
                                  acc = acc + step?; i = i + 1; }
                    Some(acc)"#;
    let blocks = script_listing(&interner, source, Context::default());
    let names = ops_of_anywhere(&blocks);
    assert!(
        !names.iter().any(|name| name == "Loop"),
        "a `return` is not straight-line, so `straight_run` stops at it and the \
         `while` prepares as blocks: {names:?}"
    );
    assert!(
        ends(&blocks).contains(&"JumpIf") && ends(&blocks).contains(&"Return<false>"),
        "the loop is blocks: its test is a `JumpIf` terminator and the `?` a \
         `Return` one — {:?}",
        ends(&blocks)
    );

    let value = run_script_mode(&interner, source, Context::default()).await;
    assert!(
        value.is_none(),
        "the `?` returns None out of the loop at i == 3: {value:?}"
    );
}

#[tokio::test]
async fn the_blocks_that_splitting_states_run_to_their_values() {
    let interner = Interner::new();
    let above = run_script_mode(&interner, TAIL_ABOVE_THE_BRANCH, Context::default()).await;
    assert_eq!(
        above.as_int(),
        0 + 2 + 4,
        "the even n below 6, summed by the arm inside the loop's body"
    );

    let mut acc = 0i64;
    for n in 0..6 {
        acc += if n % 2 == 0 { n } else { 1 };
        acc *= 2;
    }
    let below = run_script_mode(&interner, TAIL_BELOW_THE_JOIN, Context::default()).await;
    assert_eq!(
        below.as_int(),
        acc,
        "one arm then the tail, six times, as Rust runs the same steps"
    );
}
