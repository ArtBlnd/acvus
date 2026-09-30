//! Where a body's blocks begin and end, and what a region holds (RFC-0052
//! rules 1 and 3).
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
    BlockListing, PartListing, RegionListing, family_of, ops_of_anywhere, regions_named,
    script_listing,
};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// A `while` whose body holds an `if` and nothing after it that reads an arm.
/// The loops here test with `!=` because RFC-0081 and RFC-0094 turn `n < 6`
/// and `n <= 5` into a range `for`, which has no head part. The arm adds floats, which cannot trap, so
/// running it on both paths is a `Select`.
const TAIL_ABOVE_THE_BRANCH: &str = "let acc = 0.0; let n = 0; \
     while n != 6 { if n % 2 == 0 { acc = acc + 1.5; }; n = n + 1; } acc";

/// The same shape with the program's integer `+` in the arm, which traps
/// where it overflows: the select runs it as its overflowing form and traps
/// only on the path the program takes (RFC-0074 rule 2).
const TRAPPING_ARM: &str =
    "let acc = 0; let n = 0; while n != 6 { if n % 2 == 0 { acc = acc + n; }; n = n + 1; } acc";

/// The same loop with a tail that reads what both arms wrote.
const TAIL_BELOW_THE_JOIN: &str = "let acc = 0; let n = 0; while n != 6 { \
     if n % 2 == 0 { acc = acc + n; } else { acc = acc + 1; }; acc = acc * 2; n = n + 1; } acc";

fn ends(blocks: &[BlockListing]) -> Vec<&str> {
    blocks.iter().map(|b| b.end.as_str()).collect()
}

fn part_of<'r>(region: &'r RegionListing, part: &str) -> &'r PartListing {
    region
        .part(part)
        .unwrap_or_else(|| panic!("the region holds no {part} part"))
}

fn loop_of(source: &str, ty: Ty) -> Vec<BlockListing> {
    let interner = Interner::new();
    let blocks = script_listing(&interner, source, Context::default(), ty);
    assert_eq!(
        ends(&blocks),
        ["Goto", "Return<true, false>"],
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
    let blocks = loop_of(TAIL_ABOVE_THE_BRANCH, Ty::Float);
    assert_eq!(
        part_of(one_loop(&blocks), "head").ops,
        ["Neq<i64, Slot, Slot, R0>"],
        "the head is the condition alone: no terminator, because the head's \
         `JumpIf` is the `cond` the Loop reads itself — and the head's last \
         operation writes it to the argument register, which `Yield` hands \
         the `Loop` (RFC-0052 rule 3)"
    );
}

#[tokio::test]
async fn a_tail_that_reads_neither_arm_sits_before_the_select() {
    let blocks = loop_of(TAIL_ABOVE_THE_BRANCH, Ty::Float);
    let body = part_of(one_loop(&blocks), "body");
    assert_eq!(
        body.ops,
        [
            "Chain2<i64, Slot, 0, 0>",
            "Add<i64, Slot, Slot, Slot>",
            "Select<f64, Slot, Slot, 1, true>"
        ],
        "`n = n + 1` reads neither arm, so code_motion put it above the branch, \
         and the branch follows in the same list"
    );
    assert!(
        body.regions.is_empty(),
        "one arm is `acc + 1.5` and the other passes `acc` through, so the branch \
         is a `Select` holding a chain plan and not a region holding two arms"
    );
}

#[tokio::test]
async fn an_arm_that_can_trap_is_a_select() {
    let blocks = loop_of(TRAPPING_ARM, Ty::I64);
    let body = part_of(one_loop(&blocks), "body");
    assert_eq!(
        body.ops,
        [
            "Chain2<i64, Slot, 0, 0>",
            "Add<i64, Slot, Slot, Slot>",
            "Select<i64, Slot, Slot, 1, true>",
            "Mov<false, true>"
        ],
        "`acc + n` traps where it overflows, and the `Select` defers that trap \
         to the side it takes, so the branch is one operation"
    );
}

#[tokio::test]
async fn a_tail_that_reads_an_arm_sits_after_the_diamond() {
    let blocks = loop_of(TAIL_BELOW_THE_JOIN, Ty::I64);
    let body = part_of(one_loop(&blocks), "body");
    assert_eq!(
        body.ops,
        [
            "Chain2<i64, Slot, 0, 0>",
            "Add<i64, Slot, Slot, Slot>",
            "Diamond<Slot, Rejoins>",
            "Mul<i64, Slot, Slot, Slot>",
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

#[tokio::test]
async fn a_while_that_returns_is_a_region_and_the_return_travels_out_of_it() {
    let interner = Interner::new();
    let source = r#"let i = 0; let acc = 0;
                    while i < 4 { let step = if i == 3 { None } else { Some(i) };
                                  acc = acc + step?; i = i + 1; }
                    Some(acc)"#;
    let blocks = script_listing(
        &interner,
        source,
        Context::default(),
        Ty::Option(Box::new(Ty::I64)),
    );
    let names = ops_of_anywhere(&blocks);
    assert!(
        names.iter().any(|name| name == "Loop<R0, Escapes>"),
        "the `while` is one region reading its body's verdict: {names:?}"
    );
    assert!(
        names.iter().any(|name| family_of(name) == "Escape"),
        "the `?` is a branch inside the body and not a block of its own: {names:?}"
    );

    let value = run_script_mode(
        &interner,
        source,
        Context::default(),
        Ty::Option(Box::new(Ty::I64)),
    )
    .await;
    assert!(
        value.is_none(),
        "the `?` returns None out of the loop at i == 3: {value:?}"
    );
}

#[tokio::test]
async fn the_blocks_that_splitting_states_run_to_their_values() {
    let interner = Interner::new();
    let above = run_script_mode(
        &interner,
        TAIL_ABOVE_THE_BRANCH,
        Context::default(),
        Ty::Float,
    )
    .await;
    assert_eq!(
        above.as_float(),
        1.5 * 3.0,
        "the three even n below 6, each adding by the arm inside the loop's body"
    );
    let trapping = run_script_mode(&interner, TRAPPING_ARM, Context::default(), Ty::I64).await;
    assert_eq!(
        trapping.as_int(),
        0 + 2 + 4,
        "the even n below 6, summed by the arm inside the loop's body"
    );

    let mut acc = 0i64;
    for n in 0..6 {
        acc += if n % 2 == 0 { n } else { 1 };
        acc *= 2;
    }
    let below = run_script_mode(&interner, TAIL_BELOW_THE_JOIN, Context::default(), Ty::I64).await;
    assert_eq!(
        below.as_int(),
        acc,
        "one arm then the tail, six times, as Rust runs the same steps"
    );
}
