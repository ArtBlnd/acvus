//! What an exit flag costs the machine, and what `break` costs instead
//! (RFC-0057 Decision 4).
//!
//! A loop whose exit is a flag carries that flag from one iteration to the
//! next, and a carried value the machine keeps in a block argument is a `Mov`
//! per iteration. The two sources below are the same scan -- the flag idiom
//! `benches/logs.rs` case `inline` is written in, and the `break` its case
//! `inline break` is written in -- and what they measure is where the two
//! differ: this scan's flag lives in a slot rather than a block argument, so
//! the exit costs it no `Mov` either way, and what `break` does change is
//! that the loop is no longer a region.

use acvus_interpreter_test::listing::script_listing;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

const PRELUDE: &str = "\
let n = len(&@pat); \
let one = 1u64; \
let zero = 0u64; \
";

const WITH_A_FLAG: &str = "\
let i = zero; \
let hits = 0; \
let alive = true; \
while alive { \
if i < n { \
if @pat[i] == 42 { hits = hits + 1; }; \
i = i + one; \
} else { alive = false; }; \
} \
hits";

const WITH_BREAK: &str = "\
let i = zero; \
let hits = 0; \
while true { \
if i < n { \
if @pat[i] == 42 { hits = hits + 1; }; \
i = i + one; \
} else { break; }; \
} \
hits";

fn context(i: &Interner) -> Context {
    let pattern = serde_json::json!([1, 42, 3, 42]);
    std::iter::once((i.intern("pat"), value_from_json(i, &pattern))).collect::<Context>()
}

fn ops(tail: &str) -> Vec<String> {
    let i = Interner::new();
    let source = format!("{PRELUDE}{tail}");
    let blocks = script_listing(&i, &source, context(&i), Ty::I64);
    acvus_interpreter::listing::ops_of_anywhere(&blocks)
}

fn movs(tail: &str) -> usize {
    ops(tail).iter().filter(|op| op.starts_with("Mov")).count()
}

fn regions(tail: &str) -> Vec<String> {
    let i = Interner::new();
    let source = format!("{PRELUDE}{tail}");
    let blocks = script_listing(&i, &source, context(&i), Ty::I64);
    blocks
        .iter()
        .flat_map(|block| block.regions.iter())
        .map(|region| region.name.clone())
        .collect()
}

async fn value(tail: &str) -> acvus_interpreter::Value {
    let i = Interner::new();
    let source = format!("{PRELUDE}{tail}");
    run_script_mode(&i, &source, context(&i), Ty::I64).await
}

#[tokio::test]
async fn the_two_scans_count_the_same_hits() {
    assert_eq!(value(WITH_A_FLAG).await, value(WITH_BREAK).await);
}

#[test]
fn a_flag_loop_is_a_region_and_a_break_loop_is_joints() {
    assert_eq!(
        regions(WITH_A_FLAG),
        ["Loop<Slot>"],
        "{:?}",
        ops(WITH_A_FLAG)
    );
    assert_eq!(
        regions(WITH_BREAK),
        Vec::<String>::new(),
        "an arm that leaves the loop does not rejoin, so neither the loop nor \
         the branch inside it is a region (RFC-0057 Decision 4): {:?}",
        ops(WITH_BREAK)
    );
}

#[test]
fn neither_scan_carries_a_move_through_its_back_edge() {
    assert_eq!(
        (movs(WITH_A_FLAG), movs(WITH_BREAK)),
        (0, 0),
        "this scan carries its flag in a slot, not in a block argument, so the \
         exit costs it no `Mov`: {:?} and {:?}",
        ops(WITH_A_FLAG),
        ops(WITH_BREAK)
    );
}
