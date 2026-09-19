//! A `for` is a region (RFC-0057 Decision 3): the value each head produces,
//! and what the machine runs it as.

use acvus_interpreter::listing::{BlockListing, RegionListing, ops_of_anywhere, regions_named};
use acvus_interpreter_test::listing::script_listing;
use acvus_interpreter_test::*;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;

async fn int(source: &str) -> i64 {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default(), Ty::I64)
        .await
        .as_int()
}

async fn uint(source: &str) -> u64 {
    let i = Interner::new();
    run_script_mode(&i, source, Context::default(), Ty::Int(IntTy::U64))
        .await
        .bits()
}

fn i64_blocks(source: &str) -> Vec<BlockListing> {
    let i = Interner::new();
    script_listing(&i, source, Context::default(), Ty::I64)
}

fn regions(source: &str) -> Vec<String> {
    i64_blocks(source)
        .iter()
        .flat_map(|block| block.regions.iter())
        .map(|region| region.name.clone())
        .collect()
}

fn one_region<'b>(source: &str, found: &'b [BlockListing], named: &str) -> &'b RegionListing {
    let matching = regions_named(found, named);
    let [region] = matching.as_slice() else {
        panic!(
            "{source} prepares to one `{named}` region, and its operations are {:?}",
            ops_of_anywhere(found)
        )
    };
    region
}

fn part_ops(source: &str, named: &str, part: &str) -> Vec<String> {
    let found = i64_blocks(source);
    let region = one_region(source, &found, named);
    let Some(held) = region.part(part) else {
        panic!(
            "a `{named}` owns a `{part}` chain, and {source} holds parts {:?}",
            region
                .owns
                .iter()
                .map(|p| p.part.as_str())
                .collect::<Vec<_>>()
        )
    };
    held.ops.clone()
}

struct Head {
    source: &'static str,
    region: &'static str,
}

const SLICE: &str = "let v = vec([1, 2, 3]); let acc = 0; for x in &v { acc = acc + *x; } acc";
const SLICE_MUT: &str = "let v = vec([1, 2, 3]); for x in &mut v { *x = 7; } v[0] + v[1] + v[2]";
const ARRAY: &str = "let a = [1, 2, 3]; let acc = 0; for x in a { acc = acc + x; } acc";
const ARRAY_OF_OWNERS: &str = "let a = [\"ab\".to_string(), \"cde\".to_string()]; let acc = 0; \
                               for s in a { let n = len(&s) as i64; acc = acc + n; } acc";
const RANGE_U64: &str = "let n = 4u64; let acc = 0u64; for i in 0u64..n { acc = acc + i; } acc";
const RANGE_I64: &str = "let n = 4; let acc = 0; for i in 0..n { acc = acc + i; } acc";
const WHILE: &str = "let i = 0; let acc = 0; while i < 3 { acc = acc + i; i = i + 1; } acc";

// -- The four heads produce their values -------------------------------

#[tokio::test]
async fn a_shared_head_sums_the_elements() {
    assert_eq!(int(SLICE).await, 6);
}

#[tokio::test]
async fn a_mutable_head_writes_through_to_the_container() {
    assert_eq!(int(SLICE_MUT).await, 21);
}

#[tokio::test]
async fn an_array_by_value_hands_each_element_out() {
    assert_eq!(int(ARRAY).await, 6);
}

#[tokio::test]
async fn an_array_of_owners_hands_each_owner_out_and_the_exit_releases_the_rest() {
    assert_eq!(int(ARRAY_OF_OWNERS).await, 5);
}

#[tokio::test]
async fn a_range_sums_its_own_counter() {
    assert_eq!(uint(RANGE_U64).await, 6);
    assert_eq!(int(RANGE_I64).await, 6);
}

// -- A source with nothing in it ---------------------------------------

#[tokio::test]
async fn a_range_of_equal_bounds_runs_no_iteration() {
    assert_eq!(
        uint("let acc = 7u64; for i in 3u64..3u64 { acc = 0u64; } acc").await,
        7
    );
}

#[tokio::test]
async fn a_range_whose_low_bound_is_above_its_high_one_runs_no_iteration() {
    assert_eq!(int("let acc = 7; for i in 5..2 { acc = 0; } acc").await, 7);
}

#[tokio::test]
async fn a_range_below_zero_counts_up_through_it() {
    assert_eq!(
        int("let acc = 0; for i in -3..3 { acc = acc + i; } acc").await,
        -3
    );
}

// -- Loops inside loops ------------------------------------------------

#[tokio::test]
async fn a_for_inside_a_for_runs_the_product() {
    assert_eq!(
        uint(
            "let n = 3u64; let acc = 0u64; \
             for i in 0u64..n { for j in 0u64..n { acc = acc + 1u64; } } acc"
        )
        .await,
        9
    );
}

#[tokio::test]
async fn a_for_inside_a_while_runs_once_per_turn_of_the_while() {
    assert_eq!(
        uint(
            "let n = 3u64; let k = 0u64; let acc = 0u64; \
             while k < n { for j in 0u64..n { acc = acc + 1u64; } k = k + 1u64; } acc"
        )
        .await,
        9
    );
}

#[tokio::test]
async fn a_while_inside_a_for_runs_once_per_element() {
    assert_eq!(
        uint(
            "let n = 3u64; let acc = 0u64; \
             for i in 0u64..n { let k = 0u64; while k < 2u64 { acc = acc + 1u64; k = k + 1u64; } } \
             acc"
        )
        .await,
        6
    );
}

// -- What the machine runs it as ---------------------------------------

#[test]
fn every_head_prepares_to_one_for_region() {
    for head in [
        Head {
            source: SLICE,
            region: "For<Slice>",
        },
        Head {
            source: SLICE_MUT,
            region: "For<Slice>",
        },
        Head {
            source: ARRAY,
            region: "For<Array<false, true>>",
        },
        Head {
            source: ARRAY_OF_OWNERS,
            region: "For<Array<true, false>>",
        },
    ] {
        assert_eq!(regions(head.source), [head.region], "{}", head.source);
    }
}

#[test]
fn the_body_holds_no_comparison_and_no_move_of_the_counter() {
    for source in [SLICE, ARRAY, RANGE_I64] {
        let ops = part_ops(source, "For", "body");
        assert!(
            !ops.iter()
                .any(|op| op.starts_with("Lt<") || op.starts_with("Mov")),
            "{source}: {ops:?}"
        );
    }
}

#[test]
fn a_for_owns_its_body_alone_where_a_while_owns_a_head_too() {
    let found = i64_blocks(SLICE);
    let traversal = one_region(SLICE, &found, "For");
    let parts: Vec<&str> = traversal.owns.iter().map(|p| p.part.as_str()).collect();
    assert_eq!(parts, ["body"], "{:?}", ops_of_anywhere(&found));

    let found = i64_blocks(WHILE);
    let condition = one_region(WHILE, &found, "Loop");
    let parts: Vec<&str> = condition.owns.iter().map(|p| p.part.as_str()).collect();
    assert_eq!(parts, ["head", "body"], "{:?}", ops_of_anywhere(&found));
}

// -- Where this run stopped --------------------------------------------

/// The two tests below are the executable statement of an unfinished piece of
/// this run, not of a decision. `break` and `continue` reach the joints path
/// RFC-0057 Decision 4 names, and that path is not built, so `recognize_for`
/// refuses both shapes and the block emitter then meets a `For` it has no
/// operation for. Each fails, and must be deleted, when the joints path lands.
const BREAK: &str = "let v = vec([1, 2, 3]); let acc = 0; \
                     for x in &v { if *x == 2 { break; }; acc = acc + *x; } acc";
const CONTINUE: &str = "let v = vec([1, 2, 3]); let acc = 0; \
                        for x in &v { if *x == 2 { continue; }; acc = acc + *x; } acc";

#[test]
#[should_panic(expected = "the joints path RFC-0057 Decision 4 names does not run yet")]
fn a_for_a_break_leaves_is_refused() {
    i64_blocks(BREAK);
}

#[test]
#[should_panic(expected = "the joints path RFC-0057 Decision 4 names does not run yet")]
fn a_for_a_continue_returns_to_the_head_of_is_refused() {
    i64_blocks(CONTINUE);
}
