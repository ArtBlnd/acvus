//! A `for` is a region, and a `break` or a `continue` in its body is an
//! operation of that region (RFC-0057 rules 3 and 4):
//! the value each head produces, and what the machine runs each as. The one
//! traversal still on the joints path is the one whose exit edge carries a
//! drop, which puts a block between the terminator and the body.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use acvus_extern::{ExternType, Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::listing::{BlockListing, RegionListing, ops_of_anywhere, regions_named};
use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::listing::{script_listing, script_listing_with_externs};
use acvus_interpreter_test::*;
use acvus_mir::graph::optimize::Opt;
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
/// Tested with `<=` so that it stays a `while`: RFC-0081 turns `i < 3` into
/// a range `for`.
const WHILE: &str = "let i = 0; let acc = 0; while i <= 2 { acc = acc + i; i = i + 1; } acc";

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
            region: "For<Slice, Rejoins>",
        },
        Head {
            source: SLICE_MUT,
            region: "For<Slice, Rejoins>",
        },
        Head {
            source: ARRAY,
            region: "For<Array<false, true>, Rejoins>",
        },
        Head {
            source: ARRAY_OF_OWNERS,
            region: "For<Array<true, false>, Rejoins>",
        },
    ] {
        assert_eq!(regions(head.source), [head.region], "{}", head.source);
    }
}

/// RFC-0089 rule 1: in place, a stage chain is the body of the region the
/// loop as written would be, so each loop `optimize::stages` writes as
/// several stages prepares to what the same loop prepared to before stages.
/// The `anyorder` case's body spawns its call and evaluates it, which no
/// region part admits, so it ran on the joints path before stages as well.
#[test]
fn every_staged_loop_prepares_as_the_loop_as_written_did() {
    struct Partitioned {
        source: &'static str,
        prepares_to: Expected,
    }
    enum Expected {
        OneRegion(&'static str),
        Joints,
    }
    let cases = [
        Partitioned {
            source: "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
                     for x in &v { s = s + *x; p = p * *x; } s + p",
            prepares_to: Expected::OneRegion("For<Slice, Rejoins>"),
        },
        Partitioned {
            source: "let a = 0; let b = 0; let c = 0; \
                     for i in 0..10 { a = a * a + i; b = b * b + 1; c = c * c + 2; } a + b + c",
            prepares_to: Expected::OneRegion("For<Range<i64>, Rejoins>"),
        },
        Partitioned {
            source: "let a = 1; let s = 0; for i in 0..5 { a = a * a + i; s = s + i; } a + s",
            prepares_to: Expected::OneRegion("For<Range<i64>, Rejoins>"),
        },
        Partitioned {
            source: "let v = [1, 2, 3, 4]; let s = 0; let t = 0; \
                     for x in &v { let y = *x * *x + 3; s = s + y; t = t + y; } s + t",
            prepares_to: Expected::OneRegion("For<Slice, Rejoins>"),
        },
        Partitioned {
            source: "let v = [1, 2, 3]; anyorder { for x in &v { io::print(\"a\"); } } 0",
            prepares_to: Expected::Joints,
        },
        Partitioned {
            source: "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
                     for x in &v { if *x > 2 { s = s + *x; } p = p * *x; } s + p",
            prepares_to: Expected::OneRegion("For<Slice, Rejoins>"),
        },
        Partitioned {
            source: "let s = 0; let p = 1; \
                     for i in 0..10 { p = p * 2; if i % 2 == 0 { continue; } s = s + i; } s + p",
            prepares_to: Expected::OneRegion("For<Range<i64>, Escapes>"),
        },
    ];
    for case in cases {
        let i = Interner::new();
        let registries = || {
            let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
            registries.push(acvus_ext::io_registry::<AcvusRuntime>());
            registries
        };
        let ast = acvus_mir::graph::ParsedAst::Script(
            acvus_ast::parse_script(&i, case.source).expect("parse error"),
        );
        let compiled = compile_source_with_externs(
            &i,
            ast,
            &rustc_hash::FxHashMap::default(),
            registries(),
            Ty::I64,
        );
        let mir = acvus_mir::printer::dump_with(&i, &compiled.modules[&compiled.entry_qref]);
        assert!(mir.contains(" stages ["), "{}\n{mir}", case.source);
        let found =
            script_listing_with_externs(&i, case.source, Context::default(), registries(), Ty::I64);
        let names: Vec<String> = found
            .iter()
            .flat_map(|block| block.regions.iter())
            .map(|region| region.name.clone())
            .collect();
        let joints: Vec<String> = ops_of_anywhere(&found)
            .into_iter()
            .chain(found.iter().map(|block| block.end.clone()))
            .filter(|op| {
                ["ForStart<", "ForAt<", "ForStep<"]
                    .iter()
                    .any(|joint| op.starts_with(joint))
            })
            .collect();
        match case.prepares_to {
            Expected::OneRegion(region) => {
                assert_eq!(names, [region], "{}\n{mir}", case.source);
                assert!(joints.is_empty(), "{}: {joints:?}\n{mir}", case.source);
            }
            Expected::Joints => {
                assert!(names.is_empty(), "{}: {names:?}\n{mir}", case.source);
                assert_eq!(joints.len(), 3, "{}: {joints:?}\n{mir}", case.source);
            }
        }
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

/// The `else if` chain whose arms assign variables. The references at the
/// tail are what keeps `a` and `b` variables rather than block parameters,
/// and that is the whole difference: with block parameters the inner join
/// carries them and stays a block of its own, while without them
/// `optimize::forward` collapses that join into the outer one, so the inner
/// branch names the same join as the branch above it. Every example's loop
/// body has the second shape.
const IF_CHAIN: &str = "let a = 0; let b = 0; for i in 0..6 { \
                        if i % 3 == 0 { a = a + 1; } \
                        else if i % 3 == 1 { b = b + 1; } \
                        else { a = a + 2; }; \
                        } let ra = &a; let rb = &b; *ra + *rb";

#[tokio::test]
async fn an_else_if_chain_counts_every_band() {
    assert_eq!(int(IF_CHAIN).await, 8);
}

#[test]
fn an_else_if_chain_that_joins_where_the_if_joins_leaves_the_for_one_region() {
    let found = i64_blocks(IF_CHAIN);
    assert_eq!(
        regions(IF_CHAIN),
        ["For<Range<i64>, Rejoins>"],
        "{:?}",
        ops_of_anywhere(&found)
    );
    let body = part_ops(IF_CHAIN, "For", "body");
    assert!(
        !body.iter().any(|op| op.starts_with("Goto")),
        "a region's body holds no terminator, so the chain a band takes is \
         not a block the machine dispatches to: {body:?}"
    );
}

/// A `match` whose arms all rejoin. RFC-0051's terminator names the arms and
/// not the block they meet at, so the recognizer reads that block off the
/// first arm the lowering laid and requires it of every other.
const MATCH_BODY: &str = "let a = 0; for i in 0..6 { \
                          let o = if i % 2 == 0 { Some(i) } else { None }; \
                          match o { Some(v) => { a = a + v; }, None => { a = a + 1; } }; \
                          } let ra = &a; *ra";

#[tokio::test]
async fn a_match_in_a_for_body_reaches_both_sides() {
    assert_eq!(int(MATCH_BODY).await, 9);
}

#[test]
fn a_match_whose_arms_all_rejoin_leaves_the_for_one_region() {
    let found = i64_blocks(MATCH_BODY);
    assert_eq!(
        regions(MATCH_BODY),
        ["For<Range<i64>, Rejoins>"],
        "{:?}",
        ops_of_anywhere(&found)
    );
    let body = part_ops(MATCH_BODY, "For", "body");
    assert!(
        body.iter().any(|op| op.starts_with("SwitchOptionRegion")),
        "the dispatch is an operation of the body's chain, not a block the \
         machine enters: {body:?}"
    );
}

// -- break and continue ------------------------------------------------

const BREAK: &str = "let v = vec([1, 2, 3]); let acc = 0; \
                     for x in &v { if *x == 2 { break; }; acc = acc + *x; } acc";
const CONTINUE: &str = "let v = vec([1, 2, 3]); let acc = 0; \
                        for x in &v { if *x == 2 { continue; }; acc = acc + *x; } acc";
const RANGE_BREAK: &str = "let s = 0; for i in 0..10 { if i == 5 { break; }; s = s + i; } s";
/// The same `break` written in a `match` arm, which `recognize_switch`
/// refuses, so this traversal is the joints path over a `Range` head.
const RANGE_MATCH_BREAK: &str =
    "let s = 0; for i in 0..10 { match i { 5 => { break; }, _ => { s = s + i; } }; } s";

#[tokio::test]
async fn a_break_stops_the_traversal_where_it_stands() {
    assert_eq!(int(BREAK).await, 1);
}

#[tokio::test]
async fn a_continue_skips_the_rest_of_the_body_and_advances() {
    assert_eq!(int(CONTINUE).await, 4);
}

#[tokio::test]
async fn a_break_on_the_first_iteration_leaves_the_carried_value_alone() {
    assert_eq!(
        int("let acc = 7; for i in 0..5 { if i == 0 { break; }; acc = acc + 1; } acc").await,
        7
    );
}

#[tokio::test]
async fn a_continue_over_a_range_skips_the_odd_counters() {
    assert_eq!(
        int("let acc = 0; for i in 0..6 { if i % 2 == 1 { continue; }; acc = acc + i; } acc").await,
        6
    );
}

#[tokio::test]
async fn a_range_head_takes_a_break() {
    assert_eq!(int(RANGE_BREAK).await, 10);
}

#[tokio::test]
async fn an_array_of_words_takes_a_break() {
    assert_eq!(
        int("let a = [1, 2, 3]; let acc = 0; \
             for x in a { if x == 2 { break; }; acc = acc + x; } acc")
        .await,
        1
    );
}

#[tokio::test]
async fn a_mutable_head_takes_a_continue() {
    assert_eq!(
        int(
            "let v = vec([1, 2, 3]); for x in &mut v { if *x == 2 { continue; }; *x = 9; } \
             v[0] + v[1] + v[2]"
        )
        .await,
        20
    );
}

#[tokio::test]
async fn a_break_inside_a_nested_for_leaves_the_inner_loop_alone() {
    assert_eq!(
        uint(
            "let acc = 0u64; for i in 0u64..3u64 { \
             for j in 0u64..3u64 { if j == 1u64 { break; }; acc = acc + 1u64; } } acc"
        )
        .await,
        3
    );
}

#[tokio::test]
async fn a_break_in_a_for_inside_a_while_runs_once_per_turn_of_the_while() {
    assert_eq!(
        uint(
            "let k = 0u64; let acc = 0u64; while k < 3u64 { \
             for j in 0u64..5u64 { if j == 2u64 { break; }; acc = acc + 1u64; } k = k + 1u64; } acc"
        )
        .await,
        6
    );
}

/// The latch carries the counter as a jump argument, so the move that reads
/// it stands ahead of the step that advances it. With the two the other way
/// round every assignment writes the next counter and this answers 3.
#[tokio::test]
async fn a_latch_that_carries_the_counter_reads_it_before_the_step() {
    assert_eq!(
        int("let last = 0; for i in 0..5 { if i == 3 { break; }; last = i; } last").await,
        2
    );
}

/// The counter's register is the loop's for the whole loop. `ForAt` writes it
/// on every iteration, so a value live across the loop that shared it would
/// come out holding a counter, and `acc` here answers 4 instead of 100.
#[tokio::test]
async fn a_value_live_across_the_loop_does_not_share_the_counter_register() {
    assert_eq!(
        int("let v = vec([1, 2, 3]); let acc = 100; for x in &v { if *x == 2 { break; }; } acc")
            .await,
        100
    );
}

// -- What the machine runs a traversal it cannot collapse as -----------

#[test]
fn the_preheader_lays_the_counter_the_header_tests_and_the_latch_steps() {
    let blocks = i64_blocks(RANGE_MATCH_BREAK);
    let preheader = blocks.first().expect("an entry block");
    assert_eq!(
        preheader.ops.last().map(String::as_str),
        Some("ForStart<Range<i64>>"),
        "{:?}",
        preheader.ops
    );

    let header = blocks
        .iter()
        .find(|block| block.end.starts_with("ForAt<"))
        .unwrap_or_else(|| panic!("a header: {:?}", ops_of_anywhere(&blocks)));
    assert_eq!(header.ops, Vec::<String>::new(), "{:?}", header.ops);

    let latch = blocks
        .iter()
        .find(|block| block.ops.iter().any(|op| op.starts_with("ForStep<")))
        .unwrap_or_else(|| panic!("a latch: {:?}", ops_of_anywhere(&blocks)));
    assert_eq!(
        latch.ops.last().map(String::as_str),
        Some("ForStep<Range<i64>>"),
        "the step is the last operation of the latch: {:?}",
        latch.ops
    );
}

#[test]
fn a_loop_a_continue_returns_to_the_head_of_is_one_region() {
    for source in [CONTINUE, RANGE_BREAK] {
        let ops = ops_of_anywhere(&i64_blocks(source));
        assert!(
            ops.iter().any(|op| op.starts_with("For<")),
            "{source} is one region: {ops:?}"
        );
        assert!(
            !ops.iter().any(|op| op.starts_with("ForAt<")),
            "{source} holds no joint header: {ops:?}"
        );
    }
}

/// The exit edge of this traversal carries the vec's drop, and `lower` lays
/// that drop's block between the terminator and the body, where the body's
/// label has to be for `recognize_for` to match. The `break` then jumps past
/// the block rather than to it, so the region's one exit word could not name
/// where the arm goes.
#[test]
fn a_traversal_whose_exit_edge_carries_a_drop_keeps_its_break_on_the_joints_path() {
    let blocks = i64_blocks(BREAK);
    assert!(
        blocks.iter().any(|block| block.end.starts_with("ForAt<")),
        "{BREAK}: {:?}",
        ops_of_anywhere(&blocks)
    );
}

/// A `match` arm that leaves a loop is the one escape `prepare` still refuses:
/// `recognize_switch` builds no escaping dispatch form, so the loop is joints
/// and every edge into its header carries a start or a step.
#[test]
fn every_edge_into_a_header_carries_a_start_or_a_step() {
    let blocks = i64_blocks(
        "let acc = 0; \
         for i in 0..4 { match i { 2 => { continue; }, _ => { acc = acc + i; } }; } acc",
    );
    let ops = ops_of_anywhere(&blocks);
    assert_eq!(
        ops.iter().filter(|op| op.starts_with("ForStart<")).count(),
        1,
        "one preheader: {ops:?}"
    );
    assert_eq!(
        ops.iter().filter(|op| op.starts_with("ForStep<")).count(),
        2,
        "the continue and the latch each step: {ops:?}"
    );
}

// -- What a `break` releases -------------------------------------------

static RELEASES: AtomicUsize = AtomicUsize::new(0);

/// One counter, and the harness runs these tests on parallel threads.
static TRACKED_SCRIPTS: Mutex<()> = Mutex::new(());

struct Measured {
    at_start: usize,
    _scripts: MutexGuard<'static, ()>,
}

impl Measured {
    fn start() -> Self {
        let scripts = TRACKED_SCRIPTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Self {
            at_start: RELEASES.load(Ordering::SeqCst),
            _scripts: scripts,
        }
    }

    fn count(&self) -> usize {
        RELEASES.load(Ordering::SeqCst) - self.at_start
    }
}

struct Counted;

impl Drop for Counted {
    fn drop(&mut self) {
        RELEASES.fetch_add(1, Ordering::SeqCst);
    }
}

/// An extension type over a `Vec`, so the value crosses as a `Large` and its
/// release is what the counter sees. It carries no identity variable, so two
/// calls to `tracked` are values of one source and an array of them
/// typechecks.
#[derive(ExternType)]
#[repr(transparent)]
struct Tracked(Vec<Counted>);

#[extern_fn(effect = pure)]
fn tracked(n: i64) -> Tracked {
    Tracked((0..n).map(|_| Counted).collect())
}

#[extern_fn(effect = pure)]
fn rank(t: &Tracked) -> i64 {
    t.0.len() as i64
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        types: [Tracked],
        fns: [tracked, rank],
    });
    regs
}

async fn tracked_int(source: &str) -> i64 {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), regs(), Ty::I64)
        .await
        .value
        .as_int()
}

const OWNERS_IN_A_VEC: &str = "let v = vec([tracked(1), tracked(1), tracked(1)]); let seen = 0; \
                               for t in &v { if seen == 1 { break; }; seen = seen + rank(t); } seen";

const OWNERS_IN_AN_ARRAY: &str = "let a = [tracked(1), tracked(1), tracked(1)]; let seen = 0; \
     for t in a { if seen == 1 { continue; }; seen = seen + rank(&t); } seen";

/// Three elements released for two the loop reached: a traversal of borrowed
/// owners never takes an element, so the release is the container's, and the
/// lowering puts it on the `break` edge as it puts it on the terminator's.
#[tokio::test]
async fn a_break_out_of_a_vec_of_owners_releases_every_element_once() {
    let measured = Measured::start();
    assert_eq!(tracked_int(OWNERS_IN_A_VEC).await, 1);
    assert_eq!(measured.count(), 3);
}

/// An array by value hands each element out as the loop reaches it and the
/// exit releases the rest, a `continue` included.
#[tokio::test]
async fn a_continue_in_an_array_of_owners_releases_every_element_once() {
    let measured = Measured::start();
    assert_eq!(tracked_int(OWNERS_IN_AN_ARRAY).await, 1);
    assert_eq!(measured.count(), 3);
}

/// The machine emits no release of its own for a container a `break` leaves:
/// the two `DropValue`s are the lowering's, one on the terminator's exit edge
/// and one on the `break` edge.
#[test]
fn a_traversal_of_owners_a_break_leaves_runs_as_joints_and_releases_nothing_itself() {
    let i = Interner::new();
    let blocks =
        script_listing_with_externs(&i, OWNERS_IN_A_VEC, Context::default(), regs(), Ty::I64);
    let ops = ops_of_anywhere(&blocks);
    assert!(
        blocks.iter().any(|block| block.end.starts_with("ForAt<")),
        "{ops:?}"
    );
    assert_eq!(
        ops.iter().filter(|op| *op == "DropValue").count(),
        2,
        "{ops:?}"
    );
}

/// An owned element's drop is placed in the stage that reads it
/// (`drop_insertion::stage_entries`). The element is bound to a storage one
/// stage alone touches, so its release is that stage's, the product's join
/// holds none, and each element is released once.
const OWNER_READ_BY_ONE_PART: &str = "let a = [tracked(1), tracked(1), tracked(1)]; \
     let n = 0; let m = 1; for t in a { m = m * 2; n = n + rank(&t); } n * 100 + m";

/// Two sums that each read the element: the element is bound to a storage
/// the body defines and releases within an iteration, so it is no target, and
/// the reads of it are pure work ahead of the two sums' joins, one per sum.
const OWNER_READ_BY_TWO_SUMS: &str = "let a = [tracked(1), tracked(1), tracked(1)]; \
     let n = 0; let m = 0; for t in a { n = n + rank(&t); m = m + rank(&t) * 2; } n * 10 + m";

#[tokio::test]
async fn an_owned_element_one_part_reads_is_released_once_in_that_part() {
    let measured = Measured::start();
    assert_eq!(tracked_int(OWNER_READ_BY_ONE_PART).await, 308);
    assert_eq!(measured.count(), 3);

    let i = Interner::new();
    let ast = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script(&i, OWNER_READ_BY_ONE_PART).expect("parse error"),
    );
    let compiled =
        compile_source_with_externs(&i, ast, &rustc_hash::FxHashMap::default(), regs(), Ty::I64);
    let module = &compiled.modules[&compiled.entry_qref];
    let mir = acvus_mir::printer::dump_with(&i, module);
    let cfg = acvus_mir::cfg::promote(module.main.clone());
    let Some((header, stages)) = cfg.blocks.iter().find_map(|block| match &block.terminator {
        acvus_mir::cfg::Terminator::For { stages, .. } => Some((block.label, stages)),
        _ => None,
    }) else {
        panic!("the loop states its stages:\n{mir}")
    };
    let entries: Vec<usize> = stages
        .entries()
        .map(|entry| cfg.label_to_block[&entry].0)
        .collect();
    let reads_the_element: Vec<usize> = (0..entries.len())
        .filter(|&stage| {
            let end = entries.get(stage + 1).copied().unwrap_or(cfg.blocks.len());
            (entries[stage]..end).any(|at| {
                cfg.blocks[at].insts.iter().any(|inst| {
                    matches!(&inst.kind, acvus_mir::ir::InstKind::FunctionCall { .. })
                })
            })
        })
        .collect();
    let latch = (entries[0]..cfg.blocks.len())
        .find(|at| {
            matches!(&cfg.blocks[*at].terminator,
                acvus_mir::cfg::Terminator::Jump { label, .. } if *label == header)
        })
        .expect("the last stage jumps back to the header");
    let dropped_in: Vec<usize> = (entries[0]..=latch)
        .filter(|at| {
            cfg.blocks[*at]
                .insts
                .iter()
                .any(|inst| matches!(inst.kind, acvus_mir::ir::InstKind::Drop { .. }))
        })
        .map(|at| entries.iter().rposition(|entry| *entry <= at).unwrap_or(0))
        .collect();
    assert_eq!(dropped_in, reads_the_element, "{mir}");
}

#[tokio::test]
async fn an_owned_element_two_sums_read_is_released_once() {
    let measured = Measured::start();
    assert_eq!(tracked_int(OWNER_READ_BY_TWO_SUMS).await, 3 * 10 + 6);
    assert_eq!(measured.count(), 3);

    let i = Interner::new();
    let ast = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script(&i, OWNER_READ_BY_TWO_SUMS).expect("parse error"),
    );
    let compiled =
        compile_source_with_externs(&i, ast, &rustc_hash::FxHashMap::default(), regs(), Ty::I64);
    let module = &compiled.modules[&compiled.entry_qref];
    let mir = acvus_mir::printer::dump_with(&i, module);
    let cfg = acvus_mir::cfg::promote(module.main.clone());
    let joins: Vec<usize> = cfg
        .blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            acvus_mir::cfg::Terminator::For { stages, .. } => Some(
                stages
                    .iter()
                    .filter(|stage| matches!(stage, acvus_mir::ir::Stage::Join { .. }))
                    .count(),
            ),
            _ => None,
        })
        .collect();
    assert_eq!(joins, [2], "{mir}");
}

// -- An element's reference outlives its iteration ----------------------

const LIMIT: Duration = Duration::from_secs(30);

#[test]
fn corpus_child() {
    corpus::child();
}

fn outcome(source: &str, opt: Opt) -> Outcome {
    acvus_interpreter_test::attempt_within!(source, opt, Stage::Run, LIMIT)
        .unwrap_or_else(|lapse| panic!("at {opt:?}, {lapse:?}: {source}"))
}

fn runs_to(source: &str, value: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Value(got) => assert_eq!(got, value, "at {opt:?}: {source}"),
            other => panic!("at {opt:?}, expected {value}, got {other:?}: {source}"),
        }
    }
}

fn refused_with(source: &str, words: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Refused(why) => assert!(why.contains(words), "at {opt:?}: {why}"),
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {source}"),
        }
    }
}

/// The terminator hands each element out through the one `SliceMut` borrow
/// taken before the header (RFC-0057 rule 2), so a reference to an element
/// kept past its iteration holds that borrow's loan of `v` from where it is
/// kept, and not before.
#[test]
fn a_mutable_element_kept_past_the_loop_writes_through_to_the_container() {
    runs_to(
        "let v = vec([1, 2, 3]); let z = 0; let q = &mut z; \
         for x in &mut v { q = x; } *q = 9; v[2u64]",
        "9",
    );
}

#[test]
fn a_shared_element_kept_past_the_loop_reads_the_container() {
    runs_to(
        "let v = vec([1, 2, 3]); let z = 0; let q = &z; for x in &v { q = x; } *q",
        "3",
    );
}

#[test]
fn the_container_is_not_written_while_a_kept_element_is_still_used() {
    refused_with(
        "let v = vec([1, 2, 3]); let z = 0; let q = &mut z; \
         for x in &mut v { q = x; } push(&mut v, 4); *q = 9; v[2u64]",
        "`v` is written here while a reference to it is live",
    );
    refused_with(
        "let v = vec([1, 2, 3]); let z = 0; let q = &mut z; \
         for x in &mut v { q = x; } v = vec([7]); *q = 9; v[0u64]",
        "`v` is written here while a reference to it is live",
    );
}

/// RFC-0057 rule 5: the loop holds the container exclusively.
#[test]
fn the_container_is_not_written_inside_its_own_mutable_loop() {
    refused_with(
        "let v = vec([1, 2, 3]); for x in &mut v { push(&mut v, 4); } v[0u64]",
        "`v` is written here while a reference to it is live",
    );
    refused_with(
        "let v = vec([1, 2, 3]); for x in &mut v { v = vec([7]); } v[0u64]",
        "`v` is written here while a reference to it is live",
    );
}

/// Every element a `SliceMut` head hands out is a loan of the one container
/// (RFC-0057 rule 5), and no RFC tells two elements' loans apart: two kept
/// from two iterations are two live `&mut`s of `v`, and a write through
/// either is refused while the other is used after it (RFC-0018 rule 8).
#[test]
fn two_mutable_elements_kept_from_two_iterations_are_not_both_written() {
    refused_with(
        "let v = vec([1, 2, 3]); let z = 0; let w = 0; let a = &mut z; let b = &mut w; \
         for x in &mut v { b = a; a = x; } *a = 7; *b = 8; v[1u64] * 10 + v[2u64]",
        "is written here while a reference to it is live",
    );
}
