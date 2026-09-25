//! A loop a `break`, a `continue` or a `?` leaves is still one region
//! (RFC-0057 rule 4).
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

/// `!=` is no bound RFC-0081 or RFC-0094 reads, so the loop stays a
/// `while`.
#[test]
fn a_break_out_of_a_while_is_a_region() {
    let source = "let i = 0; let acc = 0; \
                  while i != 10 { i = i + 1; if i > 5 { break; }; acc = acc + i; } acc";
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

// -- What an early exit carries out --------------------------------------
//
// Each script leaves a loop by `break` or `?` with a value the body changed
// after the header carried it. These ran to the header's value while a
// loop's test edge was laid after the loop's operation, where a `break`
// reached it too and its moves overwrote what the `break` carried.

#[test]
fn a_break_out_of_a_for_in_a_while_carries_the_value_it_set() {
    answers(
        "let n = 4u64; let count = 0; let k = 0; \
         while k < 1 { k = k + 1; let pivot = n; \
             for v in 0u64..n { if v == 2u64 { pivot = v; break; }; } \
             if pivot < n { count = count + 1; }; } \
         count",
        "1",
    );
}

#[test]
fn a_break_out_of_a_top_level_for_carries_the_value_it_set() {
    answers(
        "let n = 4u64; let count = 0; let pivot = n; \
         for v in 0u64..n { if v == 2u64 { pivot = v; break; }; } \
         if pivot < n { count = count + 1; }; \
         count",
        "1",
    );
}

#[test]
fn a_break_out_of_a_while_in_a_while_carries_the_value_it_set() {
    answers(
        "let n = 4u64; let count = 0; let k = 0; \
         while k < 1 { k = k + 1; let pivot = n; let i = 0u64; \
             while i < n { if i == 2u64 { pivot = i; break; }; i = i + 1u64; } \
             if pivot < n { count = count + 1; }; } \
         count",
        "1",
    );
}

#[test]
fn a_break_out_of_a_for_in_a_for_in_a_while_carries_the_value_it_set() {
    answers(
        "let n = 4u64; let count = 0; let k = 0; \
         while k < 1 { k = k + 1; \
             for j in 0u64..2u64 { let pivot = n; \
                 for v in 0u64..n { if v == j + 1u64 { pivot = v; break; }; } \
                 if pivot < n { count = count + 1; }; } } \
         count",
        "2",
    );
}

#[test]
fn a_break_carries_every_value_it_set() {
    answers(
        "let n = 5u64; let k = 0; let total = 0u64; \
         while k < 2 { k = k + 1; let pivot = n; let seen = 0u64; \
             for v in 0u64..n { seen = seen + 1u64; if v == 3u64 { pivot = v; break; }; } \
             total = total + seen + pivot; } \
         total",
        "14",
    );
}

#[test]
fn a_break_carries_out_a_string_beside_the_value_it_set() {
    answers(
        "let n = 4u64; let out = \"\".to_string(); let k = 0; \
         while k < 2 { k = k + 1; let pivot = n; let tag = \"none\".to_string(); \
             for v in 0u64..n { if v == 2u64 { pivot = v; tag = \"two\".to_string(); break; }; } \
             if pivot < n { out = out + &tag; }; } \
         out",
        "\"twotwo\"",
    );
}

#[test]
fn a_break_carries_out_a_string_element_of_an_array() {
    answers(
        "let k = 0; let out = \"\".to_string(); \
         while k < 1 { k = k + 1; \
             let a = [\"a\".to_string(), \"b\".to_string(), \"c\".to_string()]; \
             let found = \"none\".to_string(); \
             for s in a { if s == \"b\" { found = s; break; }; } \
             out = found; } \
         out",
        "\"b\"",
    );
}

#[test]
fn a_break_over_a_slice_in_a_while_carries_the_element_it_read() {
    answers(
        "let v = [5u64, 6u64, 7u64, 8u64]; let n = 100u64; let count = 0; let k = 0; \
         while k < 1 { k = k + 1; let found = n; \
             for x in &v { if *x == 7u64 { found = *x; break; }; } \
             if found < n { count = count + 1; }; } \
         count",
        "1",
    );
}

#[test]
fn a_break_over_an_array_in_a_while_carries_the_element_it_took() {
    answers(
        "let n = 100u64; let count = 0; let k = 0; \
         while k < 1 { k = k + 1; let a = [5u64, 6u64, 7u64, 8u64]; let found = n; \
             for x in a { if x == 7u64 { found = x; break; }; } \
             if found < n { count = count + 1; }; } \
         count",
        "1",
    );
}

#[test]
fn a_break_out_of_a_while_let_call_carries_the_value_it_set() {
    // The iterator is declared above both loops. One declared inside the
    // outer loop is dropped on the inner loop's exit edge, and `prepare` then
    // runs the inner loop as blocks instead of the `For<Call<…>>` region this
    // test is about.
    answers(
        "let v = range(0, 10) | collect; let count = 0; let k = 0; let n = 99; \
         let it = as_iter(&v); \
         while k < 2 { k = k + 1; let pivot = n; \
             while let Some(x) = next(&mut it) { if *x % 4 == 3 { pivot = *x; break; }; } \
             if pivot < n { count = count + 1; }; } \
         count",
        "2",
    );
}

#[test]
fn a_value_a_nested_break_set_is_read_after_the_outer_loop() {
    answers(
        "let n = 4u64; let k = 0; let last = n; \
         while k < 3 { k = k + 1; let pivot = n; \
             for v in 0u64..n { if v == 2u64 { pivot = v; break; }; } \
             last = pivot; } \
         last",
        "2",
    );
}

#[test]
fn a_try_that_passes_in_a_nested_for_carries_the_value_it_set() {
    answers(
        "let n = 4u64; let count = 0; let k = 0; \
         while k < 1 { k = k + 1; let pivot = n; \
             for v in 0u64..n { let s = if v == 9u64 { None } else { Some(v) }; \
                 if s? == 2u64 { pivot = v; break; }; } \
             if pivot < n { count = count + 1; }; } \
         Some(count)",
        "1",
    );
}

#[test]
fn a_try_that_fails_in_a_nested_for_leaves_the_function() {
    answers(
        "let k = 0; let acc = 0; \
         while k < 2 { k = k + 1; let pivot = 9; \
             for v in 0..4 { let s = if v == 3 { None } else { Some(v) }; pivot = s?; } \
             acc = acc + pivot; } \
         Some(acc)",
        "\"None\"",
    );
}

#[test]
fn a_loops_own_exit_edge_is_a_part_of_its_region() {
    let found = blocks(
        "let n = 4u64; let count = 0; let k = 0; \
         while k < 1 { k = k + 1; let pivot = n; \
             for v in 0u64..n { if v == 2u64 { pivot = v; break; }; } \
             if pivot < n { count = count + 1; }; } \
         count",
        Ty::I64,
    );
    let loops: Vec<_> = regions_named(&found, "For")
        .into_iter()
        .chain(regions_named(&found, "Loop"))
        .collect();
    let escaping: Vec<_> = loops
        .iter()
        .filter(|region| region.name.ends_with("Escapes>"))
        .collect();
    let [inner] = escaping.as_slice() else {
        panic!("one escaping loop region: {:?}", ops_of_anywhere(&found))
    };
    let exit = inner.part("exit").expect("a loop region owns its exit edge");
    assert_eq!(exit.leaves_with, 1, "{:?}", exit.ops);
    let enclosing: Vec<_> = loops
        .iter()
        .filter_map(|region| region.part("body"))
        .filter(|body| body.ops.iter().any(|name| *name == inner.name))
        .collect();
    let [body] = enclosing.as_slice() else {
        panic!("one loop body holds the inner loop: {:?}", ops_of_anywhere(&found))
    };
    let after_inner: Vec<&str> = body
        .ops
        .iter()
        .skip_while(|name| **name != inner.name)
        .skip(1)
        .map(|name| family_of(name))
        .collect();
    assert!(!after_inner.contains(&"Mov"), "{:?}", body.ops);
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
