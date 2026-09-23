//! `if/else` as one operation: what each shape computes, and what the
//! recognizer collapses it into.
//!
//! `short_circuit.rs` holds the other half of the untaken-arm property,
//! where the arm that must not run is an extern call rather than a
//! division.

use acvus_interpreter_test::listing::{
    RegionListing, family_of, ops_of_anywhere, regions_named, script_listing,
};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[derive(Debug, PartialEq)]
struct DiamondShape {
    on_true_ops: usize,
    on_false_ops: usize,
    join_moves: usize,
}

fn shape_of(region: &RegionListing) -> DiamondShape {
    let on_true = region
        .part("on_true")
        .expect("a Diamond holds an on_true arm");
    let on_false = region
        .part("on_false")
        .expect("a Diamond holds an on_false arm");
    assert_eq!(
        on_true.leaves_with, on_false.leaves_with,
        "both arms of a diamond feed the same join parameters"
    );
    DiamondShape {
        on_true_ops: on_true.ops.len(),
        on_false_ops: on_false.ops.len(),
        join_moves: on_true.leaves_with,
    }
}

fn diamonds_innermost_first(source: &str, page: fn(&Interner) -> Context) -> Vec<String> {
    let interner = Interner::new();
    regions_named(
        &script_listing(&interner, source, page(&interner), Ty::I64),
        "Diamond",
    )
    .into_iter()
    .map(|region| format!("{:?}", shape_of(region)))
    .collect()
}

fn loop_count(source: &str) -> usize {
    loop_count_on_page(source, |_| Context::default())
}

fn loop_count_on_page(source: &str, page: fn(&Interner) -> Context) -> usize {
    let interner = Interner::new();
    let page = page(&interner);
    ops_of_anywhere(&script_listing(&interner, source, page, Ty::I64))
        .iter()
        .filter(|name| family_of(name) == "Loop")
        .count()
}

fn shape(on_true_ops: usize, on_false_ops: usize, join_moves: usize) -> String {
    format!(
        "{:?}",
        DiamondShape {
            on_true_ops,
            on_false_ops,
            join_moves,
        }
    )
}

#[tokio::test]
async fn an_if_expression_yields_the_taken_arms_value() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let n = 7; let d = if n % 2 == 0 { n / 2 } else { n * 3 + 1 }; d",
        Context::default(),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 22);
}

#[tokio::test]
async fn an_if_statement_without_a_value_runs_only_the_taken_arm() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let acc = 0; let n = 0; while n < 6 { if n % 2 == 0 { acc = acc + n; }; n = n + 1; } acc",
        Context::default(),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 6);
}

#[tokio::test]
async fn an_if_nested_in_an_arm_is_a_diamond_inside_a_diamond() {
    let i = Interner::new();
    let source = "let r = if @n > 5 { if @n > 8 { 100 } else { 10 } } else { 1 }; r";
    let v = run_script(&i, source, int_context(&i, "n", 9), Ty::I64).await;
    assert_eq!(v.as_int(), 100);
    assert_eq!(
        diamonds_innermost_first(source, |i| int_context(i, "n", 9)),
        vec![shape(1, 1, 0), shape(2, 1, 0)],
        "the outer arm counts the inner diamond among its own operations"
    );
    let interner = Interner::new();
    let outer = script_listing(&interner, source, int_context(&interner, "n", 9), Ty::I64);
    let outer = regions_named(&outer, "Diamond");
    let arm = outer
        .last()
        .expect("the outer diamond")
        .part("on_true")
        .expect("its true arm");
    assert_eq!(
        arm.ops,
        vec!["Gt<i64, Slot, Slot, R0>", "Diamond<R0, Rejoins>"],
        "the outer arm is two operations: the inner test and the inner diamond. `@n` \
         comes from the page, so the inner `@n > 8` is a live compare no fold reaches, \
         and a compare is a chain producer — its word rides in R0 into the diamond \
         rather than through a slot"
    );
}

#[tokio::test]
async fn an_untaken_arm_that_would_panic_never_runs() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let z = 0; let r = if z == 0 { 5 } else { 1 / z }; r",
        Context::default(),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 5);
}

#[tokio::test]
async fn the_dividing_arm_runs_when_it_is_the_one_taken() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let z = 2; let r = if z == 0 { 5 } else { 10 / z }; r",
        Context::default(),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 5);
}

/// This and the next loop test with `<=` so that they stay `while`s:
/// RFC-0079 turns `n < 8` into a range `for`.
#[tokio::test]
async fn a_diamond_in_a_body_leaves_the_while_recognizable() {
    let source = "let acc = 0; let n = 0; while n <= 7 { if n % 3 == 0 { acc = acc + 1; } else if n % 3 == 1 { acc = acc + 10; } else { acc = acc + 100; }; n = n + 1; } acc";
    let i = Interner::new();
    let v = run_script(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 3 + 30 + 200);
    assert_eq!(loop_count(source), 1);
    assert_eq!(
        diamonds_innermost_first(source, |_| Context::default()).len(),
        2
    );
}

#[tokio::test]
async fn a_while_in_an_arm_is_one_operation_inside_the_diamond() {
    let source = "let acc = 0; if @n > 3 { let k = 0; while k <= @n - 1 { acc = acc + k; k = k + 1; } } else { acc = 1; }; acc";
    let i = Interner::new();
    let v = run_script(&i, source, int_context(&i, "n", 5), Ty::I64).await;
    assert_eq!(v.as_int(), 10);
    assert_eq!(loop_count_on_page(source, |i| int_context(i, "n", 5)), 1);
    assert_eq!(
        diamonds_innermost_first(source, |i| int_context(i, "n", 5)).len(),
        1
    );
}

#[tokio::test]
async fn a_short_circuit_condition_leaves_the_while_recognizable() {
    let source =
        "let n = 0; let acc = 0; while n < 10 && acc < 12 { acc = acc + n; n = n + 1; } acc";
    let i = Interner::new();
    let v = run_script(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 15);
    assert_eq!(loop_count(source), 1);
    assert_eq!(
        diamonds_innermost_first(source, |_| Context::default()),
        vec![shape(1, 1, 0)]
    );
}

/// `benches/shapes.rs`'s `enum match`. `optimize::sroa` threads each
/// constant tag straight to its arm, `code_motion` sinks the arm's `acc + v`
/// into the branch that selected it, and `optimize::forward` gives the
/// branch the arm's edge -- so what reaches the recognizer is the one
/// diamond the `if` wrote, and the tag test is not an operation at all.
#[tokio::test]
async fn a_threaded_matchs_arms_leave_one_diamond_in_the_loop() {
    let source = "let acc = 0; let i = 0; while i < @n { \
         let e = if i % 2 == 0 { E::A(i) } else { E::B(i + 1) }; \
         match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v; } }; \
         i = i + 1; } acc";
    let i = Interner::new();
    let v = run_script(&i, source, int_context(&i, "n", 6), Ty::I64).await;
    assert_eq!(v.as_int(), 18);

    let interner = Interner::new();
    let listing = script_listing(&interner, source, int_context(&interner, "n", 6), Ty::I64);
    assert_eq!(
        ops_of_anywhere(&listing)
            .iter()
            .filter(|name| family_of(name) == "For")
            .count(),
        1,
        "`i < @n` counting by one is a range `for` (RFC-0079)"
    );
    let diamonds = regions_named(&listing, "Diamond");
    assert_eq!(
        diamonds.len(),
        1,
        "the tag test is gone with the arm blocks"
    );
    for part in ["on_true", "on_false"] {
        assert_eq!(
            diamonds[0]
                .part(part)
                .expect("a diamond holds both arms")
                .ops
                .len(),
            1,
            "{part} is the one sum the arm computes"
        );
    }
}

#[tokio::test]
async fn an_or_condition_is_the_same_diamond() {
    let source = "let n = 0; let acc = 0; while n < 3 || acc < 20 { acc = acc + n; n = n + 1; } n";
    let i = Interner::new();
    let v = run_script(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 7);
    assert_eq!(loop_count(source), 1);
    assert_eq!(
        diamonds_innermost_first(source, |_| Context::default()).len(),
        1
    );
}
