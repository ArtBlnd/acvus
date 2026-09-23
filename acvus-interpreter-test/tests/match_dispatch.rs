//! A `match` arrives at the value its arm names (RFC-0051 rule 5).
//!
//! The contract is the value the script returns, so each case runs every arm
//! of one `match` over the same source and asks for that arm's own answer.
//! That is what separates a dispatch which reached the wrong block from one
//! which reached the right one; the shape it took to get there is
//! `benches/programs.rs`'s to read.
//!
//! Obligation across artifacts: one case per operation `prepare::switch_op`
//! can choose — `switch::Switch` over a heap variant, with a catch-all and
//! without, by value and through a reference, a `Result` among them
//! (RFC-0050 rule 8); and `switch::SwitchOption` over the one form whose tag
//! is the value's own kind (RFC-0039).

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::{Context, int_context, run_script_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn maybe(i: i64) -> Option<i64> {
    match i {
        0 => None,
        n => Some(n * 10),
    }
}

#[extern_fn(effect = pure)]
fn checked(i: i64) -> Result<i64, String> {
    match i {
        0 => Err("zero".to_owned()),
        n => Ok(n * 10),
    }
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(extern_registry! {
        ns: "t",
        fns: [maybe, checked],
    });
    registries
}

/// `@n` is the one input, so one source answers for every arm.
async fn answer(source: &str, n: i64) -> i64 {
    let interner = Interner::new();
    run_script_with_externs(
        &interner,
        source,
        int_context(&interner, "n", n),
        regs(),
        Ty::I64,
    )
    .await
    .value
    .as_int()
}

/// The value `@n` picks, as one of five variants.
const FIVE: &str = "\
let e = if @n == 0 { E::A(1) } else { if @n == 1 { E::B(2) } else { \
if @n == 2 { E::C(3) } else { if @n == 3 { E::D(4) } else { E::E(5) } } } }; ";

/// Five arms and no catch-all: the fifth is the edge the dispatch falls out
/// of, and `validate::exhaustive` is what says it holds.
const CLOSED: &str = "match e { E::A(v) => v, E::B(v) => v * 2, \
E::C(v) => v * 3, E::D(v) => v * 4, E::E(v) => v * 5 }";

/// The same five variants, three arms and a catch-all.
const OPEN: &str = "match e { E::A(v) => v, E::B(v) => v * 2, E::C(v) => v * 3, _ => 99 }";

#[tokio::test]
async fn every_arm_of_a_closed_match_is_reached() {
    let source = format!("{FIVE}{CLOSED}");
    for (n, expected) in [(0, 1), (1, 4), (2, 9), (3, 16), (4, 25)] {
        assert_eq!(answer(&source, n).await, expected, "n={n}");
    }
}

#[tokio::test]
async fn a_tag_no_arm_names_reaches_the_catch_all() {
    let source = format!("{FIVE}{OPEN}");
    for (n, expected) in [(0, 1), (1, 4), (2, 9), (3, 99), (4, 99)] {
        assert_eq!(answer(&source, n).await, expected, "n={n}");
    }
}

#[tokio::test]
async fn a_match_on_an_option_reaches_the_side_its_value_is() {
    let source = "match maybe(@n) { Some(v) => v, None => 0 - 1 }";
    assert_eq!(answer(source, 0).await, -1);
    assert_eq!(answer(source, 7).await, 70);
}

#[tokio::test]
async fn an_option_match_written_with_a_catch_all_reaches_it() {
    let source = "match maybe(@n) { Some(v) => v, _ => 0 - 1 }";
    assert_eq!(answer(source, 0).await, -1);
    assert_eq!(answer(source, 7).await, 70);
}

#[tokio::test]
async fn a_match_on_a_result_reaches_the_side_its_value_is() {
    let source = "match checked(@n) { Ok(v) => v, Err(m) => 0 - 1 }";
    assert_eq!(answer(source, 0).await, -1);
    assert_eq!(answer(source, 7).await, 70);
}

/// A list element read through a reference is the `THROUGH` parameter every
/// operation in `ops::switch` carries, and unit variants behind it are the
/// shape `bf table` runs. The catch-all here is `prepare::switch_op`'s
/// `default` edge, which the three elements never take.
#[tokio::test]
async fn a_match_through_a_reference_reaches_the_arm() {
    let source = "\
let v = [Op::Inc, Op::Dec, Op::Out]; \
let len = len(&v); let one = len / len; let i = len - len; let acc = @n; \
while i < len { \
let picked = match &v[i] { Op::Inc => 1, Op::Dec => 4, Op::Out => 9, _ => 0 }; \
acc = acc + picked; i = i + one; } acc";
    // Every element is read once, so the three arms sum to 1 + 4 + 9 above
    // `@n`, and a dispatch that took the catch-all would add zero.
    assert_eq!(answer(source, 0).await, 14);
    assert_eq!(answer(source, 100).await, 114);
}

/// This scrutinee is a join of a join, which `optimize::sroa` does not
/// take apart, so the enum reaches the machine and its own dispatch reads
/// the tag. Scalar replacement learning this shape would leave the test
/// running arithmetic and `ops::switch` unexercised in a loop.
#[tokio::test]
async fn a_match_inside_a_loop_reaches_every_arm_in_turn() {
    let source = "\
let acc = 0; let i = 0; \
while i < @n { \
let e = if i % 3 == 0 { E::A(1) } else { if i % 3 == 1 { E::B(2) } else { E::C(3) } }; \
let picked = match e { E::A(v) => v, E::B(v) => v * 2, E::C(v) => v * 3 }; \
acc = acc + picked; \
i = i + 1; } acc";
    // 1 + 4 + 9 over three iterations, then 1 + 4 for the two that follow.
    assert_eq!(answer(source, 3).await, 14);
    assert_eq!(answer(source, 5).await, 19);
}

/// The arms scale their payload differently so that an edge which reached
/// the wrong one changes the sum; arms that all computed `acc + v` would
/// answer the same whatever the dispatch did.
#[tokio::test]
async fn a_threaded_two_armed_match_reaches_the_arm_its_edge_names() {
    let source = "\
let acc = 0; let i = 0; \
while i < @n { \
let e = if i % 2 == 0 { E::A(i) } else { E::B(i + 1) }; \
match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v * 2; } }; \
i = i + 1; } acc";
    // i even adds i, i odd adds twice i + 1: 0, 4, 2, 8, 4, 12.
    assert_eq!(answer(source, 5).await, 18);
    assert_eq!(answer(source, 6).await, 30);
}

#[tokio::test]
async fn a_threaded_three_armed_match_reaches_the_arm_its_edge_names() {
    let source = "\
let acc = 0; let i = 0; \
while i < @n { \
let e = match i % 3 { 0 => E::A(i), 1 => E::B(i + 1), _ => E::C(i + 2) }; \
match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v * 2; }, \
E::C(v) => { acc = acc + v * 3; } }; \
i = i + 1; } acc";
    // i, twice i + 1, three times i + 2, by i % 3: 0, 4, 12, 3, 10, 21.
    assert_eq!(answer(source, 3).await, 16);
    assert_eq!(answer(source, 6).await, 50);
}

/// Two of the three edges carry a tag no arm names, and each of those is
/// threaded to the `default` its tag selects.
#[tokio::test]
async fn a_threaded_edge_whose_tag_no_arm_names_reaches_the_catch_all() {
    let source = "\
let acc = 0; let i = 0; \
while i < @n { \
let e = match i % 3 { 0 => E::A(i), 1 => E::B(i + 1), _ => E::C(i + 2) }; \
match e { E::A(v) => { acc = acc + v; }, _ => { acc = acc + 99; } }; \
i = i + 1; } acc";
    // i when i % 3 is zero, 99 otherwise: 0, 99, 99, 3, 99, 99.
    assert_eq!(answer(source, 3).await, 198);
    assert_eq!(answer(source, 6).await, 399);
}

#[tokio::test]
async fn a_match_with_no_context_still_runs() {
    let interner = Interner::new();
    let ran = run_script_with_externs(
        &interner,
        "let e = E::A(6); match e { E::A(v) => v * 7 }",
        Context::default(),
        regs(),
        Ty::I64,
    )
    .await;
    assert_eq!(ran.value.as_int(), 42);
}

/// A list of two constructions: the element type is the union of both, so the
/// two arms cover it and no `_` is written. The two payload types differ, so
/// the union is what carries the set -- neither construction alone does.
const MIXED_LIST: &str = "\
let v = [E::A(1), E::B(\"xyz\"), E::A(4)]; \
let len = len(&v); let one = len / len; let i = len - len; let acc = @n; \
while i < len { ";

#[tokio::test]
async fn a_match_on_a_list_element_reaches_the_arm_without_a_catch_all() {
    let source = format!(
        "{MIXED_LIST}let picked = match &v[i] {{ E::A(x) => *x, E::B(s) => 100 }}; \
acc = acc + picked; i = i + one; }} acc"
    );
    // 1 + 100 + 4 above `@n`; an arm reached for the wrong element moves it.
    assert_eq!(answer(&source, 0).await, 105);
    assert_eq!(answer(&source, 1000).await, 1105);
}

#[tokio::test]
#[should_panic(expected = "non-exhaustive match: `E::B` is not covered")]
async fn a_match_on_a_list_element_that_misses_a_variant_is_refused() {
    let source = format!(
        "{MIXED_LIST}let picked = match &v[i] {{ E::A(x) => *x }}; \
acc = acc + picked; i = i + one; }} acc"
    );
    answer(&source, 0).await;
}
