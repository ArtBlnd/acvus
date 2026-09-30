//! Obligation across artifacts: `run_allocations.rs` counts what a run keeps
//! off the heap, `try_op.rs` says what `?` answers, and `run_shape.rs` pins the
//! same shape for a declared enum. None of the three is re-asserted here.

use acvus_interpreter::Value;
use acvus_interpreter_test::listing::{BlockListing, ops_of_anywhere, script_listing_with_externs};
use acvus_interpreter_test::{Context, int_context, run_script_mode};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// A `Result` built and matched in one body, held across an inner loop so its
/// address is taken and `optimize::sroa` leaves the aggregate to the machine.
const MATCHED: &str = "let acc = 0; let i = 0; while i < @n { \
let r = if i % 2 == 0 { Ok(i) } else { Err(i + 1) }; \
let j = 0; while j < 1 { match r { Ok(v) => { acc = acc + v; }, \
Err(v) => { acc = acc + v; } }; j = j + 1; } i = i + 1; } acc";

/// `?` on a `Result` whose every use the web covers. The `Result` the body
/// returns is a second value, which escapes and is realized on the heap.
const TRIED: &str = "let r = if @n == 0 { Ok(3) } else { Err(7) }; let v = r?; Ok(v * 10)";

/// `TRIED` with the two sides disagreeing in `Large`-ness: only `Err` owns one,
/// while `Layout::widest` marks the shared payload word `large` for both.
const TRIED_MIXED: &str = "let r = if @n == 0 { Ok(3) } else { Err(\"bad\".to_string()) }; \
let v = r?; Ok(v * 10)";

/// The same `Result` crossing into an array, which `prepare::runs::Sites`
/// refuses, so rule 4 realizes it on the heap.
const ESCAPING: &str = "let r = if @n == 0 { Ok(1) } else { Err(2) }; \
let v = [r]; let m = len(&v); let z = m - m; \
match &v[z] { Ok(w) => { w * 1 }, Err(w) => { w * 2 } }";

fn ops(source: &str, ret: Ty) -> Vec<String> {
    let interner = Interner::new();
    let blocks: Vec<BlockListing> = script_listing_with_externs(
        &interner,
        source,
        int_context(&interner, "n", 0),
        acvus_ext::std_registries(),
        ret,
    );
    let ends: Vec<String> = blocks.iter().map(|block| block.end.clone()).collect();
    ops_of_anywhere(&blocks).into_iter().chain(ends).collect()
}

fn holds(ops: &[String], name: &str) -> bool {
    ops.iter().any(|op| op.starts_with(name))
}

fn int_result() -> Ty {
    Ty::Result(Box::new(Ty::I64), Box::new(Ty::I64))
}

fn text_result() -> Ty {
    Ty::Result(Box::new(Ty::I64), Box::new(Ty::String))
}

fn n(interner: &Interner, value: i64) -> Context {
    int_context(interner, "n", value)
}

fn tag_of(interner: &Interner, v: &Value) -> String {
    // SAFETY: the script's declared return type is a `Result`.
    let variant = unsafe { v.as_variant() };
    // SAFETY: the same witness — a variant's first register is its tag.
    interner
        .resolve(unsafe { variant.tag().as_tag() })
        .to_owned()
}

fn payload_of(v: &Value) -> &Value {
    // SAFETY: as `tag_of`'s.
    unsafe { v.as_variant() }.payload()
}

#[test]
fn a_matched_result_takes_a_run_and_no_heap_variant() {
    let ops = ops(MATCHED, Ty::I64);
    assert!(
        holds(&ops, "LayRun"),
        "the two sides write the run: {ops:?}"
    );
    assert!(holds(&ops, "Project"), "`&r` is a projection: {ops:?}");
    assert!(
        holds(&ops, "SwitchRun"),
        "the dispatch reads the tag register: {ops:?}"
    );
    assert!(
        !holds(&ops, "MakeVariant"),
        "no `Make` exists for a value that stays in its body: {ops:?}"
    );
    assert!(
        !holds(&ops, "Switch<"),
        "no boxed variant is dispatched on: {ops:?}"
    );
}

/// RFC-0050 rule 1, for the one operator that reads a variant without a
/// `match`.
#[test]
fn a_question_mark_on_a_run_is_a_tag_compare_and_a_move() {
    let ops = ops(TRIED, int_result());
    assert!(
        holds(&ops, "LayRun"),
        "the two sides write the run: {ops:?}"
    );
    assert!(
        holds(&ops, "TestRun"),
        "the test reads the tag register: {ops:?}"
    );
    assert!(
        !holds(&ops, "TestVariant"),
        "no boxed variant is tested: {ops:?}"
    );
    assert!(
        !holds(&ops, "UnwrapVariant"),
        "the payload is moved out of its register: {ops:?}"
    );
}

#[tokio::test]
async fn a_matched_result_reaches_the_arm_its_tag_names() {
    let interner = Interner::new();
    for (arms, expected) in [(1, 0), (2, 2), (4, 8)] {
        let answer = run_script_mode(&interner, MATCHED, n(&interner, arms), Ty::I64)
            .await
            .as_int();
        assert_eq!(answer, expected, "n={arms}");
    }
}

#[tokio::test]
async fn a_question_mark_on_a_run_takes_the_ok_and_returns_the_err() {
    let interner = Interner::new();
    let ok = run_script_mode(&interner, TRIED, n(&interner, 0), int_result()).await;
    assert_eq!(tag_of(&interner, &ok), "Ok");
    assert_eq!(payload_of(&ok).as_int(), 30);

    let err = run_script_mode(&interner, TRIED, n(&interner, 1), int_result()).await;
    assert_eq!(tag_of(&interner, &err), "Err");
    assert_eq!(payload_of(&err).as_int(), 7);
}

#[tokio::test]
async fn a_result_whose_sides_disagree_in_largeness_releases_by_the_mark() {
    let interner = Interner::new();
    let ok = run_script_mode(&interner, TRIED_MIXED, n(&interner, 0), text_result()).await;
    assert_eq!(tag_of(&interner, &ok), "Ok");
    assert_eq!(payload_of(&ok).as_int(), 30);

    let err = run_script_mode(&interner, TRIED_MIXED, n(&interner, 1), text_result()).await;
    assert_eq!(tag_of(&interner, &err), "Err");
    // SAFETY: the `Err` side of the script's return type is a `String`.
    assert_eq!(unsafe { payload_of(&err).as_str() }, "bad");
}

#[test]
fn a_result_that_escapes_keeps_its_heap_form() {
    let ops = ops(ESCAPING, Ty::I64);
    assert!(
        holds(&ops, "MakeVariant"),
        "an escaping `Result` is realized: {ops:?}"
    );
    assert!(!holds(&ops, "LayRun"), "no run is written: {ops:?}");
    assert!(!holds(&ops, "SwitchRun"), "no run is read: {ops:?}");
}

#[tokio::test]
async fn an_escaping_result_answers_the_same() {
    let interner = Interner::new();
    for (arms, expected) in [(0, 1), (1, 4)] {
        let answer = run_script_mode(&interner, ESCAPING, n(&interner, arms), Ty::I64)
            .await
            .as_int();
        assert_eq!(answer, expected, "n={arms}");
    }
}
