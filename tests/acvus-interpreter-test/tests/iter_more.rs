//! The iterator producers, stages, consumers and aggregates added after
//! RFC-0041, each at the script contract: `range`, `range_step`,
//! `step_by`, `take_while`, `skip_while`, `chunks`, `dedup`, `count`,
//! `last`, `nth`, `position`, `sum`, `product`, `min`, `max`,
//! `min_by_key`, `max_by_key`. A trap surfaces here as a panic of the
//! harness carrying the trap's message.

use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    run_script_mode_with_externs(&i, source, Context::default(), registries, ret)
        .await
        .value
}

// -- Producers ------------------------------------------------------------------

#[tokio::test]
async fn range_is_half_open() {
    assert_eq!(run("range(0, 5) | count()", Ty::I64).await.as_int(), 5);
    assert_eq!(run("range(3, 4) | sum()", Ty::I64).await.as_int(), 3);
}

#[tokio::test]
async fn range_with_end_not_after_start_is_empty() {
    assert_eq!(run("range(5, 5) | count()", Ty::I64).await.as_int(), 0);
    assert_eq!(run("range(7, 2) | count()", Ty::I64).await.as_int(), 0);
}

#[tokio::test]
async fn range_step_counts_up_by_the_step_short_of_the_end() {
    let count = run("range_step(0, 10, 3) | count()", Ty::I64).await;
    let last = run("let xs = range_step(0, 10, 3) | collect; xs[3]", Ty::I64).await;
    assert_eq!((count.as_int(), last.as_int()), (4, 9), "0, 3, 6, 9");
}

#[tokio::test]
async fn range_step_with_a_negative_step_counts_down() {
    assert_eq!(
        run("range_step(10, 0, -3) | sum()", Ty::I64).await.as_int(),
        22,
        "10 + 7 + 4 + 1"
    );
}

#[tokio::test]
#[should_panic(expected = "range_step")]
async fn range_step_with_a_zero_step_traps() {
    run("range_step(0, 5, 0) | count()", Ty::I64).await;
}

// -- Stages ---------------------------------------------------------------------

#[tokio::test]
async fn step_by_keeps_the_first_and_every_nth_after_it() {
    let count = run("range(0, 10) | step_by(3) | count()", Ty::I64).await;
    let last = run(
        "let xs = range(0, 10) | step_by(3) | collect; xs[3]",
        Ty::I64,
    )
    .await;
    assert_eq!((count.as_int(), last.as_int()), (4, 9), "0, 3, 6, 9");
}

#[tokio::test]
#[should_panic(expected = "step_by")]
async fn step_by_zero_traps() {
    run("range(0, 3) | step_by(0) | count()", Ty::I64).await;
}

#[tokio::test]
async fn take_while_stops_at_the_first_element_that_fails() {
    assert_eq!(
        run(
            "into_iter([1, 2, 9, 3]) | take_while(|x| -> *x < 5) | count()",
            Ty::I64
        )
        .await
        .as_int(),
        2
    );
}

#[tokio::test]
async fn skip_while_resumes_at_the_first_element_that_fails_and_keeps_the_rest() {
    assert_eq!(
        run(
            "into_iter([1, 2, 9, 3]) | skip_while(|x| -> *x < 5) | sum()",
            Ty::I64
        )
        .await
        .as_int(),
        12
    );
}

/// `chain` draws its first pipeline, then its second, each through the
/// `next` that owns it: the two here are of one type and different
/// values, so a stage drawing either through the other's instance, or
/// the second first, yields another sequence.
#[tokio::test]
async fn chain_yields_the_first_pipeline_then_the_second() {
    let digits = run(
        "range(1, 3) | chain(range(7, 9)) | fold(0, |acc, x| -> acc * 10 + x)",
        Ty::I64,
    )
    .await;
    assert_eq!(digits.as_int(), 1278, "1, 2, then 7, 8");
}

#[tokio::test]
async fn chunks_cover_the_source_with_a_short_last_chunk() {
    assert_eq!(
        run("range(0, 7) | chunks(3) | count()", Ty::I64)
            .await
            .as_int(),
        3
    );
    let v = run(
        "let c = range(0, 7) | chunks(3) | last() | unwrap; c.len()",
        Ty::U64,
    )
    .await;
    assert_eq!(v.as_int(), 1);
}

#[tokio::test]
#[should_panic(expected = "chunks")]
async fn chunks_of_zero_traps() {
    run("range(0, 3) | chunks(0) | count()", Ty::I64).await;
}

#[tokio::test]
async fn dedup_collapses_consecutive_equal_elements_only() {
    let v = run(
        "let xs = into_iter([1, 1, 2, 2, 1]) | dedup | collect; xs.len()",
        Ty::U64,
    )
    .await;
    assert_eq!(v.as_int(), 3);
}

#[tokio::test]
async fn dedup_over_strings_reads_each_element_in_place() {
    let v = run(
        r#"into_iter(["a".to_string(), "a".to_string(), "b".to_string(), "b".to_string(), "a".to_string()]) | dedup | count()"#,
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 3);
}

// -- Consumers ------------------------------------------------------------------

#[tokio::test]
async fn count_is_the_number_of_elements_after_the_stages() {
    let v = run(
        "range(0, 5) | map(|x| -> x * 2) | filter(|x| -> *x > 2) | count()",
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 3, "4, 6, 8");
}

#[tokio::test]
async fn last_is_the_final_element_or_none() {
    assert_eq!(
        run("range(3, 8) | last() | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        7
    );
    assert_eq!(
        run("range(0, 0) | last() | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        -1
    );
}

#[tokio::test]
async fn nth_is_zero_based_and_none_past_the_end() {
    assert_eq!(
        run("range(10, 20) | nth(2) | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        12
    );
    assert_eq!(
        run("range(0, 2) | nth(5) | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        -1
    );
}

#[tokio::test]
async fn position_is_the_index_of_the_first_match_or_none() {
    assert_eq!(
        run(
            "into_iter([5, 6, 7]) | position(|x| -> *x == 7) | unwrap_or(-1)",
            Ty::I64
        )
        .await
        .as_int(),
        2
    );
    assert_eq!(
        run(
            "into_iter([5, 6, 7]) | position(|x| -> *x == 9) | unwrap_or(-1)",
            Ty::I64
        )
        .await
        .as_int(),
        -1
    );
}

// -- Aggregates -----------------------------------------------------------------

#[tokio::test]
async fn sum_of_a_range_of_ints() {
    assert_eq!(run("range(0, 5) | sum()", Ty::I64).await.as_int(), 10);
}

#[tokio::test]
async fn sum_of_an_empty_iter_is_zero() {
    assert_eq!(run("range(0, 0) | sum()", Ty::I64).await.as_int(), 0);
}

#[tokio::test]
async fn sum_of_floats_is_a_float() {
    assert_eq!(
        run("into_iter([1.0, 2.5]) | sum()", Ty::Float)
            .await
            .as_float(),
        3.5
    );
}

#[tokio::test]
async fn product_of_a_range_and_of_nothing() {
    assert_eq!(run("range(1, 5) | product()", Ty::I64).await.as_int(), 24);
    assert_eq!(run("range(0, 0) | product()", Ty::I64).await.as_int(), 1);
}

#[tokio::test]
async fn sum_and_product_past_the_width_wrap() {
    let v = run("into_iter([9223372036854775807, 1]) | sum()", Ty::I64).await;
    assert_eq!(v.as_int(), i64::MIN);
    let v = run("into_iter([4611686018427387904, 2]) | product()", Ty::I64).await;
    assert_eq!(v.as_int(), i64::MIN);
}

#[tokio::test]
async fn max_of_an_empty_iter_is_none() {
    assert_eq!(
        run("range(0, 0) | max() | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        -1
    );
}

#[tokio::test]
async fn min_and_max_of_ints() {
    assert_eq!(
        run("into_iter([3, 9, 2]) | max() | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        9
    );
    assert_eq!(
        run("into_iter([3, 9, 2]) | min() | unwrap_or(-1)", Ty::I64)
            .await
            .as_int(),
        2
    );
}

#[tokio::test]
async fn min_and_max_of_floats() {
    assert_eq!(
        run("into_iter([1.5, -2.0]) | min() | unwrap_or(0.0)", Ty::Float)
            .await
            .as_float(),
        -2.0
    );
    assert_eq!(
        run("into_iter([1.5, -2.0]) | max() | unwrap_or(0.0)", Ty::Float)
            .await
            .as_float(),
        1.5
    );
}

#[tokio::test]
async fn min_by_key_and_max_by_key_return_the_element_not_the_key() {
    assert_eq!(
        run(
            "into_iter([3, -7, 5]) | min_by_key(|x| -> *x * *x) | unwrap_or(0)",
            Ty::I64
        )
        .await
        .as_int(),
        3
    );
    assert_eq!(
        run(
            "into_iter([3, -7, 5]) | max_by_key(|x| -> *x * *x) | unwrap_or(0)",
            Ty::I64
        )
        .await
        .as_int(),
        -7
    );
}

#[tokio::test]
async fn min_by_key_and_max_by_key_keep_the_first_of_equal_keys() {
    assert_eq!(
        run(
            "into_iter([4, -4, 2, -2]) | min_by_key(|x| -> *x * *x) | unwrap_or(0)",
            Ty::I64
        )
        .await
        .as_int(),
        2
    );
    assert_eq!(
        run(
            "into_iter([4, -4, 2, -2]) | max_by_key(|x| -> *x * *x) | unwrap_or(0)",
            Ty::I64
        )
        .await
        .as_int(),
        4
    );
}

#[tokio::test]
async fn min_by_key_over_an_empty_iter_is_none() {
    assert_eq!(
        run(
            "range(0, 0) | min_by_key(|x| -> *x) | unwrap_or(-1)",
            Ty::I64
        )
        .await
        .as_int(),
        -1
    );
}

#[tokio::test]
async fn a_pipeline_mixing_the_new_stages_and_an_aggregate() {
    let v = run(
        "range(0, 5) | map(|x| -> x * 2) | step_by(2) | sum()",
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 12, "0, 4, 8");
}

/// The closure's own `/` panics, three synchronous stages below the
/// boundary, and the unwinder carries it out through every stage.
#[tokio::test]
#[should_panic(expected = "attempt to divide by zero")]
async fn a_panic_three_sync_stages_deep_unwinds_out_of_the_pipeline() {
    let i = Interner::new();
    let source = "range(0, 4) | map(|x| -> x) | map(|x| -> x) | map(|x| -> x / @zero) | sum()";
    let context: Context = [(
        i.intern("zero"),
        typed(acvus_mir::ty::Ty::I64, Value::int(0)),
    )]
    .into_iter()
    .collect();

    run_script_mode(&i, source, context, Ty::I64).await;
}

const THREE_STAGES_DEEP: &str =
    "range(0, 4) | map(|x| -> x) | map(|x| -> x) | map(|x| -> x + @big) | sum()";

fn big(i: &Interner, value: i64) -> Context {
    [(
        i.intern("big"),
        typed(acvus_mir::ty::Ty::I64, Value::int(value)),
    )]
    .into_iter()
    .collect()
}

/// `10 + 0` to `10 + 3`, summed.
#[tokio::test]
async fn arithmetic_three_sync_stages_deep_runs_at_each_stage() {
    let i = Interner::new();
    assert_eq!(
        run_script_mode(&i, THREE_STAGES_DEEP, big(&i, 10), Ty::I64)
            .await
            .as_int(),
        46
    );
}

/// `1 + MAX` is the program's `+` past the width, and it traps three stages
/// deep as it does anywhere (RFC-0037 rule 3).
#[tokio::test]
#[should_panic(expected = "attempt to add with overflow")]
async fn arithmetic_three_sync_stages_deep_traps_past_the_width() {
    let i = Interner::new();
    run_script_mode(&i, THREE_STAGES_DEEP, big(&i, i64::MAX), Ty::I64).await;
}
