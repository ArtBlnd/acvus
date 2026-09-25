//! What `optimize::lsr` must leave untouched: the value a loop produces and
//! the iteration on which it stops (RFC-0056).
//!
//! Each script holds the same recurrence twice. The first loop writes
//! `i * k + x` straight in the body, which the pass reduces to a derived
//! induction variable; the second writes it under an `if`, whose block does
//! not dominate the latch, so the pass leaves that multiplication where it
//! is. The two are the reduced and the unreduced form of one computation, in
//! one program, on one run, and the test reads their difference. The
//! accumulator is squared before it adds, `acc * acc % 1000 + …`: a plain
//! sum is a merge, joined `AnyOrder`, `acc * 2 + …` has the affine map law
//! (RFC-0093 rule 8), and the pass reduces only what an `InOrder` join with
//! no law reads (RFC-0066 rule 7).
//!
//! The program's `*` and `+` trap where they leave the width (RFC-0037
//! rule 3), and the pass reduces them only where the loop's word bounds,
//! `k` and `x` keep both inside the width, so each case writes its count,
//! factor and offset into the script as words.
//! The pass's own start, step and advance wrap: the edge case is a counter
//! whose advance after the last iteration passes `i64::MAX`, a value the
//! program never computes, and the two forms must still agree. The
//! iteration a loop stops on is measured separately, with a division.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::{Astr, Interner};

struct Case {
    n: i64,
    factor: i64,
    offset: i64,
}

fn ctx(i: &Interner, entries: &[(&str, i64)]) -> Context {
    entries
        .iter()
        .map(|(name, value)| -> (Astr, TypedValue) {
            (
                i.intern(name),
                typed(Ty::I64, Value::from_bits(IntTy::I64, *value as u64)),
            )
        })
        .collect()
}

/// `acc * acc % 1000 + (i * k + x)` over the iterations. A case is one
/// whose program does not overflow, and Rust's own operators here say so.
fn squaring_reference(case: &Case) -> i64 {
    (0..case.n).fold(0i64, |acc, i| acc * acc % 1000 + (i * case.factor + case.offset))
}

fn both_forms(case: &Case) -> String {
    let Case { n, factor, offset } = case;
    format!(
        "let reduced = 0; let i = 0; \
         while i < {n} {{ reduced = reduced * reduced % 1000 + (i * {factor} + {offset}); i = i + 1; }} \
         let unreduced = 0; let j = 0; \
         while j < {n} {{ \
             if j + 1 > j {{ unreduced = unreduced * unreduced % 1000 + (j * {factor} + {offset}); }}; \
             j = j + 1; \
         }} \
         reduced - unreduced"
    )
}

fn reduced_only(case: &Case) -> String {
    let Case { n, factor, offset } = case;
    format!(
        "let acc = 0; let i = 0; \
         while i < {n} {{ acc = acc * acc % 1000 + (i * {factor} + {offset}); i = i + 1; }} \
         acc"
    )
}

async fn run_case(source: String) -> i64 {
    let i = Interner::new();
    run_script(&i, &source, Context::default(), Ty::I64)
        .await
        .as_int()
}

#[tokio::test]
async fn the_reduced_form_and_the_unreduced_one_agree() {
    for case in [
        Case {
            n: 0,
            factor: 3,
            offset: 7,
        },
        Case {
            n: 1,
            factor: 3,
            offset: 7,
        },
        Case {
            n: 2,
            factor: 3,
            offset: 7,
        },
        Case {
            n: 17,
            factor: 5,
            offset: -11,
        },
        Case {
            n: 40,
            factor: -9,
            offset: 0,
        },
        Case {
            n: 50,
            factor: 1,
            offset: 1,
        },
    ] {
        let Case { n, factor, offset } = case;
        assert_eq!(
            run_case(both_forms(&case)).await,
            0,
            "n={n} k={factor} x={offset}: the reduced loop and the unreduced one disagree"
        );
        assert_eq!(
            run_case(reduced_only(&case)).await,
            squaring_reference(&case),
            "n={n} k={factor} x={offset}: the reduced loop does not compute the recurrence"
        );
    }
}

/// The second iteration's product is `5 · 10^18`, inside `i64`; the
/// reduced counter's advance after it is `10^19 + 1`, past `i64::MAX`.
const PASSES_THE_WIDTH_ON_THE_LAST_ADVANCE: i64 = 5_000_000_000_000_000_000;

#[tokio::test]
async fn a_counter_advanced_past_the_width_after_the_last_iteration_agrees() {
    let case = Case {
        n: 2,
        factor: PASSES_THE_WIDTH_ON_THE_LAST_ADVANCE,
        offset: 1,
    };
    assert!(
        case.factor.checked_mul(case.n).is_none(),
        "the advance after the last iteration leaves the width"
    );
    assert_eq!(
        run_case(both_forms(&case)).await,
        0,
        "the two forms disagree where the advance passes i64::MAX"
    );
    assert_eq!(
        run_case(reduced_only(&case)).await,
        squaring_reference(&case),
        "the reduced loop does not compute the recurrence"
    );
}

const DIVISOR_REACHES_ZERO_AT: i64 = 3;
const FACTOR: i64 = 5;
const OFFSET: i64 = 7;

/// The division raises where the divisor reaches zero (RFC-0037 rule 2).
/// The accumulator is squared here too: `acc + a` then `acc + b` alone is a
/// sum, which RFC-0089 rule 4 reads as a merge through its two steps.
fn reduced_with_a_division(n: i64) -> String {
    format!(
        "let acc = 0; let i = 0; \
         while i < {n} {{ \
             acc = acc * acc % 1000 + (i * {FACTOR} + {OFFSET}); \
             acc = acc + 10 / (@d - i); \
             i = i + 1; \
         }} \
         acc"
    )
}

async fn with_division(n: i64) -> i64 {
    let i = Interner::new();
    run_script(
        &i,
        &reduced_with_a_division(n),
        ctx(&i, &[("d", DIVISOR_REACHES_ZERO_AT)]),
        Ty::I64,
    )
    .await
    .as_int()
}

#[tokio::test]
async fn every_iteration_before_the_raise_still_runs() {
    let completed = (0..DIVISOR_REACHES_ZERO_AT).fold(0i64, |acc, i| {
        acc * acc % 1000 + (i * FACTOR + OFFSET) + 10 / (DIVISOR_REACHES_ZERO_AT - i)
    });
    assert_eq!(with_division(DIVISOR_REACHES_ZERO_AT).await, completed);
}

#[tokio::test]
#[should_panic(expected = "attempt to divide by zero")]
async fn the_raise_lands_on_the_iteration_it_landed_on() {
    with_division(DIVISOR_REACHES_ZERO_AT + 1).await;
}
