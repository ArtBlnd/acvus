//! What `optimize::lsr` must leave untouched: the value a loop produces and
//! the iteration on which it stops (RFC-0056).
//!
//! Each script holds the same sum twice. The first loop writes `i * @k + @x`
//! straight in the body, which the pass reduces to a derived induction
//! variable; the second writes it under an `if`, whose block does not
//! dominate the latch, so the pass leaves that multiplication where it is.
//! The two are the reduced and the unreduced form of one computation, in one
//! program, on one run, and the test reads their difference.
//!
//! `*` and `+` wrap at the operand's width (RFC-0037), so no product of an
//! induction variable can raise and there is no trap to move. What the
//! overflow case measures is therefore the wrap: the two forms must carry
//! the same two's-complement bits past `i64::MAX`. The iteration a loop
//! stops on is measured separately, with a division, which does raise.

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

/// The sum at the width and the wrap RFC-0037 gives `*` and `+`.
fn reference(case: &Case) -> i64 {
    (0..case.n).fold(0i64, |acc, i| {
        acc.wrapping_add(i.wrapping_mul(case.factor).wrapping_add(case.offset))
    })
}

const BOTH_FORMS: &str = "\
let reduced = 0; \
let i = 0; \
while i < @n { \
    reduced = reduced + (i * @k + @x); \
    i = i + 1; \
} \
let unreduced = 0; \
let j = 0; \
while j < @n { \
    if j + 1 > j { unreduced = unreduced + (j * @k + @x); }; \
    j = j + 1; \
} \
reduced - unreduced";

const REDUCED_ONLY: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = acc + (i * @k + @x); \
    i = i + 1; \
} \
acc";

async fn run_case(source: &str, case: &Case) -> i64 {
    let i = Interner::new();
    run_script(
        &i,
        source,
        ctx(&i, &[("n", case.n), ("k", case.factor), ("x", case.offset)]),
        Ty::I64,
    )
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
            n: 64,
            factor: -9,
            offset: 0,
        },
        Case {
            n: 100,
            factor: 1,
            offset: 1,
        },
    ] {
        let Case { n, factor, offset } = case;
        assert_eq!(
            run_case(BOTH_FORMS, &case).await,
            0,
            "n={n} k={factor} x={offset}: the reduced loop and the unreduced one disagree"
        );
        assert_eq!(
            run_case(REDUCED_ONLY, &case).await,
            reference(&case),
            "n={n} k={factor} x={offset}: the reduced loop does not compute the sum"
        );
    }
}

const WRAPS_ON_THE_FIFTH_ITERATION: i64 = 3_000_000_000_000_000_000;

#[tokio::test]
async fn a_product_that_wraps_carries_the_same_bits_in_both_forms() {
    for n in [3i64, 4, 5, 6, 9] {
        let case = Case {
            n,
            factor: WRAPS_ON_THE_FIFTH_ITERATION,
            offset: 1,
        };
        assert_eq!(
            run_case(BOTH_FORMS, &case).await,
            0,
            "n={n}: the two forms disagree past i64::MAX"
        );
        assert_eq!(
            run_case(REDUCED_ONLY, &case).await,
            reference(&case),
            "n={n}: the reduced loop does not wrap as `*` and `+` do"
        );
    }
    let five = Case {
        n: 5,
        factor: WRAPS_ON_THE_FIFTH_ITERATION,
        offset: 1,
    };
    let unwrapped: i128 = (0..five.n)
        .map(|i| i as i128 * five.factor as i128 + five.offset as i128)
        .sum();
    assert_ne!(
        reference(&five) as i128,
        unwrapped,
        "the case is an overflow case only if it overflows"
    );
}

/// Division is the one arithmetic that can raise (RFC-0037).
const REDUCED_WITH_A_DIVISION: &str = "\
let acc = 0; \
let i = 0; \
while i < @n { \
    acc = acc + (i * @k + @x); \
    acc = acc + 10 / (@d - i); \
    i = i + 1; \
} \
acc";

const DIVISOR_REACHES_ZERO_AT: i64 = 3;
const FACTOR: i64 = 5;
const OFFSET: i64 = 7;

async fn with_division(n: i64) -> i64 {
    let i = Interner::new();
    run_script(
        &i,
        REDUCED_WITH_A_DIVISION,
        ctx(
            &i,
            &[
                ("n", n),
                ("k", FACTOR),
                ("x", OFFSET),
                ("d", DIVISOR_REACHES_ZERO_AT),
            ],
        ),
        Ty::I64,
    )
    .await
    .as_int()
}

#[tokio::test]
async fn every_iteration_before_the_raise_still_runs() {
    let completed = Case {
        n: DIVISOR_REACHES_ZERO_AT,
        factor: FACTOR,
        offset: OFFSET,
    };
    let divisions: i64 = (0..DIVISOR_REACHES_ZERO_AT)
        .map(|i| 10 / (DIVISOR_REACHES_ZERO_AT - i))
        .sum();
    assert_eq!(
        with_division(DIVISOR_REACHES_ZERO_AT).await,
        reference(&completed) + divisions
    );
}

#[tokio::test]
#[should_panic(expected = "attempt to divide by zero")]
async fn the_raise_lands_on_the_iteration_it_landed_on() {
    with_division(DIVISOR_REACHES_ZERO_AT + 1).await;
}
