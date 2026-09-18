//! What `acvus-mir`'s `optimize::fold` computes is what the machine
//! computes (RFC-0055).
//!
//! Each case is one edge written twice: once with the operands the pass
//! can read as constants, and once with the same values arriving through
//! the page, where it cannot. The two runs must leave the same register
//! word, so a fold that disagreed with `ops::arith` on a wrap, a negative
//! dividend or a NaN bit pattern fails here.

use acvus_interpreter::Value;
use acvus_interpreter_test::{Context, run_script, typed};
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;

/// One edge written twice.
struct Agreement {
    folded: &'static str,
    through_the_page: &'static str,
}

fn integers(i: &Interner, k: IntTy, named: &[(&str, u64)]) -> Context {
    named
        .iter()
        .map(|(name, bits)| {
            (
                i.intern(name),
                typed(Ty::Int(k), Value::from_bits(k, *bits)),
            )
        })
        .collect()
}

fn floats(i: &Interner, named: &[(&str, f64)]) -> Context {
    named
        .iter()
        .map(|(name, held)| (i.intern(name), typed(Ty::Float, Value::float(*held))))
        .collect()
}

/// The register word both spellings leave.
async fn agreed<C>(i: &Interner, case: Agreement, ret: Ty, context: C) -> u64
where
    C: Fn(&Interner) -> Context,
{
    let folded = run_script(i, case.folded, context(i), ret.clone()).await;
    let through = run_script(i, case.through_the_page, context(i), ret).await;
    assert_eq!(
        folded.bits(),
        through.bits(),
        "`{}` and `{}` disagree",
        case.folded,
        case.through_the_page
    );
    folded.bits()
}

#[tokio::test]
async fn an_addition_past_the_width_wraps_the_same_way_folded() {
    let i = Interner::new();
    let byte = agreed(
        &i,
        Agreement {
            folded: "250 + 10",
            through_the_page: "@a + @b",
        },
        Ty::Int(IntTy::U8),
        |i| integers(i, IntTy::U8, &[("a", 250), ("b", 10)]),
    )
    .await;
    assert_eq!(IntTy::U8.read(byte), 4);

    let widest = agreed(
        &i,
        Agreement {
            folded: "9223372036854775807 + 1",
            through_the_page: "@a + @b",
        },
        Ty::I64,
        |i| integers(i, IntTy::I64, &[("a", i64::MAX as u64), ("b", 1)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(widest), i128::from(i64::MIN));
}

#[tokio::test]
async fn a_negative_dividend_keeps_its_sign_folded() {
    let i = Interner::new();
    let remainder = agreed(
        &i,
        Agreement {
            folded: "-7 % 2",
            through_the_page: "@a % @b",
        },
        Ty::I64,
        |i| integers(i, IntTy::I64, &[("a", (-7i64) as u64), ("b", 2)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(remainder), -1);

    let quotient = agreed(
        &i,
        Agreement {
            folded: "-7 / 2",
            through_the_page: "@a / @b",
        },
        Ty::I64,
        |i| integers(i, IntTy::I64, &[("a", (-7i64) as u64), ("b", 2)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(quotient), -3);
}

#[tokio::test]
async fn two_constants_joined_across_one_add_wrap_the_same() {
    let i = Interner::new();
    let byte = agreed(
        &i,
        Agreement {
            folded: "@a + 1 + 2",
            through_the_page: "@a + @b",
        },
        Ty::Int(IntTy::U8),
        |i| integers(i, IntTy::U8, &[("a", 254), ("b", 3)]),
    )
    .await;
    assert_eq!(IntTy::U8.read(byte), 1);
}

#[tokio::test]
async fn a_nan_carries_the_machines_bit_pattern() {
    let i = Interner::new();
    let held = agreed(
        &i,
        Agreement {
            folded: "0.0 / 0.0 == @z / @z",
            through_the_page: "@z / @z == @z / @z",
        },
        Ty::Bool,
        |i| floats(i, &[("z", 0.0)]),
    )
    .await;
    assert_eq!(held, 1);
}

#[tokio::test]
#[should_panic(expected = "attempt to divide by zero")]
async fn a_division_by_a_constant_zero_still_panics() {
    let i = Interner::new();
    run_script(&i, "1 / 0", Context::default(), Ty::I64).await;
}

#[tokio::test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
async fn a_remainder_by_a_constant_zero_still_panics() {
    let i = Interner::new();
    run_script(&i, "1 % 0", Context::default(), Ty::I64).await;
}
