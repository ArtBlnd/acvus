//! What `acvus-mir`'s `optimize::fold` computes is what the machine
//! computes (RFC-0055).
//!
//! Each case is one edge written twice: once with the operands the pass
//! can read as constants, and once with the same values arriving through
//! the page, where it cannot. The two runs must leave the same register
//! word or end with the same trap, so a fold that disagreed with
//! `ops::arith` on an overflow, a negative dividend or a NaN bit pattern
//! fails here.

use std::sync::Arc;

use acvus_interpreter::{HostError, SequentialExecutor, Value};
use acvus_interpreter_test::{
    Context, compile_script_mode, execute_compiled, run_script, split_context, typed,
};
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;

const ADD_OVERFLOW: &str = "attempt to add with overflow";

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

/// The text of the trap a run ends with.
async fn trap_of(i: &Interner, source: &str, context: Context, ret: Ty) -> String {
    let (context_types, snapshot) = split_context(i, context);
    let compiled = compile_script_mode(i, source, &context_types, ret);
    let (_, mut interp) = execute_compiled(i, compiled, snapshot, Arc::new(SequentialExecutor));
    match interp.execute().await {
        Err(HostError::Trapped { message }) => message,
        Ok(value) => panic!("`{source}` does not trap, it runs to {value:?}"),
        Err(other) => panic!("`{source}` does not trap, it ends with {other:?}"),
    }
}

/// The trap both spellings end with.
async fn trapped_alike<C>(i: &Interner, case: Agreement, ret: Ty, context: C) -> String
where
    C: Fn(&Interner) -> Context,
{
    let folded = trap_of(i, case.folded, context(i), ret.clone()).await;
    let through = trap_of(i, case.through_the_page, context(i), ret).await;
    assert_eq!(
        folded, through,
        "`{}` and `{}` disagree",
        case.folded, case.through_the_page
    );
    folded
}

#[tokio::test]
async fn an_addition_past_the_width_traps_the_same_way_folded() {
    let i = Interner::new();
    let byte = trapped_alike(
        &i,
        Agreement {
            folded: "250 + 10",
            through_the_page: "@a + @b",
        },
        Ty::Int(IntTy::U8),
        |i| integers(i, IntTy::U8, &[("a", 250), ("b", 10)]),
    )
    .await;
    assert_eq!(byte, ADD_OVERFLOW);

    let widest = trapped_alike(
        &i,
        Agreement {
            folded: "9223372036854775807 + 1",
            through_the_page: "@a + @b",
        },
        Ty::I64,
        |i| integers(i, IntTy::I64, &[("a", i64::MAX as u64), ("b", 1)]),
    )
    .await;
    assert_eq!(widest, ADD_OVERFLOW);
}

#[tokio::test]
async fn an_addition_up_to_the_width_is_the_same_word_folded() {
    let i = Interner::new();
    let byte = agreed(
        &i,
        Agreement {
            folded: "250 + 5",
            through_the_page: "@a + @b",
        },
        Ty::Int(IntTy::U8),
        |i| integers(i, IntTy::U8, &[("a", 250), ("b", 5)]),
    )
    .await;
    assert_eq!(IntTy::U8.read(byte), 255);

    let widest = agreed(
        &i,
        Agreement {
            folded: "9223372036854775806 + 1",
            through_the_page: "@a + @b",
        },
        Ty::I64,
        |i| integers(i, IntTy::I64, &[("a", (i64::MAX - 1) as u64), ("b", 1)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(widest), i128::from(i64::MAX));
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
async fn two_constants_joined_across_one_add_trap_the_same() {
    let i = Interner::new();
    let byte = trapped_alike(
        &i,
        Agreement {
            folded: "@a + 1 + 2",
            through_the_page: "@a + @b",
        },
        Ty::Int(IntTy::U8),
        |i| integers(i, IntTy::U8, &[("a", 254), ("b", 3)]),
    )
    .await;
    assert_eq!(byte, ADD_OVERFLOW);
}

#[tokio::test]
async fn two_constants_joined_across_one_add_reach_the_width_the_same() {
    let i = Interner::new();
    let byte = agreed(
        &i,
        Agreement {
            folded: "@a + 1 + 2",
            through_the_page: "@a + @b",
        },
        Ty::Int(IntTy::U8),
        |i| integers(i, IntTy::U8, &[("a", 252), ("b", 3)]),
    )
    .await;
    assert_eq!(IntTy::U8.read(byte), 255);
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

// -- `as` (RFC-0049) --------------------------------------------------
//
// Cross-artifact obligation: every expected value below is the value of
// the Rust `as` expression written beside it, which is what RFC-0049 rules
// both `optimize::fold`'s `cast_result` and `ops::cast`'s instance for the
// pair produce. The two spellings run the same edge with the cast's source
// constant and with it arriving through the page.

#[tokio::test]
async fn an_integer_cast_truncates_and_extends_by_width_the_same_way_folded() {
    let i = Interner::new();
    let widest = agreed(
        &i,
        Agreement {
            folded: "(0 - 1) as u64",
            through_the_page: "@a as u64",
        },
        Ty::Int(IntTy::U64),
        |i| integers(i, IntTy::I64, &[("a", (-1i64) as u64)]),
    )
    .await;
    assert_eq!(IntTy::U64.read(widest), i128::from(-1i64 as u64));

    let byte = agreed(
        &i,
        Agreement {
            folded: "300 as u8",
            through_the_page: "@a as u8",
        },
        Ty::Int(IntTy::U8),
        |i| integers(i, IntTy::I64, &[("a", 300)]),
    )
    .await;
    assert_eq!(IntTy::U8.read(byte), i128::from(300i64 as u8));

    let signed = agreed(
        &i,
        Agreement {
            folded: "200 as u8 as i8",
            through_the_page: "@a as u8 as i8",
        },
        Ty::Int(IntTy::I8),
        |i| integers(i, IntTy::I64, &[("a", 200)]),
    )
    .await;
    assert_eq!(IntTy::I8.read(signed), i128::from(200u8 as i8));

    let extended = agreed(
        &i,
        Agreement {
            folded: "(0 - 1) as i8 as i64",
            through_the_page: "@a as i64",
        },
        Ty::I64,
        |i| integers(i, IntTy::I8, &[("a", (-1i8) as u64)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(extended), i128::from(-1i8 as i64));
}

#[tokio::test]
async fn an_integer_to_a_float_rounds_the_same_way_folded() {
    let i = Interner::new();
    let widest = agreed(
        &i,
        Agreement {
            folded: "(0 - 1) as u64 as f64",
            through_the_page: "@a as u64 as f64",
        },
        Ty::Float,
        |i| integers(i, IntTy::I64, &[("a", (-1i64) as u64)]),
    )
    .await;
    assert_eq!(f64::from_bits(widest), u64::MAX as f64);

    let least = agreed(
        &i,
        Agreement {
            folded: "(0 - 9223372036854775807 - 1) as f64",
            through_the_page: "@a as f64",
        },
        Ty::Float,
        |i| integers(i, IntTy::I64, &[("a", i64::MIN as u64)]),
    )
    .await;
    assert_eq!(f64::from_bits(least), i64::MIN as f64);
}

#[tokio::test]
async fn a_float_to_an_integer_saturates_the_same_way_folded() {
    let i = Interner::new();
    let past = agreed(
        &i,
        Agreement {
            folded: "1000000000000000000000000000000.0 as i64",
            through_the_page: "@x as i64",
        },
        Ty::I64,
        |i| floats(i, &[("x", 1e30)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(past), i128::from(1e30f64 as i64));

    let below = agreed(
        &i,
        Agreement {
            folded: "(0.0 - 1.0) as u8",
            through_the_page: "@x as u8",
        },
        Ty::Int(IntTy::U8),
        |i| floats(i, &[("x", -1.0)]),
    )
    .await;
    assert_eq!(IntTy::U8.read(below), i128::from(-1.0f64 as u8));

    let toward_zero = agreed(
        &i,
        Agreement {
            folded: "(0.0 - 1.9) as i64",
            through_the_page: "@x as i64",
        },
        Ty::I64,
        |i| floats(i, &[("x", -1.9)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(toward_zero), i128::from(-1.9f64 as i64));
}

/// A NaN is never a constant a body holds — `float_result` refuses to fold
/// one (RFC-0055) — so this edge reaches the machine by both spellings and
/// the pair states that they still agree.
#[tokio::test]
async fn a_nan_cast_to_an_integer_is_zero() {
    let i = Interner::new();
    let held = agreed(
        &i,
        Agreement {
            folded: "(0.0 / 0.0) as i64",
            through_the_page: "(@z / @z) as i64",
        },
        Ty::I64,
        |i| floats(i, &[("z", 0.0)]),
    )
    .await;
    assert_eq!(IntTy::I64.read(held), i128::from(f64::NAN as i64));
}
