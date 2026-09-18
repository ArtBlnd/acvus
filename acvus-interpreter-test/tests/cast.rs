//! `expr as T` at the contract it arrives at: the value a script produces
//! (RFC-0049). Where the intent is that a cast in a chain is a *leaf* and
//! not a node, no value can show it, so those tests read the prepared code
//! as well.
//!
//! Cross-artifact obligation: every expected value here is the value of
//! the Rust `as` expression written beside it. RFC-0049 rules that the two
//! are one thing, and `acvus-mir`'s `optimize::fold` owes the same values
//! wherever it can read the source as a constant — `tests/fold_agreement.rs`
//! runs both spellings of each edge against each other.

use acvus_interpreter::code::Shape;
use acvus_interpreter::{ChainTy, LeafRead, Value};
use acvus_interpreter_test::listing::{
    ChainShape, chains_of, ops_of, prepared_script, script_listing,
};
use acvus_interpreter_test::*;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn at(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    run_script(&i, source, FxHashMap::default(), ret).await
}

async fn int_at(source: &str, k: IntTy) -> i128 {
    k.read(at(source, Ty::Int(k)).await.bits())
}

async fn float(source: &str) -> f64 {
    at(source, Ty::Float).await.as_float()
}

// -- The values (RFC-0049 rule 2) -------------------------------------

#[tokio::test]
async fn an_integer_cast_truncates_and_sign_extends_by_width() {
    assert_eq!(
        int_at("(0 - 1) as u64", IntTy::U64).await,
        i128::from(-1i64 as u64)
    );
    assert_eq!(
        int_at("300 as u8", IntTy::U8).await,
        i128::from(300i64 as u8)
    );
    assert_eq!(
        int_at("200 as u8 as i8", IntTy::I8).await,
        i128::from(200u8 as i8)
    );
    assert_eq!(
        int_at("(0 - 1) as i8 as i64", IntTy::I64).await,
        i128::from(-1i8 as i64)
    );
    assert_eq!(int_at("7 as i64", IntTy::I64).await, i128::from(7i64));
}

#[tokio::test]
async fn an_integer_to_a_float_rounds_to_nearest() {
    assert_eq!(float("(0 - 1) as u64 as f64").await, u64::MAX as f64);
    assert_eq!(
        float("(0 - 9223372036854775807 - 1) as f64").await,
        i64::MIN as f64
    );
    assert_eq!(float("1 as f64").await, 1i64 as f64);
}

#[tokio::test]
async fn a_float_to_an_integer_saturates_and_maps_nan_to_zero() {
    assert_eq!(
        int_at("1000000000000000000000000000000.0 as i64", IntTy::I64).await,
        i128::from(1e30f64 as i64)
    );
    assert_eq!(
        int_at(
            "(0.0 - 1000000000000000000000000000000.0) as i64",
            IntTy::I64
        )
        .await,
        i128::from(-1e30f64 as i64)
    );
    assert_eq!(
        int_at("(0.0 / 0.0) as i64", IntTy::I64).await,
        i128::from(f64::NAN as i64)
    );
    assert_eq!(
        int_at("(0.0 - 1.0) as u8", IntTy::U8).await,
        i128::from(-1.0f64 as u8)
    );
    assert_eq!(
        int_at("300.5 as u8", IntTy::U8).await,
        i128::from(300.5f64 as u8)
    );
    assert_eq!(
        int_at("(0.0 - 1.9) as i64", IntTy::I64).await,
        i128::from(-1.9f64 as i64)
    );
    assert_eq!(float("1.5 as f64").await, 1.5f64);
}

#[tokio::test]
async fn a_cast_the_fold_cannot_read_runs_as_an_operation() {
    let i = Interner::new();
    let widest = || -> Context {
        [(
            i.intern("a"),
            typed(Ty::Int(IntTy::U64), Value::from_bits(IntTy::U64, u64::MAX)),
        )]
        .into_iter()
        .collect()
    };

    let held = run_script(&i, "@a as f64", widest(), Ty::Float).await;
    assert_eq!(held.as_float(), u64::MAX as f64);

    let narrowed = run_script(&i, "@a as i32", widest(), Ty::Int(IntTy::I32)).await;
    assert_eq!(
        IntTy::I32.read(narrowed.bits()),
        i128::from(u64::MAX as i32)
    );
}

// -- `as` binds where Rust binds it (RFC-0049 rule 1) -----------------

#[tokio::test]
async fn a_cast_binds_tighter_than_a_binary_operator_and_looser_than_a_unary_one() {
    assert_eq!(
        float("1.0 + 1 as f64").await,
        1.0f64 + 1i64 as f64,
        "`1.0 + 1 as f64` is `1.0 + (1 as f64)`"
    );
    assert_eq!(
        int_at("(0 - 300) as u8", IntTy::U8).await,
        i128::from(-300i64 as u8),
        "the subtraction runs before the cast reads it"
    );
}

// -- A cast is a leaf (RFC-0049 rule 3) --------------------------------

fn chains(i: &Interner, source: &str, context: Context, ret: Ty) -> Vec<ChainShape> {
    let prepared = prepared_script(i, source, context, ret);
    chains_of(prepared.main.as_ref())
}

/// The `accum` bench's `float while` body, character for character;
/// `benches/accum.rs` holds the other copy and times it. Before RFC-0049
/// the conversion was `i.to_float()`, one `CallExtern1` per iteration that
/// the recognizer could not see through.
#[tokio::test]
async fn a_cast_feeding_an_operator_is_that_operator_s_leaf() {
    let i = Interner::new();
    let context = || -> Context {
        [(i.intern("n"), typed(Ty::I64, Value::int(4)))]
            .into_iter()
            .collect()
    };
    let source = "let acc = 0.0; let i = 0; while i < @n { acc = acc + i as f64; i = i + 1; } acc";

    let blocks = script_listing(&i, source, context(), Ty::Float);
    let found = ops_of(&blocks);
    assert!(
        found.iter().all(|name| !name.contains("Cast")),
        "a cast stood beside the chain that should hold it: {found:?}"
    );

    let held = chains(&i, source, context(), Ty::Float);
    let float_add = held
        .iter()
        .find(|chain| chain.reads.iter().any(|read| *read != LeafRead::Own))
        .expect("one chain reads a cast leaf");
    assert_eq!(
        float_add.shape,
        Shape::NLL,
        "the cast became a leaf, so the chain is still one node"
    );
    assert_eq!(
        float_add.reads,
        vec![LeafRead::Own, LeafRead::Cast(ChainTy::Int(IntTy::I64))],
        "the left leaf is the accumulator and the right is `i as f64`"
    );
}
