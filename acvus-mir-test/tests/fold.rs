//! A constant expression folds (RFC-0055).
//!
//! The sources are loop bodies the `accum` and `shapes` benches run, in
//! `acvus-interpreter-test/benches/{accum,shapes}.rs`. Move one and the
//! listing here stops describing the program those benches measure.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn signed_n(i: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(i.intern("n"), Ty::I64)])
}

fn optimized(source: &str, context: fn(&Interner) -> FxHashMap<Astr, Ty>) -> String {
    let i = Interner::new();
    compile_script_optimized(&i, source, &context(&i)).unwrap()
}

/// The `shapes` bench's `field read`, whose two constants scalar
/// replacement produces from the object's fields.
///
/// IV canonicalization computes `acc`, an induction variable of the promoted
/// `for`, after the loop as its step times the trip count, so the folded
/// constant is found as that step, in the wrapping `*%` the pass writes.
/// The loop stays, holding the `Check` of `acc`'s step over `@n`
/// iterations. This test looks where RFC-0066 rule 7 writes it, and moves
/// with that pass.
#[test]
fn two_constants_under_one_add_are_one_constant() {
    let listing = optimized(
        "let p = { x: 1, y: 2, }; let acc = 0; let i = 0; \
         while i < @n { acc = acc + p.x + p.y; i = i + 1; } acc",
        signed_n,
    );
    assert!(listing.contains("*% 3 ("), "no folded step: {listing}");
    assert!(
        !listing.contains(" 1 (") && !listing.contains(" 2 ("),
        "an addend stood beside the folded constant: {listing}"
    );
    insta::assert_snapshot!("field_read@folded", listing);
}

/// The `accum` bench's `collatz while`.
///
/// Reducing `i % 2 == 0` to `(i & 1) == 0` is not missing here; it was
/// built, measured and taken back out. `prepare::arith_of` claims the
/// five arithmetic operators, so the mask splits the pair the machine ran
/// as one fused `Chain2<i64>` into two dispatches, and one dispatch costs
/// more than the division it removes: `enum match` was 19 % slower and
/// `collatz` 2.3 % slower with the reduction in. It is re-admitted when
/// the chain alphabet holds the bitwise and shift operators, and those
/// numbers are its test (RFC-0055, Rejected).
#[test]
fn a_remainder_and_a_division_by_a_power_of_two_stand() {
    let listing = optimized(
        "let i = 0; let acc = 0; \
         while i < @n { let d = if i % 2 == 0 { i / 2 } else { i * 3 + 1 }; \
         acc = acc + d; i = i + 1; } acc",
        signed_n,
    );
    assert!(listing.contains("% 2 ("), "the remainder moved: {listing}");
    assert!(listing.contains("/ 2 ("), "the division moved: {listing}");
    assert!(!listing.contains(" & "), "a mask appeared: {listing}");
    insta::assert_snapshot!("collatz@folded", listing);
}

/// A cast of a constant is the constant (RFC-0049), so no `as` survives
/// the pass where its source is one. The values are the ones
/// `acvus-interpreter-test/tests/fold_agreement.rs` runs against the
/// machine; what this states is that the folded spelling is folded, which
/// that test cannot see.
#[test]
fn a_cast_of_a_constant_leaves_no_cast() {
    let widest = optimized("(0 - 1) as u64 as f64", signed_n);
    assert!(
        !widest.contains(" as "),
        "a cast of a constant stood: {widest}"
    );
    assert!(
        widest.contains("1.8446744073709552e19"),
        "the folded value is not `u64::MAX as f64`: {widest}"
    );

    let narrowed = optimized("300 as u8", signed_n);
    assert!(
        !narrowed.contains(" as "),
        "a cast of a constant stood: {narrowed}"
    );
    assert!(
        narrowed.contains("44"),
        "the folded value is not 44: {narrowed}"
    );
}

/// A cast whose source is not a constant stands, and a `NaN` is never a
/// constant a body holds (RFC-0055), so `(0.0 / 0.0) as i64` keeps both
/// its division and its cast.
#[test]
fn a_cast_of_a_value_the_pass_cannot_read_stands() {
    let listing = optimized("let x = @n as f64; (x / 0.0) as i64", signed_n);
    assert!(listing.contains(" as f64"), "the cast moved: {listing}");
    assert!(listing.contains(" as i64"), "the cast moved: {listing}");
}

fn x_y_and_c(i: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([
        (i.intern("x"), Ty::I64),
        (i.intern("y"), Ty::Int(acvus_mir::ty::IntTy::I8)),
        (i.intern("c"), Ty::Bool),
    ])
}

/// Two trapping constants join only where the joined operation traps on
/// the same `x` as the two did (RFC-0037 rule 3).
#[test]
fn a_trapping_join_keeps_the_trap_it_joins() {
    let one_sign = optimized("(@x + 1) + 2", x_y_and_c);
    assert!(one_sign.contains(" + 3 ("), "{one_sign}");

    let across_signs = optimized("(@x + 3) + -4", x_y_and_c);
    assert!(
        across_signs.contains(" + 3 (") && across_signs.contains(" + -4 ("),
        "`x + -1` does not trap at `x = i64::MAX - 2`, where `x + 3` does: \
         {across_signs}"
    );

    let through_minus_one = optimized("((@y * -1i8) * -1i8) as i64", x_y_and_c);
    assert_eq!(
        through_minus_one.matches(" * -1 (").count(),
        2,
        "`y * 1` does not trap at `y = -128`, where `y * -1` does: {through_minus_one}"
    );

    let across_blocks = optimized("let t = @x + 1; if @c { t + 2 } else { 0 }", x_y_and_c);
    assert!(
        across_blocks.contains(" + 1 (") && across_blocks.contains(" + 2 ("),
        "`x + 1` runs on the path that skips the arm too: {across_blocks}"
    );
}
