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
#[test]
fn two_constants_under_one_add_are_one_constant() {
    let listing = optimized(
        "let p = { x: 1, y: 2, }; let acc = 0; let i = 0; \
         while i < @n { acc = acc + p.x + p.y; i = i + 1; } acc",
        signed_n,
    );
    assert!(listing.contains("+ 3 ("), "no folded add: {listing}");
    assert!(!listing.contains("+ 1 (") || !listing.contains("+ 2 ("));
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
