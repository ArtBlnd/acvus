//! An aggregate that does not escape never exists (RFC-0053).
//!
//! The sources are the `shapes` bench cases
//! (`acvus-interpreter-test/benches/shapes.rs`), so the listings below and
//! the numbers that bench prints are the same programs. `option match` and
//! `vec of objects` need externs the MIR test harness does not register and
//! are measured by the bench alone; what they show -- an option is flat and
//! a collected object escapes -- the seven listings here do not contradict.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn n_is_an_int(i: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(i.intern("n"), Ty::I64)])
}

fn optimized(source: &str) -> String {
    let i = Interner::new();
    compile_script_optimized(&i, source, &n_is_an_int(&i)).unwrap()
}

fn nothing_aggregate(listing: &str) {
    for gone in ["object ", "variant ", "take ", "assign ", "drop ", "ref &"] {
        assert!(
            !listing.contains(gone),
            "`{gone}` survived scalar replacement: {listing}"
        );
    }
}

#[test]
fn constructing_an_object_per_iteration_is_the_arithmetic_it_holds() {
    let listing = optimized(
        "let acc = 0; let i = 0; while i < @n { let q = { x: i, y: i + 1, }; \
         acc = acc + q.x; i = i + 1; } acc",
    );
    nothing_aggregate(&listing);
    insta::assert_snapshot!("construct@optimized", listing);
}

#[test]
fn reading_two_fields_of_a_loop_invariant_object_reads_no_storage() {
    let listing = optimized(
        "let p = { x: 1, y: 2, }; let acc = 0; let i = 0; \
         while i < @n { acc = acc + p.x + p.y; i = i + 1; } acc",
    );
    nothing_aggregate(&listing);
    insta::assert_snapshot!("field_read@optimized", listing);
}

#[test]
fn a_field_written_every_iteration_is_a_loop_carried_register() {
    let listing = optimized(
        "let p = { x: 0, y: 0, }; let i = 0; while i < @n { p.x = p.x + i; i = i + 1; } p.x",
    );
    nothing_aggregate(&listing);
    insta::assert_snapshot!("field_write@optimized", listing);
}

#[test]
fn a_field_written_under_a_branch_gets_one_phi_and_its_sibling_none() {
    let listing = optimized(
        "let p = { x: 0, y: 7, }; let i = 0; \
         while i < @n { if let true = i % 2 == 0 { p.x = p.x + i; }; i = i + 1; } p.x + p.y",
    );
    nothing_aggregate(&listing);
    insta::assert_snapshot!("field_under_if@optimized", listing);
}

#[test]
fn a_variant_that_stays_in_the_body_is_a_tag_and_a_payload() {
    let listing = optimized(
        "let acc = 0; let i = 0; while i < @n { \
         let e = if i % 2 == 0 { E::A(i) } else { E::B(i + 1) }; \
         match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v; } }; \
         i = i + 1; } acc",
    );
    nothing_aggregate(&listing);
    assert!(
        !listing.contains(" is A") && !listing.contains(" is B"),
        "the tag is a register, not a field of a value: {listing}"
    );
    insta::assert_snapshot!("enum_match@optimized", listing);
}

#[test]
fn a_three_armed_match_is_threaded_the_same_way_as_a_two_armed_one() {
    let listing = optimized(
        "let acc = 0; let i = 0; while i < @n { \
         let e = match i % 3 { 0 => E::A(i), 1 => E::B(i + 1), _ => E::C(i + 2) }; \
         match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v; }, \
         E::C(v) => { acc = acc + v; } }; \
         i = i + 1; } acc",
    );
    nothing_aggregate(&listing);
    assert!(
        !["A -> ", "B -> ", "C -> "]
            .iter()
            .any(|tag| listing.contains(tag)),
        "a three-armed dispatch over a replaced slot leaves no `Switch` over its \
         tags: {listing}"
    );
    assert_eq!(
        listing.matches("switch ").count(),
        1,
        "the one dispatch left is the `match i % 3` that chose the constructor, \
         which is one `Switch` over its literal keys (RFC-0051): {listing}"
    );
    insta::assert_snapshot!("enum_match_three@optimized", listing);
}

/// An object that leaves the body is left alone. Scalar replacement will
/// never reach this shape, whatever it learns: RFC-0050's layout is the
/// lever on an object that genuinely lives past the body that built it.
#[test]
fn an_object_the_body_returns_is_still_built() {
    let listing = optimized("let o = { x: @n, y: @n + 1, }; o");
    assert!(
        listing.contains("object "),
        "an object that escapes its body is built: {listing}"
    );
    insta::assert_snapshot!("returned_object@optimized", listing);
}
