//! Intent tests for `optimize::reborrow`: a reference to the whole of what
//! a reference names is that reference.
//!
//! The first source is the `match` over a place that the `bf table` and
//! `bf scan` benches run in `acvus-interpreter-test/benches/programs.rs`;
//! move it and the listing here stops describing what those benches
//! measure.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_mode_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

fn optimized(source: &str, entries: &[(&str, Ty)]) -> String {
    let i = Interner::new();
    compile_script_mode_optimized(&i, source, &ctx(&i, entries)).unwrap()
}

const MATCH_OVER_A_PLACE: &str = "\
let prog = [1, 2, 3,]; \
let plen = len(&prog); let one = plen / plen; \
let acc = 0; let pc = plen - plen; \
while pc < plen { \
match &prog[pc] { 1 => { acc = acc + 1; }, 2 => { acc = acc - 1; }, _ => { acc = acc; } }; \
pc = pc + one; \
} acc";

#[test]
fn a_match_over_a_place_holds_no_reborrow() {
    let listing = optimized(MATCH_OVER_A_PLACE, &[]);
    assert!(
        !listing.contains("ref &(*"),
        "the reborrow stands: {listing}"
    );
    insta::assert_snapshot!("match_over_a_place", listing);
}
