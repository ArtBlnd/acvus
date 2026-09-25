//! The script is the attention kernel's loop shape, which
//! `acvus-interpreter-test/tests/attention_loop_shape.rs` measures at the
//! interpreter's contract. Both must move together: this file checks at the
//! listing what that one checks at the prepared code.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

const NESTED: &str = "\
let d = @n; \
let k = d + 3; \
let total = 0; \
let t = 0; \
while t < d { \
    let i = 0; \
    while i < d { \
        total = total + k; \
        i = i + 1; \
    } \
    t = t + 1; \
} \
total";

fn n_context(i: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(i.intern("n"), Ty::I64)])
}

/// Every `Lk(r1: i64, r2: i64):` of the listing as `("Lk", 2)`, in order.
fn block_params(listing: &str) -> Vec<(String, usize)> {
    listing
        .lines()
        .filter_map(|line| line.split_once("| L"))
        .filter_map(|(_, rest)| rest.split_once('('))
        .filter_map(|(label, params)| {
            let params = params.strip_suffix("):")?;
            let count = match params {
                "" => 0,
                list => list.split(',').count(),
            };
            Some((format!("L{label}"), count))
        })
        .collect()
}

#[test]
fn a_bound_and_a_factor_no_loop_assigns_are_not_block_params() {
    let i = Interner::new();
    let listing = compile_script_optimized(&i, NESTED, &n_context(&i)).expect("it compiles");
    assert_eq!(
        block_params(&listing),
        [
            ("L0".to_string(), 1),
            ("L1".to_string(), 1),
            ("L5".to_string(), 1)
        ],
        "the outer header carries the accumulator, its body the counter its \
         `for` fills, and the inner loop's exit the trip count, nothing else. \
         Both `while`s are range `for`s (RFC-0081), nothing reads `t` or `i` \
         once the comparisons are gone, and the accumulator is an induction \
         variable of the inner loop, computed at its exit from the trip \
         count (RFC-0066 rule 7). Its step by `@n + 3` can overflow; the \
         inner loop then does nothing but that step's `Check`, and is a jump \
         to its exit with the count computed above it and the check closed \
         there (RFC-0088 rules 4 and 8):\n{listing}"
    );
}
