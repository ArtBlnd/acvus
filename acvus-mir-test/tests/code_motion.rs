//! Intent tests for `optimize::code_motion`, whose rule is stated in
//! RFC-0007 and whose raising operations are RFC-0037's.

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

fn main_body(ir: &str) -> &str {
    ir.split("=== main ===")
        .nth(1)
        .unwrap()
        .split("=== ")
        .next()
        .unwrap()
}

fn at(body: &str, needle: &str) -> usize {
    body.find(needle)
        .unwrap_or_else(|| panic!("no `{needle}` in:\n{body}"))
}

#[test]
fn an_addition_in_a_loop_body_stays_below_the_test() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let acc = 0; let i = 0; while i < @n { acc = acc + i; i = i + 1; } acc",
        &ctx(&i, &[("n", Ty::I64)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "jump_if") < at(body, " + "), "{ir}");
}

#[test]
fn an_addition_used_only_in_a_then_block_stays_in_it() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let x = 0; if @c { x = @a + 1; }; x",
        &ctx(&i, &[("a", Ty::I64), ("c", Ty::Bool)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "jump_if") < at(body, " + "), "{ir}");
}

/// The merge post-dominates the branch, so the two execute under the same
/// condition and the addition may rise past it.
#[test]
fn an_addition_after_a_merge_rises_above_the_branch() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let x = 0; if @c { x = 1; } else { x = 2; }; @n = x; @a + 7",
        &ctx(&i, &[("a", Ty::I64), ("c", Ty::Bool), ("n", Ty::I64)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, " + ") < at(body, "jump_if"), "{ir}");
}
