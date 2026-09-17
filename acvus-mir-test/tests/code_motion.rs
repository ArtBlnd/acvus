//! Intent tests for `optimize::code_motion`, whose rule is stated in
//! RFC-0007 and whose raising operations are RFC-0037's.

use acvus_mir::ty::{LenTerm, Ty};
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

// -- A shared borrow of a storage -----------------------------------

fn n_ctx(i: &Interner) -> FxHashMap<Astr, Ty> {
    ctx(&i, &[("n", Ty::I64)])
}

/// The loop only reads `v`, so its borrow is built once, above the header.
#[test]
fn a_borrow_read_by_a_loop_leaves_it() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); \
         let s = 0; let i = 0; \
         while i < @n { s = s + *v.get(0); i = i + 1; } \
         s",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "ref &v") < at(body, "jump_if"), "{ir}");
}

/// The same loop, one `&mut` call added: the write is in the region the
/// borrow would newly span, so it stays in the body.
#[test]
fn a_borrow_stays_where_the_loop_writes_the_storage() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); \
         let s = 0; let i = 0; \
         while i < @n { s = s + *v.get(0); v.push_back(s); i = i + 1; } \
         s",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "jump_if") < at(body, "ref &v"), "{ir}");
}

/// A context assigned in the loop is written there like any other storage.
///
/// Two rules refuse this one, and the write condition is the second: an
/// `Assign` also *defines* its storage (`inst_info::defs`), so the storage
/// is an operand of the borrow that is not available above the loop. The
/// isolating test of the write condition is the one above it, where
/// `push_back` writes `v` without defining it.
#[test]
fn a_borrow_of_a_context_the_loop_assigns_stays() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let s = 0.0; let i = 0; \
         while i < @n { s = s + *@x.get(0); @x = [s, s]; i = i + 1; } \
         s",
        &ctx(
            &i,
            &[
                ("n", Ty::I64),
                ("x", Ty::Array(Box::new(Ty::Float), LenTerm::Known(2))),
            ],
        ),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "jump_if") < at(body, "ref &@x"), "{ir}");
}

/// A borrow through a reference is a memory op: it stays in order with the
/// other ops through that reference, wherever they are.
#[test]
fn a_borrow_through_a_reference_never_moves() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); \
         let r = &v; \
         let s = 0; let i = 0; \
         while i < @n { s = s + *r.get(0); i = i + 1; } \
         s",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "jump_if") < at(body, "ref &(*"), "{ir}");
}

/// Neither loop writes `v`, so the borrow leaves both.
#[test]
fn a_borrow_leaves_both_of_two_nested_loops() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); \
         let s = 0; let t = 0; \
         while t < @n { let j = 0; while j < @n { s = s + *v.get(0); j = j + 1; } t = t + 1; } \
         s",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "ref &v") < at(body, "jump_if"), "{ir}");
}
