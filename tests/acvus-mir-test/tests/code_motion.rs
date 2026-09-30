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

const IF_TERMINATOR: &str = " if ";
const FOR_TERMINATOR: &str = "for range";

fn at(body: &str, needle: &str) -> usize {
    body.find(needle)
        .unwrap_or_else(|| panic!("no `{needle}` in:\n{body}"))
}

fn count(body: &str, needle: &str) -> usize {
    body.matches(needle).count()
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
    assert!(at(body, FOR_TERMINATOR) < at(body, " + "), "{ir}");
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
    assert!(at(body, IF_TERMINATOR) < at(body, " + "), "{ir}");
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
    assert!(at(body, " + ") < at(body, IF_TERMINATOR), "{ir}");
}

/// A division can trap, so it moves only onto exactly the paths it ran on,
/// and it is not ordered with effects (RFC-0048 rule 8): written after the
/// branch that pushes, it rises into the entry, above the branch and the
/// push in it.
#[test]
fn a_division_after_a_merge_rises_above_an_effect_in_the_branch() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let a = @a; let b = @b; let v = vec([1]); if @c { v.push(2); }; \
         a / b + v.len() as i64",
        &ctx(&i, &[("a", Ty::I64), ("b", Ty::I64), ("c", Ty::Bool)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, " / ") < at(body, IF_TERMINATOR), "{ir}");
    assert!(at(body, " / ") < at(body, "&mut v"), "{ir}");
}

/// The same division inside the branch runs only where the branch is taken,
/// so it stays below the test.
#[test]
fn a_division_behind_a_branch_stays_behind_it() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let a = @a; let b = @b; let x = 0; if @c { x = a / b; }; x",
        &ctx(&i, &[("a", Ty::I64), ("b", Ty::I64), ("c", Ty::Bool)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, IF_TERMINATOR) < at(body, " / "), "{ir}");
}

// -- Loop depth -----------------------------------------------------

fn loop_tests_before(body: &str, needle: &str) -> usize {
    count(&body[..at(body, needle)], FOR_TERMINATOR)
}

/// The exit post-dominates the header, so post-dominance alone would take
/// this multiplication into the header and run it once per iteration.
///
/// The loop sums `s` so that it stays a loop: a body left doing nothing is
/// removed (RFC-0088), and there would be no header to stay out of.
#[test]
fn a_multiplication_after_a_loop_stays_after_it() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let i = 0; let s = 0; while i < @n { s = s + i; i = i + 1; } i * 2 + s",
        &ctx(&i, &[("n", Ty::I64)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, FOR_TERMINATOR) < at(body, " * "), "{ir}");
}

/// The same instruction one level in: written between the inner loop and
/// the outer one, it belongs to the outer loop's body and not to the inner
/// loop's head. The inner loop sums `s` for the reason the loop above does.
#[test]
fn a_multiplication_between_two_loops_stays_between_them() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let total = 1; let t = 0; \
         while t < @n { let j = 0; let s = 0; while j < @n { s = s + j; j = j + 1; } \
         total = total * j + s; t = t + 1; } \
         total",
        &ctx(&i, &[("n", Ty::I64)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(loop_tests_before(body, " * "), 2, "{ir}");
}

/// The header post-dominates the entry and is no deeper than it, so a
/// multiplication the condition rebuilds every iteration still leaves the
/// loop entirely. Tested with `<=` so that it stays a `while`: RFC-0081
/// turns `i < @n * 2` into a range `for` and writes the multiplication
/// above the header itself.
#[test]
fn a_loop_invariant_multiplication_in_the_header_rises_to_the_entry() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let s = 0; let i = 0; while i <= @n * 2 { s = s + i; i = i + 1; } s",
        &ctx(&i, &[("n", Ty::I64)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, " * ") < at(body, "jump_if"), "{ir}");
}

/// `pop_front` writes `v` in the header, which is what stops the borrow
/// above the loop; the depth clause is what stops it inside.
#[test]
fn a_borrow_after_a_loop_the_header_writes_stays_after_the_loop() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); v.push_back(2); \
         let s = 0; \
         while let Some(x) = pop_front(&mut v) { s = s + x; } \
         s + len(&v)",
        &ctx(&i, &[("n", Ty::I64)]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, "jump_if") < at(body, "ref &v"), "{ir}");
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
    assert!(at(body, "ref &v") < at(body, FOR_TERMINATOR), "{ir}");
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
    assert!(at(body, FOR_TERMINATOR) < at(body, "ref &v"), "{ir}");
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
         while i < @n { s = s + @x[0]; @x = [s, s]; i = i + 1; } \
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
    assert!(at(body, FOR_TERMINATOR) < at(body, "ref &@x"), "{ir}");
}

/// A borrow through a reference is a memory op: it stays in order with the
/// other ops through that reference, wherever they are. The borrow is one
/// field deep because `optimize::reborrow` folds a borrow of the *whole*
/// of what a reference names back into that reference, leaving no op here
/// to hold in place.
#[test]
fn a_borrow_through_a_reference_never_moves() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = { d: deque(), }; v.d.push_back(1); \
         let r = &v; \
         let s = 0; let i = 0; \
         while i < @n { s = s + *r.d.get(0); i = i + 1; } \
         s",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert!(at(body, FOR_TERMINATOR) < at(body, "ref &(*"), "{ir}");
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
    assert!(at(body, "ref &v") < at(body, FOR_TERMINATOR), "{ir}");
}

// -- A second shared borrow in one block ----------------------------

/// Two reads of one storage in one block take one borrow between them.
#[test]
fn a_second_borrow_in_a_block_is_the_first() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); \
         let a = *v.get(0); let b = *v.get(0); \
         a + b",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "ref &v"), 1, "{ir}");
}

/// The same block with a `&mut` call between them: `push_back` carries a
/// `Mut` loan of `v`, so the first borrow does not survive it.
#[test]
fn a_write_between_two_borrows_keeps_them_two() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); \
         let a = *v.get(0); v.push_back(2); let b = *v.get(0); \
         a + b",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "ref &v"), 2, "{ir}");
}

/// A context is a storage like any other, and the assignment writes it.
#[test]
fn an_assignment_between_two_borrows_of_a_context_keeps_them_two() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let a = @x[0]; @x = [a, a]; let b = @x[0]; a + b",
        &ctx(
            &i,
            &[("x", Ty::Array(Box::new(Ty::Float), LenTerm::Known(2)))],
        ),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "ref &@x"), 2, "{ir}");
}

/// An exclusive borrow is never merged: each `&mut` is a loan of its own,
/// and two of them may not be one value held across both calls.
#[test]
fn two_exclusive_borrows_are_two() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); v.push_back(2); *v.get(0)",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "ref &mut v"), 2, "{ir}");
}

/// A borrow through a reference is a memory op, not a name for a storage:
/// two of them stay two, wherever they stand. The borrow is one field deep
/// for the reason `a_borrow_through_a_reference_never_moves` gives.
#[test]
fn two_borrows_through_a_reference_are_two() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = { d: deque(), }; v.d.push_back(1); \
         let r = &v; \
         let a = *r.d.get(0); let b = *r.d.get(0); \
         a + b",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "ref &(*"), 2, "{ir}");
}

/// The merge reads one block and no more. The loop writes `v`, so neither
/// the body's borrow nor the tail's rises out of where it stands, and the
/// two blocks keep a borrow each. A pair the hoist *can* bring into one
/// block is the case above; across blocks the hoist decides, or nobody.
#[test]
fn two_borrows_in_two_blocks_stay_two() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let v = deque(); v.push_back(1); \
         let s = 0; let i = 0; \
         while i < @n { s = s + *v.get(0); v.push_back(s); i = i + 1; } \
         s + *v.get(0)",
        &n_ctx(&i),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "ref &v"), 2, "{ir}");
}

// -- What a borrow is of ---------------------------------------------

const TWO_ROWS: &str = "let m = vec([vec([1, 2]), vec([3, 4])]); \
     let z = len(&m) - len(&m); let one = z + 1u64; ";

/// `m[z]` and `m[one]` are two elements of one container, so the slice of
/// each is a borrow of its own: three in all, one of `m` and one of each
/// element.
#[test]
fn a_slice_of_one_element_is_not_a_slice_of_another() {
    let i = Interner::new();
    let ir =
        compile_script_mode_optimized(&i, &format!("{TWO_ROWS}m[z][z] + m[one][z]"), &ctx(&i, &[]))
            .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 3, "{ir}");
}

/// Both reads go through one element reference, so the two slices of it
/// have one operand between them and become one.
#[test]
fn two_slices_of_one_element_reference_are_one() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        &format!("{TWO_ROWS}let r = &m[z]; r[z] + r[z]"),
        &ctx(&i, &[]),
    )
    .unwrap();
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 2, "{ir}");
}
