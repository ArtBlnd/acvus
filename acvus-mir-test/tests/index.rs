//! `a[i]` is a place (RFC-0047): what the checker admits, what it refuses,
//! and where the container's `as_slice` ends up.

use acvus_mir::ty::Ty;
use acvus_mir_test::{compile_script_mode_optimized, compile_script_optimized};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

fn ir(source: &str) -> String {
    let i = Interner::new();
    compile_script_optimized(&i, source, &FxHashMap::default())
        .unwrap_or_else(|e| panic!("{source}\n{e}"))
}

/// Every refusal of `source`, each without the harness's own prefix.
fn refusals(source: &str) -> Vec<String> {
    let i = Interner::new();
    compile_script_optimized(&i, source, &FxHashMap::default())
        .expect_err(&format!("{source} is refused"))
        .lines()
        .map(|line| line.split("] ").nth(1).unwrap_or(line).to_string())
        .collect()
}

fn refusal(source: &str) -> String {
    let found = refusals(source);
    assert_eq!(found.len(), 1, "{source}: {found:?}");
    found.into_iter().next().expect("one refusal")
}

fn main_body(ir: &str) -> &str {
    ir.split("=== main ===")
        .nth(1)
        .expect("an entry body")
        .split("=== ")
        .next()
        .expect("the entry body ends at the next section")
}

fn count(body: &str, needle: &str) -> usize {
    body.matches(needle).count()
}

/// The `Index` operations of a body: `rD = rS[rI]`, printed with the
/// slice register before the bracket.
fn index_ops(body: &str) -> usize {
    body.lines()
        .filter(|line| {
            let Some((before, _)) = line.split_once('[') else {
                return false;
            };
            before.ends_with(|c: char| c.is_ascii_digit()) && before.contains('r')
        })
        .count()
}

fn loop_head_of(body: &str) -> usize {
    body.find("| L0")
        .unwrap_or_else(|| panic!("a loop head in:\n{body}"))
}

// -- What it admits ---------------------------------------------------

#[test]
fn an_index_of_a_vec_and_of_an_array_is_one_as_slice_and_one_index() {
    for source in [
        "let v = vec([1, 2, 3]); v[1]",
        "let a = [1, 2, 3]; a[1]",
        "let a = [1, 2, 3]; let r = &a; r[1]",
    ] {
        let ir = ir(source);
        let body = main_body(&ir);
        assert_eq!(count(body, "as_slice"), 1, "{source}:\n{body}");
        assert_eq!(index_ops(body), 1, "{source}:\n{body}");
    }
}

#[test]
fn a_borrow_of_an_element_is_an_index_that_leaves_a_reference() {
    let ir = ir("let a = [1, 2, 3]; let r = &a[1]; *r");
    let body = main_body(&ir);
    assert!(body.contains("= &"), "{body}");
}

#[test]
fn a_store_into_an_element_takes_the_container_exclusively() {
    let ir = ir("let a = [1, 2, 3]; a[1] = 9; a[1]");
    let body = main_body(&ir);
    assert!(body.contains("ref &mut a"), "{body}");
    assert!(body.contains("] = 9") || body.contains("] proven = 9"), "{body}");
}

/// A mutable projection demands its object mutably (RFC-0018).
#[test]
fn a_store_through_a_field_takes_every_container_below_it_mutably() {
    let ir = ir("let v = [{ h: [1, 2, ], }, ]; v[0u64].h[1u64] = 4; v[0u64].h[1u64]");
    let body = main_body(&ir);
    assert_eq!(
        count(body, "as_slice &mut"),
        2,
        "the element's container and the field's:\n{body}"
    );
    assert!(body.contains("] = 4") || body.contains("] proven = 4"), "{body}");
}

#[test]
fn a_mutable_reach_through_a_field_of_a_shared_reference_is_refused() {
    let through_a_shared_reference =
        "cannot store through `r`, of type &{g: Array<i64, 2>}: not a `&mut`; bind it with `&mut`";
    assert_eq!(
        refusal("let o = { g: [1, 2, ], }; let r = &o; r.g[0u64] = 5; o.g[0u64]"),
        through_a_shared_reference
    );
    assert_eq!(
        refusal("let o = { g: [1, 2, ], }; let r = &o; let x = &mut r.g[0u64]; 0"),
        through_a_shared_reference
    );
}

#[test]
fn a_nested_index_takes_a_slice_of_the_row_it_indexed() {
    let ir = ir("let m = [[1, 2], [3, 4]]; m[1][0]");
    let body = main_body(&ir);
    assert_eq!(
        count(body, "as_slice"),
        2,
        "one per index expression:\n{body}"
    );
}

#[test]
fn an_element_is_a_method_receiver() {
    let ir = ir("let m = [[1, 2], [3, 4]]; m[0].len()");
    assert!(main_body(&ir).contains("as_slice"), "{ir}");
}

// -- What it refuses --------------------------------------------------

#[test]
fn an_element_that_moves_is_not_read_by_value() {
    assert_eq!(
        refusal("let m = [[1, 2], [3, 4]]; m[0]"),
        "cannot move out of index of `Array<Array<i64, 2>, 2>`"
    );
}

#[test]
fn a_container_without_a_slice_is_not_indexed() {
    assert_eq!(
        refusal("let d = deque(); push_back(&mut d, 1); d[0]"),
        "cannot index into a value of type `Deque<i64>`"
    );
}

#[test]
fn an_index_that_is_not_a_u64_is_the_ordinary_unification_refusal() {
    assert_eq!(
        refusal("let a = [1, 2, 3]; let x = 1.5; let n = x as i64; a[n]"),
        "type mismatch: expected u64, got i64"
    );
}

/// The view is the coercion `a[i]` takes, and a script may also ask for it
/// by name. Named, it is an ordinary call of the container's own
/// declaration — not the `as_slice` instruction the coercion lowers to —
/// and what it yields is the view at the mutability the name asked for.
#[test]
fn a_script_names_the_view_and_gets_the_containers_own() {
    let shared = ir("let a = [1, 2, 3]; let s = as_slice(&a); s.len()");
    assert!(
        main_body(&shared).contains("as_slice(...)) : &[i64]"),
        "{shared}"
    );
    let exclusive = ir("let a = [1, 2, 3]; let s = as_slice_mut(&mut a); s.len()");
    assert!(
        main_body(&exclusive).contains("as_slice_mut(...)) : &mut [i64]"),
        "{exclusive}"
    );
}

#[test]
fn an_exclusive_element_borrow_while_a_shared_one_is_live_is_refused() {
    let errors = refusals("let a = [1, 2, 3]; let r = &mut a[0]; let q = &a[1]; *r + *q");
    assert!(errors.iter().any(|e| e.contains("is live")), "{errors:?}");
}

// -- Where the `AsSlice` ends up --------------------------------------

#[test]
fn a_read_only_loop_takes_its_slice_once() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let a = [1, 2, 3]; let s = 0; let k = 0; while k < 3 { s = s + a[k]; k = k + 1; } s",
        &ctx(&i, &[]),
    )
    .expect("a read-only loop compiles");
    let body = main_body(&ir);
    assert!(!body[loop_head_of(body)..].contains("as_slice"), "{body}");
}

#[test]
fn a_loop_that_writes_its_container_takes_its_slice_inside() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let a = [1, 2, 3]; let k = 0; while k < 3 { a[k] = a[0] + 1; k = k + 1; } a[0]",
        &ctx(&i, &[]),
    )
    .expect("a writing loop compiles");
    let body = main_body(&ir);
    assert!(body[loop_head_of(body)..].contains("as_slice"), "{body}");
}

/// `acvus_interpreter_test::scripts::ATTENTION` binds each row before its
/// inner loop and cites this test for why.
#[test]
fn a_row_bound_before_a_loop_takes_its_slice_once_and_an_inline_row_does_not() {
    let i = Interner::new();
    let bound = compile_script_mode_optimized(
        &i,
        "let m = [[1, 2], [3, 4]]; let s = 0; let k = 0; let row = &m[0]; \
         while k < 2 { s = s + row[k]; k = k + 1; } s",
        &ctx(&i, &[]),
    )
    .expect("the bound row compiles");
    let inline = compile_script_mode_optimized(
        &i,
        "let m = [[1, 2], [3, 4]]; let s = 0; let k = 0; \
         while k < 2 { s = s + m[0][k]; k = k + 1; } s",
        &ctx(&i, &[]),
    )
    .expect("the inline row compiles");

    let bound = main_body(&bound);
    let inline = main_body(&inline);
    assert!(
        !bound[loop_head_of(bound)..].contains("as_slice"),
        "{bound}"
    );
    assert_eq!(
        count(&inline[loop_head_of(inline)..], "as_slice"),
        1,
        "{inline}"
    );
}
