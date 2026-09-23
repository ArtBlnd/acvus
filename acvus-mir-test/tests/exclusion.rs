//! Exclusion as the source wrote it (RFC-0029): a loan held in a storage
//! is a live holder, a borrow of a reference is a reborrow, and a
//! parameter of reference type is a storage its reborrows hold.

use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn check(src: &str) -> Result<String, String> {
    let i = Interner::new();
    compile_script_mode_optimized(&i, src, &FxHashMap::default())
}

#[test]
fn a_reference_kept_in_a_variable_holds_its_loan() {
    let err = check("let x = 1; let r = &x; x = 2; *r").expect_err("x is written while r names it");
    assert!(err.contains("while a reference to it is live"), "{err}");
    check("let x = 1; let r = &x; let y = *r; x = 2; y").expect("r is dead before the write");
}

#[test]
fn a_borrow_of_a_reference_is_a_reborrow() {
    let ir = check("let x = 1; let r = &x; let rr = &r; *rr").expect("`&r` is `&Int`");
    assert!(ir.contains(": &i64"), "{ir}");
    assert!(!ir.contains("&&"), "{ir}");
    let err = check("let x = 1; let r = &x; let rr = &r; x = 2; *rr")
        .expect_err("x is written while the reborrow names it");
    assert!(err.contains("while a reference to it is live"), "{err}");
}

#[test]
fn a_shared_reference_is_not_borrowed_mutably() {
    let err = check("let x = 1; let r = &x; let m = &mut r; *m").expect_err("`&mut` of a `&`");
    assert!(
        err.contains("`r` is a shared reference and cannot be borrowed mutably"),
        "{err}"
    );
}

#[test]
fn a_reference_read_twice_from_its_variable_is_one_holder() {
    check("let x = 1; let r = &x; let a = *r + *r; a").expect("both reads are r's own");
}

fn refused_at(src: &str) -> Vec<String> {
    let i = Interner::new();
    let refusals = refuse_script_mode_optimized(&i, src, &FxHashMap::default())
        .expect_err("the program is refused");
    refusals
        .iter()
        .map(|refusal| format!("{} at `{}`", refusal.message, refusal.at(src)))
        .collect()
}

/// A storage lends what the assignments reaching an instruction gave it,
/// not what it is given later on some path: `q` holds only `z` where `v` is
/// borrowed, and `v`'s loan from where `q = s` reaches.
#[test]
fn a_storage_holds_a_loan_from_where_it_is_assigned() {
    check("let v = 1; let z = 0; let q = &mut z; let s = &mut v; if true { q = s; } *q = 9; v")
        .expect("q holds no loan of v where v is borrowed");
}

/// The element kept in `q` holds the loop's loan of `v` after the loop, so
/// the refusal is the write after the loop, not the loop's own borrow.
#[test]
fn a_kept_element_refuses_the_write_after_the_loop_and_not_the_loop_head() {
    let src = "let v = vec([1, 2, 3]); let z = 0; let q = &mut z; \
               for x in &mut v { q = x; } push(&mut v, 4); *q = 9; v[2u64]";
    assert_eq!(
        refused_at(src),
        ["`v` is written here while a reference to it is live at `&mut v`"]
    );
}

/// Every element is a loan of the one container (RFC-0057 rule 5): taking
/// the second element's `&mut` out of `x` while `b` keeps the first's and is
/// used later is two live `&mut`s of `v`.
#[test]
fn a_second_kept_element_is_refused_where_it_is_taken() {
    let src = "let v = vec([1, 2, 3]); let z = 0; let w = 0; let a = &mut z; let b = &mut w; \
               for x in &mut v { b = a; a = x; } *a = 7; *b = 8; v[1u64] * 10 + v[2u64]";
    assert_eq!(
        refused_at(src),
        ["the storage is written here while a reference to it is live at `x`"]
    );
}
