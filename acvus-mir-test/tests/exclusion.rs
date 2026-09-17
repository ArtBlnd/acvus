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
    assert!(err.contains("is touched while the reference"), "{err}");
    check("let x = 1; let r = &x; let y = *r; x = 2; y").expect("r is dead before the write");
}

#[test]
fn a_borrow_of_a_reference_is_a_reborrow() {
    let ir = check("let x = 1; let r = &x; let rr = &r; *rr").expect("`&r` is `&Int`");
    assert!(ir.contains(": &i64"), "{ir}");
    assert!(!ir.contains("&&"), "{ir}");
    let err = check("let x = 1; let r = &x; let rr = &r; x = 2; *rr")
        .expect_err("x is written while the reborrow names it");
    assert!(err.contains("is touched while the reference"), "{err}");
}

#[test]
fn a_shared_reference_is_not_borrowed_mutably() {
    let err = check("let x = 1; let r = &x; let m = &mut r; *m").expect_err("`&mut` of a `&`");
    assert!(
        err.contains("shared reference cannot be borrowed mutably"),
        "{err}"
    );
}

#[test]
fn a_reference_read_twice_from_its_variable_is_one_holder() {
    check("let x = 1; let r = &x; let a = *r + *r; a").expect("both reads are r's own");
}
