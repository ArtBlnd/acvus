//! Integer widths (RFC-0037): a literal takes the width its use demands,
//! is `i64` alone, is refused where it does not fit, and widths never mix.

use acvus_mir::ty::{IntTy, Ty};
use acvus_mir_test::compile_script_ir;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<acvus_utils::Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

#[test]
fn a_literal_takes_the_width_its_use_demands() {
    let i = Interner::new();
    let c = ctx(&i, &[("b", Ty::U8)]);
    let ir = compile_script_ir(&i, "@b + 1", &c).unwrap();
    assert!(ir.contains(": u8"), "the sum is a u8:\n{ir}");
    assert!(!ir.contains(": i64"), "nothing here is an i64:\n{ir}");
}

#[test]
fn a_literal_alone_is_i64() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "1 + 2", &FxHashMap::default()).unwrap();
    assert!(ir.contains(": i64"), "the sum is an i64:\n{ir}");
}

#[test]
fn a_literal_that_does_not_fit_its_width_is_refused() {
    let i = Interner::new();
    let c = ctx(&i, &[("b", Ty::U8)]);
    let err = compile_script_ir(&i, "@b + 300", &c).unwrap_err();
    assert!(err.contains("literal 300 does not fit u8"), "{err}");
    let c = ctx(&i, &[("n", Ty::I8)]);
    let err = compile_script_ir(&i, "@n + 128", &c).unwrap_err();
    assert!(err.contains("literal 128 does not fit i8"), "{err}");
    let ir = compile_script_ir(&i, "@n + 127", &c).unwrap();
    assert!(ir.contains(": i8"), "{ir}");
}

#[test]
fn negation_needs_a_signed_integer() {
    let i = Interner::new();
    let c = ctx(&i, &[("b", Ty::U8)]);
    let err = compile_script_ir(&i, "-@b", &c).unwrap_err();
    assert!(err.contains("type mismatch in `-`"), "{err}");
    let err = compile_script_ir(&i, "@b + -1", &c).unwrap_err();
    assert!(
        err.contains("u8"),
        "a negated literal is signed and cannot be a u8:\n{err}"
    );
    let ir = compile_script_ir(&i, "-1", &FxHashMap::default()).unwrap();
    assert!(ir.contains(": i64"), "{ir}");
}

#[test]
fn widths_do_not_mix() {
    let i = Interner::new();
    let c = ctx(&i, &[("a", Ty::U8), ("b", Ty::I64)]);
    let err = compile_script_ir(&i, "@a + @b", &c).unwrap_err();
    assert!(err.contains("type mismatch"), "{err}");
    let c = ctx(&i, &[("a", Ty::U32), ("b", Ty::U32)]);
    let ir = compile_script_ir(&i, "@a * @b", &c).unwrap();
    assert!(ir.contains(": u32"), "{ir}");
}
