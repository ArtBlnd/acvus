//! The harness itself: a case admitted prints its modules, a refused one
//! its refusals, and each is one snapshot.

use acvus_mir::ty::{IntTy, Mutability, TyKind, TypeArg};
use acvus_mir_test::{At, Case, Level};

/// `core::add` at `i64`, `(&i64, &i64) -> i64`.
fn with_add(case: &mut Case) {
    let int = case.ty(TyKind::Int(IntTy::I64));
    let shared = case.ty(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let add = case.function(&[shared, shared], int, acvus_mir::ty::Effect::pure());
    let item = case.core().add;
    case.lang_impls(item, &[add]);
}

#[test]
fn an_admitted_case_prints_its_modules() {
    let mut case = Case::new();
    with_add(&mut case);
    let int = case.ty(TyKind::Int(IntTy::I64));
    let inc = case.script(&["lib"], "inc", "$x + 1");
    let main = case.script(&["app"], "main", "lib::inc(x: $n)");
    case.entry(main, &[("n", int)], Some(int));
    let _ = inc;
    insta::assert_snapshot!(case.at(At::Optimized(Level::Full)).admitted());
}

#[test]
fn a_refused_case_prints_its_refusals() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    let a = case.script(&["m"], "a", "m::b(x: $x)");
    case.script(&["m"], "b", "m::a(x: $x)");
    case.entry(a, &[("x", int)], None);
    insta::assert_snapshot!(case.refused());
}

#[test]
fn a_case_admitted_twice_prints_the_same() {
    let build = || {
        let mut case = Case::new();
        with_add(&mut case);
        let int = case.ty(TyKind::Int(IntTy::I64));
        let main = case.script(&["app"], "main", "let y = $n + 1; y + y");
        case.entry(main, &[("n", int)], None);
        case.admitted()
    };
    assert_eq!(build(), build());
}
