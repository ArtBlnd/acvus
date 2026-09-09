//! Identity as a parameter of a user-defined type, at its contract: a value
//! with an identity is one source and moves; two sources do not mix; a
//! type without an identity copies.

use acvus_mir::ty::{LenTerm, Ty};
use acvus_mir_test::compile_script_ir;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn items(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("items"),
        Ty::Array(Box::new(Ty::Int), LenTerm::Known(2)),
    )])
}

#[test]
fn two_iterators_from_different_sources_do_not_unify() {
    let i = Interner::new();
    let err = compile_script_ir(
        &i,
        "a = @items | iter; b = @items | iter; [a, b]; 0",
        &items(&i),
    )
    .unwrap_err();
    assert!(err.contains("heterogeneous list"), "{err}");
}

#[test]
fn chain_joins_two_sources_into_a_new_one() {
    let i = Interner::new();
    compile_script_ir(
        &i,
        "a = @items | iter; b = @items | iter; chain(a, b) | collect | len",
        &items(&i),
    )
    .unwrap();
}

#[test]
fn a_derived_iterator_keeps_its_source_and_still_moves() {
    let i = Interner::new();
    let err = compile_script_ir(
        &i,
        "it = @items | iter | map(|x| -> x + 1); it | collect; it | collect",
        &items(&i),
    )
    .unwrap_err();
    assert!(err.contains("UseAfterMove"), "{err}");
}

#[test]
fn a_type_without_identity_copies() {
    let i = Interner::new();
    compile_script_ir(&i, "xs = @items | list; len(xs) + len(xs)", &items(&i)).unwrap();
}
