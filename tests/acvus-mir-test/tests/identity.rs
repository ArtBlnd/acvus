//! Identity as a parameter of a user-defined type, at its contract: a value
//! with an identity is one source and moves; two sources do not mix; a
//! type without an identity copies.

use acvus_mir::ty::{LenTerm, Ty};
use acvus_mir_test::{Marked, Refusal, compile_script_ir, refuse_script_mode_optimized};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn items(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("items"),
        Ty::Array(Box::new(Ty::I64), LenTerm::Known(2)),
    )])
}

fn words(refusals: &[Refusal]) -> Vec<String> {
    refusals.iter().map(Refusal::to_string).collect()
}

fn begins_here(name: &str) -> Marked {
    Marked {
        source: Some("into_iter".to_string()),
        text: format!("`{name}`'s source begins here"),
    }
}

#[test]
fn two_iterators_from_different_sources_do_not_unify() {
    let i = Interner::new();
    let source = "let a = @items | into_iter; let b = @items | into_iter; [a, b]; 0";
    let refusals = refuse_script_mode_optimized(&i, source, &items(&i)).unwrap_err();
    let [refusal] = refusals.as_slice() else {
        panic!("expected one refusal, got {:#?}", words(&refusals));
    };
    assert_eq!(
        refusal.message,
        "`a` and `b` are values of one type from two different sources, \
         and one place cannot hold both"
    );
    assert_eq!(refusal.marked(source), [begins_here("a"), begins_here("b")]);
}

#[test]
fn a_source_with_no_name_is_shown_as_the_value_it_is() {
    let i = Interner::new();
    let source = "[@items | into_iter, @items | into_iter]; 0";
    let refusals = refuse_script_mode_optimized(&i, source, &items(&i)).unwrap_err();
    let [refusal] = refusals.as_slice() else {
        panic!("expected one refusal, got {:#?}", words(&refusals));
    };
    assert_eq!(
        refusal.message,
        "this value and this value are values of one type from two different sources, \
         and one place cannot hold both"
    );
}

#[test]
fn chain_joins_two_sources_into_a_new_one() {
    let i = Interner::new();
    compile_script_ir(
        &i,
        "let a = [1, 2] | into_iter; let b = [1, 2] | into_iter; let c = chain(a, b) | collect; len(&c)",
        &items(&i),
    )
    .unwrap();
}

#[test]
fn a_derived_iterator_keeps_its_source_and_still_moves() {
    let i = Interner::new();
    let err = compile_script_ir(
        &i,
        "let it = @items | into_iter | map(|x| -> x + 1); it | collect; it | collect",
        &items(&i),
    )
    .unwrap_err();
    assert!(err.contains("after it was moved"), "{err}");
}
