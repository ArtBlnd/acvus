use acvus_mir::decl::{LabelledSignature, Lifetime, Outlives, SegmentLabel};
use acvus_mir::ty::FlowEnd;
use acvus_utils::{Interner, QualifiedRef};

use super::{At, EndNames, Written, labelled};
use crate::decl::DeclTy;

fn shared(target: DeclTy) -> DeclTy {
    DeclTy::Ref { mutable: false, target: Box::new(target) }
}

fn exclusive(target: DeclTy) -> DeclTy {
    DeclTy::Ref { mutable: true, target: Box::new(target) }
}

fn extern_ty(names: &Interner, name: &str, args: Vec<DeclTy>, lifetimes: u32) -> DeclTy {
    DeclTy::Extern { name: QualifiedRef::root(names.intern(name)), args, effect: None, identity: None, lifetimes }
}

fn labels(signature: &LabelledSignature, end: FlowEnd) -> &[SegmentLabel] {
    &signature.ends.iter().find(|labelled| labelled.end == end).expect("the end is labelled").labels
}

fn lifetime(label: SegmentLabel) -> Lifetime {
    match label {
        SegmentLabel::Lifetime(lifetime) => lifetime,
        SegmentLabel::TypeVar(var) => panic!("a type variable's part, {var}, where a lifetime is"),
    }
}

/// `fn get<'m, 'k>(map: &'m Map<K, V>, key: &'k K) -> Option<&'m V>`: the
/// result takes the map's loan, not the key's.
#[test]
fn a_result_takes_the_loan_its_name_says() {
    let names = Interner::new();
    let map = extern_ty(&names, "Map", vec![DeclTy::Param(0), DeclTy::Param(1)], 0);
    let ends = [
        EndNames { at: At::Param(0), ty: shared(map), outer: Some("'m"), inner: &[] },
        EndNames { at: At::Param(1), ty: shared(DeclTy::Param(0)), outer: Some("'k"), inner: &[] },
        EndNames { at: At::Result, ty: DeclTy::Option(Box::new(shared(DeclTy::Param(1)))), outer: None, inner: &["'m"] },
    ];

    let signature = labelled(&ends, &[]);

    let map = labels(&signature, FlowEnd::Param(0));
    let key = labels(&signature, FlowEnd::Param(1));
    let result = labels(&signature, FlowEnd::Result);
    assert_eq!(map[1..], [SegmentLabel::TypeVar(0), SegmentLabel::TypeVar(1)]);
    assert_eq!(result[1], SegmentLabel::TypeVar(1));
    assert_eq!(result[0], map[0]);
    assert_ne!(result[0], key[0]);
}

/// `fn next<'r, 'a>(it: &'r mut Refs<'a, T>) -> Option<&'a T>`: the result
/// takes what the iterator holds, not the loan of the iterator itself, and
/// what it holds outlives that loan.
#[test]
fn an_outermost_reference_keeps_its_own_label() {
    let names = Interner::new();
    let refs = extern_ty(&names, "Refs", vec![DeclTy::Param(0)], 1);
    let ends = [
        EndNames { at: At::Param(0), ty: exclusive(refs), outer: Some("'r"), inner: &["'a"] },
        EndNames { at: At::Result, ty: DeclTy::Option(Box::new(shared(DeclTy::Param(0)))), outer: None, inner: &["'a"] },
    ];

    let signature = labelled(&ends, &[]);

    let it = labels(&signature, FlowEnd::Param(0));
    let result = labels(&signature, FlowEnd::Result);
    assert_eq!(it[2], SegmentLabel::TypeVar(0));
    assert_ne!(it[0], it[1]);
    assert_eq!(result[0], it[1]);
    assert_eq!(*signature.outlives, [Outlives { longer: lifetime(it[1]), shorter: lifetime(it[0]) }]);
}

/// `p: (&'a Pair, &'b Number)`, declared `&(Pair, Number)`: one loan, so its
/// two names are one label, and a result at either takes it.
#[test]
fn the_names_inside_an_end_are_one_label() {
    let names = Interner::new();
    let pair = extern_ty(&names, "Pair", vec![], 0);
    let number = extern_ty(&names, "Number", vec![], 0);
    let ends = [
        EndNames { at: At::Param(0), ty: shared(DeclTy::Tuple(vec![pair, number.clone()])), outer: None, inner: &["'a", "'b"] },
        EndNames { at: At::Result, ty: shared(number), outer: Some("'b"), inner: &[] },
    ];

    let signature = labelled(&ends, &[]);

    assert_eq!(labels(&signature, FlowEnd::Result)[0], labels(&signature, FlowEnd::Param(0))[0]);
}

/// `x: Refs<T>`, its lifetime elided in the path: no name is written at it,
/// so it is the end's own, apart from every other end's.
#[test]
fn a_lifetime_elided_in_a_path_is_its_ends_own() {
    let names = Interner::new();
    let refs = extern_ty(&names, "Refs", vec![DeclTy::Param(0)], 1);
    let ends = [
        EndNames { at: At::Param(0), ty: refs.clone(), outer: None, inner: &[] },
        EndNames { at: At::Param(1), ty: refs, outer: None, inner: &[] },
    ];

    let signature = labelled(&ends, &[]);

    assert_ne!(labels(&signature, FlowEnd::Param(0))[0], labels(&signature, FlowEnd::Param(1))[0]);
}

/// A bound written between two ends' lifetimes is kept; one naming a
/// lifetime no label takes is not.
#[test]
fn a_written_bound_is_kept_between_labelled_lifetimes() {
    let text = shared(DeclTy::Str);
    let ends = [
        EndNames { at: At::Param(0), ty: text.clone(), outer: Some("'a"), inner: &[] },
        EndNames { at: At::Param(1), ty: text, outer: Some("'b"), inner: &[] },
    ];

    let signature = labelled(
        &ends,
        &[Written { longer: "'a", shorter: "'b" }, Written { longer: "'a", shorter: "'unnamed" }],
    );

    let a = lifetime(labels(&signature, FlowEnd::Param(0))[0]);
    let b = lifetime(labels(&signature, FlowEnd::Param(1))[0]);
    assert_eq!(*signature.outlives, [Outlives { longer: a, shorter: b }]);
}
