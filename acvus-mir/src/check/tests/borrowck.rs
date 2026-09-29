//! borrowck over programs built from source (ownership §5): each reborrow a
//! loan of its own within the loan it is made through, an argument's loans
//! made at its call, and nothing that outlives a body holding a loan on its
//! own storage.

use super::{Fixture, covers, span_of};
use crate::check::{Refusal, RefusalKind};
use crate::ty::{Mutability, TyKind, TypeArg};

fn conflict(loan: acvus_ast::Span, span: acvus_ast::Span) -> Refusal {
    Refusal { kind: RefusalKind::LoanConflict { loan }, span }
}

/// Two `&mut *r`, each a loan of its own within `r`'s. The second is made
/// while the first lives, since `a` is used after it; storing it into `b`
/// reborrows it again, still while the first lives; and `*a = 2` reaches
/// `x` through `a` while the second lives, since `b` is used after it.
/// Each is one conflict, named at the outermost loan it meets.
#[test]
fn two_live_reborrows_of_one_mut_reference_are_refused_naming_the_first() {
    let fx = Fixture::new();
    let source = "let x = 1; let r = &mut x; let a = &mut *r; let b = &mut *r; *a = 2; *b = 3; x";
    let refusals = fx.refusals(source);
    let (first, second) = (span_of(source, "&mut *r", 0), span_of(source, "&mut *r", 1));
    let [at_second, at_binding, at_write] = &refusals[..] else {
        panic!("three LoanConflicts: {refusals:?}");
    };
    assert_eq!(*at_second, conflict(first, second));
    assert_eq!(at_binding.kind, RefusalKind::LoanConflict { loan: first });
    assert!(covers(span_of(source, "let b = &mut *r;", 0), at_binding.span), "at `b`'s binding: {at_binding:?}");
    assert_eq!(at_write.kind, RefusalKind::LoanConflict { loan: second });
    assert!(covers(span_of(source, "*a = 2;", 0), at_write.span), "at the write through `a`: {at_write:?}");
}

/// A loan on `r.a` and a write of `r.b` name places that do not overlap.
#[test]
fn a_write_of_a_field_beside_a_live_loan_on_another_is_admitted() {
    let fx = Fixture::new();
    let refusals = fx.refusals("let r = { a: 1, b: 2, }; let a = &mut r.a; *a = 3; r.b = 4; r");
    assert_eq!(refusals, []);
}

/// A write of `r.a` while `a`, lending it exclusively, is used after it.
#[test]
fn a_write_of_a_field_a_live_loan_lends_is_refused() {
    let fx = Fixture::new();
    let source = "let r = { a: 1, b: 2, }; let a = &mut r.a; r.a = 4; *a = 3; r";
    let refusals = fx.refusals(source);
    let [refusal] = &refusals[..] else { panic!("one LoanConflict: {refusals:?}") };
    assert_eq!(refusal.kind, RefusalKind::LoanConflict { loan: span_of(source, "&mut r.a", 0) });
    assert!(covers(span_of(source, "r.a = 4;", 0), refusal.span), "at the write: {refusal:?}");
}

/// Each argument `r` is reborrowed where the call starts: two exclusive
/// loans on `x`, made at one site, meet each other.
#[test]
fn one_mut_reference_passed_twice_to_one_call_is_refused() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let mut_int = fx.ty(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(int)));
    fx.declare("f", &[mut_int, mut_int], int);
    let source = "let x = 1; let r = &mut x; f(r, r)";
    let refusals = fx.refusals(source);
    let [refusal] = &refusals[..] else { panic!("one LoanConflict: {refusals:?}") };
    let RefusalKind::LoanConflict { loan } = refusal.kind else { panic!("a LoanConflict: {refusal:?}") };
    assert_eq!(loan, refusal.span, "the loan is made at the call it is refused at: {refusal:?}");
    assert!(covers(refusal.span, span_of(source, "f(r, r)", 0)), "at the call: {refusal:?}");
}

/// The receiver's `&mut v` is made where `push` starts, after `v.len()`
/// has lent `v` shared and ended.
#[test]
fn a_mut_receiver_and_a_shared_use_among_the_other_arguments_is_admitted() {
    let mut fx = Fixture::new();
    let vec = fx.vecs();
    let int = fx.int();
    let unit = fx.ty(TyKind::Unit);
    let shared_vec = fx.ty(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(vec)));
    let mut_vec = fx.ty(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(vec)));
    fx.declare("len", &[shared_vec], int);
    fx.declare("push", &[mut_vec, int], unit);
    let refusals = fx.refusals("let v = make(); v.push(v.len()); consume(v)");
    assert_eq!(refusals, []);
}

/// A lambda stores `&y`, a loan on its own local, into the target of the
/// `&mut` it captures `o` by: when it returns, that storage outside it
/// holds a loan on storage that ends with it.
#[test]
fn a_lambda_storing_a_loan_on_its_local_through_a_mut_capture_is_refused() {
    let fx = Fixture::new();
    let source = "let z = 1; let o = { r: &z, n: 1, }; let f = |u| -> { let y = u; o.r = &y; 0 }; f(2); o.n";
    let refusals = fx.refusals(source);
    let [refusal] = &refusals[..] else { panic!("one LoanOutlivesBody: {refusals:?}") };
    assert_eq!(refusal.kind, RefusalKind::LoanOutlivesBody { loan: span_of(source, "&y", 0) });
    assert_eq!(refusal.span, span_of(source, "|u| -> { let y = u; o.r = &y; 0 }", 0), "at the lambda's return");
}

/// The same through a `&mut` parameter's target.
#[test]
fn a_lambda_storing_a_loan_on_its_local_through_a_mut_parameter_is_refused() {
    let fx = Fixture::new();
    let source = "let z = 1; let o = { r: &z, n: 1, }; let f = |p| -> { let y = 2; p.r = &y; 0 }; f(&mut o); o.n";
    let refusals = fx.refusals(source);
    let [refusal] = &refusals[..] else { panic!("one LoanOutlivesBody: {refusals:?}") };
    assert_eq!(refusal.kind, RefusalKind::LoanOutlivesBody { loan: span_of(source, "&y", 0) });
    assert_eq!(refusal.span, span_of(source, "|p| -> { let y = 2; p.r = &y; 0 }", 0), "at the lambda's return");
}

/// Moving `m` out whole meets the live exclusive loan on `m.b`, which `s`
/// holds and uses after the move.
#[test]
fn a_move_of_a_whole_a_live_loan_lends_a_part_of_is_refused() {
    let mut fx = Fixture::new();
    let vec = fx.vecs();
    let int = fx.int();
    let object = fx.object(&[("a", vec), ("b", int)]);
    fx.declare("g", &[object], int);
    let source = "let m = { a: make(), b: 1, }; let s = &mut m.b; g(m); *s = 1; 0";
    let refusals = fx.refusals(source);
    let [refusal] = &refusals[..] else { panic!("one LoanConflict: {refusals:?}") };
    assert_eq!(refusal.kind, RefusalKind::LoanConflict { loan: span_of(source, "&mut m.b", 0) });
    assert!(covers(span_of(source, "g(m)", 0), refusal.span), "at the move of `m`: {refusal:?}");
}
