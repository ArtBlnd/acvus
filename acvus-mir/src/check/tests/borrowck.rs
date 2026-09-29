//! borrowck over programs built from source (ownership §5): each reborrow a
//! loan of its own within the loan it is made through, an argument's loans
//! made at its call, and nothing that outlives a body holding a loan on its
//! own storage.

use acvus_ast::Span;

use super::{Fixture, covers, span_of};
use crate::check::{ConflictTrace, Refusal, RefusalKind};
use crate::ty::{Mutability, TyKind, TypeArg};

fn conflict(span: Span, earlier: Vec<Span>, later: Vec<Span>, uses: Vec<Span>) -> Refusal {
    Refusal { kind: RefusalKind::LoanConflict { trace: ConflictTrace { earlier, later, uses } }, span }
}

/// The span of the name `let name` binds in `source`, where the store of
/// its value stands.
fn binder(source: &str, name: &str) -> Span {
    let start = span_of(source, &format!("let {name} "), 0).start + "let ".len();
    Span { start, end: start + name.len() }
}

/// The trace of the one `LoanConflict` of `refusals`, and its span.
fn one_conflict(refusals: &[Refusal]) -> (Span, &ConflictTrace) {
    let [Refusal { kind: RefusalKind::LoanConflict { trace }, span }] = refusals else {
        panic!("one LoanConflict: {refusals:?}");
    };
    (*span, trace)
}

/// Two `&mut *r`, each a loan of its own within `r`'s store's, within
/// `&mut x`: one fact. The second is made while the first, held by `a`'s
/// store, lives, and `*a = 2` reaches `x` through the first while the
/// second lives: one refusal at the second, `*a = 2;` a use.
#[test]
fn two_live_reborrows_of_one_mut_reference_are_refused_as_one() {
    let fx = Fixture::new();
    let source = "let x = 1; let r = &mut x; let a = &mut *r; let b = &mut *r; *a = 2; *b = 3; x";
    let refusals = fx.refusals(source);
    let (first, second, on_x) = (span_of(source, "&mut *r", 0), span_of(source, "&mut *r", 1), span_of(source, "&mut x", 0));
    let (a, r) = (binder(source, "a"), binder(source, "r"));
    let uses = vec![span_of(source, "*a = 2;", 0)];
    assert_eq!(refusals, [conflict(second, vec![a, first, r, on_x], vec![second, r, on_x], uses)]);
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
    let (span, trace) = one_conflict(&refusals);
    let earlier = vec![binder(source, "a"), span_of(source, "&mut r.a", 0)];
    assert_eq!(*trace, ConflictTrace { earlier, later: Vec::new(), uses: Vec::new() });
    assert!(covers(span_of(source, "r.a = 4;", 0), span), "at the write: {span:?}");
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
    let (span, trace) = one_conflict(&refusals);
    assert!(covers(span, span_of(source, "f(r, r)", 0)), "at the call: {span:?}");
    let chain = vec![span, binder(source, "r"), span_of(source, "&mut x", 0)];
    assert_eq!(*trace, ConflictTrace { earlier: chain.clone(), later: chain, uses: Vec::new() }, "both made at the call within `r`'s");
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
    let (span, trace) = one_conflict(&refusals);
    let earlier = vec![binder(source, "s"), span_of(source, "&mut m.b", 0)];
    assert_eq!(*trace, ConflictTrace { earlier, later: Vec::new(), uses: Vec::new() });
    assert!(covers(span_of(source, "g(m)", 0), span), "at the move of `m`: {span:?}");
}

/// All elements of a sequence are one part (ownership §5): `a` and `b`,
/// lent `&mut` at `[0]` and `[1]`, overlap. `b` is made while `a` lives,
/// and `*a = 3` reaches the part through `a` while `b` lives: one refusal.
#[test]
fn two_mut_loans_on_two_elements_of_one_array_are_refused() {
    let fx = Fixture::new();
    let source = "let v = [1, 2]; match &mut v { [a, b] => { *a = 3; *b = 4; 0 }, }";
    let refusals = fx.refusals(source);
    let (span, trace) = one_conflict(&refusals);
    let (pattern, on_v) = (span_of(source, "[a, b]", 0), span_of(source, "&mut v", 0));
    assert!(covers(pattern, trace.earlier[0]) && covers(pattern, span) && trace.earlier[0].start < span.start, "`b` meets `a`: {refusals:?}");
    assert_eq!((trace.earlier.last(), trace.later.last()), (Some(&on_v), Some(&on_v)), "both within the scrutinee's `&mut v`: {trace:?}");
    assert_eq!(trace.uses, [span_of(source, "*a = 3;", 0)]);
}

/// What a path of references reaches is held by the first reference on it
/// (ownership §5): `x`, read through `r`, keeps `r`'s store's loan, within
/// `&mut o`, alive while it lives, so reading `o.b` by name meets it. `y`,
/// the `&mut v` stored in `o.b` reborrowed at its store, is made while `x`
/// lives, and `*x = 2` reaches `v` while `y` lives: one fact, at `y`'s
/// store. `r.b` lowers to a `Take` of `b` through `r`, a path of `&mut`
/// alone, so `x`'s chain runs through that `Take`'s reborrow, within both
/// the reference stored in `o.b`, made by the object within `&mut v`, and
/// `r`'s store; `y`'s through the stored one.
#[test]
fn a_reference_read_through_another_keeps_the_first_reference_s_loan_alive() {
    let fx = Fixture::new();
    let source = "let v = 1; let o = { b: &mut v, }; let r = &mut o; let x = r.b; let y = o.b; *x = 2; *y = 3; 0";
    let refusals = fx.refusals(source);
    let (on_o, on_v, object) = (span_of(source, "&mut o", 0), span_of(source, "&mut v", 0), span_of(source, "{ b: &mut v, }", 0));
    let (r, x, y) = (binder(source, "r"), binder(source, "x"), binder(source, "y"));
    let [read, reborrowed] = &refusals[..] else { panic!("two LoanConflicts: {refusals:?}") };
    assert!(covers(span_of(source, "let y = o.b;", 0), read.span), "at the read of `o.b`: {read:?}");
    let RefusalKind::LoanConflict { trace } = &read.kind else { panic!("a LoanConflict: {read:?}") };
    assert_eq!(*trace, ConflictTrace { earlier: vec![r, on_o], later: Vec::new(), uses: Vec::new() });
    let read_through = span_of(source, "r.b", 0);
    let uses = vec![span_of(source, "*x = 2;", 0)];
    let earlier = vec![x, read_through, object, r, on_v, on_o];
    assert_eq!(*reborrowed, conflict(y, earlier, vec![y, object, on_v], uses));
}

