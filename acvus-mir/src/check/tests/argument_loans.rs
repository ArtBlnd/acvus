//! Loans an argument makes where its call starts (ownership §5): a lend
//! written as the argument itself, and the reborrow that passing a `&mut`
//! makes, are each a loan of its own, and each meets what the same lend
//! bound to a name first would meet.

use super::borrowck::{lifetimes, one_conflict};
use super::{Fixture, covers, span_of};
use crate::ty::{FlowEnd, Mutability, Poly, TyId, TyKind};

/// `&mut i64`.
fn mut_int(fx: &mut Fixture) -> TyId<Poly> {
    let int = fx.int();
    fx.ty(TyKind::Ref(Mutability::Mut, int))
}

/// `set(p: &mut i64)`, stating no lifetimes.
fn declare_set(fx: &mut Fixture) {
    let (mut_int, unit) = (mut_int(fx), fx.ty(TyKind::Unit));
    fx.declare("set", &[mut_int], unit);
}

/// `g(p: &mut i64) -> &mut i64`: with `labelled`, `g<'a>(p: &'a mut i64)
/// -> &'a mut i64`, else stating no lifetimes.
fn declare_g(fx: &mut Fixture, labelled: bool) {
    let mut_int = mut_int(fx);
    match labelled {
        true => {
            let labels = lifetimes(&[(FlowEnd::Result, &[0]), (FlowEnd::Param(0), &[0])]);
            fx.declare_labelled("g", &[mut_int], mut_int, labels);
        }
        false => {
            fx.declare("g", &[mut_int], mut_int);
        }
    }
}

/// The one `LoanConflict` of `source`'s refusals lies within `within`.
fn refused_within(fx: Fixture, source: &str, within: &str) {
    let refusals = fx.refusals(source);
    let (span, _) = one_conflict(&refusals);
    assert!(covers(span_of(source, within, 0), span), "within {within:?}: {span:?}");
}

/// The lend `&mut a`, made where `f` starts, meets the shared loan `r`
/// holds, which lives on to `*r`.
#[test]
fn an_inline_mut_lend_to_a_lambda_meets_a_live_shared_loan() {
    let fx = Fixture::new();
    refused_within(fx, "let a = 1; let r = &a; let f = |p| -> { *p = 2; 0 }; f(&mut a); *r", "f(&mut a)");
}

/// The same lend bound to `w` first.
#[test]
fn a_named_mut_lend_to_a_lambda_meets_a_live_shared_loan() {
    let fx = Fixture::new();
    refused_within(fx, "let a = 1; let r = &a; let f = |p| -> { *p = 2; 0 }; let w = &mut a; f(w); *r", "&mut a");
}

/// The lend `&mut a` as an extern fn's argument.
#[test]
fn an_inline_mut_lend_to_an_extern_fn_meets_a_live_shared_loan() {
    let mut fx = Fixture::new();
    declare_set(&mut fx);
    refused_within(fx, "let a = 1; let r = &a; set(&mut a); *r", "set(&mut a)");
}

/// Two lends `&mut x` as arguments of one call, both made where it
/// starts: two exclusive loans on `x` that meet each other.
#[test]
fn two_inline_mut_lends_of_one_place_to_one_call_meet() {
    let mut fx = Fixture::new();
    let (mut_int, int) = (mut_int(&mut fx), fx.int());
    fx.declare("f2", &[mut_int, mut_int], int);
    refused_within(fx, "let x = 1; f2(&mut x, &mut x)", "f2(&mut x, &mut x)");
}

/// extern-fn §4: `g` states no lifetimes, so its result takes the loan of
/// `&mut x`. Writing `x` while `r` lives meets it.
#[test]
fn a_write_meets_what_an_unlabelled_result_takes_from_an_inline_mut_lend() {
    let mut fx = Fixture::new();
    declare_g(&mut fx, false);
    refused_within(fx, "let x = 1; let r = g(&mut x); x = 5; *r", "x = 5;");
}

/// The same where `g`'s lifetimes give its result the loan of `&mut x`.
#[test]
fn a_write_meets_what_a_labelled_result_takes_from_an_inline_mut_lend() {
    let mut fx = Fixture::new();
    declare_g(&mut fx, true);
    refused_within(fx, "let x = 1; let r = g(&mut x); x = 5; *r", "x = 5;");
}

/// The same lend bound to `w` first.
#[test]
fn a_write_meets_what_a_result_takes_from_a_named_mut_lend() {
    let mut fx = Fixture::new();
    declare_g(&mut fx, true);
    refused_within(fx, "let x = 1; let w = &mut x; let r = g(w); x = 5; *r", "x = 5;");
}

/// ownership §5: a lambda's flows are what its body does, so `f`'s result
/// takes the loan of `&mut x`.
#[test]
fn a_write_meets_what_a_lambdas_result_takes_from_an_inline_mut_lend() {
    let fx = Fixture::new();
    refused_within(fx, "let x = 1; let f = |p| -> p; let r = f(&mut x); x = 5; *r", "x = 5;");
}

/// The lend of `a`'s element as `set`'s argument meets the shared loan on
/// all of `a` that `r` holds: every element of a sequence is its one part.
#[test]
fn an_inline_mut_lend_of_an_element_meets_a_live_shared_loan() {
    let mut fx = Fixture::new();
    declare_set(&mut fx);
    refused_within(fx, "let a = [1, 2]; let r = &a; set(&mut a[0]); r[1]", "set(&mut a[0])");
}

/// Each run of the loop body makes its own lend where `set` starts, and
/// nothing holds the previous run's after its call: nothing is met.
#[test]
fn an_inline_mut_lend_in_a_loop_meets_nothing() {
    let mut fx = Fixture::new();
    declare_set(&mut fx);
    assert_eq!(fx.refusals("let a = 1; for i in 0..3 { set(&mut a); } a"), []);
}

/// The same where the call's result takes the lend's loan and is used
/// within the run that made it.
#[test]
fn an_inline_mut_lend_a_result_takes_in_a_loop_meets_nothing() {
    let mut fx = Fixture::new();
    declare_g(&mut fx, true);
    assert_eq!(fx.refusals("let a = 1; for i in 0..3 { let r = g(&mut a); *r = i; } a"), []);
}
