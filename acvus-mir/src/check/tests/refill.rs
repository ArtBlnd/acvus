//! moveck's refill and move-through judgments over programs built from
//! source (ownership §4, §5; context §1; functions §2).

use super::{Fixture, covers, span_of};
use crate::check::{Refusal, RefusalKind};
use crate::ty::TyKind;

/// A fixture with `make`, `consume` and a context `@c: Vec<i64>`.
fn with_context() -> Fixture {
    let mut fx = Fixture::new();
    let vec = fx.vecs();
    fx.context("c", vec);
    fx
}

/// context §1, §3: a context taken out is filled again before its commit
/// gives it back.
#[test]
fn a_context_taken_and_not_written_back_refuses_its_commit() {
    let source = "consume(@c); 0";
    let refusals = with_context().refusals(source);
    let [Refusal { kind: RefusalKind::NotRefilled { taken }, span }] = refusals[..] else {
        panic!("one NotRefilled: {refusals:?}");
    };
    assert_eq!(taken, span_of(source, "@c", 0), "naming the take");
    assert!(covers(span, span_of(source, "0", 0)), "at the commit, which the return makes: {span:?}");
}

/// context §1: a context written again after its take is filled at the
/// return.
#[test]
fn a_context_taken_and_written_back_is_admitted() {
    let refusals = with_context().refusals("consume(@c); @c = make(); 0");
    assert_eq!(refusals, []);
}

/// functions §2, ownership §5: a lambda without `move` captures `v` by a
/// shared reference here, and what a shared reference lends is never taken
/// from: the move out of it is refused, and nothing is taken.
#[test]
fn a_lambda_moving_a_capture_it_borrows_is_refused() {
    let source = "let v = make(); let f = |x| -> consume(v); f(1)";
    let refusals = with_context().refusals(source);
    let [Refusal { kind: RefusalKind::MoveThroughShared, span }] = refusals[..] else {
        panic!("one MoveThroughShared: {refusals:?}");
    };
    assert_eq!(span, span_of(source, "v", 1), "at the move through the capture");
}

/// functions §2: with `move`, the lambda owns `v`.
#[test]
fn a_move_lambda_moving_its_capture_is_admitted() {
    let refusals = with_context().refusals("let v = make(); let f = move |x| -> consume(v); f(1)");
    assert_eq!(refusals, []);
}

/// ownership §4: a part taken out through a `&mut` parameter and written
/// back on every path to the return.
#[test]
fn a_function_writing_back_what_it_took_through_a_mut_parameter_is_admitted() {
    let source = "let f = |r| -> { let t = *r; *r = make(); consume(t) }; let v = make(); f(&mut v)";
    let refusals = with_context().refusals(source);
    assert_eq!(refusals, []);
}

/// ownership §4: what a `&mut` parameter lends is whole whenever the body
/// returns.
#[test]
fn a_function_not_writing_back_what_it_took_through_a_mut_parameter_is_refused() {
    let source = "let f = |r| -> { let t = *r; consume(t) }; let v = make(); f(&mut v)";
    let refusals = with_context().refusals(source);
    let [Refusal { kind: RefusalKind::NotRefilled { taken }, span }] = refusals[..] else {
        panic!("one NotRefilled: {refusals:?}");
    };
    assert_eq!(taken, span_of(source, "*r", 0), "naming the take through the parameter");
    assert!(covers(span, span_of(source, "|r| -> { let t = *r; consume(t) }", 0)), "at the lambda's return: {span:?}");
}

/// ownership §5: what a shared reference lends is never taken from.
#[test]
fn a_move_through_a_shared_reference_is_refused() {
    let source = "let v = make(); let r = &v; let t = *r; consume(t)";
    let refusals = with_context().refusals(source);
    let [Refusal { kind: RefusalKind::MoveThroughShared, span }] = refusals[..] else {
        panic!("one MoveThroughShared: {refusals:?}");
    };
    assert_eq!(span, span_of(source, "*r", 0), "at the move");
}

/// ownership §4: a reference that lends one of two slots, by the path taken,
/// lends no one place to take from.
#[test]
fn a_move_through_a_reference_lending_one_of_two_places_is_refused() {
    let mut fx = with_context();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("b", bool_ty);
    let source = "let x = make(); let y = make(); let r = if $b { &mut x } else { &mut y }; let t = *r; *r = make(); consume(t)";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::MoveThroughSeveral, span }] = refusals[..] else {
        panic!("one MoveThroughSeveral: {refusals:?}");
    };
    assert_eq!(span, span_of(source, "*r", 0), "at the move");
}

/// ownership §2, §4: a reference that lends one slot on every path moves
/// out of that slot, which is not used after.
#[test]
fn a_move_through_a_reference_lending_one_place_moves_that_place() {
    let source = "let x = make(); let r = &mut x; let t = *r; consume(t); consume(x)";
    let refusals = with_context().refusals(source);
    let [Refusal { kind: RefusalKind::UsedAfterMove { moved }, span }] = refusals[..] else {
        panic!("one UsedAfterMove: {refusals:?}");
    };
    assert_eq!(moved, span_of(source, "*r", 0), "naming the move through `r`");
    assert!(covers(span, span_of(source, "x", 2)), "at the use of `x`: {span:?}");
}
