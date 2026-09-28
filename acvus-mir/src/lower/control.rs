//! Control forms, each one region with one entry and one exit (control-flow). `if`/`if let` and short-circuit `&&`/`||` are a `Diamond` the Builder makes with its join known; `match` is `pat`'s; `for` is a `For` by typeck's `Iteration` (`&`/`&mut` through the recorded `as_slice(_mut)`, arrays by value, ranges); `while`/`while let` a `While`; `break`/`continue` go to the innermost loop's exit/header; `return` is the body's `Return`; `?` tests the tag and returns the `Err`/`None` early.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_if<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, cond: &Expr<S>, then: &[Stmt<S>], then_tail: Option<&Expr<S>>, otherwise: Option<&ElseBranch<S>>) -> ValueId {
    todo!("lower::control::lower_if")
}

pub(super) fn lower_short_circuit<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, op: BinOp, left: &Expr<S>, right: &Expr<S>) -> ValueId {
    todo!("lower::control::lower_short_circuit")
}

pub(super) fn lower_for<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, head: &ForHead<S>, body: &[Stmt<S>]) {
    todo!("lower::control::lower_for")
}

pub(super) fn lower_while<S: Slot>(cx: &mut Lcx<'_, '_>, cond: &Expr<S>, body: &[Stmt<S>]) {
    todo!("lower::control::lower_while")
}

pub(super) fn lower_while_let<S: Slot>(cx: &mut Lcx<'_, '_>, pattern: &Pattern<S>, source: &Expr<S>, body: &[Stmt<S>]) {
    todo!("lower::control::lower_while_let")
}

pub(super) fn lower_try<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, inner: &Expr<S>, span: Span) -> ValueId {
    todo!("lower::control::lower_try")
}
