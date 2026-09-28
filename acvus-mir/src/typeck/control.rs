//! Control flow (control-flow): branches `enter` one slot (types §2-3), a
//! diverging block is `!` (types §4), and `for` and `?` are decided by type.

use acvus_ast::{AstId, ElseBranch, Expr, ForHead, MatchExprArm, Pattern, Slot, Span, Stmt};

use super::ctxt::Ctxt;
use crate::infer::Term;

pub(super) fn check_if<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    cond: &Expr<S>,
    then_body: &[Stmt<S>],
    then_tail: Option<&Expr<S>>,
    else_branch: Option<&ElseBranch<S>>,
) -> Term {
    todo!("typeck::control::check_if")
}

pub(super) fn check_if_let<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    pattern: &Pattern<S>,
    source: &Expr<S>,
    then_body: &[Stmt<S>],
    then_tail: Option<&Expr<S>>,
    else_branch: Option<&ElseBranch<S>>,
) -> Term {
    todo!("typeck::control::check_if_let")
}

pub(super) fn check_match<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, scrutinee: &Expr<S>, arms: &[MatchExprArm<S>]) -> Term {
    todo!("typeck::control::check_match")
}

pub(super) fn check_while<S: Slot>(cx: &mut Ctxt<'_>, cond: &Expr<S>, body: &[Stmt<S>]) {
    todo!("typeck::control::check_while")
}

pub(super) fn check_while_let<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>, source: &Expr<S>, body: &[Stmt<S>]) {
    todo!("typeck::control::check_while_let")
}

/// A `for`: an `Iterate` decision on its head's type (control-flow); by
/// reference, its elements reach `as_slice(_mut)` at `callee_site`
/// (operators §3).
pub(super) fn check_for<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    callee_site: AstId,
    binder: &acvus_ast::Binder,
    head: &ForHead<S>,
    body: &[Stmt<S>],
) {
    todo!("typeck::control::check_for")
}

/// `?`: a `Try` decision on its operand's type (control-flow).
pub(super) fn check_try<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, inner: &Expr<S>, span: Span) -> Term {
    todo!("typeck::control::check_try")
}

/// `return v`: `v` enters the body's result; the form is `!`.
pub(super) fn check_return<S: Slot>(cx: &mut Ctxt<'_>, value: &Expr<S>, span: Span) -> Term {
    todo!("typeck::control::check_return")
}
