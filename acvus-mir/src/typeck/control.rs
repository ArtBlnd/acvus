//! Control flow (control-flow): branches `enter` one slot (types §2-3), a
//! diverging block is `!` (types §4), and `for` and `?` are decided by type.

use acvus_ast::{AstId, Binder, ElseBranch, Expr, ForHead, MatchExprArm, Pattern, Slot, Span, Stmt};

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::expr::check_expr;
use super::output::Iteration;
use super::pat::check_pattern;
use super::place::Demand;
use super::scope::Binding;
use super::stmt::{check_body, check_stmts};
use crate::infer::Term;

pub(super) fn check_if<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    cond: &Expr<S>,
    then_body: &[Stmt<S>],
    then_tail: Option<&Expr<S>>,
    else_branch: Option<&ElseBranch<S>>,
) -> Term {
    check_condition(cx, cond);
    let result = cx.fresh();
    branch(cx, site, result, then_body, then_tail, cond.span());
    check_else(cx, site, result, else_branch, cond.span());
    result
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
    let scrutinee = check_expr(cx, source, Demand::Read);
    let result = cx.fresh();
    cx.scope.push();
    check_pattern(cx, pattern, scrutinee);
    let then = check_body(cx, then_body, then_tail);
    cx.scope.pop();
    enter_branch(cx, site, then, then_tail, result, pattern.span());
    check_else(cx, site, result, else_branch, pattern.span());
    result
}

pub(super) fn check_match<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, scrutinee: &Expr<S>, arms: &[MatchExprArm<S>]) -> Term {
    let scrutinee = check_expr(cx, scrutinee, Demand::Read);
    let result = cx.fresh();
    for arm in arms {
        cx.scope.push();
        check_pattern(cx, &arm.pattern, scrutinee);
        let term = check_body(cx, &arm.body, arm.tail.as_deref());
        cx.scope.pop();
        enter_branch(cx, arm.id, term, arm.tail.as_deref(), result, arm.span);
    }
    result
}

pub(super) fn check_while<S: Slot>(cx: &mut Ctxt<'_>, cond: &Expr<S>, body: &[Stmt<S>]) {
    check_condition(cx, cond);
    cx.scope.push();
    check_stmts(cx, body);
    cx.scope.pop();
}

pub(super) fn check_while_let<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>, source: &Expr<S>, body: &[Stmt<S>]) {
    let scrutinee = check_expr(cx, source, Demand::Read);
    cx.scope.push();
    check_pattern(cx, pattern, scrutinee);
    check_stmts(cx, body);
    cx.scope.pop();
}

/// A `for`: an `Iterate` decision on its head's type (control-flow); by
/// reference, its elements reach `as_slice(_mut)` at `callee_site`
/// (operators §3). A range iterates its integers by value.
pub(super) fn check_for<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    callee_site: AstId,
    binder: &Binder,
    head: &ForHead<S>,
    body: &[Stmt<S>],
) {
    let item = match head {
        ForHead::Value(expr) => {
            let head = check_expr(cx, expr, Demand::Read);
            let item = cx.fresh();
            cx.decisions.push(Decision::Iterate {
                site,
                callee_site,
                span: expr.span(),
                head,
                item,
            });
            item
        }
        ForHead::Range { lo, hi } => {
            let low = check_expr(cx, lo, Demand::Read);
            let high = check_expr(cx, hi, Demand::Read);
            // A range's ends are integers of one type.
            let span = lo.span();
            let integer = cx.integer(span);
            cx.eq(low, integer, span);
            cx.eq(high, integer, hi.span());
            cx.out.iteration.insert(site, Iteration::Range);
            integer
        }
    };
    cx.record(binder.id, item);
    cx.scope.push();
    cx.scope.define(
        binder.name,
        Binding {
            binder: binder.id,
            term: item,
        },
    );
    check_stmts(cx, body);
    cx.scope.pop();
}

/// `?`: a `Try` decision on its operand's type (control-flow); what it
/// leaves with enters the body's result.
pub(super) fn check_try<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, inner: &Expr<S>, span: Span) -> Term {
    let operand = check_expr(cx, inner, Demand::Read);
    let value = cx.fresh();
    let exit = cx.fresh();
    cx.decisions.push(Decision::Try {
        site,
        span,
        operand,
        value,
        exit,
    });
    let result = body_result(cx);
    cx.enter(site, exit, result, span);
    value
}

/// `return v`: `v` enters the body's result; the form is `!`.
pub(super) fn check_return<S: Slot>(cx: &mut Ctxt<'_>, value: &Expr<S>, span: Span) -> Term {
    let term = check_expr(cx, value, Demand::Read);
    let result = body_result(cx);
    cx.enter(value.id(), term, result, span);
    cx.never()
}

/// The slot the innermost body leaves through; the run's is named the
/// first time something leaves through it.
pub(super) fn body_result(cx: &mut Ctxt<'_>) -> Term {
    if let Some(result) = cx.scope.body_result() {
        return result;
    }
    let result = cx.fresh();
    cx.scope.set_run_result(result);
    result
}

fn check_condition<S: Slot>(cx: &mut Ctxt<'_>, cond: &Expr<S>) {
    let term = check_expr(cx, cond, Demand::Read);
    let bool_ty = cx.bool();
    cx.eq(term, bool_ty, cond.span());
}

/// A branch's block in its own scope, entering `result`; a refusal there
/// points at its tail, or at `head` where it has none.
fn branch<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, result: Term, body: &[Stmt<S>], tail: Option<&Expr<S>>, head: Span) {
    cx.scope.push();
    let term = check_body(cx, body, tail);
    cx.scope.pop();
    let span = tail.map_or(head, Expr::span);
    enter_branch(cx, site, term, tail, result, span);
}

/// A branch's value enters the one slot every branch enters (types §3), at
/// its tail's site, or at the construct's where it has none.
fn enter_branch<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, term: Term, tail: Option<&Expr<S>>, result: Term, span: Span) {
    let at = tail.map_or(site, Expr::id);
    cx.enter(at, term, result, span);
}

/// The `else`: without one, the missing branch is `()` (control-flow),
/// refused at `head` where that does not fit.
fn check_else<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, result: Term, else_branch: Option<&ElseBranch<S>>, head: Span) {
    match else_branch {
        None => {
            let unit = cx.unit();
            cx.enter(site, unit, result, head);
        }
        Some(ElseBranch::ElseIf(expr)) => {
            let term = check_expr(cx, expr, Demand::Read);
            cx.enter(expr.id(), term, result, expr.span());
        }
        Some(ElseBranch::Else { body, tail, span }) => {
            cx.scope.push();
            let term = check_body(cx, body, tail.as_deref());
            cx.scope.pop();
            enter_branch(cx, site, term, tail.as_deref(), result, *span);
        }
    }
}
