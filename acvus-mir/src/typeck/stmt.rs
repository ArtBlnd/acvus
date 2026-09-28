//! Statements, and a body of statements and a tail.

use acvus_ast::{Expr, Slot, Stmt};

use super::ctxt::Ctxt;
use super::expr::check_expr;
use super::place::{self, Demand};
use super::refusal::{self, RefusalKind};
use super::scope::{self, Binding};
use super::{control, template};
use crate::infer::Term;
use crate::ty::TyKind;

/// Statements then a tail: the tail's term, or without one `()`, or `!`
/// where the last statement diverges (control-flow).
pub(super) fn check_body<S: Slot>(cx: &mut Ctxt<'_>, stmts: &[Stmt<S>], tail: Option<&Expr<S>>) -> Term {
    check_stmts(cx, stmts);
    match tail {
        Some(tail) => check_expr(cx, tail, Demand::Read),
        None if stmts.last().is_some_and(|last| diverges(cx, last)) => cx.never(),
        None => cx.unit(),
    }
}

/// Whether `stmt` never finishes: a `break`, a `continue`, or an
/// expression of type `!` as far as it is known when the block is checked.
///
/// NOTE: a statement whose type becomes `!` only after solving leaves its
/// block `()`; a block's type is fixed when it is checked.
fn diverges<S: Slot>(cx: &Ctxt<'_>, stmt: &Stmt<S>) -> bool {
    match stmt {
        Stmt::Break { .. } | Stmt::Continue { .. } => true,
        Stmt::Expr(expr) => cx
            .out
            .types
            .get(&expr.id())
            .is_some_and(|term| matches!(cx.infer.types().kind(cx.infer.shallow(*term)), TyKind::Never)),
        _ => false,
    }
}

pub(super) fn check_stmts<S: Slot>(cx: &mut Ctxt<'_>, stmts: &[Stmt<S>]) {
    for stmt in stmts {
        check_stmt(cx, stmt);
    }
}

/// `Store`, `DerefStore`, `LetBind`, `LetUninit`, `Assign`, `While`,
/// `WhileLet`, `For`, `Break`, `Continue`, `Anyorder`, `Append`, `Expr`.
/// A `FnDecl` is not a script's (programs §2): it is refused as a form.
pub(super) fn check_stmt<S: Slot>(cx: &mut Ctxt<'_>, stmt: &Stmt<S>) {
    match stmt {
        Stmt::Store { id, place, expr, span } => place::check_store(cx, *id, place, expr, *span),
        Stmt::DerefStore { id, target, expr, span } => place::check_deref_store(cx, *id, target, expr, *span),
        Stmt::Expr(expr) => {
            check_expr(cx, expr, Demand::Read);
        }
        Stmt::LetBind { binder, expr, .. } => {
            let value = check_expr(cx, expr, Demand::Read);
            let term = cx.fresh();
            cx.enter(expr.id(), value, term, expr.span());
            cx.record(binder.id, term);
            cx.scope.define(
                binder.name,
                Binding {
                    binder: binder.id,
                    term,
                },
            );
        }
        Stmt::LetUninit { binder, .. } => {
            let term = cx.fresh();
            cx.record(binder.id, term);
            cx.scope.define(
                binder.name,
                Binding {
                    binder: binder.id,
                    term,
                },
            );
        }
        Stmt::Assign {
            id,
            name,
            name_span,
            expr,
            ..
        } => {
            let value = check_expr(cx, expr, Demand::Read);
            scope::check_assign(cx, *id, *name, value, *name_span);
        }
        Stmt::While { cond, body, .. } => control::check_while(cx, cond, body),
        Stmt::WhileLet {
            pattern, source, body, ..
        } => control::check_while_let(cx, pattern, source, body),
        Stmt::For {
            id,
            callee_id,
            binder,
            head,
            body,
            ..
        } => control::check_for(cx, *id, *callee_id, binder, head, body),
        Stmt::Break { .. } | Stmt::Continue { .. } => {}
        // typeck checks an `anyorder` body as plain statements; the order
        // token it cuts is the lowerer's (D29, effects §3).
        Stmt::Anyorder { body, .. } => {
            cx.scope.push();
            check_stmts(cx, body);
            cx.scope.pop();
        }
        Stmt::FnDecl(decl) => refusal::push(cx, RefusalKind::FnDeclInScript, decl.span),
        Stmt::Append { id, expr, span } => template::check_append(cx, *id, expr, *span),
        Stmt::Error(_) => {}
    }
}
