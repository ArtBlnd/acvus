//! Statements, and a body of statements and a tail.

use acvus_ast::{Expr, Slot, Span, Stmt};

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::expr::check_expr;
use super::place::{self, Demand};
use super::refusal::{self, RefusalKind};
use super::scope::{self, Binding};
use super::{control, template};
use crate::decl::{Base, Head};
use crate::infer::{Closed, Term};

/// Statements then a tail: the tail's term, or without one `()`, or `!`
/// where the last statement diverges (control-flow). A `break` or a
/// `continue` diverges as written; an expression statement diverges where
/// its type is `!`, decided once that type is known (types §6).
pub(super) fn check_body<S: Slot>(cx: &mut Ctxt<'_>, stmts: &[Stmt<S>], tail: Option<&Expr<S>>) -> Term {
    if let Some(tail) = tail {
        check_stmts(cx, stmts);
        return check_expr(cx, tail, Demand::Read);
    }
    let Some((last, leading)) = stmts.split_last() else {
        return cx.unit();
    };
    check_stmts(cx, leading);
    match last {
        Stmt::Expr(expr) => {
            let last = check_expr(cx, expr, Demand::Read);
            let block = cx.fresh();
            let ends = Decision::BlockEnd {
                span: expr.span(),
                last,
                block,
            };
            super::decision::decide(cx, ends);
            block
        }
        Stmt::Break { .. } | Stmt::Continue { .. } => {
            check_stmt(cx, last);
            cx.never()
        }
        _ => {
            check_stmt(cx, last);
            cx.unit()
        }
    }
}

/// A block with no tail whose last statement is an expression, fired on
/// that expression's type: `!` where it is `!`, and `()` otherwise
/// (control-flow).
pub(super) fn fire_block_end(cx: &mut Ctxt<'_>, span: Span, last: Closed, block: Term) {
    let ends = match last.head {
        Head::Base(Base::Never) => cx.never(),
        _ => cx.unit(),
    };
    cx.eq(block, ends, span);
}

pub(super) fn check_stmts<S: Slot>(cx: &mut Ctxt<'_>, stmts: &[Stmt<S>]) {
    for stmt in stmts {
        check_stmt(cx, stmt);
    }
}

/// `Store`, `DerefStore`, `LetBind`, `LetUninit`, `Assign`, `While`,
/// `WhileLet`, `For`, `Break`, `Continue`, `Anyorder`, `Append`, `Text`,
/// `Expr`.
/// A `FnDecl` is not a script's (programs §2): it is refused as a form.
pub(super) fn check_stmt<S: Slot>(cx: &mut Ctxt<'_>, stmt: &Stmt<S>) {
    match stmt {
        Stmt::Store { id, place, expr, span } => place::check_store(cx, *id, place, expr, *span),
        Stmt::DerefStore { id, target, expr, span } => place::check_deref_store(cx, *id, target, expr, *span),
        Stmt::Expr(expr) => {
            check_expr(cx, expr, Demand::Read);
        }
        Stmt::LetBind { binder, expr, .. } => {
            // The value, and its entering the binding, are the binding's own
            // walk (S7), which a generalized binding's scheme reads (G).
            let (value, term) = super::generalize::walk(cx, binder.id, |cx| {
                let value = check_expr(cx, expr, Demand::Read);
                let term = cx.fresh();
                cx.enter(expr.id(), value, term, crate::ty::Reach::Named, expr.span());
                (value, term)
            });
            cx.record(binder.id, term);
            let scheme = super::generalize::generalizes(cx, value).then(|| super::generalize::register(cx, binder.id, term));
            cx.scope.define(
                binder.name,
                Binding {
                    binder: binder.id,
                    term,
                    scheme,
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
                    scheme: None,
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
            // A value given a generalized binding is its binding's walk, as
            // the `let`'s is (G): its scheme is settled from every value it
            // is given.
            let assign = |cx: &mut Ctxt<'_>| {
                let value = check_expr(cx, expr, Demand::Read);
                scope::check_assign(cx, *id, *name, value, *name_span);
            };
            match cx.scope.lookup(*name).filter(|found| found.binding.scheme.is_some()) {
                Some(found) => super::generalize::walk(cx, found.binding.binder, assign),
                None => assign(cx),
            }
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
        Stmt::Break { span, .. } | Stmt::Continue { span, .. } => {
            if !cx.scope.in_loop() {
                refusal::push(cx, RefusalKind::OutsideLoop, *span);
            }
        }
        // typeck checks an `anyorder` body as plain statements; the order
        // token it cuts is the lowerer's (D29, effects §3).
        Stmt::Anyorder { body, .. } => {
            cx.scope.push();
            check_stmts(cx, body);
            cx.scope.pop();
        }
        Stmt::FnDecl(decl) => refusal::push(cx, RefusalKind::FnDeclInScript, decl.span),
        Stmt::Append { id, expr, span } => template::check_append(cx, *id, expr, *span),
        // Text is appended as written; no type decides anything of it
        // (programs §1).
        Stmt::Text { .. } => {}
        Stmt::Error(_) => {}
    }
}
