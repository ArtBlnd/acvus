//! Statements and bodies. A binding is a slot (`Builder::slot`); promotion
//! to SSA is a pass. A statement whose value is `!` ends its block: no run
//! gets past it (types §4).

use acvus_ast::{Expr, Slot, Span, Stmt};

use super::call::convert;
use super::control::{lower_for, lower_loop_exit, lower_while, lower_while_let};
use super::ctxt::{Home, Lcx};
use super::expr::lower_expr;
use super::place::{Place, assign, binding_place, lower_deref_store, lower_store};
use super::template::{lower_append, lower_text};
use crate::ir::cfg::Terminator;
use crate::ir::{RefTarget, ValueId};
use crate::ty::TyKind;
use crate::typeck::Name;

/// Statements then a tail: the tail's value, or without one `()`.
pub(super) fn lower_body<S: Slot>(cx: &mut Lcx<'_, '_>, stmts: &[Stmt<S>], tail: Option<&Expr<S>>, span: Span) -> ValueId {
    lower_stmts(cx, stmts);
    match tail {
        Some(tail) => lower_expr(cx, tail),
        None => cx.unit(span),
    }
}

pub(super) fn lower_stmts<S: Slot>(cx: &mut Lcx<'_, '_>, stmts: &[Stmt<S>]) {
    for stmt in stmts {
        lower_stmt(cx, stmt);
    }
}

pub(super) fn lower_stmt<S: Slot>(cx: &mut Lcx<'_, '_>, stmt: &Stmt<S>) {
    match stmt {
        Stmt::Expr(expr) => {
            lower_expr(cx, expr);
            if matches!(cx.kind(cx.ty(expr.id())), TyKind::Never) && cx.reachable {
                cx.end_block(Terminator::Diverge, expr.span());
            }
        }
        Stmt::LetBind { binder, expr, .. } => {
            let value = lower_expr(cx, expr);
            let value = convert(cx, expr.id(), value, expr.span());
            let ty = cx.ty(binder.id);
            let slot = cx.builder.slot(ty, binder.span);
            assign(cx, Place { target: RefTarget::Var(slot), path: Vec::new(), ty }, value, binder.span);
            cx.bind(binder.id, Home::Slot(slot), binder.span);
        }
        Stmt::LetUninit { binder, .. } => {
            let ty = cx.ty(binder.id);
            let slot = cx.builder.slot(ty, binder.span);
            cx.bind(binder.id, Home::Slot(slot), binder.span);
        }
        Stmt::Assign { id, expr, span, .. } => {
            let Some(Name::Binding(binder)) = cx.typeck.names.get(id).copied() else {
                panic!("an assignment names a binding in an admitted program")
            };
            let value = lower_expr(cx, expr);
            let value = convert(cx, *id, value, expr.span());
            let ty = cx.ty(binder);
            let place = binding_place(cx, binder, ty);
            assign(cx, place, value, *span);
        }
        Stmt::Store { place, expr, span, .. } => lower_store(cx, place, expr, *span),
        Stmt::DerefStore { target, expr, span, .. } => lower_deref_store(cx, target, expr, *span),
        Stmt::While { cond, body, span, .. } => lower_while(cx, cond, body, *span),
        Stmt::WhileLet { pattern, source, body, span, .. } => lower_while_let(cx, pattern, source, body, *span),
        Stmt::For { id, callee_id, binder, head, body, span } => lower_for(cx, *id, *callee_id, binder, head, body, *span),
        Stmt::Break { span, .. } => lower_loop_exit(cx, true, *span),
        Stmt::Continue { span, .. } => lower_loop_exit(cx, false, *span),
        Stmt::Anyorder { body, .. } => {
            cx.builder.open_anyorder();
            cx.anyorder += 1;
            lower_stmts(cx, body);
            cx.anyorder -= 1;
            cx.builder.close_anyorder();
        }
        Stmt::Text { value, span, .. } => lower_text(cx, value, *span),
        Stmt::Append { id, expr, span } => lower_append(cx, *id, expr, *span),
        Stmt::FnDecl(_) => panic!("a `fn` in a script is refused (FnDeclInScript), and never lowered"),
        Stmt::Error(_) => panic!("a statement the parse left in error is refused, and never lowered"),
    }
}

