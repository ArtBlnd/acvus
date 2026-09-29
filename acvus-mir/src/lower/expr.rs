//! Expression dispatch: an expression becomes a typed value in the current
//! block, typed by `Typeck::types` at its site.

use acvus_ast::*;

use super::call::{convert, lower_binary, lower_call, lower_method, lower_neg, lower_not, lower_pipe};
use super::closure::lower_lambda;
use super::context::lower_context_read;
use super::control::{lower_if, lower_if_let, lower_return, lower_try};
use super::ctxt::Lcx;
use super::pat::lower_match;
use super::place::{Operand, lend, lower_borrow, lower_deref, lower_index, operand, read};
use super::stmt::lower_body;
use crate::ir::cfg::Op;
use crate::ir::ValueId;
use crate::ty::{LenTerm, Mutability, TyKind};
use crate::typeck::Name;

pub(super) fn lower_expr<S: Slot>(cx: &mut Lcx<'_, '_>, expr: &Expr<S>) -> ValueId {
    acvus_utils::grow(|| lower_form(cx, expr))
}

fn lower_form<S: Slot>(cx: &mut Lcx<'_, '_>, expr: &Expr<S>) -> ValueId {
    match expr {
        Expr::Ident { id, span, .. } => lower_ident(cx, *id, expr, *span),
        Expr::Literal { id, value, span } => lower_literal(cx, *id, value, *span),
        Expr::BinaryOp { id, left, op, right, span } => lower_binary(cx, *id, *op, left, right, *span),
        Expr::UnaryOp { id, op: UnaryOp::Deref, operand, span } => lower_deref(cx, *id, operand, *span),
        Expr::UnaryOp { id, op: UnaryOp::Neg, operand, span } => lower_neg(cx, *id, operand, *span),
        Expr::UnaryOp { id, op: UnaryOp::Not, operand, span } => lower_not(cx, *id, operand, *span),
        Expr::FieldAccess { span, .. } => {
            let field = operand(cx, expr, Mutability::Shared);
            read(cx, field, *span)
        }
        Expr::Index { id, callee_id, object, index, span } => lower_index(cx, *id, *callee_id, object, index, *span),
        Expr::FuncCall { id, func, args, span } => lower_call(cx, *id, func, args, *span),
        Expr::MethodCall { id, callee_id, receiver, args, span, .. } => {
            lower_method(cx, *id, *callee_id, receiver, args, *span)
        }
        Expr::Pipe { id, left, right, span } => lower_pipe(cx, *id, left, right, *span),
        Expr::Lambda { id, params, body, span, .. } => lower_lambda(cx, *id, params, body, *span),
        Expr::Paren { inner, .. } => lower_expr(cx, inner),
        Expr::Borrow { mutable, place, span, .. } => lower_borrow(cx, *mutable, place, *span),
        Expr::List { id, head, tail, span, .. } => lower_list(cx, *id, head, tail, *span),
        Expr::Object { id, fields, span } => {
            let fields = fields
                .iter()
                .map(|field| {
                    let value = lower_expr(cx, &field.value);
                    (field.key, value)
                })
                .collect();
            let ty = cx.ty(*id);
            cx.builder.push(Op::MakeObject { fields }, ty, *span)
        }
        Expr::Tuple { id, elements, span } => lower_tuple(cx, *id, elements, *span),
        Expr::Block { stmts, tail, span, .. } => lower_body(cx, stmts, Some(tail), *span),
        Expr::ContextRef { id, name, span } => {
            let ty = cx.ty(*id);
            lower_context_read(cx, *name, ty, *span)
        }
        Expr::Variant { id, tag, payload, span, .. } => {
            let payload = payload.as_deref().map(|payload| lower_expr(cx, payload));
            let ty = cx.ty(*id);
            cx.builder.push(Op::MakeVariant { tag: *tag, payload }, ty, *span)
        }
        Expr::Cast { id, expr: source, span, .. } => {
            let value = lower_expr(cx, source);
            let ty = cx.ty(*id);
            match cx.typeck.casts.get(id) {
                Some(cast) => cx.builder.push(Op::Cast { src: value, to: cast.to }, ty, *span),
                None => {
                    assert!(
                        matches!(cx.kind(cx.value_ty(value)), TyKind::Never),
                        "typeck records every `as` but one of a `!` source"
                    );
                    cx.unreached_value(ty, *span)
                }
            }
        }
        Expr::Try { id, inner, span } => lower_try(cx, *id, inner, *span),
        Expr::Return { id, value, span } => lower_return(cx, *id, value, *span),
        Expr::If { id, cond, then_body, then_tail, else_branch, span } => {
            lower_if(cx, *id, cond, then_body, then_tail.as_deref(), else_branch.as_deref(), *span)
        }
        Expr::IfLet { id, pattern, source, then_body, then_tail, else_branch, span } => lower_if_let(
            cx,
            *id,
            pattern,
            source,
            then_body,
            then_tail.as_deref(),
            else_branch.as_deref(),
            *span,
        ),
        Expr::Match { id, scrutinee, arms, span } => lower_match(cx, *id, scrutinee, arms, *span),
        Expr::Group { .. } => panic!("a group standing as a value is refused (PatternInValue), and never lowered"),
        Expr::Error(_) => panic!("an expression the parse left in error is refused, and never lowered"),
    }
}

/// A name: a binding's value, a declaration as a function value, a
/// context's value, or an input's.
fn lower_ident<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, expr: &Expr<S>, span: Span) -> ValueId {
    let ty = cx.ty(site);
    match cx.typeck.names.get(&site).copied() {
        Some(Name::Declaration(declaration)) => cx.builder.push(Op::LoadFunction { id: declaration }, ty, span),
        Some(Name::Context(context)) => lower_context_read(cx, context, ty, span),
        Some(Name::Binding(_) | Name::Input(_)) => {
            let place = operand(cx, expr, Mutability::Shared);
            read(cx, place, span)
        }
        None => panic!("typeck names every name of an admitted program, and not {site:?}"),
    }
}

/// A literal at its recorded type. A string literal is a `&str` (operators §1); a
/// byte string a reference to the array of its bytes.
pub(super) fn lower_literal(cx: &mut Lcx<'_, '_>, site: AstId, literal: &Literal, span: Span) -> ValueId {
    let ty = cx.ty(site);
    match literal {
        Literal::String(text) => cx.builder.push(Op::ConstStr { text: text.clone() }, ty, span),
        Literal::Bytes(_) => {
            let TyKind::Ref(_, array) = cx.kind(ty) else {
                panic!("a byte string is a reference to an array of its bytes")
            };
            let array = array.ty();
            let bytes = cx.constant(literal.clone(), array, span);
            lend(cx, Operand::Value(bytes), Mutability::Shared, span)
        }
        Literal::Int(_) | Literal::IntOf(_) | Literal::Float(_) | Literal::Char(_) | Literal::Bool(_) | Literal::List(_) | Literal::Unit => {
            cx.constant(literal.clone(), ty, span)
        }
    }
}

/// `[a, b]`: an empty array, each element pushed onto its end in order,
/// each entering the element type at its own site.
fn lower_list<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, head: &[Expr<S>], tail: &[Expr<S>], span: Span) -> ValueId {
    let ty = cx.ty(site);
    let TyKind::Array(element, _, _) = cx.kind(ty) else {
        panic!("a list is an array")
    };
    let element = *element;
    let items: Vec<&Expr<S>> = head.iter().chain(tail).collect();
    let empty = cx.builder.intern(TyKind::Array(element, LenTerm::Known(0), ()));
    let mut array = cx.builder.push(Op::ArrayBegin { capacity: items.len() }, empty, span);
    for (at, item) in items.iter().enumerate() {
        let value = lower_expr(cx, item);
        let value = convert(cx, item.id(), value, item.span());
        let longer = cx.builder.intern(TyKind::Array(element, LenTerm::Known(at + 1), ()));
        array = cx.builder.push(Op::ArrayPush { array, value }, longer, item.span());
    }
    array
}

/// `(a, b)`: a tuple of its parts; `()` is the unit.
fn lower_tuple<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, elements: &[TupleElem<S>], span: Span) -> ValueId {
    let ty = cx.ty(site);
    if elements.is_empty() {
        return cx.constant(Literal::Unit, ty, span);
    }
    let elements = elements
        .iter()
        .map(|element| match element {
            TupleElem::Expr(expr) => lower_expr(cx, expr),
            TupleElem::Wildcard(_) => panic!("`_` standing as a value is refused (PatternInValue), and never lowered"),
        })
        .collect();
    cx.builder.push(Op::MakeTuple { elements }, ty, span)
}
