//! Dispatch over expression forms; literals (operators §1). Every form is
//! its own function; the large ones live in the files that own their rule.

use acvus_ast::{Expr, Literal, ObjectExprField, RefKind, Slot, Span, TupleElem, UnaryOp};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use super::scope::{Capturing, capturing_binding, note_captures};
use super::{call, cast, closure, context, control, method, op, place, scope, stmt, variant};
use crate::infer::{Bound, Term};
use crate::ty::{IntTy, LenTerm, Label, Mutability, ObjectTy, TyKind, TypeArg};

/// The term of `expr`, read for `demand` (operators §3). An `Error` node is
/// a fresh variable and is not refused again: the parse reported it.
pub(super) fn check_expr<S: Slot>(cx: &mut Ctxt<'_>, expr: &Expr<S>, demand: Demand) -> Term {
    let term = acvus_utils::grow(|| check_form(cx, expr, demand));
    cx.record(expr.id(), term);
    if demand == Demand::Read {
        closure::note_whole_use(cx, expr, term);
    }
    term
}

/// A method call's receiver: as `check_expr` reads it, but not yet a whole
/// use, since its adjustment decides whether it is one (types §6).
pub(super) fn check_receiver<S: Slot>(cx: &mut Ctxt<'_>, expr: &Expr<S>) -> Term {
    let term = acvus_utils::grow(|| check_form(cx, expr, Demand::Read));
    cx.record(expr.id(), term);
    term
}

/// A name read for more than a shared read: each lambda that captures it
/// captures it as `demand` asks (functions §2). An input is found in no
/// frame; it crosses every lambda around it.
fn note_demanded_capture(cx: &mut Ctxt<'_>, name: QualifiedRef, ref_kind: RefKind, term: Term, demand: Demand, span: Span) {
    match ref_kind {
        RefKind::ExternParam => {
            let crossed = cx.scope.lambdas();
            let capturing = Capturing {
                name: name.name,
                of: super::output::CaptureOf::Input(name.name),
                crossed: &crossed,
                term,
            };
            note_captures(cx, capturing, demand, span);
        }
        RefKind::Value => {
            if name.namespace.is_none()
                && let Some(found) = cx.scope.lookup(name.name)
            {
                note_captures(cx, capturing_binding(name.name, &found), demand, span);
            }
        }
    }
}

fn check_form<S: Slot>(cx: &mut Ctxt<'_>, expr: &Expr<S>, demand: Demand) -> Term {
    match expr {
        Expr::Ident {
            id,
            name,
            ref_kind,
            span,
        } => {
            let term = scope::check_ident(cx, *id, *name, *ref_kind, *span);
            if demand != Demand::Read {
                note_demanded_capture(cx, *name, *ref_kind, term, demand, *span);
            }
            term
        }
        Expr::Literal { value, span, .. } => check_literal(cx, value, *span),
        Expr::BinaryOp {
            id,
            left,
            op,
            right,
            span,
        } => op::check_binary(cx, *id, left, *op, right, *span),
        Expr::UnaryOp {
            id,
            op: UnaryOp::Deref,
            operand,
            span,
        } => place::check_deref(cx, *id, operand, *span),
        Expr::UnaryOp { id, op, operand, span } => op::check_unary(cx, *id, *op, operand, *span),
        Expr::FieldAccess {
            id,
            object,
            field,
            span,
        } => place::check_field(cx, *id, object, *field, demand, *span),
        Expr::Index {
            id,
            callee_id,
            object,
            index,
            span,
        } => place::check_index(cx, *id, *callee_id, object, index, demand, *span),
        Expr::FuncCall { id, func, args, span } => call::check_call(cx, *id, func, args, *span),
        Expr::MethodCall {
            id,
            callee_id,
            receiver,
            name,
            args,
            span,
            ..
        } => method::check_method_call(cx, *id, *callee_id, receiver, *name, args, *span),
        Expr::Pipe { id, left, right, span } => call::check_pipe(cx, *id, left, right, *span),
        Expr::Lambda {
            id,
            captures_by_move,
            params,
            body,
            span,
        } => closure::check_lambda(cx, *id, *captures_by_move, params, body, *span),
        Expr::Paren { inner, .. } => check_expr(cx, inner, demand),
        Expr::Borrow {
            id,
            mutable,
            place,
            span,
        } => place::check_borrow(cx, *id, *mutable, place, *span),
        Expr::List {
            head,
            rest,
            tail,
            span,
            ..
        } => check_list(cx, head, rest.is_some(), tail, *span),
        // A group is a lambda's parameter list, which the parser reads as
        // one; standing as a value it is a pattern's shape.
        Expr::Group { elements, span, .. } => {
            for element in elements {
                check_expr(cx, element, Demand::Read);
            }
            refusal::push(cx, RefusalKind::PatternInValue, *span);
            cx.fresh()
        }
        Expr::Object { fields, .. } => check_object(cx, fields),
        Expr::Tuple { elements, .. } => check_tuple(cx, elements),
        Expr::Block { stmts, tail, .. } => {
            cx.scope.push();
            let term = stmt::check_body(cx, stmts, Some(tail));
            cx.scope.pop();
            term
        }
        Expr::ContextRef { id, name, span } => context::check_context_ref(cx, *id, *name, demand, *span),
        Expr::Variant {
            id,
            enum_name,
            tag,
            payload,
            span,
        } => variant::check_variant(cx, *id, *enum_name, *tag, payload.as_deref(), *span),
        Expr::Cast {
            id,
            expr,
            target,
            target_span,
            span,
        } => cast::check_cast(cx, *id, expr, *target, *target_span, *span),
        Expr::Try { id, inner, span } => control::check_try(cx, *id, inner, *span),
        Expr::Return { value, span, .. } => control::check_return(cx, value, *span),
        Expr::If {
            id,
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => control::check_if(cx, *id, cond, then_body, then_tail.as_deref(), else_branch.as_deref()),
        Expr::Match {
            id, scrutinee, arms, ..
        } => control::check_match(cx, *id, scrutinee, arms),
        Expr::IfLet {
            id,
            pattern,
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => control::check_if_let(cx, *id, pattern, source, then_body, then_tail.as_deref(), else_branch.as_deref()),
        Expr::Error(_) => cx.fresh(),
    }
}

/// An integer literal's width waits for its uses and defaults to `i64`
/// (types §5); every other literal has its type.
pub(super) fn check_literal(cx: &mut Ctxt<'_>, literal: &Literal, span: Span) -> Term {
    match literal {
        Literal::Int(value) => {
            let among: Box<[IntTy]> = IntTy::ALL.into_iter().filter(|width| width.holds(*value)).collect();
            let term = cx.infer.fresh(Bound::Integer {
                signed: true,
                among,
                literal: true,
            });
            cx.decisions.push(Decision::Settle { span, term });
            term
        }
        Literal::IntOf(suffixed) => cx.node(TyKind::Int(suffixed.width.into())),
        Literal::Float(_) => cx.node(TyKind::Float),
        Literal::Char(_) => cx.node(TyKind::Char),
        Literal::Bool(_) => cx.bool(),
        Literal::Unit => cx.unit(),
        Literal::String(_) => string_literal(cx),
        // Rust's `b"…"`: a reference to an array of its bytes.
        Literal::Bytes(bytes) => {
            let byte = cx.node(TyKind::Int(IntTy::U8));
            let array = cx.node(TyKind::Array(byte, LenTerm::Known(bytes.len()), Label::Here));
            cx.node(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(array)))
        }
        Literal::List(items) => {
            let element = cx.fresh();
            for item in items {
                let term = check_literal(cx, item, span);
                cx.eq(term, element, span);
            }
            cx.held(TyKind::Array(element, LenTerm::Known(items.len()), Label::Here))
        }
    }
}

/// A string literal is a `&str`, as in Rust (operators §1).
fn string_literal(cx: &mut Ctxt<'_>) -> Term {
    let str_ty = cx.node(TyKind::Str);
    cx.node(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(str_ty)))
}

/// `[a, b]`: every element enters the one element slot (types §3). A `..`
/// is a pattern's, never a value's.
fn check_list<S: Slot>(cx: &mut Ctxt<'_>, head: &[Expr<S>], rest: bool, tail: &[Expr<S>], span: Span) -> Term {
    let element = cx.fresh();
    for item in head.iter().chain(tail) {
        let term = check_expr(cx, item, Demand::Read);
        cx.enter(item.id(), term, element, item.span());
    }
    if rest {
        refusal::push(cx, RefusalKind::PatternInValue, span);
    }
    let len = LenTerm::Known(head.len() + tail.len());
    cx.held(TyKind::Array(element, len, Label::Here))
}

/// `{ a: e, .. }`: a transparent object the program writes (types §1).
fn check_object<S: Slot>(cx: &mut Ctxt<'_>, fields: &[ObjectExprField<S>]) -> Term {
    let mut typed: Vec<(Astr, Term)> = fields
        .iter()
        .map(|field| (field.key, check_expr(cx, &field.value, Demand::Read)))
        .collect();
    typed.sort_by_key(|(name, _)| *name);
    cx.held(TyKind::Object(ObjectTy::Structural(typed.into(), Label::Here)))
}

/// `(a, b)`: a tuple of its parts' own types, and `()` is the unit. `_`
/// is a pattern's.
fn check_tuple<S: Slot>(cx: &mut Ctxt<'_>, elements: &[TupleElem<S>]) -> Term {
    if elements.is_empty() {
        return cx.unit();
    }
    let mut parts = Vec::with_capacity(elements.len());
    for element in elements {
        match element {
            TupleElem::Expr(expr) => parts.push(check_expr(cx, expr, Demand::Read)),
            TupleElem::Wildcard(at) => {
                refusal::push(cx, RefusalKind::PatternInValue, *at);
                parts.push(cx.fresh());
            }
        }
    }
    cx.held(TyKind::Tuple(parts.into(), Label::Here))
}
