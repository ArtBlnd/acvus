//! Calls: to the `Chosen` typeck recorded, at the callee type it recorded
//! (extern-fn §5-6). An operator is a call of its lang item's impl on
//! shared references to its operands (operators §2); `^ & | << >> !` are
//! the language's operations; `as` is a cast; `.unwrap()` is the
//! language's own (operators §2). A coercion is each step of its chain in
//! order: a view through its impl as an `AsSlice` (operators §5). An
//! effectful call goes through `push_effect`, bracketed by the contexts its
//! effect touches (context §3).

use acvus_ast::{AstId, BinOp, Expr, Literal, RefKind, Slot, Span};

use super::context::bracket_call;
use super::control::lower_short_circuit;
use super::ctxt::Lcx;
use super::expr::lower_expr;
use super::place::{lend, operand};
use crate::ir::cfg::Op;
use crate::ir::{Callee, UnaryOp, ValueId, fn_effect};
use crate::ty::{Mutability, Reissue, Ty, TyKind};
use crate::typeck::Comparison;

/// `value`, written at `site`, coerced by each step typeck recorded there.
pub(super) fn convert(cx: &mut Lcx<'_, '_>, site: AstId, value: ValueId, span: Span) -> ValueId {
    let Some(steps) = cx.typeck.conversions.get(&site) else {
        return value;
    };
    let core = cx.decls.core();
    let views = [core.as_slice, core.as_slice_mut, core.as_str];
    let mut at = value;
    for step in steps.iter() {
        if !views.contains(&step.via.declaration) {
            todo!(
                "D75: a declared cast or conversion step through {:?} has no callee type until `Conversion.via_ty` lands",
                step.via.declaration
            );
        }
        let TyKind::Ref(mutability, _) = cx.kind(step.to) else {
            panic!("a view's result is a reference (operators §5)")
        };
        let op = Op::AsSlice { container: at, mutability: *mutability, r#impl: cx.extern_impl(&step.via) };
        at = cx.builder.push(op, step.to, span);
    }
    at
}

/// A call of a callee of type `callee_ty`: pushed as an effect, bracketed
/// by the contexts it touches, where the callee's effect is not pure.
fn issue(cx: &mut Lcx<'_, '_>, callee: Callee, callee_ty: Ty, args: Vec<ValueId>, span: Span) -> ValueId {
    let (ret, effect) = match cx.kind(callee_ty) {
        TyKind::Fn { ret, .. } => {
            let effect = fn_effect(cx.builder.types(), callee_ty).expect("a function type has an effect").clone();
            (*ret, effect)
        }
        other => panic!("a callee's type is a function type, not {other:?}"),
    };
    let op = Op::FunctionCall { callee, callee_ty, args };
    match effect.reissue() {
        Reissue::Pure => cx.builder.push(op, ret, span),
        Reissue::Idempotent | Reissue::Opaque => {
            bracket_call(cx, &effect, span, |cx| cx.builder.push_effect(op, ret, span))
        }
    }
}

/// A call of the declaration typeck reached at `site`, at the callee type
/// it recorded there.
pub(super) fn call_reached(cx: &mut Lcx<'_, '_>, site: AstId, args: Vec<ValueId>, span: Span) -> ValueId {
    let callee = cx.callee(cx.reached(site));
    let callee_ty = *cx
        .typeck
        .callees
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records the callee type at every reached call site, and not at {site:?}"));
    issue(cx, callee, callee_ty, args, span)
}

/// Each argument's value, coerced at its own site.
fn arguments<S: Slot>(cx: &mut Lcx<'_, '_>, args: &[&Expr<S>]) -> Vec<ValueId> {
    args.iter()
        .map(|arg| {
            let value = lower_expr(cx, arg);
            convert(cx, arg.id(), value, arg.span())
        })
        .collect()
}

/// `f(args)`: of a declaration typeck reached at the call; of a variant
/// `Enum::Tag(payload)`, which the parser writes as a call; or of a value.
pub(super) fn lower_call<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, func: &Expr<S>, args: &[Expr<S>], span: Span) -> ValueId {
    let args: Vec<&Expr<S>> = args.iter().collect();
    call(cx, site, func, &args, span)
}

/// `left | func`: `func(left)`.
pub(super) fn lower_pipe<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, left: &Expr<S>, func: &Expr<S>, span: Span) -> ValueId {
    call(cx, site, func, &[left], span)
}

fn call<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, func: &Expr<S>, args: &[&Expr<S>], span: Span) -> ValueId {
    if cx.typeck.reached.contains_key(&site) {
        let args = arguments(cx, args);
        return call_reached(cx, site, args, span);
    }
    if let Expr::Ident { name, ref_kind: RefKind::Value, .. } = func
        && name.namespace.is_some()
        && cx.decls.scheme(*name).is_none()
    {
        return variant_call(cx, site, name.name, args, span);
    }
    let callee = lower_expr(cx, func);
    let callee_ty = cx.value_ty(callee);
    let args = arguments(cx, args);
    issue(cx, Callee::Indirect(callee), callee_ty, args, span)
}

/// `Enum::Tag(payload)` or `Enum::Tag()`, a variant the parser leaves as a
/// call.
fn variant_call<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, tag: acvus_utils::Astr, args: &[&Expr<S>], span: Span) -> ValueId {
    let payload = match args {
        [] => None,
        [payload] => Some(lower_expr(cx, payload)),
        _ => panic!("typeck refuses a variant of more than one payload (VariantArity)"),
    };
    let ty = cx.ty(site);
    cx.builder.push(Op::MakeVariant { tag, payload }, ty, span)
}

/// `receiver.name(args)`, reached at `callee_site`: the receiver is the
/// first argument as it stands. `.unwrap()` is the language's own
/// (operators §2).
pub(super) fn lower_method<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    callee_site: AstId,
    receiver: &Expr<S>,
    args: &[Expr<S>],
    span: Span,
) -> ValueId {
    let mut all: Vec<&Expr<S>> = vec![receiver];
    all.extend(args.iter());
    if cx.reached(callee_site).declaration == cx.decls.core().unwrap {
        let [wrapped] = &arguments(cx, &all)[..] else {
            panic!("`unwrap` takes its receiver alone (operators §2)")
        };
        let ty = cx.ty(site);
        return cx.builder.push(Op::UnwrapVariant { src: *wrapped }, ty, span);
    }
    let args = arguments(cx, &all);
    call_reached(cx, callee_site, args, span)
}

/// A shared reference to an operand, coerced at the operand's site: what a
/// lang item takes (operators §2).
fn lent_operand<S: Slot>(cx: &mut Lcx<'_, '_>, expr: &Expr<S>) -> ValueId {
    let place = operand(cx, expr, Mutability::Shared);
    let lent = lend(cx, place, Mutability::Shared, expr.span());
    convert(cx, expr.id(), lent, expr.span())
}

pub(super) fn lower_binary<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    op: BinOp,
    left: &Expr<S>,
    right: &Expr<S>,
    span: Span,
) -> ValueId {
    match op {
        BinOp::And | BinOp::Or => lower_short_circuit(cx, site, op, left, right, span),
        BinOp::Xor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
            let lhs = lower_expr(cx, left);
            let rhs = lower_expr(cx, right);
            if cx.value_ty(lhs) != cx.value_ty(rhs) {
                todo!(
                    "a shift whose amount's width differs from the shifted value's: the IR's `BinOp` takes one width, \
                     and no rule states how the amount is carried to it"
                );
            }
            let ty = cx.ty(site);
            cx.builder.push(Op::BinOp { op: op.into(), left: lhs, right: rhs }, ty, span)
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            let args = vec![lent_operand(cx, left), lent_operand(cx, right)];
            call_reached(cx, site, args, span)
        }
        BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
            compare(cx, site, op, left, right, span)
        }
    }
}

/// `==` or an ordering by the comparison typeck recorded (operators §4):
/// through the `core::eq` or `core::cmp` impl it reached, an ordering being
/// `cmp`'s answer compared with `0` (operators §2).
fn compare<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, op: BinOp, left: &Expr<S>, right: &Expr<S>, span: Span) -> ValueId {
    let comparison = cx
        .typeck
        .comparisons
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records how every comparison compares, and not at {site:?}"));
    match comparison {
        Comparison::Reached(_) => {}
        Comparison::Parts(_) => todo!(
            "a comparison part by part (operators §4): a part's `Comparison::Reached` carries no site, so the callee \
             type typeck recorded for it in `Typeck.callees` cannot be found"
        ),
    }
    let lhs = lent_operand(cx, left);
    let rhs = lent_operand(cx, right);
    let lhs = convert(cx, site, lhs, span);
    let rhs = convert(cx, site, rhs, span);
    let answer = call_reached(cx, site, vec![lhs, rhs], span);
    let bool_ty = cx.ty(site);
    match op {
        BinOp::Eq => answer,
        BinOp::Neq => cx.builder.push(Op::UnaryOp { op: UnaryOp::Not, operand: answer }, bool_ty, span),
        BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
            let answer_ty = cx.value_ty(answer);
            let zero = cx.constant(Literal::Int(0), answer_ty, span);
            cx.builder.push(Op::BinOp { op: op.into(), left: answer, right: zero }, bool_ty, span)
        }
        other => panic!("{other:?} is not a comparison"),
    }
}

/// `-x`: `core::neg` on `&x` (operators §2).
pub(super) fn lower_neg<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, operand: &Expr<S>, span: Span) -> ValueId {
    let args = vec![lent_operand(cx, operand)];
    call_reached(cx, site, args, span)
}

/// `!x`: on a `bool` its negation; on an integer, as Rust defines it, the
/// value with every bit flipped, `x ^ !0`.
pub(super) fn lower_not<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, operand: &Expr<S>, span: Span) -> ValueId {
    let value = lower_expr(cx, operand);
    let ty = cx.ty(site);
    match cx.kind(ty) {
        TyKind::Bool => cx.builder.push(Op::UnaryOp { op: UnaryOp::Not, operand: value }, ty, span),
        TyKind::Int(width) => {
            let every_bit = match width.signed() {
                true => -1,
                false => width.max(),
            };
            let mask = cx.constant(Literal::Int(every_bit), ty, span);
            cx.builder.push(Op::BinOp { op: crate::ir::BinOp::Xor, left: value, right: mask }, ty, span)
        }
        other => panic!("typeck admits `!` only on `bool` and the integers, not {other:?}"),
    }
}
