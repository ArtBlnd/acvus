//! Calls: to the `Chosen` typeck recorded, at the callee type it recorded
//! (extern-fn §5-6). An operator is a call of its lang item's impl on
//! shared references to its operands (operators §2); `^ & | << >> !` are
//! the language's operations, a shift's amount keeping its own width
//! (D79); `as` is a cast; `.unwrap()` is the
//! language's own (operators §2); a comparison and `.clone()` follow
//! typeck's part tree (`part_tree`). A coercion is each step of its chain in
//! order: a view through its impl as an `AsSlice` (operators §5), a
//! declared cast or conversion as a call of its `via` (types §3, §7). An
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
use crate::typeck::Adjustment;

/// `value`, written at `site`, coerced by each step typeck recorded there.
pub(super) fn convert(cx: &mut Lcx<'_, '_>, site: AstId, value: ValueId, span: Span) -> ValueId {
    let Some(steps) = cx.typeck.conversions.get(&site) else {
        return value;
    };
    let core = cx.decls.core();
    let views = [core.as_slice, core.as_slice_mut, core.as_str];
    let mut at = value;
    for step in steps.iter() {
        at = match views.contains(&step.via.declaration) {
            true => {
                let TyKind::Ref(mutability, _) = cx.kind(step.to) else {
                    panic!("a view's result is a reference (operators §5)")
                };
                let op = Op::AsSlice { container: at, mutability: *mutability, r#impl: cx.extern_impl(&step.via) };
                cx.builder.push(op, step.to, span)
            }
            false => {
                let callee = cx.callee(&step.via);
                issue(cx, callee, step.via_ty, vec![at], span)
            }
        };
    }
    at
}

/// A call of a callee of type `callee_ty`: pushed as an effect, bracketed
/// by the contexts it touches, where the callee's effect is not pure.
pub(super) fn issue(cx: &mut Lcx<'_, '_>, callee: Callee, callee_ty: Ty, args: Vec<ValueId>, span: Span) -> ValueId {
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
/// first argument, as written or lent as typeck adjusted it (types §6).
/// `.unwrap()` is the language's own (operators §2).
pub(super) fn lower_method<S: Slot>(
    cx: &mut Lcx<'_, '_>,
    site: AstId,
    callee_site: AstId,
    receiver: &Expr<S>,
    args: &[Expr<S>],
    span: Span,
) -> ValueId {
    let adjustment = *cx
        .typeck
        .adjustments
        .get(&callee_site)
        .unwrap_or_else(|| panic!("typeck records the receiver adjustment of every reached method call, and not {callee_site:?}"));
    let taken = match adjustment {
        Adjustment::AsWritten => lower_expr(cx, receiver),
        Adjustment::Ref(mutability) => {
            let place = operand(cx, receiver, mutability);
            lend(cx, place, mutability, receiver.span())
        }
    };
    let taken = convert(cx, receiver.id(), taken, receiver.span());
    let rest: Vec<&Expr<S>> = args.iter().collect();
    let mut all = vec![taken];
    all.extend(arguments(cx, &rest));
    if cx.reached(callee_site).declaration == cx.decls.core().unwrap {
        let [wrapped] = &all[..] else {
            panic!("`unwrap` takes its receiver alone (operators §2)")
        };
        let ty = cx.ty(site);
        return cx.builder.push(Op::UnwrapVariant { src: *wrapped }, ty, span);
    }
    call_reached(cx, callee_site, all, span)
}

/// A shared reference to an operand, coerced at the operand's site: what a
/// lang item takes (operators §2).
pub(super) fn lent_operand<S: Slot>(cx: &mut Lcx<'_, '_>, expr: &Expr<S>) -> ValueId {
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
        BinOp::Xor | BinOp::BitAnd | BinOp::Shl | BinOp::Shr => {
            let lhs = lower_expr(cx, left);
            let rhs = lower_expr(cx, right);
            let ty = cx.ty(site);
            cx.builder.push(Op::BinOp { op: op.into(), left: lhs, right: rhs }, ty, span)
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            let args = vec![lent_operand(cx, left), lent_operand(cx, right)];
            call_reached(cx, site, args, span)
        }
        BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
            super::part_tree::compare(cx, site, op, left, right, span)
        }
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
