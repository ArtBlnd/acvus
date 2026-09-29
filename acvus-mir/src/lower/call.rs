//! Calls: to the `Chosen` typeck recorded, at the callee type it recorded
//! (extern-fn §5-6). An operator is a call of its lang item's impl on
//! shared references to its operands (operators §2); `^ & | << >> !` are
//! the language's operations, a shift's amount keeping its own width
//! (D79); `as` is a cast; `.unwrap()` is the
//! language's own (operators §2). A coercion is each step of its chain in
//! order: a view through its impl as an `AsSlice` (operators §5), a
//! declared cast or conversion as a call of its `via` (types §3, §7). An
//! effectful call goes through `push_effect`, bracketed by the contexts its
//! effect touches (context §3).

use acvus_ast::{AstId, BinOp, Expr, Literal, RefKind, Slot, Span};
use acvus_utils::Astr;

use super::context::bracket_call;
use super::control::diamond;
use super::control::lower_short_circuit;
use super::ctxt::Lcx;
use super::expr::lower_expr;
use super::place::{lend, operand};
use crate::ir::cfg::Op;
use crate::ir::{Callee, PathSeg, RefTarget, UnaryOp, ValueId, fn_effect, known_len};
use crate::ty::{IntTy, Mutability, Reissue, Ty, TyKind};
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
            compare(cx, site, op, left, right, span)
        }
    }
}

/// What a comparison asks of its operands (operators §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    /// `==`: a `bool`.
    Equal,
    /// An ordering: `cmp`'s `i64`, `-1`, `0` or `1` (operators §2).
    Order,
}

/// `==` or an ordering by the comparison typeck recorded (operators §4):
/// part by part for a structural type, each part through the `core::eq` or
/// `core::cmp` impl reached for it; an ordering is `cmp`'s answer compared
/// with `0` (operators §2).
fn compare<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, op: BinOp, left: &Expr<S>, right: &Expr<S>, span: Span) -> ValueId {
    let comparison = cx
        .typeck
        .comparisons
        .get(&site)
        .unwrap_or_else(|| panic!("typeck records how every comparison compares, and not at {site:?}"));
    let asked = match op {
        BinOp::Eq | BinOp::Neq => Asked::Equal,
        _ => Asked::Order,
    };
    let compared_ty = cx.ty(left.id());
    let lhs = lent_operand(cx, left);
    let rhs = lent_operand(cx, right);
    let lhs = convert(cx, site, lhs, span);
    let rhs = convert(cx, site, rhs, span);
    let answer = compared(cx, comparison, asked, compared_ty, lhs, rhs, span);
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

/// `a` against `b`, two operands of a node of type `ty` as a comparison
/// reads them: a shared reference to the node's value, or, where the node
/// is itself a reference, that reference, since a reference's target is
/// never a reference (types §4).
fn compared(
    cx: &mut Lcx<'_, '_>,
    node: &Comparison,
    asked: Asked,
    ty: Ty,
    a: ValueId,
    b: ValueId,
    span: Span,
) -> ValueId {
    let parts = match node {
        Comparison::Reached { chosen, callee } => {
            let called = cx.callee(chosen);
            return issue(cx, called, *callee, vec![a, b], span);
        }
        Comparison::Parts(parts) => parts,
    };
    let only = || parts.first().expect("a node of one part type has its part");
    match cx.kind(ty).clone() {
        TyKind::Ref(_, target) => compared(cx, only(), asked, target.ty(), a, b, span),
        TyKind::Tuple(items) => {
            let parts: Vec<Part<'_>> = items
                .iter()
                .zip(parts.iter())
                .enumerate()
                .map(|(at, (item, node))| Part { seg: PathSeg::Index(at), ty: *item, node })
                .collect();
            in_order(cx, asked, &parts, a, b, span)
        }
        TyKind::Object(object) => {
            let parts: Vec<Part<'_>> = object
                .fields()
                .iter()
                .zip(parts.iter())
                .map(|((name, field), node)| Part { seg: PathSeg::Field(*name), ty: *field, node })
                .collect();
            in_order(cx, asked, &parts, a, b, span)
        }
        TyKind::Array(element, len) => {
            let node = only();
            let parts: Vec<Part<'_>> =
                (0..known_len(&len)).map(|at| Part { seg: PathSeg::Index(at), ty: element, node }).collect();
            in_order(cx, asked, &parts, a, b, span)
        }
        TyKind::Option(payload) => {
            let tags = cx.decls.core().tags;
            let payload = Part { seg: PathSeg::Payload, ty: payload, node: only() };
            tagged(cx, asked, tags.some, Tagged::Option(payload), a, b, span)
        }
        TyKind::Result(ok, err) => {
            let tags = cx.decls.core().tags;
            let [ok_node, err_node] = &parts[..] else { panic!("a `Result` compares by its two payloads") };
            let ok = Part { seg: PathSeg::Payload, ty: ok, node: ok_node };
            let err = Part { seg: PathSeg::Payload, ty: err, node: err_node };
            tagged(cx, asked, tags.ok, Tagged::Result { ok, err }, a, b, span)
        }
        TyKind::Never => answered(cx, asked, span),
        other => panic!("typeck compares part by part only a structural type or a reference, not {other:?}"),
    }
}

/// One part of a node: the step to it, its type, and how it compares.
#[derive(Clone, Copy)]
struct Part<'c> {
    seg: PathSeg,
    ty: Ty,
    node: &'c Comparison,
}

/// A variant type's payloads: `Option`'s one, or `Result`'s two.
enum Tagged<'c> {
    Option(Part<'c>),
    Result { ok: Part<'c>, err: Part<'c> },
}

/// What a comparison of no part answers: equal.
fn answered(cx: &mut Lcx<'_, '_>, asked: Asked, span: Span) -> ValueId {
    match asked {
        Asked::Equal => cx.bool_const(true, span),
        Asked::Order => order_const(cx, 0, span),
    }
}

fn order_const(cx: &mut Lcx<'_, '_>, value: i128, span: Span) -> ValueId {
    let i64_ty = cx.builder.intern(TyKind::Int(IntTy::I64));
    cx.constant(Literal::Int(value), i64_ty, span)
}

/// The operand of `part` of the node `base` is the operand of.
fn part_operand(cx: &mut Lcx<'_, '_>, base: ValueId, part: Part<'_>, span: Span) -> ValueId {
    let target = RefTarget::Through(base);
    let path = vec![part.seg];
    match cx.kind(part.ty) {
        TyKind::Ref(..) => cx.builder.push(Op::Take { target, path, taken_out: false }, part.ty, span),
        _ => {
            let ty = cx.ref_ty(Mutability::Shared, part.ty);
            cx.builder.push(Op::Ref { target, path, mutability: Mutability::Shared }, ty, span)
        }
    }
}

fn part_compared(cx: &mut Lcx<'_, '_>, asked: Asked, part: Part<'_>, a: ValueId, b: ValueId, span: Span) -> ValueId {
    let a = part_operand(cx, a, part, span);
    let b = part_operand(cx, b, part, span);
    compared(cx, part.node, asked, part.ty, a, b, span)
}

/// Parts compared in order, as Rust's derived comparisons do: equal while
/// every part is, and ordered by the first part that is not equal. A part
/// is compared only where the parts before it left the answer open.
fn in_order(cx: &mut Lcx<'_, '_>, asked: Asked, parts: &[Part<'_>], a: ValueId, b: ValueId, span: Span) -> ValueId {
    let mut answer: Option<ValueId> = None;
    for part in parts {
        let next = |cx: &mut Lcx<'_, '_>| part_compared(cx, asked, *part, a, b, span);
        answer = Some(match answer {
            None => next(cx),
            Some(before) => match asked {
                Asked::Equal => {
                    let bool_ty = cx.bool_ty();
                    diamond(cx, before, bool_ty, span, next, |cx| cx.bool_const(false, span))
                }
                Asked::Order => {
                    let order_ty = cx.value_ty(before);
                    let zero = order_const(cx, 0, span);
                    let bool_ty = cx.bool_ty();
                    let open = cx.builder.push(Op::BinOp { op: crate::ir::BinOp::Eq, left: before, right: zero }, bool_ty, span);
                    diamond(cx, open, order_ty, span, next, |_| before)
                }
            },
        });
    }
    match answer {
        Some(answer) => answer,
        None => answered(cx, asked, span),
    }
}

/// An `Option` or a `Result` compared: by tag first, `None` before `Some`
/// and `Ok` before `Err` as Rust derives them, then by the payload the two
/// share.
fn tagged(cx: &mut Lcx<'_, '_>, asked: Asked, first: Astr, payloads: Tagged<'_>, a: ValueId, b: ValueId, span: Span) -> ValueId {
    let bool_ty = cx.bool_ty();
    let a_first = cx.builder.push(Op::TestVariant { src: a, tag: first }, bool_ty, span);
    let b_first = cx.builder.push(Op::TestVariant { src: b, tag: first }, bool_ty, span);
    let same = cx.builder.push(Op::BinOp { op: crate::ir::BinOp::Eq, left: a_first, right: b_first }, bool_ty, span);
    let answer_ty = match asked {
        Asked::Equal => bool_ty,
        Asked::Order => cx.builder.intern(TyKind::Int(IntTy::I64)),
    };
    let payloads_compared = |cx: &mut Lcx<'_, '_>| match payloads {
        Tagged::Option(payload) => {
            diamond(cx, a_first, answer_ty, span, |cx| part_compared(cx, asked, payload, a, b, span), |cx| answered(cx, asked, span))
        }
        Tagged::Result { ok, err } => diamond(
            cx,
            a_first,
            answer_ty,
            span,
            |cx| part_compared(cx, asked, ok, a, b, span),
            |cx| part_compared(cx, asked, err, a, b, span),
        ),
    };
    let tags_differ = |cx: &mut Lcx<'_, '_>| match asked {
        Asked::Equal => cx.bool_const(false, span),
        Asked::Order => {
            let (when_first, otherwise) = match payloads {
                Tagged::Option(_) => (1, -1),
                Tagged::Result { .. } => (-1, 1),
            };
            let i64_ty = answer_ty;
            diamond(cx, a_first, i64_ty, span, |cx| order_const(cx, when_first, span), |cx| order_const(cx, otherwise, span))
        }
    };
    diamond(cx, same, answer_ty, span, payloads_compared, tags_differ)
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
