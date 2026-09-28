//! Operators: each a call of its lang item through `call`
//! (operators §2); `&&`, `||`, `!`, `^`, `&`, `|`, `<<` and `>>` are the
//! language's own (operators §2).

use acvus_ast::{AstId, BinOp, Expr, Slot, Span, UnaryOp};

use super::call::{reach_call, reach_in};
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig, Decision};
use super::output::Comparing;
use super::expr::check_expr;
use super::place::{self, Demand};
use super::refusal::{self, RefusalKind};
use crate::decl::{Base, Head};
use crate::infer::{Closed, Term};
use crate::ty::{EffectTerm, Infer as InferPhase, Mutability, ObjectTy, TyKind};

/// `==` and the orderings compare part by part where the operands' type is
/// structural, and through `core::eq` or `core::cmp` otherwise
/// (operators §4): a `Compare` decision on the operands' one type.
pub(super) fn check_binary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, left: &Expr<S>, op: BinOp, right: &Expr<S>, span: Span) -> Term {
    // A lang item takes its operands by `&` (operators §2), so they are
    // lent, not used whole.
    let demand = match op {
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Eq
        | BinOp::Neq
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte => Demand::Lend(Mutability::Shared),
        BinOp::And | BinOp::Or | BinOp::Xor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => Demand::Read,
    };
    let lhs = check_expr(cx, left, demand);
    let rhs = check_expr(cx, right, demand);
    let operands = Operands { left, lhs, right, rhs };
    match op {
        BinOp::Add => lang_item(cx, site, CoreSig::Add, operands, span),
        BinOp::Sub => lang_item(cx, site, CoreSig::Sub, operands, span),
        BinOp::Mul => lang_item(cx, site, CoreSig::Mul, operands, span),
        BinOp::Div => lang_item(cx, site, CoreSig::Div, operands, span),
        BinOp::Mod => lang_item(cx, site, CoreSig::Rem, operands, span),
        BinOp::Eq | BinOp::Neq => compare(cx, site, CoreSig::Eq, operands, span),
        // An ordering is `core::cmp`'s answer compared with `0`: the
        // operator is a `bool` (operators §2).
        BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => compare(cx, site, CoreSig::Cmp, operands, span),
        BinOp::And | BinOp::Or => {
            let bool_ty = cx.bool();
            cx.eq(lhs, bool_ty, left.span());
            cx.eq(rhs, bool_ty, right.span());
            bool_ty
        }
        BinOp::Xor => {
            cx.eq(lhs, rhs, span);
            let operand = Decision::BoolOrInteger { span, operand: lhs };
            super::decision::decide(cx, operand);
            lhs
        }
        BinOp::BitAnd | BinOp::BitOr => {
            let integer = cx.integer(span);
            cx.eq(lhs, integer, left.span());
            cx.eq(rhs, integer, right.span());
            integer
        }
        BinOp::Shl | BinOp::Shr => {
            let shifted = cx.integer(left.span());
            cx.eq(lhs, shifted, left.span());
            let by = cx.integer(right.span());
            cx.eq(rhs, by, right.span());
            shifted
        }
    }
}

struct Operands<'e, S> {
    left: &'e Expr<S>,
    lhs: Term,
    right: &'e Expr<S>,
    rhs: Term,
}

/// A binary operator as a call of its lang item on shared references to
/// its operands (operators §2).
fn lang_item<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, sig: CoreSig, operands: Operands<'_, S>, span: Span) -> Term {
    let args = vec![
        (operands.left.id(), cx.lend(operands.lhs, Mutability::Shared, operands.left.span())),
        (operands.right.id(), cx.lend(operands.rhs, Mutability::Shared, operands.right.span())),
    ];
    reach_call(cx, site, Callee::Core(sig), args, span)
}

/// A comparison: the lang items take both operands at one type
/// (operators §2), and how that type compares is decided once it is
/// known.
fn compare<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, sig: CoreSig, operands: Operands<'_, S>, span: Span) -> Term {
    cx.eq(operands.lhs, operands.rhs, operands.right.span());
    let body = super::effect::body_effect(cx);
    let compare = Decision::Compare {
        node: site,
        span,
        sig,
        operand: operands.lhs,
        body,
    };
    super::decision::decide(cx, compare);
    cx.bool()
}

/// A comparison fired at `node` on its operands' type (operators §4): a
/// structural type compares its parts, each a node of its own; a reference
/// compares what it names, since a lang item's `&T` is never a reference to
/// a reference (types §4); any other type reaches `core::eq` or `core::cmp`
/// at `node`.
pub(super) fn fire_compare(
    cx: &mut Ctxt<'_>,
    node: AstId,
    span: Span,
    sig: CoreSig,
    operand: Closed,
    body: &EffectTerm<InferPhase>,
) {
    let parts: Option<Vec<Term>> = match cx.infer.types().kind(operand.term) {
        TyKind::Tuple(items) => Some(items.to_vec()),
        TyKind::Array(element, _) | TyKind::Option(element) => Some(vec![*element]),
        TyKind::Result(ok, err) => Some(vec![*ok, *err]),
        TyKind::Object(ObjectTy::Structural(fields, _)) => Some(fields.iter().map(|(_, field)| *field).collect()),
        TyKind::Ref(_, target) => Some(vec![target.ty()]),
        TyKind::Never => Some(Vec::new()),
        _ => None,
    };
    let Some(parts) = parts else {
        cx.out.comparisons.insert(node, Comparing::Reached);
        let lent = cx.lend(operand.term, Mutability::Shared, span);
        reach_in(cx, body, node, Callee::Core(sig), vec![(node, lent), (node, lent)], span);
        return;
    };
    let nodes: Vec<AstId> = parts.iter().map(|_| AstId::alloc()).collect();
    cx.out.comparisons.insert(node, Comparing::Parts(nodes.clone().into()));
    for (part, part_node) in parts.into_iter().zip(nodes) {
        let compare = Decision::Compare {
            node: part_node,
            span,
            sig,
            operand: part,
            body: body.clone(),
        };
        super::decision::decide(cx, compare);
    }
}

/// `^` fired on its operands' one type (operators §2): a `bool` or an
/// integer. A `!` operand is never read.
pub(super) fn fire_bool_or_integer(cx: &mut Ctxt<'_>, span: Span, operand: Closed) {
    match operand.head {
        Head::Base(Base::Bool | Base::Int(_) | Base::Never) => {}
        Head::Base(_) | Head::Ref(..) => refusal::push(cx, RefusalKind::NotBoolOrInteger, span),
    }
}

/// `-x`, `!x`. `*r` is `place::check_deref`.
///
/// NOTE: operators §2 lists `!` on `bool` alone, where Rust also has it on
/// the integers. Which the language means is not settled yet, so its
/// operand is not constrained here.
pub(super) fn check_unary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, op: UnaryOp, operand: &Expr<S>, span: Span) -> Term {
    match op {
        UnaryOp::Neg => {
            let value = check_expr(cx, operand, Demand::Lend(Mutability::Shared));
            let arg = cx.lend(value, Mutability::Shared, operand.span());
            reach_call(cx, site, Callee::Core(CoreSig::Neg), vec![(operand.id(), arg)], span)
        }
        UnaryOp::Not => check_expr(cx, operand, Demand::Read),
        UnaryOp::Deref => place::check_deref(cx, site, operand, span),
    }
}
