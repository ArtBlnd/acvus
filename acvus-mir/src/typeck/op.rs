//! Operators: each a call of its lang item through `call`
//! (operators §2); `&&`, `||`, `!`, `^`, `&`, `|`, `<<` and `>>` are the
//! language's own (operators §2).

use acvus_ast::{AstId, BinOp, Expr, Slot, Span, UnaryOp};

use super::call::reach_call;
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig};
use super::expr::check_expr;
use super::place::{self, Demand};
use crate::infer::Term;
use crate::ty::Mutability;

/// NOTE: `==` and the orderings reach `core::eq` and `core::cmp` at the
/// operands' type whatever it is. The structural comparison part by part
/// (operators §4) is not built: its parts' reaches have no site to be
/// recorded at (question Q15 to the coordinator).
pub(super) fn check_binary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, left: &Expr<S>, op: BinOp, right: &Expr<S>, span: Span) -> Term {
    let lhs = check_expr(cx, left, Demand::Read);
    let rhs = check_expr(cx, right, Demand::Read);
    let operands = Operands { left, lhs, right, rhs };
    match op {
        BinOp::Add => lang_item(cx, site, CoreSig::Add, operands, span),
        BinOp::Sub => lang_item(cx, site, CoreSig::Sub, operands, span),
        BinOp::Mul => lang_item(cx, site, CoreSig::Mul, operands, span),
        BinOp::Div => lang_item(cx, site, CoreSig::Div, operands, span),
        BinOp::Mod => lang_item(cx, site, CoreSig::Rem, operands, span),
        BinOp::Eq | BinOp::Neq => lang_item(cx, site, CoreSig::Eq, operands, span),
        // An ordering is `core::cmp`'s answer compared with `0`: the
        // operator is a `bool` (operators §2).
        BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
            lang_item(cx, site, CoreSig::Cmp, operands, span);
            cx.bool()
        }
        BinOp::And | BinOp::Or => {
            let bool_ty = cx.bool();
            cx.eq(lhs, bool_ty, left.span());
            cx.eq(rhs, bool_ty, right.span());
            bool_ty
        }
        // NOTE: `^` is on `bool` and the integers; no bound states "a
        // `bool` or an integer", so only its operands' sameness is checked.
        BinOp::Xor => {
            cx.eq(lhs, rhs, span);
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
        cx.lend(operands.lhs, Mutability::Shared, operands.left.span()),
        cx.lend(operands.rhs, Mutability::Shared, operands.right.span()),
    ];
    reach_call(cx, site, Callee::Core(sig), args, span)
}

/// `-x`, `!x`. `*r` is `place::check_deref`.
///
/// NOTE: `!` is on `bool` and the integers; as for `^`, no bound states
/// that pair, so its operand is not constrained here.
pub(super) fn check_unary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, op: UnaryOp, operand: &Expr<S>, span: Span) -> Term {
    match op {
        UnaryOp::Neg => {
            let value = check_expr(cx, operand, Demand::Read);
            let arg = cx.lend(value, Mutability::Shared, operand.span());
            reach_call(cx, site, Callee::Core(CoreSig::Neg), vec![arg], span)
        }
        UnaryOp::Not => check_expr(cx, operand, Demand::Read),
        UnaryOp::Deref => place::check_deref(cx, site, operand, span),
    }
}
