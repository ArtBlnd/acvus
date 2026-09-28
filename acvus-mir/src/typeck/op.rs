//! Operators: each a call of its lang item through `call`
//! (operators §2); equality and ordering structural part by part, or
//! through `core::eq` and `core::cmp` (operators §4).

use acvus_ast::{AstId, BinOp, Expr, Slot, Span, UnaryOp};

use super::ctxt::Ctxt;
use crate::infer::Term;

pub(super) fn check_binary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, left: &Expr<S>, op: BinOp, right: &Expr<S>, span: Span) -> Term {
    todo!("typeck::op::check_binary")
}

/// `-x`, `!x`. `*r` is `place::check_deref`.
pub(super) fn check_unary<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, op: UnaryOp, operand: &Expr<S>, span: Span) -> Term {
    todo!("typeck::op::check_unary")
}
