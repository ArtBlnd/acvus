//! Dispatch over expression forms; literals (operators §1). Every form is
//! its own function; the large ones live in the files that own their rule.

use acvus_ast::{Expr, Literal, Slot, Span};

use super::ctxt::Ctxt;
use super::place::Demand;
use crate::infer::Term;

/// The term of `expr`, read for `demand` (operators §3). An `Error` node is
/// a fresh variable and is not refused again: the parse reported it.
pub(super) fn check_expr<S: Slot>(cx: &mut Ctxt<'_>, expr: &Expr<S>, demand: Demand) -> Term {
    todo!("typeck::expr::check_expr")
}

/// An integer literal's width waits for its uses and defaults to `i64`
/// (types §5); every other literal has its type.
pub(super) fn check_literal(cx: &mut Ctxt<'_>, literal: &Literal, span: Span) -> Term {
    todo!("typeck::expr::check_literal")
}
