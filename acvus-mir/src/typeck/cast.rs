//! `e as T` (operators §5).

use acvus_ast::{AstId, Expr, Slot, Span};
use acvus_utils::Astr;

use super::ctxt::Ctxt;
use crate::infer::Term;

pub(super) fn check_cast<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, expr: &Expr<S>, target: Astr, target_span: Span, span: Span) -> Term {
    todo!("typeck::cast::check_cast")
}
