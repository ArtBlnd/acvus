//! `e as T` (operators §5).

use acvus_ast::{AstId, Expr, Slot, Span};
use acvus_utils::Astr;

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::expr::check_expr;
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use crate::infer::Term;

/// The cast's value has the type it names; whether Rust casts the source's
/// type to it is decided once that type is known (types §6).
pub(super) fn check_cast<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, expr: &Expr<S>, target: Astr, target_span: Span, span: Span) -> Term {
    let source = check_expr(cx, expr, Demand::Read);
    let Some(to) = cx.decls.core().cast_target(target) else {
        refusal::push(cx, RefusalKind::UnknownCastTarget { name: target }, target_span);
        return cx.fresh();
    };
    cx.decisions.push(Decision::Cast { site, span, source, to }, cx.infer.level());
    cx.node(to.into())
}
