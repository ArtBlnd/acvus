//! A template's text and `{{ }}` (programs §1).

use acvus_ast::{AstId, Expr, Slot, Span};

use super::call::reach_call;
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig};
use super::expr::check_expr;
use super::place::Demand;
use crate::ty::{Mutability, TyKind};

/// `Append`, a `{{ }}` tag: a value formatted by `core::display`
/// (programs §1), reached through `call`: `display(&T, out: &mut String)`
/// (operators §2).
pub(super) fn check_append<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, expr: &Expr<S>, span: Span) {
    let value = check_expr(cx, expr, Demand::Lend(Mutability::Shared));
    let shown = cx.lend(value, Mutability::Shared, expr.span());
    let string = cx.node(TyKind::String);
    let out = cx.node(TyKind::Ref(Mutability::Mut, string));
    reach_call(cx, site, Callee::Core(CoreSig::Display), vec![(expr.id(), shown), (site, out)], span);
}
