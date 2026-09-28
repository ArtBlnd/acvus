//! A template's text and `{{ }}` (programs §1).

use acvus_ast::{AstId, Expr, Slot, Span};

use super::ctxt::Ctxt;

/// `Append`: a text line, or a value formatted by `core::display`
/// (operators §7), reached through `call`.
pub(super) fn check_append<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, expr: &Expr<S>, span: Span) {
    todo!("typeck::template::check_append")
}
