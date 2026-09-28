//! Expression dispatch: an expression becomes a typed value in the current block, typed by `Typeck::types` at its site.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_expr<S: Slot>(cx: &mut Lcx<'_, '_>, expr: &Expr<S>) -> ValueId {
    todo!("lower::expr::lower_expr")
}

pub(super) fn lower_literal(cx: &mut Lcx<'_, '_>, site: AstId, literal: &Literal, span: Span) -> ValueId {
    todo!("lower::expr::lower_literal")
}
