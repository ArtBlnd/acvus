//! Statements and bodies. A binding is a slot (`Builder::slot`); promotion to SSA is a pass.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_body<S: Slot>(cx: &mut Lcx<'_, '_>, stmts: &[Stmt<S>], tail: Option<&Expr<S>>) -> ValueId {
    todo!("lower::stmt::lower_body")
}

pub(super) fn lower_stmt<S: Slot>(cx: &mut Lcx<'_, '_>, stmt: &Stmt<S>) {
    todo!("lower::stmt::lower_stmt")
}
