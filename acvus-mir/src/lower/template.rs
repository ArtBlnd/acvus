//! A template's body: one `String` slot it returns; text is appended as a literal and `{{ x }}` through `core::display(&x, &mut out)` (programs §1, operators §2); `%` lines lower as a script's statements.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_append<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, expr: &Expr<S>) {
    todo!("lower::template::lower_append")
}
