//! A lambda is a `MakeClosure` of a body of its own, lowered over a nested Builder: ordered where its effect is not pure. A capture by move passes the value; by reference, a `&T` value (functions §2).

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_lambda<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, params: &[Binder], body: &Expr<S>) -> ValueId {
    todo!("lower::closure::lower_lambda")
}
