//! Calls: to the `Chosen` typeck recorded (extern-fn §5-6). An operator or a view is a call of a lang item's impl; `^ & | << >> !` are the language's operations; `as` is a cast; a recorded conversion is a call of its `via`, at the impl its `Chosen` names (types §7, operators §5). An effectful call goes through `push_effect`.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_call<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, func: &Expr<S>, args: &[Expr<S>]) -> ValueId {
    todo!("lower::call::lower_call")
}

pub(super) fn lower_binary<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, op: BinOp, left: &Expr<S>, right: &Expr<S>) -> ValueId {
    todo!("lower::call::lower_binary")
}
