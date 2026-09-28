//! Places: a field, `a[i]` through the recorded `as_slice(_mut)` (operators §3), `&`/`&mut` of a place (a reborrow of a reference: types §4), `*r` as a take (operators §6), and stores into places and through `&mut`.

use acvus_ast::*;

use super::ctxt::Lcx;
use crate::ir::ValueId;

pub(super) fn lower_borrow<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, mutable: bool, place: &Expr<S>) -> ValueId {
    todo!("lower::place::lower_borrow")
}

pub(super) fn lower_deref<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, reference: &Expr<S>) -> ValueId {
    todo!("lower::place::lower_deref")
}

pub(super) fn lower_store<S: Slot>(cx: &mut Lcx<'_, '_>, site: AstId, place: &Place<S>, value: &Expr<S>) {
    todo!("lower::place::lower_store")
}
