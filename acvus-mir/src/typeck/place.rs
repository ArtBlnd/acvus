//! Places: field access, `&` and `&mut`, `a[i]`, `*r`, and stores.

use acvus_ast::{AstId, Expr, Place, Slot, Span};
use acvus_utils::Astr;

use super::ctxt::Ctxt;
use crate::infer::Term;
use crate::ty::Mutability;

/// What an expression is read for, passed down and never held: `a[i]`
/// reaches `as_slice` or `as_slice_mut` by it (operators §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Demand {
    Read,
    Lend(Mutability),
    Write,
}

/// `a.f`: a transparent object grows to have `f`; an opaque one must
/// (types §1-2).
pub(super) fn check_field<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, object: &Expr<S>, field: Astr, demand: Demand, span: Span) -> Term {
    todo!("typeck::place::check_field")
}

/// `a[i]`: its container's `as_slice` or `as_slice_mut` by `demand`,
/// reached at `callee_site` (operators §3).
pub(super) fn check_index<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    callee_site: AstId,
    object: &Expr<S>,
    index: &Expr<S>,
    demand: Demand,
    span: Span,
) -> Term {
    todo!("typeck::place::check_index")
}

/// `&p`, `&mut p`.
pub(super) fn check_borrow<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, mutable: bool, place: &Expr<S>, span: Span) -> Term {
    todo!("typeck::place::check_borrow")
}

/// `*r` read: a take of what `r` names (operators §6).
pub(super) fn check_deref<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, reference: &Expr<S>, span: Span) -> Term {
    todo!("typeck::place::check_deref")
}

/// `p = v` into a place; a store into a binding's field joins its object
/// type (ownership §4, types §2).
pub(super) fn check_store<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, place: &Place<S>, value: &Expr<S>, span: Span) {
    todo!("typeck::place::check_store")
}

/// `*r = v`: `r` must be a `&mut` (operators §6).
pub(super) fn check_deref_store<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, target: &Expr<S>, value: &Expr<S>, span: Span) {
    todo!("typeck::place::check_deref_store")
}
