//! `recv.f(args)`: a call reaching by the same rule, the receiver first
//! (types §6).

use acvus_ast::{AstId, Expr, Slot, Span};
use acvus_utils::Astr;

use super::ctxt::Ctxt;
use crate::infer::Term;

pub(super) fn check_method_call<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    callee_site: AstId,
    receiver: &Expr<S>,
    name: Astr,
    args: &[Expr<S>],
    span: Span,
) -> Term {
    todo!("typeck::method::check_method_call")
}
