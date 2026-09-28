//! `recv.f(args)`: a call reaching by the same rule, the receiver first
//! (types §6). `.unwrap()` is core's lang item (operators §2).

use acvus_ast::{AstId, Expr, Slot, Span};
use acvus_utils::Astr;

use super::call::reach_call;
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig};
use super::expr::check_expr;
use super::place::Demand;
use crate::infer::Term;

/// The reach is recorded at `callee_site`, where the callee is written.
pub(super) fn check_method_call<S: Slot>(
    cx: &mut Ctxt<'_>,
    _site: AstId,
    callee_site: AstId,
    receiver: &Expr<S>,
    name: Astr,
    args: &[Expr<S>],
    span: Span,
) -> Term {
    let mut terms = vec![receiver_arg(cx, receiver)];
    terms.extend(args.iter().map(|arg| (arg.id(), check_expr(cx, arg, Demand::Read))));
    let callee = match name == cx.decls.core().unwrap.name {
        true => Callee::Core(CoreSig::Unwrap),
        false => Callee::Method(name),
    };
    reach_call(cx, callee_site, callee, terms, span)
}

/// The receiver as the call's first argument.
///
/// PARKED(P8): the ledger does not state Rust's receiver adjustment
/// (auto-ref and auto-deref). Provisionally the receiver is the argument
/// as it stands, with no `&` added; this function is the one place that
/// decides it.
fn receiver_arg<S: Slot>(cx: &mut Ctxt<'_>, receiver: &Expr<S>) -> (AstId, Term) {
    (receiver.id(), check_expr(cx, receiver, Demand::Read))
}
