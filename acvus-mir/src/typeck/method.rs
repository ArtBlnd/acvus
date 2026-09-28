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
///
/// NOTE: the receiver is the first argument as it stands, with no `&`
/// added; the ledger states no adjustment (question Q18 to the
/// coordinator).
pub(super) fn check_method_call<S: Slot>(
    cx: &mut Ctxt<'_>,
    _site: AstId,
    callee_site: AstId,
    receiver: &Expr<S>,
    name: Astr,
    args: &[Expr<S>],
    span: Span,
) -> Term {
    let mut terms = vec![check_expr(cx, receiver, Demand::Read)];
    terms.extend(args.iter().map(|arg| check_expr(cx, arg, Demand::Read)));
    let callee = match name == cx.decls.core().unwrap.name {
        true => Callee::Core(CoreSig::Unwrap),
        false => Callee::Method(name),
    };
    reach_call(cx, callee_site, callee, terms, span)
}
