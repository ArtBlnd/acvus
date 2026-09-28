//! A call: of a name, a `Reach` decision over the declarations it may reach
//! (types §6, extern-fn §5); of a value, a function form at the call's
//! arity. Arguments `enter` parameters (types §7). Every other form that
//! reaches a declaration comes here.

use acvus_ast::{AstId, Expr, Slot, Span};

use super::ctxt::Ctxt;
use super::decision::Callee;
use crate::infer::Term;

/// `f(args)`.
pub(super) fn check_call<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, func: &Expr<S>, args: &[Expr<S>], span: Span) -> Term {
    todo!("typeck::call::check_call")
}

/// `left | func`: `func(left)`.
pub(super) fn check_pipe<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, left: &Expr<S>, func: &Expr<S>, span: Span) -> Term {
    todo!("typeck::call::check_pipe")
}

/// The one path to a declaration: a `Reach` at `site` over `callee` with
/// argument terms `args`; the result's term.
pub(super) fn reach_call(cx: &mut Ctxt<'_>, site: AstId, callee: Callee, args: Vec<Term>, span: Span) -> Term {
    todo!("typeck::call::reach_call")
}
