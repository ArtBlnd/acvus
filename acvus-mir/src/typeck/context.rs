//! Contexts (context): a fixed, invariant type; a value stored into one, and
//! a run's result, have types with no positions (ownership §5).

use acvus_ast::{AstId, Span};
use acvus_utils::QualifiedRef;

use super::ctxt::Ctxt;
use super::decision::Decision;
use super::output::Name;
use super::refusal::{self, RefusalKind};
use crate::infer::Term;
use crate::ty::{LifetimesOf, TyKind};

/// `@x`: the declared type, invariant (context §2).
pub(super) fn check_context_ref(cx: &mut Ctxt<'_>, site: AstId, context: QualifiedRef, span: Span) -> Term {
    match context_term(cx, site, context, span) {
        Some(term) => {
            cx.out.names.insert(site, Name::Context(context));
            term
        }
        None => cx.fresh(),
    }
}

/// A context's declared type at this body, every node `Opaque`, so it never
/// grows and equals only itself (context §2, types §1).
pub(super) fn context_term(cx: &mut Ctxt<'_>, _site: AstId, context: QualifiedRef, span: Span) -> Option<Term> {
    match cx.decls.context(context) {
        Some(ty) => Some(cx.declared(ty)),
        None => {
            refusal::push(cx, RefusalKind::UnknownContext { context }, span);
            None
        }
    }
}

/// A value stored into a context outlives the run, so the context's type
/// has no positions (context §3, ownership §5).
pub(super) fn refuse_held_loans(cx: &mut Ctxt<'_>, context: Term, span: Span) {
    if holds_positions(cx, context) {
        refusal::push(cx, RefusalKind::HoldsLoans, span);
    }
}

/// A run's result holds no loan (ownership §5), read once its type is
/// solved. The tail enters the run's result at `site`, its own, as a
/// `return` or a `?` enters it at theirs (types §3, §7).
pub(super) fn check_run_result(cx: &mut Ctxt<'_>, site: AstId, tail: Term, span: Span) {
    let result = super::control::body_result(cx);
    cx.enter(site, tail, result, span);
    cx.decisions.push(Decision::HoldsNoLoans { span, term: result });
}

/// Whether a value of a closed term holds a loan at some position
/// (ownership §5): a reference, a function value, an extern-defined type's
/// declared lifetime, or any of these in a part.
///
/// # Panics
/// `term` holds an unbound variable: it is read only on a declared type,
/// which has none, or after solving, which binds every one.
pub(super) fn holds_positions(cx: &Ctxt<'_>, term: Term) -> bool {
    let mut stack = vec![term];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(at) = stack.pop() {
        let at = cx.infer.shallow(at);
        if !seen.insert(at) {
            continue;
        }
        match cx.infer.types().kind(at) {
            TyKind::Ref(..) | TyKind::Fn { .. } | TyKind::Handle(_) => return true,
            TyKind::UserDefined { id, .. } if cx.decls.lifetimes_of(*id) > 0 => return true,
            TyKind::Var(var) => panic!("holds_positions: {var:?} is unbound on a declared or solved type"),
            kind => kind.for_each_child(|child| stack.push(child)),
        }
    }
    false
}
