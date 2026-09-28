//! Contexts (context): a fixed, invariant type; a value stored into one, and
//! a run's result, have types with no positions (ownership §5).

use acvus_ast::{AstId, Span};
use acvus_utils::QualifiedRef;

use super::ctxt::Ctxt;
use crate::infer::Term;

/// `@x`: the declared type, invariant (context §2).
pub(super) fn check_context_ref(cx: &mut Ctxt<'_>, site: AstId, context: QualifiedRef, span: Span) -> Term {
    todo!("typeck::context::check_context_ref")
}

/// A run's result holds no loan (ownership §5).
pub(super) fn check_run_result(cx: &mut Ctxt<'_>, result: Term, span: Span) {
    todo!("typeck::context::check_run_result")
}
