//! Lambdas (functions §2-4): captures by move or by reference, once or
//! many, and no recursion. A lambda's flows are claimed at the type level,
//! by union over positions (ownership §5); validate's borrow check verifies
//! the claim.

use acvus_ast::{AstId, Binder, Expr, Slot, Span};

use super::ctxt::Ctxt;
use crate::infer::Term;

pub(super) fn check_lambda<S: Slot>(cx: &mut Ctxt<'_>, site: AstId, params: &[Binder], body: &Expr<S>, span: Span) -> Term {
    todo!("typeck::closure::check_lambda")
}
