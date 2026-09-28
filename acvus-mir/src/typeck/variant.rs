//! A variant expression (types §2): with its enum's name, that enum, grown
//! where transparent; without, only `Some`, `None`, `Ok` and `Err`, of the
//! primitive `Option` and `Result`; any other tag without a name is refused.

use acvus_ast::{AstId, Expr, Slot, Span};
use acvus_utils::Astr;

use super::ctxt::Ctxt;
use crate::infer::Term;

pub(super) fn check_variant<S: Slot>(
    cx: &mut Ctxt<'_>,
    site: AstId,
    enum_name: Option<Astr>,
    tag: Astr,
    payload: Option<&Expr<S>>,
    span: Span,
) -> Term {
    todo!("typeck::variant::check_variant")
}
