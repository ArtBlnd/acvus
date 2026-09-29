//! A variant expression (types §2): with its enum's name, that enum, grown
//! where transparent; without, only `Some`, `None`, `Ok` and `Err`, of the
//! primitive `Option` and `Result`; any other tag without a name is refused.

use acvus_ast::{AstId, Expr, Slot, Span};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use super::expr::check_expr;
use super::place::Demand;
use super::refusal::{self, RefusalKind};
use crate::infer::Term;
use crate::ty::{Label, TyKind};

pub(super) fn check_variant<S: Slot>(
    cx: &mut Ctxt<'_>,
    _site: AstId,
    enum_name: Option<Astr>,
    tag: Astr,
    payload: Option<&Expr<S>>,
    span: Span,
) -> Term {
    let payload = payload.map(|payload| check_expr(cx, payload, Demand::Read));
    let tags = cx.decls.core().tags;
    let (shape, carries) = match enum_name {
        Some(name) => {
            let shape = TyKind::Enum {
                name: QualifiedRef::root(name),
                variants: Box::new([(tag, payload)]),
                label: Label::Transparent,
            };
            (shape, payload.is_some())
        }
        None if tag == tags.some => (TyKind::Option(payload_or_fresh(cx, payload)), true),
        None if tag == tags.none => (TyKind::Option(cx.fresh()), false),
        None if tag == tags.ok => {
            let err = cx.fresh();
            (TyKind::Result(payload_or_fresh(cx, payload), err), true)
        }
        None if tag == tags.err => {
            let ok = cx.fresh();
            (TyKind::Result(ok, payload_or_fresh(cx, payload)), true)
        }
        None => {
            refusal::push(cx, RefusalKind::UnnamedVariant { tag }, span);
            return cx.fresh();
        }
    };
    if carries != payload.is_some() {
        refusal::push(cx, RefusalKind::VariantArity { tag }, span);
    }
    cx.held(shape)
}

/// A primitive variant's payload; one missing is refused by the caller,
/// and the variant's type stands on a fresh part meanwhile.
fn payload_or_fresh(cx: &mut Ctxt<'_>, payload: Option<Term>) -> Term {
    match payload {
        Some(payload) => payload,
        None => cx.fresh(),
    }
}
