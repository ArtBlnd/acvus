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
use crate::ty::structural::{PartKey, VariantKey, structure};
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
    let (variant, shape) = match (enum_name, cx.decls.core().tags.primitive(tag)) {
        (Some(name), _) => {
            let shape = TyKind::Enum {
                name: QualifiedRef::root(name),
                variants: Box::new([(tag, payload)]),
                label: Label::Here,
            };
            (VariantKey::Enum(tag), shape)
        }
        (None, Some(VariantKey::Some)) => (VariantKey::Some, TyKind::Option(payload_or_fresh(cx, payload), Label::Here)),
        (None, Some(VariantKey::None)) => (VariantKey::None, TyKind::Option(cx.fresh(), Label::Here)),
        (None, Some(VariantKey::Ok)) => {
            let err = cx.fresh();
            (VariantKey::Ok, TyKind::Result(payload_or_fresh(cx, payload), err, Label::Here))
        }
        (None, Some(VariantKey::Err)) => {
            let ok = cx.fresh();
            (VariantKey::Err, TyKind::Result(ok, payload_or_fresh(cx, payload), Label::Here))
        }
        (None, Some(VariantKey::Enum(_))) => unreachable!("`Tags::primitive` names no enum's variant"),
        (None, None) => {
            refusal::push(cx, RefusalKind::UnnamedVariant { tag }, span);
            return cx.fresh();
        }
    };
    let made = cx.held(shape);
    let carries = structure(cx.infer.types(), cx.infer.shallow(made))
        .and_then(|structure| structure.part(PartKey::Payload(variant)))
        .is_some();
    if carries != payload.is_some() {
        refusal::push(cx, RefusalKind::VariantArity { tag }, span);
    }
    made
}

/// A primitive variant's payload; one missing is refused by the caller,
/// and the variant's type stands on a fresh part meanwhile.
fn payload_or_fresh(cx: &mut Ctxt<'_>, payload: Option<Term>) -> Term {
    match payload {
        Some(payload) => payload,
        None => cx.fresh(),
    }
}
