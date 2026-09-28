//! Patterns (patterns §1): a pattern naming a variant grows the enum where
//! transparent, and is refused where it is not (patterns §2); a `!`
//! scrutinee reaches no arm (patterns §4). Exhaustiveness is validate's.
//!
//! A pattern relates to its scrutinee by `join`: a transparent scrutinee
//! grows by what the pattern names, an opaque one must already have it, and
//! `join` with `!` leaves every slot of the pattern open, so each becomes
//! `!` (patterns §4, types §5).

use acvus_ast::{ObjectPatternField, Pattern, RefKind, Slot, Span, TuplePatternElem};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use super::output::Name;
use super::refusal::{self, RefusalKind};
use super::scope::Binding;
use crate::decl::{Base, Head};
use crate::infer::Term;
use crate::ty::{Label, LenTerm, Mutability, ObjectTy, TyKind};

/// Binds `pattern`'s names against a scrutinee of term `scrutinee`.
pub(super) fn check_pattern<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>, scrutinee: Term) {
    at(cx, pattern, scrutinee, None);
}

/// `pattern` against `scrutinee`, its bindings lent by `lent` where the
/// match passed through a reference (Rust's default binding modes,
/// patterns §1).
///
/// NOTE: whether the scrutinee is a reference is read through the
/// closed-head query when the pattern is checked. A scrutinee not closed by
/// then is matched as it stands (question Q12 to the coordinator).
fn at<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>, scrutinee: Term, lent: Option<Mutability>) {
    acvus_utils::grow(|| at_level(cx, pattern, scrutinee, lent));
}

fn at_level<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>, scrutinee: Term, lent: Option<Mutability>) {
    let destructures = !matches!(
        pattern,
        Pattern::Binding { .. } | Pattern::Wildcard { .. } | Pattern::ContextBind { .. } | Pattern::Error(_)
    );
    if destructures && let Some(Head::Ref(mutability, _)) = cx.closed(scrutinee).map(|closed| closed.head) {
        let through = match lent {
            Some(Mutability::Shared) => Mutability::Shared,
            Some(Mutability::Mut) | None => mutability,
        };
        let target = cx.referent(scrutinee);
        return at(cx, pattern, target, Some(through));
    }
    cx.record(pattern.id(), scrutinee);
    match pattern {
        Pattern::Binding {
            id,
            name,
            ref_kind,
            span,
        } => match ref_kind {
            RefKind::ExternParam => refusal::push(cx, RefusalKind::InputAssigned { input: *name }, *span),
            RefKind::Value => {
                let value = bound_value(cx, scrutinee, lent, *span);
                let term = cx.fresh();
                cx.eq(term, value, *span);
                cx.record(*id, term);
                cx.scope.define(*name, Binding { binder: *id, term });
            }
        },
        Pattern::ContextBind { id, name, span } => {
            let value = bound_value(cx, scrutinee, lent, *span);
            if let Some(context) = super::context::context_term(cx, *id, *name, *span) {
                cx.out.names.insert(*id, Name::Context(*name));
                cx.enter(*id, value, context, *span);
            }
        }
        Pattern::Literal { value, span, .. } => {
            let literal = super::expr::check_literal(cx, value, *span);
            cx.join(scrutinee, literal, *span);
        }
        Pattern::List {
            head, rest, tail, span, ..
        } => {
            let element = cx.fresh();
            let shape = match cx.closed(scrutinee).map(|closed| closed.head) {
                Some(Head::Base(Base::Slice)) => TyKind::Slice(element),
                _ => {
                    let len = match rest {
                        Some(_) => LenTerm::Var(cx.infer.fresh_len()),
                        None => LenTerm::Known(head.len() + tail.len()),
                    };
                    TyKind::Array(element, len)
                }
            };
            let shape = cx.held(shape);
            cx.join(scrutinee, shape, *span);
            for item in head.iter().chain(tail) {
                at(cx, item, element, lent);
            }
        }
        Pattern::Object { fields, span, .. } => object(cx, fields, scrutinee, lent, *span),
        Pattern::Tuple { elements, span, .. } => {
            let parts: Vec<Term> = elements.iter().map(|_| cx.fresh()).collect();
            let shape = cx.held(TyKind::Tuple(parts.clone().into()));
            cx.join(scrutinee, shape, *span);
            for (element, part) in elements.iter().zip(parts) {
                if let TuplePatternElem::Pattern(inner) = element {
                    at(cx, inner, part, lent);
                }
            }
        }
        Pattern::Wildcard { .. } | Pattern::Error(_) => {}
        Pattern::Variant {
            enum_name,
            tag,
            payload,
            span,
            ..
        } => variant(cx, *enum_name, *tag, payload.as_deref(), scrutinee, lent, *span),
    }
}

/// What a binding holds: the part itself, or where the match passed
/// through a reference, a reference to it.
fn bound_value(cx: &mut Ctxt<'_>, part: Term, lent: Option<Mutability>, span: Span) -> Term {
    match lent {
        Some(mutability) => cx.lend(part, mutability, span),
        None => part,
    }
}

/// `{ f: p, .. }`: a declared struct must have each field; any other
/// object grows by them where transparent (types §1-2).
fn object<S: Slot>(
    cx: &mut Ctxt<'_>,
    fields: &[ObjectPatternField<S>],
    scrutinee: Term,
    lent: Option<Mutability>,
    span: Span,
) {
    let declared = match cx.closed(scrutinee).map(|closed| closed.term) {
        Some(node) => match cx.infer.types().kind(node) {
            TyKind::Object(ObjectTy::Declared(_, declared)) => Some(declared.clone()),
            _ => None,
        },
        None => None,
    };
    if let Some(declared) = declared {
        for field in fields {
            match declared.iter().find(|(name, _)| *name == field.key) {
                Some((_, part)) => at(cx, &field.pattern, *part, lent),
                None => refusal::push(cx, RefusalKind::GrowsOpaque, field.span),
            }
        }
        return;
    }
    let parts: Vec<(Astr, Term)> = fields.iter().map(|field| (field.key, cx.fresh())).collect();
    let mut sorted = parts.clone();
    sorted.sort_by_key(|(name, _)| *name);
    let shape = cx.held(TyKind::Object(ObjectTy::Structural(sorted.into(), Label::Transparent)));
    cx.join(scrutinee, shape, span);
    for (field, (_, part)) in fields.iter().zip(parts) {
        at(cx, &field.pattern, part, lent);
    }
}

/// A variant pattern: a named enum's variant grows the enum where
/// transparent (patterns §2); without a name, only `Some`, `None`, `Ok` and
/// `Err` of the primitive `Option` and `Result` (types §2).
fn variant<S: Slot>(
    cx: &mut Ctxt<'_>,
    enum_name: Option<Astr>,
    tag: Astr,
    payload: Option<&Pattern<S>>,
    scrutinee: Term,
    lent: Option<Mutability>,
    span: Span,
) {
    let tags = cx.decls.core().tags;
    let (shape, part) = match enum_name {
        Some(name) => {
            let part = payload.map(|_| cx.fresh());
            let shape = TyKind::Enum {
                name: QualifiedRef::root(name),
                variants: Box::new([(tag, part)]),
                label: Label::Transparent,
            };
            (shape, part)
        }
        None if tag == tags.some || tag == tags.none => {
            let inner = cx.fresh();
            (TyKind::Option(inner), (tag == tags.some).then_some(inner))
        }
        None if tag == tags.ok || tag == tags.err => {
            let (ok, err) = (cx.fresh(), cx.fresh());
            (TyKind::Result(ok, err), Some(if tag == tags.ok { ok } else { err }))
        }
        None => {
            refusal::push(cx, RefusalKind::UnnamedVariant { tag }, span);
            return;
        }
    };
    let shape = cx.held(shape);
    cx.join(scrutinee, shape, span);
    match (payload, part) {
        (Some(payload), Some(part)) => at(cx, payload, part, lent),
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => refusal::push(cx, RefusalKind::VariantArity { tag }, span),
    }
}
