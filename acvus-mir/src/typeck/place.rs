//! Places: field access, `&` and `&mut`, `a[i]`, `*r`, and stores.
//!
//! NOTE: where a place's rule depends on whether a value is a reference,
//! an array or a declared struct (auto-deref of `r.f`, `a[i]` of an array,
//! `*r`), it reads the closed-head query when the place is checked; a term
//! not closed by then takes the general rule (questions Q12 and Q14 to the
//! coordinator).

use acvus_ast::{AstId, Expr, Place, PlaceBase, Root, Slot, Span};
use acvus_utils::{Astr, QualifiedRef};

use super::call::reach_call;
use super::closure::refuse_recursion;
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig};
use super::expr::check_expr;
use super::output::Name;
use super::refusal::{self, RefusalKind};
use super::scope::note_captures;
use crate::decl::{Base, Head};
use crate::infer::Term;
use crate::ty::{Label, Mutability, ObjectTy, TyKind, TypeArg};

/// What an expression is read for, passed down and never held: `a[i]`
/// reaches `as_slice` or `as_slice_mut` by it (operators §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Demand {
    Read,
    Lend(Mutability),
    Write,
}

impl Demand {
    fn mutability(self) -> Mutability {
        match self {
            Demand::Read | Demand::Lend(Mutability::Shared) => Mutability::Shared,
            Demand::Write | Demand::Lend(Mutability::Mut) => Mutability::Mut,
        }
    }

    /// What a place's container is read for when a part of it is: a part
    /// read whole leaves the container in place, lent for the part's read.
    fn of_container(self) -> Demand {
        match self {
            Demand::Read => Demand::Lend(Mutability::Shared),
            other => other,
        }
    }
}

/// `a.f`: a transparent object grows to have `f`; an opaque one must
/// (types §1-2).
pub(super) fn check_field<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, object: &Expr<S>, field: Astr, demand: Demand, span: Span) -> Term {
    let object = check_expr(cx, object, demand.of_container());
    field_of(cx, object, field, span)
}

/// Field `field` of a value of term `object`, through a reference as Rust
/// reads a field through one.
fn field_of(cx: &mut Ctxt<'_>, object: Term, field: Astr, span: Span) -> Term {
    let object = match cx.closed(object).map(|closed| closed.head) {
        Some(Head::Ref(..)) => cx.referent(object),
        _ => object,
    };
    let declared = match cx.closed(object).map(|closed| cx.infer.types().kind(closed.term)) {
        Some(TyKind::Object(ObjectTy::Declared(_, fields))) => Some(fields.clone()),
        _ => None,
    };
    if let Some(fields) = declared {
        return match fields.iter().find(|(name, _)| *name == field) {
            Some((_, part)) => *part,
            None => {
                refusal::push(cx, RefusalKind::GrowsOpaque, span);
                cx.fresh()
            }
        };
    }
    let part = cx.fresh();
    let shape = cx.held(TyKind::Object(ObjectTy::Structural(
        Box::new([(field, part)]),
        Label::Transparent,
    )));
    cx.join(object, shape, span);
    part
}

/// `a[i]`: its container's `as_slice` or `as_slice_mut` by `demand`,
/// reached at `callee_site` (operators §3). An array or a slice is the
/// language's own, since no declaration carries an array (extern-fn §1).
pub(super) fn check_index<S: Slot>(
    cx: &mut Ctxt<'_>,
    _site: AstId,
    callee_site: AstId,
    object: &Expr<S>,
    index: &Expr<S>,
    demand: Demand,
    span: Span,
) -> Term {
    let container = check_expr(cx, object, demand.of_container());
    let at = check_expr(cx, index, Demand::Read);
    let integer = cx.integer(index.span());
    cx.eq(at, integer, index.span());
    element(cx, callee_site, container, demand.mutability(), span)
}

fn element(cx: &mut Ctxt<'_>, callee_site: AstId, container: Term, mutability: Mutability, span: Span) -> Term {
    let sequence = match cx.closed(container).map(|closed| closed.head) {
        Some(Head::Base(Base::Array)) => Some(container),
        Some(Head::Ref(_, Base::Array | Base::Slice)) => Some(cx.referent(container)),
        _ => None,
    };
    if let Some(sequence) = sequence {
        return match cx.infer.types().kind(cx.infer.shallow(sequence)) {
            TyKind::Array(element, _) | TyKind::Slice(element) => *element,
            other => panic!("a term closed as an array or a slice is one, not {other:?}"),
        };
    }
    let element = cx.fresh();
    let slice = cx.node(TyKind::Slice(element));
    let lent_slice = cx.held(TyKind::Ref(mutability, TypeArg::Uniform(slice)));
    let lent = cx.lend(container, mutability, span);
    let reached = reach_call(cx, callee_site, Callee::Core(CoreSig::AsSlice(mutability)), vec![lent], span);
    cx.eq(reached, lent_slice, span);
    element
}

/// `&p`, `&mut p`.
pub(super) fn check_borrow<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, mutable: bool, place: &Expr<S>, span: Span) -> Term {
    let mutability = match mutable {
        true => Mutability::Mut,
        false => Mutability::Shared,
    };
    let target = check_expr(cx, place, Demand::Lend(mutability));
    cx.lend(target, mutability, span)
}

/// `*r` read: a take of what `r` names (operators §6).
///
/// NOTE: an `r` not closed when read is taken to be a `&`, the one
/// reference both mutabilities reach (question Q12 to the coordinator).
pub(super) fn check_deref<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, reference: &Expr<S>, span: Span) -> Term {
    let reference = check_expr(cx, reference, Demand::Read);
    match cx.closed(reference).map(|closed| closed.head) {
        Some(Head::Ref(..)) => cx.referent(reference),
        Some(Head::Base(Base::Never)) => cx.never(),
        Some(Head::Base(_)) => {
            refusal::push(cx, RefusalKind::NotAReference, span);
            cx.fresh()
        }
        None => {
            let target = cx.fresh();
            let shape = cx.held(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(target)));
            cx.eq(reference, shape, span);
            target
        }
    }
}

/// `p = v` into a place; a store into a binding's field joins its object
/// type (ownership §4, types §2).
pub(super) fn check_store<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, place: &Place<S>, value: &Expr<S>, span: Span) {
    let stored = check_expr(cx, value, Demand::Read);
    if let Place::Base(PlaceBase::Root {
        root: Root::Local(name), ..
    }) = place
    {
        refuse_recursion(cx, stored, *name, span);
    }
    let Some(slot) = place_slot(cx, place) else {
        return;
    };
    cx.enter(value.id(), stored, slot, value.span());
}

/// The term a store into `place` enters, `None` where the place is refused.
fn place_slot<S: Slot>(cx: &mut Ctxt<'_>, place: &Place<S>) -> Option<Term> {
    acvus_utils::grow(|| {
        let slot = match place {
            Place::Base(PlaceBase::Root { id, root, span }) => root_slot(cx, *id, *root, *span)?,
            Place::Base(PlaceBase::Element {
                id,
                callee_id,
                container,
                index,
                span,
            }) => check_index(cx, *id, *callee_id, container.expr(), index, Demand::Write, *span),
            Place::Field {
                object, field, span, ..
            } => {
                let object = place_slot(cx, object)?;
                field_of(cx, object, *field, *span)
            }
        };
        cx.record(place.id(), slot);
        Some(slot)
    })
}

fn root_slot(cx: &mut Ctxt<'_>, site: AstId, root: Root, span: Span) -> Option<Term> {
    match root {
        Root::Local(name) => match cx.scope.lookup(name) {
            Some(found) => {
                cx.out.names.insert(site, Name::Binding(found.binding.binder));
                note_captures(cx, name, &found, Demand::Write, span);
                Some(found.binding.term)
            }
            None => {
                refusal::push(cx, RefusalKind::UnknownName { name }, span);
                None
            }
        },
        // An input is never assigned (programs §4).
        Root::ExternParam(input) => {
            refusal::push(cx, RefusalKind::InputAssigned { input }, span);
            None
        }
        Root::Context(name) => {
            let context = QualifiedRef::root(name);
            let term = super::context::context_term(cx, site, context, span)?;
            super::context::refuse_held_loans(cx, term, span);
            cx.out.names.insert(site, Name::Context(context));
            Some(term)
        }
    }
}

/// `*r = v`: `r` must be a `&mut` (operators §6).
pub(super) fn check_deref_store<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, target: &Expr<S>, value: &Expr<S>, span: Span) {
    let reference = check_expr(cx, target, Demand::Write);
    let stored = check_expr(cx, value, Demand::Read);
    if let Some(Head::Ref(Mutability::Shared, _)) = cx.closed(reference).map(|closed| closed.head) {
        refusal::push(cx, RefusalKind::StoreThroughShared, span);
        return;
    }
    let slot = cx.fresh();
    let shape = cx.held(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(slot)));
    cx.eq(reference, shape, target.span());
    cx.enter(value.id(), stored, slot, value.span());
}
