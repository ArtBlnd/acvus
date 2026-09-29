//! Places: field access, `&` and `&mut`, `a[i]`, `*r`, and stores. What
//! each means depends on its operand's head (a reference, an array, a
//! declared struct), so each is a decision fired once that head is known
//! (types §6).

use acvus_ast::{AstId, Expr, Place, PlaceBase, Root, Slot, Span};
use acvus_utils::{Astr, QualifiedRef};

use super::call::reach_in;
use super::closure::refuse_recursion;
use super::context::Access;
use super::ctxt::Ctxt;
use super::decision::{Callee, CoreSig, Decision};
use super::expr::check_expr;
use super::output::Name;
use super::refusal::{self, RefusalKind};
use super::scope::{capturing_binding, note_captures};
use crate::decl::{Base, Head};
use crate::infer::{Closed, Term};
use crate::ty::{EffectTerm, Infer as InferPhase, IntTy, Label, Mutability, ObjectTy, TyKind, TypeArg};

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

/// Field `field` of a value of term `object`: a decision on its head.
fn field_of(cx: &mut Ctxt<'_>, object: Term, field: Astr, span: Span) -> Term {
    let part = cx.fresh();
    super::decision::decide(
        cx,
        Decision::Field {
            span,
            object,
            field,
            part,
        },
    );
    part
}

/// A field read fired: through a reference to its target, as Rust
/// auto-derefs; of a declared struct, the field it has; of `!`, `!`; of
/// any other object, the field it grows to have where transparent
/// (types §1-2). `head` is `None` where the object's use decides it at a
/// stall, which makes it an object and never a reference.
pub(super) fn fire_field(cx: &mut Ctxt<'_>, span: Span, object: Term, head: Option<Head>, field: Astr, part: Term) {
    match head {
        Some(Head::Ref(..)) => {
            let target = cx.referent(object);
            let reached = field_of(cx, target, field, span);
            cx.eq(part, reached, span);
            return;
        }
        Some(Head::Base(Base::Never)) => {
            let never = cx.never();
            cx.eq(part, never, span);
            return;
        }
        Some(Head::Base(_)) | None => {}
    }
    let declared = match head {
        Some(_) => match cx.infer.types().kind(cx.infer.shallow(object)) {
            TyKind::Object(ObjectTy::Declared(_, fields)) => Some(fields.clone()),
            _ => None,
        },
        None => None,
    };
    if let Some(fields) = declared {
        match fields.iter().find(|(name, _)| *name == field) {
            Some((_, declared_part)) => cx.eq(part, *declared_part, span),
            None => refusal::push(cx, RefusalKind::GrowsOpaque, span),
        }
        return;
    }
    let shape = cx.held(TyKind::Object(ObjectTy::Structural(
        Box::new([(field, part)]),
        Label::Here,
    )));
    cx.join(object, shape, span);
}

/// `a[i]`: its container's `as_slice` or `as_slice_mut` by `demand`,
/// reached at `callee_site` (operators §3). An array or a slice is the
/// language's own, since no declaration carries an array (extern-fn §1).
/// `i` enters a `u64`, as Rust's `usize` (operators §3).
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
    let usize_ty = cx.node(TyKind::Int(IntTy::U64));
    cx.enter(index.id(), at, usize_ty, index.span());
    element(cx, callee_site, container, demand.mutability(), span)
}

fn element(cx: &mut Ctxt<'_>, callee_site: AstId, container: Term, mutability: Mutability, span: Span) -> Term {
    let element = cx.fresh();
    let body = super::effect::body_effect(cx);
    let index = Decision::Index {
        callee_site,
        span,
        container,
        mutability,
        element,
        body,
    };
    super::decision::decide(cx, index);
    element
}

/// An index fired: an array's or a slice's element is the language's own,
/// since no declaration carries an array (extern-fn §1), through a
/// reference too; any other container's is reached through its
/// `as_slice(_mut)` (operators §3).
pub(super) fn fire_index(
    cx: &mut Ctxt<'_>,
    callee_site: AstId,
    span: Span,
    container: Closed,
    mutability: Mutability,
    element: Term,
    body: &EffectTerm<InferPhase>,
) {
    let sequence = match container.head {
        Head::Base(Base::Array) => Some(container.term),
        Head::Ref(_, Base::Array | Base::Slice) => Some(cx.referent(container.term)),
        Head::Base(Base::Never) => {
            let never = cx.never();
            cx.eq(element, never, span);
            return;
        }
        Head::Base(_) | Head::Ref(..) => None,
    };
    if let Some(sequence) = sequence {
        let part = match cx.infer.types().kind(cx.infer.shallow(sequence)) {
            TyKind::Array(part, _, _) | TyKind::Slice(part) => *part,
            other => panic!("a term closed as an array or a slice is one, not {other:?}"),
        };
        cx.eq(element, part, span);
        return;
    }
    let slice = cx.node(TyKind::Slice(element));
    let lent_slice = cx.held(TyKind::Ref(mutability, TypeArg::Uniform(slice)));
    let lent = cx.lend(container.term, mutability, span);
    let callee = Callee::Core(CoreSig::AsSlice(mutability));
    let reached = reach_in(cx, body, callee_site, callee, vec![(callee_site, lent)], span);
    cx.eq(reached, lent_slice, span);
}

/// A lend fired. A reference is reborrowed: a `&mut` of a `&` is refused,
/// and a `&` of a `&mut` is a `&` of its target. Any other value, and one
/// whose use decides it at a stall (`head` `None`), is lent as it is.
pub(super) fn fire_lend(cx: &mut Ctxt<'_>, span: Span, value: Option<Closed>, mutability: Mutability, result: Term, target: Term) {
    let lent = match value.map(|closed| closed.head) {
        Some(Head::Ref(held, _)) => {
            if !held.reaches(mutability) {
                let wanted = cx.fresh();
                let wanted = cx.node(TyKind::Ref(mutability, TypeArg::Uniform(wanted)));
                cx.eq(target, wanted, span);
            }
            match held == mutability {
                true => target,
                false => {
                    let inner = cx.referent(target);
                    cx.node(TyKind::Ref(mutability, TypeArg::Uniform(inner)))
                }
            }
        }
        Some(Head::Base(_)) | None => cx.node(TyKind::Ref(mutability, TypeArg::Uniform(target))),
    };
    cx.eq(result, lent, span);
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
pub(super) fn check_deref<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, reference: &Expr<S>, span: Span) -> Term {
    let reference = check_expr(cx, reference, Demand::Read);
    deref(cx, reference, false, span)
}

fn deref(cx: &mut Ctxt<'_>, reference: Term, store: bool, span: Span) -> Term {
    let target = cx.fresh();
    let deref = Decision::Deref {
        span,
        reference,
        target,
        store,
    };
    super::decision::decide(cx, deref);
    target
}

/// A deref fired: a reference's target, which a store reaches only through
/// a `&mut` (operators §6); `!` of `!`; anything else is refused.
pub(super) fn fire_deref(cx: &mut Ctxt<'_>, span: Span, reference: Closed, target: Term, store: bool) {
    match reference.head {
        Head::Ref(Mutability::Shared, _) if store => refusal::push(cx, RefusalKind::StoreThroughShared, span),
        Head::Ref(..) => {
            let referent = cx.referent(reference.term);
            cx.eq(target, referent, span);
        }
        Head::Base(Base::Never) => {
            let never = cx.never();
            cx.eq(target, never, span);
        }
        Head::Base(_) => refusal::push(cx, RefusalKind::NotAReference, span),
    }
}

/// `p = v` into a place; a store into a binding's field joins its object
/// type (ownership §4, types §2).
pub(super) fn check_store<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, place: &Place<S>, value: &Expr<S>, span: Span) {
    // A value given a generalized binding is its binding's walk, as the
    // `let`'s is (G): its scheme is settled from every value it is given.
    let generalized = match place {
        Place::Base(PlaceBase::Root { root: Root::Local(name), .. }) => {
            cx.scope.lookup(*name).filter(|found| found.binding.scheme.is_some()).map(|found| found.binding.binder)
        }
        _ => None,
    };
    let store = |cx: &mut Ctxt<'_>| {
        let stored = check_expr(cx, value, Demand::Read);
        if let Place::Base(PlaceBase::Root {
            root: Root::Local(name), ..
        }) = place
        {
            refuse_recursion(cx, stored, *name, span);
        }
        let Some(slot) = place_slot(cx, place, Access::Write) else {
            return;
        };
        cx.enter(value.id(), stored, slot, value.span());
    };
    match generalized {
        Some(binder) => super::generalize::walk(cx, binder, store),
        None => store(cx),
    }
}

/// The term a store into `place` enters, `None` where the place is refused.
/// A context at its root is touched by `access`: written where it is the
/// whole place, and read and written where a part of it is stored into
/// (context §3).
fn place_slot<S: Slot>(cx: &mut Ctxt<'_>, place: &Place<S>, access: Access) -> Option<Term> {
    acvus_utils::grow(|| {
        let slot = match place {
            Place::Base(PlaceBase::Root { id, root, span }) => root_slot(cx, *id, *root, access, *span)?,
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
                let object = place_slot(cx, object, Access::ReadWrite)?;
                field_of(cx, object, *field, *span)
            }
        };
        cx.record(place.id(), slot);
        Some(slot)
    })
}

fn root_slot(cx: &mut Ctxt<'_>, site: AstId, root: Root, access: Access, span: Span) -> Option<Term> {
    match root {
        Root::Local(name) => match cx.scope.lookup(name) {
            Some(found) => {
                if access != Access::Write && found.binding.scheme.is_some() {
                    refusal::push(cx, RefusalKind::MutOfGeneralized { name }, span);
                }
                cx.out.names.insert(site, Name::Binding(found.binding.binder));
                note_captures(cx, capturing_binding(name, &found), Demand::Write, span);
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
            super::context::note_access(cx, context, access, span);
            Some(term)
        }
    }
}

/// `*r = v`: `r` must be a `&mut` (operators §6).
pub(super) fn check_deref_store<S: Slot>(cx: &mut Ctxt<'_>, _site: AstId, target: &Expr<S>, value: &Expr<S>, span: Span) {
    let reference = check_expr(cx, target, Demand::Write);
    let stored = check_expr(cx, value, Demand::Read);
    let slot = deref(cx, reference, true, span);
    cx.enter(value.id(), stored, slot, value.span());
}
