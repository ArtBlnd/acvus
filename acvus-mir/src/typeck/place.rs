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
use super::decision::{Callee, CoreSig, Decision, Lent};
use super::expr::check_expr;
use super::output::Name;
use super::refusal::{self, RefusalKind};
use super::scope::{capturing_binding, note_captures};
use crate::decl::{Base, Head};
use crate::infer::{Closed, Term};
use crate::ty::{EffectTerm, Infer as InferPhase, IntTy, Label, Mutability, Reach, TyKind};

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
    let part = field_of(cx, object, field, span);
    read_for(cx, part, demand, span)
}

/// A place read for its value is a value of its own, which a reference
/// part reached through references makes a reborrow (ownership §5); a
/// place stored into or lent is the place itself.
fn read_for(cx: &mut Ctxt<'_>, place: Term, demand: Demand, span: Span) -> Term {
    match demand {
        Demand::Read => cx.lend_as(place, Lent::Read, span),
        Demand::Lend(_) | Demand::Write => place,
    }
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

/// `term`, a place's step, reached through references at `least` where it
/// is: recorded, so that every step after it on its path is too (types §1),
/// with the least mutability on the path (ownership §5).
fn record_reach(cx: &mut Ctxt<'_>, term: Term, least: Option<Mutability>) {
    if let Some(least) = least {
        cx.out.through.insert(term, least);
    }
}

/// The least mutability on the path to `term`, where references are on
/// it: its record, or, where `reach` is through a step still pending,
/// shared, the least there is.
pub(super) fn path_of(cx: &Ctxt<'_>, term: Term, reach: Reach) -> Option<Mutability> {
    match cx.out.through.get(&term) {
        Some(least) => Some(*least),
        None => (reach == Reach::Through).then_some(Mutability::Shared),
    }
}

/// The path to a step past a reference of `mutability` from `from`,
/// reached by `reach`.
fn past(cx: &Ctxt<'_>, from: Term, reach: Reach, mutability: Mutability) -> Mutability {
    path_of(cx, from, reach).map_or(mutability, |least| least.least(mutability))
}

/// A field read fired, of an object reached by `reach`: through a
/// reference to its target, reached through it, as Rust auto-derefs; of an
/// object, the field it has or grows to have where transparent at `reach`
/// (types §1-2). A field of `!` is `never::settle`'s.
/// The part is reached as its object was, or through the reference. `head`
/// is `None` where the object's use decides it at a stall, which makes it
/// an object and never a reference.
pub(super) fn fire_field(cx: &mut Ctxt<'_>, span: Span, (object, reach): (Term, Reach), head: Option<Head>, field: Astr, part: Term) {
    match head {
        Some(Head::Ref(mutability, _)) => {
            let least = past(cx, object, reach, mutability);
            record_reach(cx, part, Some(least));
            let target = cx.fresh();
            let referent = cx.referent(object);
            cx.eq(target, referent, span);
            record_reach(cx, target, Some(least));
            let reached = field_of(cx, target, field, span);
            cx.eq(part, reached, span);
            return;
        }
        Some(Head::Base(_)) | None => {}
    }
    record_reach(cx, part, path_of(cx, object, reach));
    let shape = cx.held(TyKind::Object {
        fields: Box::new([(field, part)]),
        label: Label::Here,
    });
    cx.join(object, shape, reach, span);
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
    cx.enter(index.id(), at, usize_ty, Reach::Named, index.span());
    let element = element(cx, callee_site, container, demand.mutability(), span);
    read_for(cx, element, demand, span)
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
/// `as_slice(_mut)` (operators §3). The element is reached as the
/// container was, `reach`, or through the reference.
pub(super) fn fire_index(
    cx: &mut Ctxt<'_>,
    callee_site: AstId,
    span: Span,
    (container, of, reach): (Closed, Term, Reach),
    mutability: Mutability,
    element: Term,
    body: &EffectTerm<InferPhase>,
) {
    let least = match container.head {
        Head::Ref(mutability, _) => Some(past(cx, of, reach, mutability)),
        Head::Base(_) => path_of(cx, of, reach),
    };
    record_reach(cx, element, least);
    let sequence = match container.head {
        Head::Base(Base::Array) => Some(container.term),
        Head::Ref(_, Base::Array | Base::Slice) => Some(cx.referent(container.term)),
        Head::Base(_) | Head::Ref(..) => None,
    };
    if let Some(sequence) = sequence {
        let part = super::decision::element_of(cx, sequence);
        cx.eq(element, part, span);
        return;
    }
    let slice = cx.node(TyKind::Slice(element));
    let lent_slice = cx.held(TyKind::Ref(mutability, slice));
    let lent = cx.lend(container.term, mutability, span);
    let callee = Callee::Core(CoreSig::AsSlice(mutability));
    let reached = reach_in(cx, body, callee_site, callee, vec![(callee_site, lent)], span);
    cx.eq(reached, lent_slice, span);
}

/// A lend fired, as `lent` says. A reference borrowed is reborrowed: a
/// `&mut` of a `&` is refused, and a `&` of a `&mut` is a `&` of its target;
/// any other value is lent as it is. A reference bound or read past
/// references is reborrowed at the least on its path, never refused
/// (ownership §5); any other value bound is lent, and read is itself. One
/// whose use decides it at a stall (`value` `None`) is no reference.
pub(super) fn fire_lend(cx: &mut Ctxt<'_>, span: Span, value: Option<Closed>, lent: Lent, result: Term, target: Term) {
    if lent == Lent::Borrow(Mutability::Mut) && cx.out.through.get(&target) == Some(&Mutability::Shared) {
        refusal::push(cx, RefusalKind::LendMutThroughShared, span);
    }
    let held = match value.map(|closed| closed.head) {
        Some(Head::Ref(held, _)) => Some(held),
        Some(Head::Base(_)) | None => None,
    };
    let taken = match (lent, held) {
        (Lent::Borrow(mutability), Some(held)) => {
            if !held.reaches(mutability) {
                let wanted = cx.fresh();
                let wanted = cx.node(TyKind::Ref(mutability, wanted));
                cx.eq(target, wanted, span);
            }
            reborrow(cx, target, held, mutability)
        }
        (Lent::Borrow(mutability) | Lent::Bind(mutability), None) => cx.node(TyKind::Ref(mutability, target)),
        (Lent::Bind(least), Some(held)) => reborrow(cx, target, held, held.least(least)),
        (Lent::Read, Some(held)) => match cx.out.through.get(&target).copied() {
            Some(least) => reborrow(cx, target, held, held.least(least)),
            None => target,
        },
        (Lent::Read, None) => target,
    };
    cx.eq(result, taken, span);
}

/// `reference`, a reference of `held`, taken at `mutability`: itself where
/// the two are one, and a reference of `mutability` to its target where
/// they are not.
fn reborrow(cx: &mut Ctxt<'_>, reference: Term, held: Mutability, mutability: Mutability) -> Term {
    match held == mutability {
        true => reference,
        false => {
            let inner = cx.referent(reference);
            cx.node(TyKind::Ref(mutability, inner))
        }
    }
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
/// a `&mut` (operators §6); anything else is refused. A deref of `!` is
/// `never::settle`'s.
pub(super) fn fire_deref(cx: &mut Ctxt<'_>, span: Span, (reference, of): (Closed, Term), target: Term, store: bool) {
    match reference.head {
        Head::Ref(Mutability::Shared, _) if store => refusal::push(cx, RefusalKind::StoreThroughShared, span),
        Head::Ref(mutability, _) => {
            let referent = cx.referent(reference.term);
            cx.eq(target, referent, span);
            let least = past(cx, of, Reach::Named, mutability);
            record_reach(cx, target, Some(least));
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
        cx.enter(value.id(), stored, slot, Reach::Named, value.span());
    };
    match generalized {
        Some(binder) => super::generalize::walk(cx, binder, store),
        None => store(cx),
    }
}

/// The term a store into `place` enters, `None` where the place is
/// refused; how each step is reached is recorded as it fires. A context at
/// its root is touched by `access`: written where it is the whole place,
/// and read and written where a part of it is stored into (context §3).
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
    cx.enter(value.id(), stored, slot, Reach::Through, value.span());
}
