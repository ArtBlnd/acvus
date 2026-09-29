//! `recv.f(args)`: a call reaching by the same rule, the receiver first
//! (types §6). `.unwrap()` and `.clone()` are core's lang items
//! (operators §2); `.clone()` is `clone.rs`'s.
//!
//! The receiver is adjusted as Rust's method probe adjusts it (D85): once
//! its head is known, it is taken as written, then as `&`, then as `&mut`
//! where it is a writable place, and the first that some candidate's
//! receiver parameter matches is taken. The reach then goes on over every
//! argument with the adjusted receiver. A `&mut` receiver lends its place
//! mutably: a captured name is captured `&mut`, a context is read and
//! written, and each index on the way reaches `as_slice_mut`, as Rust's
//! `Index` becomes `IndexMut`. A receiver taken as written is a whole use.

use acvus_ast::{AstId, Expr, RefKind, Slot, Span};
use acvus_utils::{Astr, QualifiedRef};

use super::call::reach_in;
use super::ctxt::Ctxt;
use super::decision::{self, Callee, CoreSig, Decision, Offer, Offers, Reading};
use super::expr::{check_expr, check_receiver};
use super::output::{Adjustment, CaptureOf, CaptureUse, WholeUse};
use super::place::Demand;
use super::reach::{self, Candidate, NoReach};
use super::refusal::{self, RefusalKind};
use crate::decl::{Base, Head};
use crate::infer::{Closed, Term};
use crate::ty::{Contexts, Effect, EffectTerm, Infer as InferPhase, Mutability, Reissue, SplitCost};

/// The reach is recorded at `callee_site`, where the callee is written, and
/// the receiver's adjustment there too.
pub(super) fn check_method_call<S: Slot>(
    cx: &mut Ctxt<'_>,
    _site: AstId,
    callee_site: AstId,
    receiver: &Expr<S>,
    name: Astr,
    args: &[Expr<S>],
    span: Span,
) -> Term {
    let term = check_receiver(cx, receiver);
    let place = receiver_place(cx, receiver);
    let args: Vec<(AstId, Term)> = args.iter().map(|arg| (arg.id(), check_expr(cx, arg, Demand::Read))).collect();
    let core = cx.decls.core();
    let callee = if name == core.unwrap.name {
        Callee::Core(CoreSig::Unwrap)
    } else {
        Callee::Method(name)
    };
    let result = cx.fresh();
    let body = super::effect::body_effect(cx);
    let adjust = Adjust {
        callee_site,
        span,
        callee,
        receiver: term,
        receiver_site: receiver.id(),
        args,
        result,
        body,
        place,
    };
    cx.decisions.push(Decision::Adjust(Box::new(adjust)));
    result
}

/// A method call waiting for its receiver's head.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Adjust {
    callee_site: AstId,
    pub span: Span,
    callee: Callee,
    pub receiver: Term,
    receiver_site: AstId,
    /// The arguments after the receiver, each at its site.
    args: Vec<(AstId, Term)>,
    pub result: Term,
    /// The effect of the body the call is written in.
    body: EffectTerm<InferPhase>,
    place: Option<ReceiverPlace>,
}

impl Adjust {
    pub fn terms(&self) -> Vec<Term> {
        let mut terms = vec![self.receiver, self.result];
        terms.extend(self.args.iter().map(|(_, term)| *term));
        if let Some(place) = &self.place {
            terms.extend(place.indexes.iter().flat_map(|index| [index.container, index.element]));
        }
        terms
    }
}

/// A receiver that is a place: what it is rooted at, whether it may be
/// written (a binding, a context, or a field or index of one), and each
/// index on its way.
#[derive(Debug, Clone, PartialEq)]
struct ReceiverPlace {
    root: Root,
    writable: bool,
    indexes: Vec<IndexOnTheWay>,
}

#[derive(Debug, Clone, PartialEq)]
enum Root {
    /// A binding or an input: a capture of each lambda `crossed`, the ones
    /// written `move` listed apart.
    Name { name: Astr, of: CaptureOf, crossed: Box<[AstId]>, moving: Box<[AstId]>, term: Term },
    Context(QualifiedRef),
}

/// `container[i]` on the way to the receiver, reached at `callee_site`.
#[derive(Debug, Clone, PartialEq)]
struct IndexOnTheWay {
    callee_site: AstId,
    span: Span,
    container: Term,
    element: Term,
}

/// `receiver` as a place, where it is one.
fn receiver_place<S: Slot>(cx: &mut Ctxt<'_>, receiver: &Expr<S>) -> Option<ReceiverPlace> {
    match receiver {
        Expr::Paren { inner, .. } | Expr::FieldAccess { object: inner, .. } => receiver_place(cx, inner),
        Expr::Index { id, callee_id, object, span, .. } => {
            let mut place = receiver_place(cx, object)?;
            place.indexes.push(IndexOnTheWay {
                callee_site: *callee_id,
                span: *span,
                container: cx.out.types[&object.id()],
                element: cx.out.types[id],
            });
            Some(place)
        }
        Expr::Ident { name, ref_kind: RefKind::Value, .. } if name.namespace.is_none() => {
            let found = cx.scope.lookup(name.name)?;
            let root = name_root(cx, name.name, CaptureOf::Binding(found.binding.binder), found.crossed, found.binding.term);
            Some(ReceiverPlace { root, writable: true, indexes: Vec::new() })
        }
        Expr::Ident { id, name, ref_kind: RefKind::ExternParam, .. } => {
            let crossed = cx.scope.lambdas();
            let root = name_root(cx, name.name, CaptureOf::Input(name.name), crossed, cx.out.types[id]);
            Some(ReceiverPlace { root, writable: false, indexes: Vec::new() })
        }
        Expr::ContextRef { name, .. } => Some(ReceiverPlace { root: Root::Context(*name), writable: true, indexes: Vec::new() }),
        _ => None,
    }
}

fn name_root(cx: &Ctxt<'_>, name: Astr, of: CaptureOf, crossed: Vec<AstId>, term: Term) -> Root {
    let moving = crossed.iter().copied().filter(|lambda| cx.scope.captures_by_move(*lambda)).collect();
    Root::Name { name, of, crossed: crossed.into(), moving, term }
}

/// The adjustment fired on the receiver's head, then the call's reach. It
/// waits where a `&mut` receiver lends an index still undecided.
pub(super) fn fire_adjust(cx: &mut Ctxt<'_>, adjust: Adjust, receiver: Closed, reading: Reading) -> Result<(), Adjust> {
    let adjustment = adjustment(cx, &adjust, receiver);
    if adjustment == Adjustment::Ref(Mutability::Mut) {
        let indexes = adjust.place.iter().flat_map(|place| &place.indexes);
        let waiting = indexes.clone().any(|index| decision::index_pending(cx, index.callee_site))
            || indexes.clone().any(|index| decision::closed_for(cx, index.container, reading).is_none());
        if waiting {
            return Err(adjust);
        }
    }
    let Adjust { callee_site, span, callee, receiver: term, receiver_site, args, result, body, place } = adjust;
    cx.out.adjustments.insert(callee_site, adjustment);
    let lent = match adjustment {
        Adjustment::AsWritten => term,
        Adjustment::Ref(mutability) => cx.lend(term, mutability, span),
    };
    if let Some(place) = place {
        lends(cx, &place, adjustment, term, &body, reading, span);
    }
    let mut all = vec![(receiver_site, lent)];
    all.extend(args);
    let reached = reach_in(cx, &body, callee_site, callee, all, span);
    cx.eq(result, reached, span);
    Ok(())
}

/// The first of as written, `&` and `&mut` whose receiver some candidate's
/// first parameter matches, by the reach rule at that one position; as
/// written where none does, which the reach then refuses.
fn adjustment(cx: &Ctxt<'_>, adjust: &Adjust, receiver: Closed) -> Adjustment {
    let head = receiver.head;
    let Offers { concrete, generic } = decision::offers(cx, &adjust.callee);
    let arity = adjust.args.len() + 1;
    let candidates: Vec<Offer> = concrete
        .into_iter()
        .chain(generic)
        .filter(|offer| decision::params_of(cx.decls.types(), offer.ty).len() == arity)
        .collect();
    let mut rows: Vec<Vec<Option<Head>>> = candidates
        .iter()
        .map(|offer| vec![decision::param_head_at(cx.decls.types(), decision::params_of(cx.decls.types(), offer.ty), &[0])])
        .collect();
    if let Some(lifted) = super::structure::lifted_receiver(cx, &adjust.callee, arity, receiver) {
        rows.push(vec![Some(lifted)]);
    }
    let rows: Vec<Candidate<'_>> = rows.iter().map(|row| Candidate { params: row }).collect();
    let matches = |head: Head| match reach::reach(&rows, &[decision::chain_up(cx, head)]) {
        Ok(_) | Err(NoReach::Several(_)) => true,
        Err(NoReach::None) => false,
    };
    let writable = adjust.place.as_ref().is_some_and(|place| place.writable);
    let tried = [
        Some((Adjustment::AsWritten, head)),
        Some((Adjustment::Ref(Mutability::Shared), lent_head(head, Mutability::Shared))),
        match writable {
            true => mut_lent_head(head).map(|lent| (Adjustment::Ref(Mutability::Mut), lent)),
            false => None,
        },
    ];
    tried
        .into_iter()
        .flatten()
        .find(|(_, head)| matches(*head))
        .map_or(Adjustment::AsWritten, |(adjustment, _)| adjustment)
}

/// The head of `&` of a value of head `head`: a reference is reborrowed,
/// since a reference's target is never one (types §4).
fn lent_head(head: Head, mutability: Mutability) -> Head {
    match head {
        Head::Ref(_, base) | Head::Base(base) => Head::Ref(mutability, base),
    }
}

/// The head of `&mut` of a value of head `head`; none of a `&`, which is
/// never lent mutably.
fn mut_lent_head(head: Head) -> Option<Head> {
    match head {
        Head::Ref(Mutability::Shared, _) => None,
        head => Some(lent_head(head, Mutability::Mut)),
    }
}

/// What the adjustment makes of the receiver's place: as written, a whole
/// use; lent `&mut`, a capture by `&mut`, a context read and written, and
/// every index on the way through `as_slice_mut`.
fn lends(
    cx: &mut Ctxt<'_>,
    place: &ReceiverPlace,
    adjustment: Adjustment,
    term: Term,
    body: &EffectTerm<InferPhase>,
    reading: Reading,
    span: Span,
) {
    match adjustment {
        Adjustment::AsWritten => {
            if let Root::Name { moving, .. } = &place.root
                && !moving.is_empty()
            {
                cx.out.whole_uses.push(WholeUse { lambdas: moving.clone(), term });
            }
        }
        Adjustment::Ref(Mutability::Shared) => {}
        Adjustment::Ref(Mutability::Mut) => {
            match &place.root {
                Root::Name { name, of, crossed, term, .. } => {
                    if !crossed.is_empty() {
                        let used = CaptureUse { name: *name, of: *of, crossed: crossed.clone(), mutability: Mutability::Mut, term: *term };
                        cx.out.capture_uses.push(used);
                    }
                }
                Root::Context(context) => {
                    let touched = Contexts::from([*context]);
                    let effect = Effect::new(Reissue::Opaque, false, touched.clone(), touched, SplitCost::Unavailable);
                    super::effect::flows_into(cx, &EffectTerm::Known(effect), body, span);
                }
            }
            for index in &place.indexes {
                let container = decision::closed_for(cx, index.container, reading)
                    .expect("a `&mut` receiver's adjustment waits for every index container on its way");
                match container.head {
                    Head::Ref(Mutability::Shared, Base::Array | Base::Slice) => {
                        refusal::push(cx, RefusalKind::StoreThroughShared, index.span);
                    }
                    _ => super::place::fire_index(cx, index.callee_site, index.span, container, Mutability::Mut, index.element, body),
                }
            }
        }
    }
}
