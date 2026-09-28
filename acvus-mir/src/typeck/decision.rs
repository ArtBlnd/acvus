//! Decisions (types §6): what waits for a closed type. One set, one
//! fixpoint (S2, S3).
//!
//! `solve` fires every decision whose terms are closed, each once, until
//! none fires. At a stall, in this order and resuming after the first that
//! changes anything: a value with no head yet takes the closed slot it
//! enters, since that use decides it (types §5, D27); every integer literal
//! no use decided takes its default width; every type variable no pending
//! decision reads is `!`; then every type variable at all is `!`; then a
//! decision closed only up to a cycle through decisions fires on what the
//! sources outside the cycle fixed (typeck-structure section 4). What is
//! still open after that is refused, naming the construct left open.
//!
//! Cost: a round is O(pending²); each firing removes one decision and
//! pushes at most the chosen declaration's arity, requirements and bounded
//! variables, finite since requirements end (extern-fn §6); the stall steps
//! bind at least one variable each, so they run at most once per variable
//! (compilation).

use acvus_ast::{AstId, Span};
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashSet;

use super::ctxt::Ctxt;
use super::output::{AsCast, Chosen, Conversion, Impl, Iteration, TryExit};
use super::reach::{self, Candidate, NoReach};
use super::refusal::{self, RefusalKind};
use crate::decl::{Base, Head, ImplSig, Scheme, TyVarBound};
use crate::infer::{Bound, Closed, Entered, Instantiated, OpenSources, RequiredAt, Term};
use crate::ty::{
    CastTy, EffectTerm, Infer as InferPhase, IntTy, Mutability, ObjectTy, Poly, TyId, TyKind, TyVarId, TypeArg, Types,
};

/// A lang item a form calls (operators §2, §3, §7). `&&`, `||`,
/// `!`, `^`, `&`, `|`, `<<` and `>>` are the language's own operations, not
/// calls (operators §2): `op` types them without a reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum CoreSig {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Eq,
    Cmp,
    AsSlice(Mutability),
    AsStr,
    Display,
    Clone,
    Unwrap,
}

/// What a reach decides among.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Callee {
    /// Every declaration a bare name may reach.
    Name(Astr),
    /// A qualified name: one declaration.
    Declaration(QualifiedRef),
    /// A method: the declarations of that name, the receiver first.
    Method(Astr),
    Core(CoreSig),
    /// An impl a reached declaration requires.
    Required(RequiredAt),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Decision {
    /// Which declaration a call reaches, and at which impl (types §6).
    /// `effect` is the call's effect as its body already joined it
    /// (effects §1): the reached declaration's effect joins into it. A
    /// required impl is no call and has none.
    Reach {
        site: AstId,
        span: Span,
        callee: Callee,
        args: Box<[Term]>,
        result: Term,
        effect: Option<EffectTerm<InferPhase>>,
    },
    /// What a `for` iterates by (control-flow); by reference, its
    /// container's `as_slice(_mut)` is reached at `callee_site`.
    Iterate { site: AstId, callee_site: AstId, span: Span, head: Term, item: Term },
    /// What a `?` unwraps (control-flow).
    Try { site: AstId, span: Span, operand: Term, value: Term, exit: Term },
    /// A value entering a slot (types §3, §7).
    Enter { site: AstId, span: Span, value: Term, slot: Term },
    /// `as` from its source's type (operators §5).
    Cast { site: AstId, span: Span, source: Term, to: CastTy },
    /// A bounded variable some use must answer (types §5), refused where
    /// none does.
    Settle { span: Span, term: Term },
    /// A value that outlives the run, whose type has no positions
    /// (ownership §5), read once no variable is left in it.
    HoldsNoLoans { span: Span, term: Term },
}

impl Decision {
    /// The terms this decision may still bind or enter into.
    fn outputs(&self) -> impl Iterator<Item = Term> {
        let outputs: Vec<Term> = match self {
            Decision::Reach { result, .. } => vec![*result],
            Decision::Iterate { item, .. } => vec![*item],
            Decision::Try { value, exit, .. } => vec![*value, *exit],
            Decision::Enter { slot, .. } => vec![*slot],
            Decision::Cast { .. } | Decision::Settle { .. } | Decision::HoldsNoLoans { .. } => Vec::new(),
        };
        outputs.into_iter()
    }

    /// Every term this decision reads or writes.
    fn terms(&self) -> Vec<Term> {
        match self {
            Decision::Reach { args, result, .. } => args.iter().copied().chain([*result]).collect(),
            Decision::Iterate { head, item, .. } => vec![*head, *item],
            Decision::Try { operand, value, exit, .. } => vec![*operand, *value, *exit],
            Decision::Enter { value, slot, .. } => vec![*value, *slot],
            Decision::Cast { source, .. } => vec![*source],
            Decision::Settle { term, .. } | Decision::HoldsNoLoans { term, .. } => vec![*term],
        }
    }
}

/// The pending decisions, and what they may still change.
#[derive(Default)]
pub(super) struct Decisions {
    pending: Vec<Decision>,
}

impl Decisions {
    pub fn push(&mut self, decision: Decision) {
        self.pending.push(decision);
    }
}

impl OpenSources for Decisions {
    /// Whether a pending decision outside the cycles through `term` may
    /// still enter into it or bind it (types §6). A decision being fired is
    /// taken out of the pending set first, so its own outputs never keep
    /// its inputs open.
    ///
    /// NOTE: this compares terms as written, since a decision holds no
    /// view of B's bindings. A term linked to a pending slot through
    /// another variable is not seen as that slot.
    fn may_change(&self, term: Term) -> bool {
        self.pending.iter().any(|decision| decision.outputs().any(|output| output == term))
    }
}

/// No pending decision changes anything: what a decision in a cycle
/// through decisions reads, once the sources outside the cycle are fixed.
struct NothingOpen;

impl OpenSources for NothingOpen {
    fn may_change(&self, _term: Term) -> bool {
        false
    }
}

/// Whether a firing waits for a term or goes ahead on what is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reading {
    Closed,
    CycleBroken,
}

/// The fixpoint of the module doc.
pub(super) fn solve(cx: &mut Ctxt<'_>) {
    loop {
        if fire_round(cx, Reading::Closed)
            || slots_decide(cx)
            || default_literals(cx)
            || default_unconstrained(cx, true)
            || default_unconstrained(cx, false)
            || fire_round(cx, Reading::CycleBroken)
        {
            continue;
        }
        break;
    }
    for decision in std::mem::take(&mut cx.decisions.pending) {
        refuse_left_open(cx, decision);
    }
}

/// `decision`, an `Enter`, fired now where its value is closed at the top
/// of its cast tree, which no later decision can change (types §6);
/// pending otherwise.
pub(super) fn enter(cx: &mut Ctxt<'_>, decision: Decision) {
    let Decision::Enter { value, .. } = decision else {
        panic!("decision::enter: only an Enter is entered");
    };
    let top = cx
        .infer
        .closed(cx.decls.casts(), &NothingOpen, value)
        .is_some_and(|closed| cx.decls.casts().step_up(closed.head).is_none());
    match top {
        true => fire(cx, decision, Reading::Closed).expect("a value closed at the top of its tree fires"),
        false => cx.decisions.push(decision),
    }
}

/// Fires every decision whose terms `reading` admits; whether any fired.
fn fire_round(cx: &mut Ctxt<'_>, reading: Reading) -> bool {
    let mut fired = false;
    let mut at = 0;
    while at < cx.decisions.pending.len() {
        let decision = cx.decisions.pending.remove(at);
        match fire(cx, decision, reading) {
            Ok(()) => fired = true,
            Err(waiting) => {
                cx.decisions.pending.insert(at, waiting);
                at += 1;
            }
        }
    }
    fired
}

fn closed_for(cx: &Ctxt<'_>, term: Term, reading: Reading) -> Option<Closed> {
    match reading {
        Reading::Closed => cx.closed(term),
        Reading::CycleBroken => cx.infer.closed(cx.decls.casts(), &NothingOpen, term),
    }
}

/// Fires `decision` if its inputs are closed; hands it back otherwise.
fn fire(cx: &mut Ctxt<'_>, decision: Decision, reading: Reading) -> Result<(), Decision> {
    match decision {
        Decision::Enter { site, span, value, slot } => {
            let Some(closed) = closed_for(cx, value, reading) else {
                return Err(Decision::Enter { site, span, value, slot });
            };
            fire_enter(cx, site, span, closed, slot);
            Ok(())
        }
        Decision::Iterate {
            site,
            callee_site,
            span,
            head,
            item,
        } => {
            let Some(closed) = closed_for(cx, head, reading) else {
                return Err(Decision::Iterate {
                    site,
                    callee_site,
                    span,
                    head,
                    item,
                });
            };
            fire_iterate(cx, site, callee_site, span, closed, item);
            Ok(())
        }
        Decision::Try {
            site,
            span,
            operand,
            value,
            exit,
        } => {
            let Some(closed) = closed_for(cx, operand, reading) else {
                return Err(Decision::Try {
                    site,
                    span,
                    operand,
                    value,
                    exit,
                });
            };
            fire_try(cx, site, span, closed, value, exit);
            Ok(())
        }
        Decision::Cast { site, span, source, to } => {
            let Some(closed) = closed_for(cx, source, reading) else {
                return Err(Decision::Cast { site, span, source, to });
            };
            fire_cast(cx, site, span, closed, to);
            Ok(())
        }
        Decision::Settle { span, term } => match cx.infer.types().kind(cx.infer.shallow(term)) {
            TyKind::Var(_) => Err(Decision::Settle { span, term }),
            _ => Ok(()),
        },
        Decision::HoldsNoLoans { span, term } => {
            if !unbound_vars(cx, [term]).is_empty() {
                return Err(Decision::HoldsNoLoans { span, term });
            }
            if super::context::holds_positions(cx, term) {
                refusal::push(cx, RefusalKind::HoldsLoans, span);
            }
            Ok(())
        }
        Decision::Reach {
            site,
            span,
            callee,
            args,
            result,
            effect,
        } => {
            let decision = Reach {
                site,
                span,
                callee,
                args,
                result,
                effect,
            };
            fire_reach(cx, decision, reading).map_err(Reach::into_decision)
        }
    }
}

fn fire_enter(cx: &mut Ctxt<'_>, site: AstId, span: Span, value: Closed, slot: Term) {
    let from = value.head;
    match cx.infer.enter(cx.decls, value, slot) {
        Ok(Entered::Folded) => note_view(cx, site, from, slot),
        Ok(Entered::Converted(conversion)) => {
            let via = Chosen {
                declaration: conversion.rule().via,
                r#impl: None,
                required: Box::new([]),
            };
            cx.out.conversions.insert(site, Conversion { via, to: slot });
        }
        Err(mismatch) => refusal::mismatch(cx, mismatch, span),
    }
}

/// A value that climbed the cast tree into its slot is coerced there by
/// the step it climbed first (operators §5, types §3).
///
/// NOTE: a climb of several steps records only its first; `Conversion`
/// names one function. Lower will need the whole path.
fn note_view(cx: &mut Ctxt<'_>, site: AstId, from: Head, slot: Term) {
    let Some(to) = cx.infer.closed(cx.decls.casts(), &NothingOpen, slot).map(|closed| closed.head) else {
        return;
    };
    if from == to || from == Head::Base(Base::Never) {
        return;
    }
    let Some(step) = cx.decls.casts().step_up(from) else {
        return;
    };
    let via = Chosen {
        declaration: step.rule().via,
        r#impl: None,
        required: Box::new([]),
    };
    cx.out.conversions.insert(site, Conversion { via, to: slot });
}

/// `for` by the kind its head's type gives (control-flow): a reference to
/// an array or a slice by reference, the language's own, since no
/// declaration carries an array (extern-fn §1); a reference to any other
/// container by reference, through its `as_slice(_mut)` (operators §3); an
/// array by value.
fn fire_iterate(cx: &mut Ctxt<'_>, site: AstId, callee_site: AstId, span: Span, head: Closed, item: Term) {
    let iteration = match head.head {
        Head::Base(Base::Array) => {
            let element = element_of(cx, head.term);
            cx.eq(item, element, span);
            Iteration::ByValue
        }
        Head::Base(Base::Never) => {
            let never = cx.never();
            cx.eq(item, never, span);
            Iteration::ByValue
        }
        Head::Ref(mutability, Base::Array | Base::Slice) => {
            let element = element_of(cx, cx.referent(head.term));
            let lent = cx.lend(element, mutability, span);
            cx.eq(item, lent, span);
            Iteration::ByRef(mutability)
        }
        Head::Ref(mutability, _) => {
            let element = cx.fresh();
            let slice = cx.node(TyKind::Slice(element));
            let lent_slice = cx.held(TyKind::Ref(mutability, TypeArg::Uniform(slice)));
            reach_core(cx, callee_site, span, CoreSig::AsSlice(mutability), vec![head.term], lent_slice);
            let lent = cx.lend(element, mutability, span);
            cx.eq(item, lent, span);
            Iteration::ByRef(mutability)
        }
        Head::Base(_) => {
            refusal::push(cx, RefusalKind::NotIterable, span);
            return;
        }
    };
    cx.out.iteration.insert(site, iteration);
}

/// The element of a closed array or slice node.
///
/// # Panics
/// `sequence` is neither: callers match its head first.
fn element_of(cx: &Ctxt<'_>, sequence: Term) -> Term {
    match cx.infer.types().kind(cx.infer.shallow(sequence)) {
        TyKind::Array(element, _) | TyKind::Slice(element) => *element,
        other => panic!("element_of: a term closed as an array or a slice is one, not {other:?}"),
    }
}

/// A lang item's reach at `site`, pushed from a firing: the call's effect
/// joins the run's, since a decision fires after every body is walked.
///
/// NOTE: a reach pushed from a firing inside a lambda's body (a `for` by
/// reference there) joins the run's effect rather than the lambda's; the
/// decision does not know which body pushed it.
fn reach_core(cx: &mut Ctxt<'_>, site: AstId, span: Span, sig: CoreSig, args: Vec<Term>, result: Term) {
    let effect = EffectTerm::Var(cx.infer.fresh_effect());
    super::effect::note_call_effect(cx, &effect);
    cx.decisions.push(Decision::Reach {
        site,
        span,
        callee: Callee::Core(sig),
        args: args.into(),
        result,
        effect: Some(effect),
    });
}

/// `?` on an `Option` or a `Result` (control-flow): its value is the
/// payload, and it leaves through the same primitive at the body's type.
fn fire_try(cx: &mut Ctxt<'_>, site: AstId, span: Span, operand: Closed, value: Term, exit: Term) {
    let kind = cx.infer.types().kind(operand.term).clone();
    let tried = match kind {
        TyKind::Option(payload) => {
            cx.eq(value, payload, span);
            let any = cx.fresh();
            let leaves = cx.node(TyKind::Option(any));
            cx.eq(exit, leaves, span);
            TryExit::Option(exit)
        }
        TyKind::Result(payload, error) => {
            cx.eq(value, payload, span);
            let any = cx.fresh();
            let leaves = cx.node(TyKind::Result(any, error));
            cx.eq(exit, leaves, span);
            TryExit::Result(exit)
        }
        TyKind::Never => {
            let never = cx.never();
            cx.eq(value, never, span);
            return;
        }
        _ => {
            refusal::push(cx, RefusalKind::NotTryable, span);
            return;
        }
    };
    cx.out.tries.insert(site, tried);
}

/// `as` between the numbers, `bool` and `char`, as Rust casts them
/// (operators §5).
fn fire_cast(cx: &mut Ctxt<'_>, site: AstId, span: Span, source: Closed, to: CastTy) {
    let kind = cx.infer.types().kind(source.term);
    if matches!(kind, TyKind::Never) {
        return;
    }
    match CastTy::of_kind(kind) {
        Some(from) if from.admits(to) => {
            cx.out.casts.insert(site, AsCast { from, to });
        }
        _ => refusal::push(cx, RefusalKind::CastNotAdmitted { to }, span),
    }
}

/// A reach decision's parts, moved out of the `Decision` while it fires.
struct Reach {
    site: AstId,
    span: Span,
    callee: Callee,
    args: Box<[Term]>,
    result: Term,
    effect: Option<EffectTerm<InferPhase>>,
}

impl Reach {
    fn into_decision(self) -> Decision {
        Decision::Reach {
            site: self.site,
            span: self.span,
            callee: self.callee,
            args: self.args,
            result: self.result,
            effect: self.effect,
        }
    }
}

/// One candidate of a reach: a declaration without impls, or one impl of a
/// generic function.
#[derive(Debug, Clone, Copy)]
struct Offer {
    declaration: QualifiedRef,
    r#impl: Option<Impl>,
    ty: TyId<Poly>,
}

/// One position of a call's arguments read front to back: an argument, or
/// a part of a tuple one holds (types §6), with the heads up the tree from
/// its exact type. `None` where the position is not closed yet.
struct Position {
    path: Vec<usize>,
    chain: Option<Vec<Head>>,
}

fn fire_reach(cx: &mut Ctxt<'_>, decision: Reach, reading: Reading) -> Result<(), Reach> {
    let Offers { concrete, generic } = offers(cx, &decision.callee);
    let arity = decision.args.len();
    let fits_arity = |offer: &&Offer| params_of(cx.decls.types(), offer.ty).len() == arity;
    let offers: Vec<Offer> = concrete.iter().filter(fits_arity).copied().collect();
    let generic: Vec<Offer> = generic.iter().filter(fits_arity).copied().collect();
    let positions = positions(cx, &decision.args, reading);
    let complete = positions.iter().all(|position| position.chain.is_some());
    let chains: Vec<Vec<Head>> = positions.iter().map_while(|position| position.chain.clone()).collect();

    let chosen = match choose(cx, &offers, &positions, &chains) {
        Ok(chosen) => chosen,
        Err(NoReach::None) => match choose(cx, &generic, &positions, &chains) {
            Ok(chosen) => chosen,
            Err(NoReach::None) => {
                let name = callee_name(cx, &decision.callee);
                refusal::push(cx, RefusalKind::NoDeclaration { name }, decision.span);
                return Ok(());
            }
            Err(NoReach::Several(left)) => {
                return several(cx, decision, &generic, &left, complete, reading);
            }
        },
        Err(NoReach::Several(left)) => return several(cx, decision, &offers, &left, complete, reading),
    };
    reached(cx, decision, chosen);
    Ok(())
}

/// Several offers left: a call whose arguments are all read is refused;
/// one with a position still open waits for it.
fn several(
    cx: &mut Ctxt<'_>,
    decision: Reach,
    offers: &[Offer],
    left: &[usize],
    complete: bool,
    reading: Reading,
) -> Result<(), Reach> {
    if !complete && reading == Reading::Closed {
        return Err(decision);
    }
    let name = callee_name(cx, &decision.callee);
    let left = left.iter().map(|at| offers[*at].declaration).collect();
    refusal::push(cx, RefusalKind::SeveralDeclarations { name, left }, decision.span);
    Ok(())
}

/// The one offer the closed prefix of `positions` reaches.
fn choose(cx: &Ctxt<'_>, offers: &[Offer], positions: &[Position], chains: &[Vec<Head>]) -> Result<Offer, NoReach> {
    let rows: Vec<Vec<Option<Head>>> = offers
        .iter()
        .map(|offer| {
            let params = params_of(cx.decls.types(), offer.ty);
            positions[..chains.len()]
                .iter()
                .map(|position| param_head_at(cx.decls.types(), params, &position.path))
                .collect()
        })
        .collect();
    let candidates: Vec<Candidate<'_>> = rows.iter().map(|row| Candidate { params: row }).collect();
    reach::reach(&candidates, chains).map(|at| offers[at])
}

/// Every offer a callee may reach: the concrete impls and the declarations
/// without impls, and apart, the generic impls, which a call reaches only
/// where no concrete one is left (`Impls`: the concrete ones in order, then
/// the generic one).
///
/// NOTE: the generic impl as the fallback is read from `Impls`' order, not
/// from the ledger (question Q8 to the coordinator).
struct Offers {
    concrete: Vec<Offer>,
    generic: Vec<Offer>,
}

fn offers(cx: &Ctxt<'_>, callee: &Callee) -> Offers {
    let mut concrete = Vec::new();
    let mut generic = Vec::new();
    let mut offer_of = |declaration: QualifiedRef, keep: &dyn Fn(&crate::decl::DeclaredFacts) -> bool| {
        let scheme = cx
            .decls
            .scheme(declaration)
            .expect("a callee names a declared function: names come from the declarations");
        match &scheme.impls {
            None => concrete.push(Offer {
                declaration,
                r#impl: None,
                ty: scheme.ty,
            }),
            Some(impls) => {
                for (at, sig) in impls.concrete.iter().enumerate() {
                    if keep(&sig.facts) {
                        concrete.push(Offer {
                            declaration,
                            r#impl: Some(Impl::Concrete(at)),
                            ty: sig.ty,
                        });
                    }
                }
                if let Some(sig) = &impls.generic
                    && keep(&sig.facts)
                {
                    generic.push(Offer {
                        declaration,
                        r#impl: Some(Impl::Generic),
                        ty: scheme.ty,
                    });
                }
            }
        }
    };
    let any = |_: &crate::decl::DeclaredFacts| true;
    match callee {
        Callee::Name(name) | Callee::Method(name) => {
            for declaration in cx.decls.overloads(*name) {
                offer_of(*declaration, &any);
            }
        }
        Callee::Declaration(declaration) => offer_of(*declaration, &any),
        Callee::Core(sig) => {
            let declaration = core_name(cx, *sig);
            if cx.decls.scheme(declaration).is_some() {
                offer_of(declaration, &any);
            }
        }
        Callee::Required(required) => {
            let law = required.law;
            let states = move |facts: &crate::decl::DeclaredFacts| law.is_none_or(|law| law.stated_by(&facts.laws));
            if cx.decls.scheme(required.generic_function).is_some() {
                offer_of(required.generic_function, &states);
            }
        }
    }
    Offers { concrete, generic }
}

fn core_name(cx: &Ctxt<'_>, sig: CoreSig) -> QualifiedRef {
    let core = cx.decls.core();
    match sig {
        CoreSig::Add => core.add,
        CoreSig::Sub => core.sub,
        CoreSig::Mul => core.mul,
        CoreSig::Div => core.div,
        CoreSig::Rem => core.rem,
        CoreSig::Neg => core.neg,
        CoreSig::Eq => core.eq,
        CoreSig::Cmp => core.cmp,
        CoreSig::AsSlice(Mutability::Shared) => core.as_slice,
        CoreSig::AsSlice(Mutability::Mut) => core.as_slice_mut,
        CoreSig::AsStr => core.as_str,
        CoreSig::Display => core.display,
        CoreSig::Clone => core.clone,
        CoreSig::Unwrap => core.unwrap,
    }
}

/// The name a refusal of a reach names: what the program wrote.
fn callee_name(cx: &Ctxt<'_>, callee: &Callee) -> Astr {
    match callee {
        Callee::Name(name) | Callee::Method(name) => *name,
        Callee::Declaration(declaration) => declaration.name,
        Callee::Core(sig) => core_name(cx, *sig).name,
        Callee::Required(required) => required.generic_function.name,
    }
}

/// # Panics
/// `ty` is not a function type: verify refuses a signature or impl that is
/// not one (`NotAFunction`).
fn params_of(types: &Types<Poly>, ty: TyId<Poly>) -> &[TyId<Poly>] {
    match types.kind(ty) {
        TyKind::Fn { params, .. } => params,
        _ => panic!("a verified signature is a function type (NotAFunction)"),
    }
}

/// The arguments' positions front to back, stopping after the first that
/// is not closed: a later one is not read before it (types §6).
fn positions(cx: &Ctxt<'_>, args: &[Term], reading: Reading) -> Vec<Position> {
    let mut out = Vec::new();
    for (at, arg) in args.iter().enumerate() {
        if !position(cx, *arg, vec![at], reading, &mut out) {
            break;
        }
    }
    out
}

/// Pushes `term`'s position and, for a tuple, its parts'; whether every
/// one was closed.
fn position(cx: &Ctxt<'_>, term: Term, path: Vec<usize>, reading: Reading, out: &mut Vec<Position>) -> bool {
    let Some(closed) = closed_for(cx, term, reading) else {
        out.push(Position { path, chain: None });
        return false;
    };
    out.push(Position {
        path: path.clone(),
        chain: Some(chain_up(cx, closed.head)),
    });
    let TyKind::Tuple(items) = cx.infer.types().kind(closed.term) else {
        return true;
    };
    for (at, item) in items.clone().iter().enumerate() {
        let mut part = path.clone();
        part.push(at);
        if !position(cx, *item, part, reading, out) {
            return false;
        }
    }
    true
}

/// `head` and every head up the cast tree from it (types §3).
fn chain_up(cx: &Ctxt<'_>, head: Head) -> Vec<Head> {
    let mut chain = vec![head];
    let mut at = head;
    while let Some(step) = cx.decls.casts().step_up(at) {
        at = poly_head(cx.decls.types(), step.rule().to)
            .expect("a tree step joins two heads: verify refuses one without (CastWithoutHead)");
        chain.push(at);
    }
    chain
}

/// A declared parameter's head at `path`, `None` where any head matches:
/// a type variable, a reference to one, or a part below a head that is not
/// a tuple, which its own position already decided.
fn param_head_at(types: &Types<Poly>, params: &[TyId<Poly>], path: &[usize]) -> Option<Head> {
    let (first, parts) = path.split_first().expect("a position names its argument");
    let mut at = params[*first];
    for part in parts {
        match types.kind(at) {
            TyKind::Tuple(items) => at = items[*part],
            _ => return None,
        }
    }
    poly_head(types, at)
}

/// The head of a declared type, `None` for a type variable or a reference
/// to one, which matches any head.
///
/// # Panics
/// `ty` is the order token or a started call, or a reference to a
/// reference: verify refuses a declaration that writes one
/// (`InternalType`, `ReferenceToReference`).
fn poly_head(types: &Types<Poly>, ty: TyId<Poly>) -> Option<Head> {
    match types.kind(ty) {
        TyKind::Ref(mutability, target) => poly_base(types.kind(target.ty())).map(|base| Head::Ref(*mutability, base)),
        kind => poly_base(kind).map(Head::Base),
    }
}

fn poly_base(kind: &TyKind<Poly>) -> Option<Base> {
    Some(match kind {
        TyKind::Var(_) => return None,
        TyKind::Never => Base::Never,
        TyKind::Int(width) => Base::Int(*width),
        TyKind::Float => Base::Float,
        TyKind::Char => Base::Char,
        TyKind::String => Base::String,
        TyKind::Bool => Base::Bool,
        TyKind::Unit => Base::Unit,
        TyKind::Str => Base::Str,
        TyKind::Array(..) => Base::Array,
        TyKind::Slice(_) => Base::Slice,
        TyKind::Tuple(items) => Base::Tuple(items.len()),
        TyKind::Option(_) => Base::Option,
        TyKind::Result(..) => Base::Result,
        TyKind::Object(ObjectTy::Structural(..)) => Base::Object,
        TyKind::Object(ObjectTy::Declared(id, _)) | TyKind::UserDefined { id, .. } => Base::Named(*id),
        TyKind::Enum { name, .. } => Base::Enum(*name),
        TyKind::Fn { .. } => Base::Fn,
        TyKind::Ref(..) => panic!("a declared reference's target is never a reference (ReferenceToReference)"),
        TyKind::Order | TyKind::Handle(_) => panic!("no declaration writes the order token or a started call (InternalType)"),
    })
}

/// The chosen offer's type at this use, its arguments entering its
/// parameters (types §7), its result the call's, its effect the call's
/// (effects §1), and each of its requirements a reach of its own
/// (extern-fn §6).
fn reached(cx: &mut Ctxt<'_>, decision: Reach, offer: Offer) {
    let Reach {
        site,
        span,
        callee,
        args,
        result,
        effect,
    } = decision;
    let Instantiated { ty, requirements } = instantiate(cx, offer, span);
    let TyKind::Fn {
        params,
        ret,
        effect: declared_effect,
        ..
    } = cx.infer.types().kind(cx.infer.shallow(ty)).clone()
    else {
        panic!("an instantiated signature is a function type (NotAFunction)");
    };
    let chosen = Chosen {
        declaration: offer.declaration,
        r#impl: offer.r#impl,
        required: Box::new([]),
    };
    match &callee {
        Callee::Required(required) => {
            cx.join(required.pattern, ty, span);
            let parent = cx
                .out
                .reached
                .get_mut(&site)
                .expect("a requirement is reached after the call that states it");
            let mut all = parent.required.to_vec();
            all.push(chosen);
            parent.required = all.into();
        }
        _ => {
            for (arg, param) in args.iter().zip(params.iter()) {
                cx.enter(site, *arg, *param, span);
            }
            cx.eq(result, ret, span);
            if let Some(effect) = effect {
                cx.infer.join_effects(&effect, &declared_effect);
            }
            cx.out.reached.insert(site, chosen);
        }
    }
    settle_bounded(cx, ty, span);
    for required in requirements {
        push_required(cx, site, span, required);
    }
}

/// A requirement a reached declaration states, as a reach of its own over
/// the required generic function's impls, by the pattern's parameters
/// (extern-fn §6).
pub(super) fn push_required(cx: &mut Ctxt<'_>, site: AstId, span: Span, required: RequiredAt) {
    let TyKind::Fn { params: args, .. } = cx.infer.types().kind(cx.infer.shallow(required.pattern)).clone() else {
        // An impl is a function (extern-fn §1), so a pattern of another type
        // is answered by none.
        let name = required.generic_function.name;
        refusal::push(cx, RefusalKind::NoDeclaration { name }, span);
        return;
    };
    let result = cx.fresh();
    cx.decisions.push(Decision::Reach {
        site,
        span,
        callee: Callee::Required(required),
        args,
        result,
        effect: None,
    });
}

/// `offer`'s type at fresh variables with the generic function's bounds,
/// and every requirement the generic function and the impl state.
fn instantiate(cx: &mut Ctxt<'_>, offer: Offer, span: Span) -> Instantiated {
    let scheme = cx
        .decls
        .scheme(offer.declaration)
        .expect("an offer names a declared function");
    let generic = cx.infer.instantiate(cx.decls, scheme);
    let Some(Impl::Concrete(at)) = offer.r#impl else {
        return generic;
    };
    let sig = &scheme.impls.as_ref().expect("a concrete impl's scheme has impls").concrete[at];
    let own = cx.infer.instantiate(cx.decls, &impl_scheme(cx.decls.types(), sig));
    cx.eq(generic.ty, own.ty, span);
    let mut requirements = generic.requirements;
    requirements.extend(own.requirements);
    Instantiated { ty: own.ty, requirements }
}

/// A concrete impl as a scheme of its own, each of its type variables
/// unbounded: the generic function's bounds reach them through `eq` with
/// the generic function's instantiation.
///
/// NOTE: infer instantiates only a `Scheme`; an impl's own variables have
/// no declared bounds, so they are stated here as `Any`, one per index up
/// to the largest the impl writes.
fn impl_scheme(types: &Types<Poly>, sig: &ImplSig) -> Scheme {
    let mut largest: Option<u32> = None;
    let mut seen = FxHashSet::default();
    let mut stack = vec![sig.ty];
    stack.extend(sig.requires.iter().map(|required| required.pattern));
    while let Some(at) = stack.pop() {
        if !seen.insert(at) {
            continue;
        }
        let kind = types.kind(at);
        if let TyKind::Var(var) = kind {
            largest = Some(largest.map_or(*var, |known| known.max(*var)));
        }
        kind.for_each_child(|child| stack.push(child));
    }
    let count = largest.map_or(0, |largest| largest as usize + 1);
    Scheme {
        ty: sig.ty,
        param_names: sig.param_names.clone(),
        bounds: vec![TyVarBound::Any; count].into(),
        impls: None,
        requires: sig.requires.clone(),
    }
}

/// Every variable of `ty` a bound constrains waits for a use to answer it
/// (types §5).
pub(super) fn settle_bounded(cx: &mut Ctxt<'_>, ty: Term, span: Span) {
    let bounded: FxHashSet<TyVarId> = cx
        .infer
        .unbound()
        .filter(|(_, bound)| matches!(bound, Bound::Settled | Bound::Integer { literal: false, .. }))
        .map(|(var, _)| var)
        .collect();
    for var in unbound_vars(cx, [ty]) {
        if bounded.contains(&var) {
            let term = cx.node(TyKind::Var(var));
            cx.decisions.push(Decision::Settle { span, term });
        }
    }
}

/// The unbound variables `terms` hold, bound variables followed.
fn unbound_vars(cx: &Ctxt<'_>, terms: impl IntoIterator<Item = Term>) -> FxHashSet<TyVarId> {
    let mut vars = FxHashSet::default();
    let mut seen = FxHashSet::default();
    let mut stack: Vec<Term> = terms.into_iter().collect();
    while let Some(at) = stack.pop() {
        let at = cx.infer.shallow(at);
        if !seen.insert(at) {
            continue;
        }
        match cx.infer.types().kind(at) {
            TyKind::Var(var) => {
                vars.insert(*var);
            }
            kind => kind.for_each_child(|child| stack.push(child)),
        }
    }
    vars
}

/// At a stall, a value with no head yet entering a closed slot takes the
/// slot's type: the use decides it (types §5, D27). Whether any did.
///
/// NOTE: provisional (question Q7 to the coordinator). Without it an
/// integer literal passed to an `u8` parameter would default to `i64`
/// before its use is read.
fn slots_decide(cx: &mut Ctxt<'_>) -> bool {
    let mut decided = false;
    let mut at = 0;
    while at < cx.decisions.pending.len() {
        let Decision::Enter { value, slot, span, .. } = cx.decisions.pending[at] else {
            at += 1;
            continue;
        };
        let headless = cx.infer.closed(cx.decls.casts(), &NothingOpen, value).is_none();
        let slot_closed = cx.infer.closed(cx.decls.casts(), &NothingOpen, slot).is_some();
        if headless && slot_closed {
            cx.decisions.pending.remove(at);
            cx.eq(value, slot, span);
            decided = true;
        } else {
            at += 1;
        }
    }
    decided
}

/// types §5 and D16: an integer literal no use decided is `i64` where its
/// bound still admits it, its one width where one is left, and otherwise
/// waits for its `Settle` to refuse it. Whether any was bound.
fn default_literals(cx: &mut Ctxt<'_>) -> bool {
    let widths: Vec<(TyVarId, IntTy)> = cx
        .infer
        .unbound()
        .filter_map(|(var, bound)| match bound {
            Bound::Integer {
                signed,
                among,
                literal: true,
            } => {
                let admits = |width: &IntTy| among.contains(width) && (*signed || !width.signed());
                let admitted: Vec<IntTy> = among.iter().copied().filter(admits).collect();
                match admitted[..] {
                    _ if admitted.contains(&IntTy::I64) => Some((var, IntTy::I64)),
                    [one] => Some((var, one)),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    for (var, width) in &widths {
        let var = cx.node(TyKind::Var(*var));
        let int = cx.node(TyKind::Int(*width));
        cx.infer
            .eq(var, int)
            .expect("a literal's variable admits the width its own bound admits");
    }
    !widths.is_empty()
}

/// types §5: a type variable nothing constrains is `!`. With
/// `outside_pending`, only those no pending decision reads; without, every
/// one left, since a decision still waiting on one is not a use that will
/// answer it. Whether any was bound.
fn default_unconstrained(cx: &mut Ctxt<'_>, outside_pending: bool) -> bool {
    let read: FxHashSet<TyVarId> = match outside_pending {
        true => unbound_vars(cx, cx.decisions.pending.iter().flat_map(Decision::terms).collect::<Vec<_>>()),
        false => FxHashSet::default(),
    };
    let free: Vec<TyVarId> = cx
        .infer
        .unbound()
        .filter(|(var, bound)| matches!(bound, Bound::Any) && !read.contains(var))
        .map(|(var, _)| var)
        .collect();
    for var in &free {
        let var = cx.node(TyKind::Var(*var));
        let never = cx.never();
        cx.infer
            .eq(var, never)
            .expect("an unbound variable of no bound takes `!`: it is no reference, so no reference target refuses it");
    }
    !free.is_empty()
}

/// What is still open after every default is refused (types §5-6).
fn refuse_left_open(cx: &mut Ctxt<'_>, decision: Decision) {
    match decision {
        Decision::Settle { span, .. } => refusal::push(cx, RefusalKind::BoundUnanswered, span),
        // Its type holds a variable some other refusal left open.
        Decision::HoldsNoLoans { .. } => {}
        Decision::Reach { span, callee, .. } => {
            let open = callee_name(cx, &callee);
            refusal::push(cx, RefusalKind::NeverKnown { open }, span);
        }
        Decision::Enter { span, .. }
        | Decision::Iterate { span, .. }
        | Decision::Try { span, .. }
        | Decision::Cast { span, .. } => refusal::push(cx, RefusalKind::Undecided, span),
    }
}
