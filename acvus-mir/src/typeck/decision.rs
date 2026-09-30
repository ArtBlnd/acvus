//! Decisions (types §6): what waits for a closed type. One set, one
//! fixpoint (S2, S3).
//!
//! `solve` fires every decision whose terms are closed, each once, until
//! none fires. At a stall, in this order and resuming after the first that
//! changes anything: a form whose head is open is decided by its own use
//! where it can be (`uses_decide`); every integer literal
//! no use decided takes its default width; every type variable no pending
//! decision reads is `!`; then every type variable at all is `!`; then
//! every effect nothing constrains is opaque, every open `#` is not
//! specialized, and every flow set no lambda claims states every flow
//! (types §5); then a
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
use super::output::{AsCast, Chosen, Converted, Impl, Iteration, TryExit};
use super::reach::{self, Candidate, Match, NoReach};
use super::refusal::{self, RefusalKind};
use super::pat::Shape;
use crate::decl::{Base, Head, ImplSig, Scheme, Step as TreeStep};
use crate::infer::{Bound, Closed, Entered, Infer, Instantiated, OpenSources, RequiredAt, Term};
use crate::ty::structural::structure;
use crate::ty::{
    CastTy, EffectTerm, Infer as InferPhase, IntTy, Label, Mutability, ObjectTy, Poly, TyId, TyKind, TyVarId, Types,
};

/// A lang item a form calls (operators §2, §3, §7). `&&`, `||`,
/// `!`, `^`, `&`, `<<` and `>>` are the language's own operations, not
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
    Unwrap,
}

/// The types one of the language's own operators takes (operators §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Domain {
    /// `!`.
    Bool,
    /// `&`, `<<`, `>>`.
    Integer,
    /// `^`.
    BoolOrInteger,
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
    /// A part's impl a lifted impl answers by (types §1), reached at the
    /// part's own node: `generic_function` at the part's type, stating
    /// `law` where the lift combines.
    Part(RequiredAt),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Decision {
    /// Which declaration a call reaches, and at which impl (types §6).
    /// `effect` is the call's effect as its body already joined it
    /// (effects §1): the reached declaration's effect joins into it. For a
    /// required impl it is the effect of the call the requirement serves,
    /// which the reached declaration's own declared effect already covers,
    /// so nothing joins it.
    Reach {
        site: AstId,
        span: Span,
        callee: Callee,
        /// Each argument at the site it is written, where its conversion
        /// into its parameter is recorded.
        args: Box<[(AstId, Term)]>,
        result: Term,
        effect: EffectTerm<InferPhase>,
    },
    /// What a `for` iterates by (control-flow); by reference, its
    /// container's `as_slice(_mut)` is reached at `callee_site`.
    /// `body` is the effect of the body the `for` is in.
    Iterate {
        site: AstId,
        callee_site: AstId,
        span: Span,
        head: Term,
        item: Term,
        body: EffectTerm<InferPhase>,
    },
    /// What a `?` unwraps (control-flow).
    Try { site: AstId, span: Span, operand: Term, value: Term, exit: Term },
    /// A value entering a slot (types §3, §7), in the body whose effect is
    /// `body`: a coercion's function joins it (effects §1). `reach` is how
    /// the slot was reached: by name, or through a reference, which grows
    /// nothing (types §1).
    Enter {
        site: AstId,
        span: Span,
        value: Term,
        slot: Term,
        body: EffectTerm<InferPhase>,
        reach: crate::ty::Reach,
    },
    /// `as` from its source's type (operators §5).
    Cast { site: AstId, span: Span, source: Term, to: CastTy },
    /// A bounded variable some use must answer (types §5), refused where
    /// none does.
    Settle { span: Span, term: Term },
    /// A value that outlives the run, whose type has no positions
    /// (ownership §5), read once no variable is left in it.
    HoldsNoLoans { span: Span, term: Term },
    /// `&value` or `&mut value`: a reborrow where `value` is a reference,
    /// since a reference's target is never one (types §4).
    Lend { span: Span, value: Term, lent: Lent, result: Term },
    /// `object.field`, through a reference as Rust auto-derefs; a declared
    /// struct must have the field, and any other object grows by it where
    /// transparent (types §1-2).
    Field { span: Span, object: Term, field: Astr, part: Term },
    /// `container[i]`: the language's own for an array or a slice, and
    /// `as_slice(_mut)` at `callee_site` otherwise (operators §3).
    /// `body` is the effect of the body the index is in.
    Index {
        callee_site: AstId,
        span: Span,
        container: Term,
        mutability: Mutability,
        element: Term,
        body: EffectTerm<InferPhase>,
    },
    /// `*reference`: its target; a store through it needs a `&mut`
    /// (operators §6).
    Deref { span: Span, reference: Term, target: Term, store: bool },
    /// A pattern against its scrutinee, through a reference as Rust's
    /// default binding modes do (patterns §1).
    Destructure { shape: Box<Shape>, scrutinee: Term, lent: Option<Mutability> },
    /// A method call's receiver adjustment (types §6), once the receiver's
    /// head is known; the call's reach follows it.
    Adjust(Box<super::method::Adjust>),
    /// A block with no tail whose last statement is the expression of
    /// term `last`: `block` is `!` where `last` is, and `()` otherwise
    /// (control-flow).
    BlockEnd { span: Span, last: Term, block: Term },
    /// An operand of one of the language's own operators, whose type is
    /// in the operator's domain (operators §2).
    Operand { span: Span, operand: Term, domain: Domain },
    /// `==` or an ordering at `node`: part by part for a structural type,
    /// through `core::eq` or `core::cmp` otherwise (operators §4).
    /// `body` is the effect of the body the comparison is in.
    Compare {
        node: AstId,
        span: Span,
        sig: CoreSig,
        operand: Term,
        body: EffectTerm<InferPhase>,
    },
    /// A use of a generalized binding, waiting until its scheme is settled
    /// (functions §2): the instance's term is `result`.
    Instantiate { scheme: super::generalize::SchemeId, site: AstId, span: Span, result: Term },
    /// A requirement made at an instance: `decision`, whose sites are the
    /// lambda body's, recorded at `instance` (functions §2).
    In { instance: super::generalize::InstanceId, decision: Box<Decision> },
}

/// How a place's value is taken as a reference (ownership §5, patterns §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Lent {
    /// `&` or `&mut`, written: a reference is reborrowed, and a `&mut` of a
    /// `&` is refused.
    Borrow(Mutability),
    /// A name a pattern binds past references of this least mutability: a
    /// reference part is reborrowed at the least of its own and this, and
    /// any other part lent at this.
    Bind(Mutability),
    /// A place read for its value: a reference part reached through
    /// references is reborrowed at the least of its own and its path's,
    /// and any other part is itself.
    Read,
}

impl Decision {
    /// The terms this decision may still bind or enter into.
    fn outputs(&self) -> impl Iterator<Item = Term> {
        let outputs: Vec<Term> = match self {
            Decision::Reach { result, .. } => vec![*result],
            Decision::Iterate { item, .. } => vec![*item],
            Decision::Try { value, exit, .. } => vec![*value, *exit],
            Decision::Enter { slot, .. } => vec![*slot],
            Decision::Lend { result, .. } => vec![*result],
            Decision::Field { part, .. } => vec![*part],
            Decision::Index { element, .. } => vec![*element],
            Decision::Deref { target, .. } => vec![*target],
            Decision::BlockEnd { block, .. } => vec![*block],
            Decision::Adjust(adjust) => vec![adjust.result],
            Decision::Destructure { shape, .. } => shape.bindings(),
            Decision::Instantiate { result, .. } => vec![*result],
            Decision::In { decision, .. } => decision.outputs().collect(),
            Decision::Compare { .. }
            | Decision::Operand { .. }
            | Decision::Cast { .. }
            | Decision::Settle { .. }
            | Decision::HoldsNoLoans { .. } => Vec::new(),
        };
        outputs.into_iter()
    }

    /// The terms this decision may still rebind when it fires: its
    /// outputs, and an `Enter`'s value where the slot is not a structure
    /// written elsewhere, since a fold links the value to the slot.
    fn grows(&self, infer: &Infer) -> Vec<Term> {
        match self {
            Decision::Enter { value, slot, .. } => {
                let written_elsewhere =
                    structure(infer.types(), infer.shallow(*slot)).is_some_and(|slot| slot.label() == Label::Elsewhere);
                match written_elsewhere {
                    true => vec![*slot],
                    false => vec![*value, *slot],
                }
            }
            Decision::In { decision, .. } => decision.grows(infer),
            other => other.outputs().collect(),
        }
    }

    /// Every term this decision reads or writes.
    pub(super) fn terms(&self) -> Vec<Term> {
        match self {
            Decision::Reach { args, result, .. } => args.iter().map(|(_, arg)| *arg).chain([*result]).collect(),
            Decision::Iterate { head, item, .. } => vec![*head, *item],
            Decision::Try { operand, value, exit, .. } => vec![*operand, *value, *exit],
            Decision::Enter { value, slot, .. } => vec![*value, *slot],
            Decision::Cast { source, .. } => vec![*source],
            Decision::Lend { value, result, .. } => vec![*value, *result],
            Decision::Field { object, part, .. } => vec![*object, *part],
            Decision::Index { container, element, .. } => vec![*container, *element],
            Decision::Deref { reference, target, .. } => vec![*reference, *target],
            Decision::BlockEnd { last, block, .. } => vec![*last, *block],
            Decision::Adjust(adjust) => adjust.terms(),
            Decision::Destructure { shape, scrutinee, .. } => {
                let mut terms = shape.bindings();
                terms.push(*scrutinee);
                terms
            }
            Decision::Compare { operand, .. } | Decision::Operand { operand, .. } => {
                vec![*operand]
            }
            Decision::Settle { term, .. }
            | Decision::HoldsNoLoans { term, .. } => vec![*term],
            Decision::Instantiate { result, .. } => vec![*result],
            Decision::In { decision, .. } => decision.terms(),
        }
    }

    /// Every effect this decision joins or flows a call's effect into.
    pub(super) fn effects(&self) -> Vec<EffectTerm<InferPhase>> {
        match self {
            Decision::Reach { effect, .. } => vec![effect.clone()],
            Decision::Iterate { body, .. } | Decision::Enter { body, .. } | Decision::Index { body, .. } | Decision::Compare { body, .. } => {
                vec![body.clone()]
            }
            Decision::Adjust(adjust) => adjust.effects(),
            Decision::Destructure { shape, .. } => shape.effects(),
            Decision::In { decision, .. } => decision.effects(),
            Decision::Try { .. }
            | Decision::Cast { .. }
            | Decision::Settle { .. }
            | Decision::HoldsNoLoans { .. }
            | Decision::Lend { .. }
            | Decision::Field { .. }
            | Decision::Deref { .. }
            | Decision::BlockEnd { .. }
            | Decision::Operand { .. }
            | Decision::Instantiate { .. } => Vec::new(),
        }
    }

    /// This decision with each term and each effect at an instance: a
    /// requirement made there (functions §2).
    pub(super) fn map_terms(&self, f: &mut dyn MapTerms) -> Decision {
        match self.clone() {
            Decision::Reach { site, span, callee, args, result, effect } => Decision::Reach {
                site: f.site(site),
                span: f.span(span),
                callee: callee.map_terms(f),
                args: args.iter().map(|(at, arg)| (f.site(*at), f.ty(*arg))).collect(),
                result: f.ty(result),
                effect: f.effect(&effect),
            },
            Decision::Iterate { site, callee_site, span, head, item, body } => Decision::Iterate {
                site: f.site(site),
                callee_site: f.site(callee_site),
                span: f.span(span),
                head: f.ty(head),
                item: f.ty(item),
                body: f.effect(&body),
            },
            Decision::Try { site, span, operand, value, exit } => {
                Decision::Try { site: f.site(site), span: f.span(span), operand: f.ty(operand), value: f.ty(value), exit: f.ty(exit) }
            }
            Decision::Enter { site, span, value, slot, body, reach } => {
                Decision::Enter { site: f.site(site), span: f.span(span), value: f.ty(value), slot: f.ty(slot), body: f.effect(&body), reach }
            }
            Decision::Cast { site, span, source, to } => Decision::Cast { site: f.site(site), span: f.span(span), source: f.ty(source), to },
            Decision::Settle { span, term } => Decision::Settle { span: f.span(span), term: f.ty(term) },
            Decision::HoldsNoLoans { span, term } => Decision::HoldsNoLoans { span: f.span(span), term: f.ty(term) },
            Decision::Lend { span, value, lent, result } => Decision::Lend { span: f.span(span), value: f.ty(value), lent, result: f.ty(result) },
            Decision::Field { span, object, field, part } => Decision::Field { span: f.span(span), object: f.ty(object), field, part: f.ty(part) },
            Decision::Index { callee_site, span, container, mutability, element, body } => Decision::Index {
                callee_site: f.site(callee_site),
                span: f.span(span),
                container: f.ty(container),
                mutability,
                element: f.ty(element),
                body: f.effect(&body),
            },
            Decision::Deref { span, reference, target, store } => {
                Decision::Deref { span: f.span(span), reference: f.ty(reference), target: f.ty(target), store }
            }
            Decision::Destructure { shape, scrutinee, lent } => {
                Decision::Destructure { shape: Box::new(shape.map_terms(f)), scrutinee: f.ty(scrutinee), lent }
            }
            Decision::Adjust(adjust) => Decision::Adjust(Box::new(adjust.map_terms(f))),
            Decision::BlockEnd { span, last, block } => Decision::BlockEnd { span: f.span(span), last: f.ty(last), block: f.ty(block) },
            Decision::Operand { span, operand, domain } => Decision::Operand { span: f.span(span), operand: f.ty(operand), domain },
            Decision::Compare { node, span, sig, operand, body } => {
                Decision::Compare { node: f.site(node), span: f.span(span), sig, operand: f.ty(operand), body: f.effect(&body) }
            }
            Decision::Instantiate { scheme, site, span, result } => {
                Decision::Instantiate { scheme: f.scheme(scheme), site: f.site(site), span: f.span(span), result: f.ty(result) }
            }
            Decision::In { instance, decision } => Decision::In { instance: f.instance(instance), decision: Box::new(decision.map_terms(f)) },
        }
    }
}

/// What a decision's terms and effects become at an instance.
pub(super) trait MapTerms {
    fn ty(&mut self, term: Term) -> Term;
    fn effect(&mut self, effect: &EffectTerm<InferPhase>) -> EffectTerm<InferPhase>;

    /// A scheme a decision names, where the decision moves to another
    /// check's tables; itself otherwise.
    fn scheme(&mut self, scheme: super::generalize::SchemeId) -> super::generalize::SchemeId {
        scheme
    }

    /// An instance a decision is made at, likewise.
    fn instance(&mut self, instance: super::generalize::InstanceId) -> super::generalize::InstanceId {
        instance
    }

    /// A site a decision records at; itself unless the decision is being
    /// compared without its sites (`Interface::same`).
    fn site(&mut self, site: AstId) -> AstId {
        site
    }

    /// A span a decision refuses at, likewise.
    fn span(&mut self, span: Span) -> Span {
        span
    }
}

impl Callee {
    fn map_terms(self, f: &mut dyn MapTerms) -> Callee {
        let at = |required: RequiredAt, f: &mut dyn MapTerms| RequiredAt { pattern: f.ty(required.pattern), ..required };
        match self {
            Callee::Required(required) => Callee::Required(at(required, f)),
            Callee::Part(required) => Callee::Part(at(required, f)),
            other => other,
        }
    }
}

/// The pending decisions, and what they may still change.
#[derive(Default)]
pub(super) struct Decisions {
    pending: Vec<Decision>,
    /// Each pending decision's owner, by position in `pending`: the
    /// innermost lambda it was made in, `None` for the run (S7).
    owners: Vec<Option<AstId>>,
    /// The owner a decision pushed now takes: the lambda whose body is
    /// being walked, or the owner of the decision being fired.
    pub owner: Option<AstId>,
    /// The instance whose requirement is firing, where one is: a decision
    /// pushed now is made at it too (`Decision::In`).
    pub within: Option<super::generalize::InstanceId>,
}

impl Decisions {
    /// `decision` pending, owned by `self.owner`; made while a requirement
    /// fires at an instance, it is that instance's too (S7, G).
    pub fn push(&mut self, decision: Decision) {
        let decision = match self.within {
            Some(instance) if !matches!(decision, Decision::In { .. }) => Decision::In { instance, decision: Box::new(decision) },
            _ => decision,
        };
        self.pending.push(decision);
        self.owners.push(self.owner);
    }

    fn remove(&mut self, at: usize) -> (Decision, Option<AstId>) {
        (self.pending.remove(at), self.owners.remove(at))
    }

    fn insert(&mut self, at: usize, decision: Decision, owner: Option<AstId>) {
        self.pending.insert(at, decision);
        self.owners.insert(at, owner);
    }

    fn take_all(&mut self) -> Vec<Decision> {
        self.owners.clear();
        std::mem::take(&mut self.pending)
    }

    /// Every pending decision owned by one of `owned`, taken out with its
    /// owner, in order.
    pub(super) fn take_owned(&mut self, owned: &dyn Fn(Option<AstId>) -> bool) -> Vec<(Decision, Option<AstId>)> {
        let mut taken = Vec::new();
        let mut at = 0;
        while at < self.pending.len() {
            match owned(self.owners[at]) {
                true => taken.push(self.remove(at)),
                false => at += 1,
            }
        }
        taken
    }

    /// `decision`, owned by `owner`, pending again.
    pub(super) fn restore(&mut self, decision: Decision, owner: Option<AstId>) {
        self.pending.push(decision);
        self.owners.push(owner);
    }

    /// Each pending decision with its owner.
    pub(super) fn iter(&self) -> impl Iterator<Item = (&Decision, Option<AstId>)> {
        self.pending.iter().zip(self.owners.iter().copied())
    }
}

/// The pending decisions, each output seen through B's links.
pub(super) struct Pending<'a> {
    pub decisions: &'a Decisions,
    pub infer: &'a Infer,
}

impl OpenSources for Pending<'_> {
    /// Whether a pending decision outside the cycles through `term` may
    /// still enter into it or bind it (types §6): whether some pending
    /// output is held by a variable whose rebinding changes `term`'s head.
    /// A decision being fired is taken out of the pending set first, so
    /// its own outputs never keep its inputs open.
    fn may_change(&self, term: Term) -> bool {
        let reads: FxHashSet<TyVarId> = self.infer.head_variables(term).collect();
        self.decisions
            .pending
            .iter()
            .flat_map(Decision::outputs)
            .filter_map(|output| self.infer.last_variable(output))
            .any(|var| reads.contains(&var))
    }

    /// Whether a pending decision may still rebind a variable `term`'s
    /// members hang on: its outputs, and the value of an `Enter` that may
    /// fold, which links the value to its slot (`Infer::enter`). An `Enter`
    /// into a structure written elsewhere coerces, and changes no value.
    fn may_grow(&self, term: Term) -> bool {
        let reads: FxHashSet<TyVarId> = self.infer.head_variables(term).collect();
        self.decisions
            .pending
            .iter()
            .flat_map(|decision| decision.grows(self.infer))
            .filter_map(|grown| self.infer.last_variable(grown))
            .any(|var| reads.contains(&var))
    }
}

/// No pending decision changes anything: what a decision in a cycle
/// through decisions reads, once the sources outside the cycle are fixed.
struct NothingOpen;

impl OpenSources for NothingOpen {
    fn may_change(&self, _term: Term) -> bool {
        false
    }

    fn may_grow(&self, _term: Term) -> bool {
        false
    }
}

/// Whether a firing waits for a term or goes ahead on what is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Reading {
    Closed,
    CycleBroken,
}

/// The fixpoint of the module doc.
pub(super) fn solve(cx: &mut Ctxt<'_>) {
    loop {
        if fire_round(cx, Reading::Closed)
            || uses_decide(cx)
            || default_literals(cx)
            || default_unconstrained(cx, true)
            || default_unconstrained(cx, false)
            || default_non_type_terms(cx)
            || fire_round(cx, Reading::CycleBroken)
        {
            continue;
        }
        break;
    }
    super::effect::refuse_effects_exceeded(cx);
    for decision in cx.decisions.take_all() {
        refuse_left_open(cx, decision);
    }
}

/// The stall step of types §5 for what is not a type: effects, flow sets,
/// and function types' holds and calls; whether any took its
/// default. A lambda's flow set is its claim's (`writeback::claim_flows`),
/// made once its type is solved, and its holds and calls are what settling
/// its captures finds (`closure::settle_captures`), after the fixpoint; no
/// default takes them (D84).
fn default_non_type_terms(cx: &mut Ctxt<'_>) -> bool {
    let effects = cx.infer.default_effects();
    let claimed = super::writeback::lambda_flows(cx);
    let flows = cx.infer.default_unclaimed_flows(&claimed);
    let settled_later: Vec<_> = cx.out.lambdas.iter().map(|lambda| (lambda.holds, lambda.calls)).collect();
    let holds_and_calls = cx.infer.default_unsettled_holds_and_calls(&settled_later);
    effects || flows || holds_and_calls
}

/// Whether an index at `callee_site` is still to be decided.
pub(super) fn index_pending(cx: &Ctxt<'_>, callee_site: AstId) -> bool {
    cx.decisions
        .pending
        .iter()
        .any(|decision| matches!(decision, Decision::Index { callee_site: at, .. } if *at == callee_site))
}

/// The one term whose head a decision reads before it fires, where it has
/// exactly one.
fn read_head(decision: &Decision) -> Option<Term> {
    match decision {
        Decision::Enter { value, .. } | Decision::Lend { value, .. } => Some(*value),
        Decision::Field { object, .. } => Some(*object),
        Decision::Index { container, .. } => Some(*container),
        Decision::Deref { reference, .. } => Some(*reference),
        Decision::Destructure { scrutinee, .. } => Some(*scrutinee),
        Decision::Compare { operand, .. } | Decision::Operand { operand, .. } => {
            Some(*operand)
        }
        Decision::BlockEnd { last, .. } => Some(*last),
        Decision::Adjust(adjust) => Some(adjust.receiver),
        Decision::Iterate { head, .. } => Some(*head),
        Decision::Try { operand, .. } => Some(*operand),
        Decision::Cast { source, .. } => Some(*source),
        Decision::In { decision, .. } => read_head(decision),
        Decision::Reach { .. }
        | Decision::Instantiate { .. }
        | Decision::Settle { .. }
        | Decision::HoldsNoLoans { .. } => None,
    }
}

/// `decision` fired now where the head it reads is closed at the top of
/// its cast tree, which no later decision can change (types §6); pending
/// otherwise.
pub(super) fn decide(cx: &mut Ctxt<'_>, decision: Decision) {
    // A value entering a slot still unbound waits for every other value
    // entering it (`meeting_waits`), and the walk has not pushed them all
    // yet: it is decided with the others once solving starts.
    if let Decision::Enter { slot, .. } = &decision
        && matches!(cx.infer.types().kind(cx.infer.shallow(*slot)), TyKind::Var(_))
    {
        cx.decisions.push(decision);
        return;
    }
    let top = read_head(&decision).is_some_and(|term| {
        cx.infer
            .closed(cx.decls.casts(), &NothingOpen, term)
            .is_some_and(|closed| cx.decls.casts().step_up(closed.head).is_none())
    });
    match top {
        // A value entering a slot may still wait for the others meeting
        // there (`meeting_waits`), though its own head is closed.
        true => {
            if let Err(waiting) = fire(cx, decision, Reading::Closed) {
                cx.decisions.push(waiting);
            }
        }
        false => cx.decisions.push(decision),
    }
}

/// Fires every decision whose terms `reading` admits; whether any fired.
fn fire_round(cx: &mut Ctxt<'_>, reading: Reading) -> bool {
    let mut fired = false;
    let mut at = 0;
    while at < cx.decisions.pending.len() {
        let (decision, owner) = cx.decisions.remove(at);
        match fire_owned(cx, decision, owner, reading) {
            Ok(()) => fired = true,
            Err(waiting) => {
                cx.decisions.insert(at, waiting, owner);
                at += 1;
            }
        }
    }
    fired
}

/// Fires `decision` as `owner`'s: what it pushes is `owner`'s too (S7).
pub(super) fn fire_owned(cx: &mut Ctxt<'_>, decision: Decision, owner: Option<AstId>, reading: Reading) -> Result<(), Decision> {
    let enclosing = std::mem::replace(&mut cx.decisions.owner, owner);
    let fired = fire(cx, decision, reading);
    cx.decisions.owner = enclosing;
    fired
}

/// Fires every decision `owned` admits and `reading` admits; whether any
/// fired. A generalized binding's local fixpoint reads only its own.
pub(super) fn fire_owned_round(cx: &mut Ctxt<'_>, owned: &dyn Fn(&Decision, Option<AstId>) -> bool, reading: Reading) -> bool {
    let mut fired = false;
    let mut at = 0;
    while at < cx.decisions.pending.len() {
        if !owned(&cx.decisions.pending[at], cx.decisions.owners[at]) {
            at += 1;
            continue;
        }
        let (decision, owner) = cx.decisions.remove(at);
        match fire_owned(cx, decision, owner, reading) {
            Ok(()) => fired = true,
            Err(waiting) => {
                cx.decisions.insert(at, waiting, owner);
                at += 1;
            }
        }
    }
    fired
}

/// How the place `term` is reached (types §1): through a reference where
/// the step that made it recorded so as it fired, or where that step, a
/// `Deref`, `Field`, `Index` or `Destructure`, is still pending, since a
/// reach not yet known grows nothing; and `made`, the reach the reading
/// decision was made with, otherwise.
pub(super) fn reach_of(cx: &Ctxt<'_>, term: Term, made: crate::ty::Reach) -> crate::ty::Reach {
    let pending = cx.decisions.pending.iter().any(|decision| makes_place(decision, term));
    match cx.out.through.contains_key(&term) || pending {
        true => crate::ty::Reach::Through,
        false => made,
    }
}

/// Whether `decision` is a place's step that makes `term` and records how
/// it reached it when it fires.
fn makes_place(decision: &Decision, term: Term) -> bool {
    match decision {
        Decision::Deref { .. } | Decision::Field { .. } | Decision::Index { .. } | Decision::Destructure { .. } => {
            decision.outputs().any(|output| output == term)
        }
        Decision::In { decision, .. } => makes_place(decision, term),
        _ => false,
    }
}

pub(super) fn closed_for(cx: &Ctxt<'_>, term: Term, reading: Reading) -> Option<Closed> {
    match reading {
        Reading::Closed => cx.closed(term),
        Reading::CycleBroken => cx.infer.closed(cx.decls.casts(), &NothingOpen, term),
    }
}

/// Fires `decision` if its inputs are closed; hands it back otherwise.
fn fire(cx: &mut Ctxt<'_>, decision: Decision, reading: Reading) -> Result<(), Decision> {
    match decision {
        Decision::Enter {
            site,
            span,
            value,
            slot,
            body,
            reach,
        } => {
            let closed = closed_for(cx, value, reading).filter(|_| !meeting_waits(cx, value, slot, reading));
            // Under `Reading::Closed` an entry waits for how its slot is
            // reached (types §1), and one coerced by shape waits until the
            // members of both sides are fixed (types §3, §6).
            let waits = |cx: &Ctxt<'_>, closed: &Closed| {
                reading == Reading::Closed
                    && (cx.decisions.pending.iter().any(|decision| makes_place(decision, slot))
                        || (cx.infer.coerces(closed.term, slot, reach_of(cx, slot, reach))
                            && !(cx.fixed(value) && cx.fixed(slot))))
            };
            let Some(closed) = closed.filter(|closed| !waits(cx, closed)) else {
                return Err(Decision::Enter {
                    site,
                    span,
                    value,
                    slot,
                    body,
                    reach,
                });
            };
            let reach = reach_of(cx, slot, reach);
            // A value entering a place reached through a shared reference is
            // stored through it (operators §6, ownership §5).
            if cx.out.through.get(&slot) == Some(&Mutability::Shared) {
                refusal::push(cx, RefusalKind::StoreThroughShared, span);
            }
            fire_enter(cx, site, span, (value, closed), (slot, reach), body);
            Ok(())
        }
        Decision::Iterate {
            site,
            callee_site,
            span,
            head,
            item,
            body,
        } => {
            let Some(closed) = closed_for(cx, head, reading) else {
                return Err(Decision::Iterate {
                    site,
                    callee_site,
                    span,
                    head,
                    item,
                    body,
                });
            };
            fire_iterate(cx, site, callee_site, span, closed, item, &body);
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
        Decision::Lend {
            span,
            value,
            lent,
            result,
        } => {
            let Some(closed) = closed_for(cx, value, reading) else {
                return Err(Decision::Lend {
                    span,
                    value,
                    lent,
                    result,
                });
            };
            super::place::fire_lend(cx, span, Some(closed), lent, result, value);
            Ok(())
        }
        Decision::Field {
            span,
            object,
            field,
            part,
        } => {
            let Some(closed) = closed_for(cx, object, reading) else {
                return Err(Decision::Field {
                    span,
                    object,
                    field,
                    part,
                });
            };
            let reach = reach_of(cx, object, crate::ty::Reach::Named);
            super::place::fire_field(cx, span, (object, reach), Some(closed.head), field, part);
            Ok(())
        }
        Decision::Index {
            callee_site,
            span,
            container,
            mutability,
            element,
            body,
        } => {
            let Some(closed) = closed_for(cx, container, reading) else {
                return Err(Decision::Index {
                    callee_site,
                    span,
                    container,
                    mutability,
                    element,
                    body,
                });
            };
            let reach = reach_of(cx, container, crate::ty::Reach::Named);
            super::place::fire_index(cx, callee_site, span, (closed, container, reach), mutability, element, &body);
            Ok(())
        }
        Decision::Deref {
            span,
            reference,
            target,
            store,
        } => {
            let Some(closed) = closed_for(cx, reference, reading) else {
                return Err(Decision::Deref {
                    span,
                    reference,
                    target,
                    store,
                });
            };
            super::place::fire_deref(cx, span, (closed, reference), target, store);
            Ok(())
        }
        Decision::Destructure { shape, scrutinee, lent } => {
            let Some(closed) = closed_for(cx, scrutinee, reading) else {
                return Err(Decision::Destructure { shape, scrutinee, lent });
            };
            let reach = reach_of(cx, scrutinee, crate::ty::Reach::Named);
            super::pat::fire_destructure(cx, *shape, (scrutinee, reach), Some(closed.head), lent);
            Ok(())
        }
        Decision::Compare {
            node,
            span,
            sig,
            operand,
            body,
        } => {
            // A comparison is decided part by part (operators §4), so it
            // waits until the operand's members are fixed (types §6).
            let fixed = reading == Reading::CycleBroken || cx.fixed(operand);
            let Some(closed) = closed_for(cx, operand, reading).filter(|_| fixed) else {
                return Err(Decision::Compare {
                    node,
                    span,
                    sig,
                    operand,
                    body,
                });
            };
            super::op::fire_compare(cx, node, span, sig, closed, &body);
            Ok(())
        }
        Decision::Adjust(adjust) => {
            let Some(closed) = closed_for(cx, adjust.receiver, reading) else {
                return Err(Decision::Adjust(adjust));
            };
            super::method::fire_adjust(cx, *adjust, closed, reading).map_err(|adjust| Decision::Adjust(Box::new(adjust)))
        }
        Decision::BlockEnd { span, last, block } => {
            let Some(closed) = closed_for(cx, last, reading) else {
                return Err(Decision::BlockEnd { span, last, block });
            };
            super::stmt::fire_block_end(cx, span, closed, block);
            Ok(())
        }
        Decision::Operand { span, operand, domain } => {
            let Some(closed) = closed_for(cx, operand, reading) else {
                return Err(Decision::Operand { span, operand, domain });
            };
            super::op::fire_operand(cx, span, closed, domain);
            Ok(())
        }
        Decision::Instantiate { scheme, site, span, result } => {
            super::generalize::fire_instantiate(cx, scheme, site, span, result).map_err(|()| Decision::Instantiate { scheme, site, span, result })
        }
        Decision::In { instance, decision } => super::generalize::in_instance(cx, instance, |cx| fire(cx, *decision, reading))
            .map_err(|decision| Decision::In { instance, decision: Box::new(decision) }),
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

/// Whether a value entering the still unbound slot `slot` waits for the
/// other values entering it (types §1, §6): the slot's type is the one a
/// value of a structured type written elsewhere brings, where any does,
/// and every other value is coerced to it; so the slot is bound only once
/// every value entering it is closed, by such a value first. The outcome
/// does not depend on which value was checked first.
///
/// Under `Reading::CycleBroken` a value still open is not waited for: no
/// other decision will close it, and waiting would leave the slot unbound.
/// Nor is one that can never be a transparent structured type.
fn meeting_waits(cx: &Ctxt<'_>, value: Term, slot: Term, reading: Reading) -> bool {
    let slot = cx.infer.shallow(slot);
    if !matches!(cx.infer.types().kind(slot), TyKind::Var(_)) {
        return false;
    }
    let written_elsewhere = |closed: &Closed| {
        structure(cx.infer.types(), closed.term).is_some_and(|structure| structure.label() == Label::Elsewhere)
    };
    if closed_for(cx, value, reading).is_some_and(|closed| written_elsewhere(&closed)) {
        return false;
    }
    let others = cx.decisions.pending.iter().filter_map(|decision| match decision {
        Decision::Enter { value: other, slot: into, .. } if cx.infer.shallow(*into) == slot => Some(*other),
        _ => None,
    });
    let mut waits = false;
    for other in others {
        match closed_for(cx, other, reading) {
            Some(closed) => waits |= written_elsewhere(&closed),
            // An open value that can never be a structure brings no type
            // written elsewhere: waiting for it would only keep a literal
            // from being decided by the slot (types §5).
            None => waits |= reading == Reading::Closed && cx.infer.may_be_structured(other),
        }
    }
    waits
}

fn fire_enter(cx: &mut Ctxt<'_>, site: AstId, span: Span, (of, value): (Term, Closed), (slot, reach): (Term, crate::ty::Reach), body: EffectTerm<InferPhase>) {
    match cx.infer.enter(cx.decls, of, value, slot, reach) {
        Ok(Entered::Folded) => cx.out.entered.push(super::output::Entered {
            site,
            span,
            value: value.term,
            slot,
            body,
        }),
        Ok(Entered::Converted(conversion)) => {
            let node = AstId::alloc();
            let step = Step {
                node,
                span,
                via: conversion.rule().via,
                view: None,
                body: &body,
            };
            let via_ty = reach_step(cx, step, value.term, slot);
            cx.out.conversions.insert(site, Converted { node, via_ty, to: slot });
        }
        Ok(Entered::Coerced(coerced)) => super::structure::fire_coerced(cx, site, span, (slot, reach), coerced, body),
        Err(mismatch) => refusal::mismatch(cx, mismatch, span),
    }
}

/// `for` by the kind its head's type gives (control-flow): a reference to
/// an array or a slice by reference, the language's own, since no
/// declaration carries an array (extern-fn §1); a reference to any other
/// container by reference, through its `as_slice(_mut)` (operators §3); an
/// array by value.
fn fire_iterate(
    cx: &mut Ctxt<'_>,
    site: AstId,
    callee_site: AstId,
    span: Span,
    head: Closed,
    item: Term,
    body: &EffectTerm<InferPhase>,
) {
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
            let lent_slice = cx.held(TyKind::Ref(mutability, slice));
            let callee = Callee::Core(CoreSig::AsSlice(mutability));
            let reached = super::call::reach_in(cx, body, callee_site, callee, vec![(site, head.term)], span);
            cx.eq(reached, lent_slice, span);
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
        TyKind::Array(element, _, _) | TyKind::Slice(element) => *element,
        other => panic!("element_of: a term closed as an array or a slice is one, not {other:?}"),
    }
}

/// `?` on an `Option` or a `Result` (control-flow): its value is the
/// payload, and it leaves through the same primitive at the body's type.
fn fire_try(cx: &mut Ctxt<'_>, site: AstId, span: Span, operand: Closed, value: Term, exit: Term) {
    let kind = cx.infer.types().kind(operand.term).clone();
    let tried = match kind {
        TyKind::Option(payload, _) => {
            cx.eq(value, payload, span);
            let any = cx.fresh();
            let leaves = cx.node(TyKind::Option(any, Label::Here));
            cx.eq(exit, leaves, span);
            TryExit::Option(exit)
        }
        TyKind::Result(payload, error, _) => {
            cx.eq(value, payload, span);
            let any = cx.fresh();
            let leaves = cx.node(TyKind::Result(any, error, Label::Here));
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
        from => refusal::push(cx, RefusalKind::CastNotAdmitted { from, to }, span),
    }
}

/// A reach decision's parts, moved out of the `Decision` while it fires.
pub(super) struct Reach {
    pub site: AstId,
    pub span: Span,
    pub callee: Callee,
    pub args: Box<[(AstId, Term)]>,
    pub result: Term,
    pub effect: EffectTerm<InferPhase>,
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
pub(super) struct Offer {
    pub declaration: QualifiedRef,
    pub r#impl: Option<Impl>,
    pub ty: TyId<Poly>,
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
    let mut offers: Vec<Offer> = concrete.iter().filter(fits_arity).copied().collect();
    // A transparent structured type has an impl by its generic function's
    // lift (types §1): one candidate among the concrete ones.
    offers.extend(super::structure::lifted_offer(cx, &decision.callee, arity));
    let generic: Vec<Offer> = generic.iter().filter(fits_arity).copied().collect();
    let positions = positions(cx, &decision.args, reading);
    let complete = positions.iter().all(|position| position.chain.is_some());
    let chains: Vec<Vec<Head>> = positions.iter().map_while(|position| position.chain.clone()).collect();
    let result = result_head(cx, decision.result, reading);
    let read = Read { positions: &positions, chains: &chains, complete, result };

    // The language's own answer comes before any impl (ownership §2,
    // operators §2). A requirement asks for a declared function to pass,
    // which an own answer is not, so there a concrete impl comes first.
    // `as_ref` and `as_mut` are the language's alone, given no impls
    // (operators §2).
    match super::structure::lend_at(cx, &decision, reading) {
        Some(super::structure::LendAt::Wait) => return Err(decision),
        Some(super::structure::LendAt::Refused(kind)) => {
            refusal::push(cx, kind, decision.span);
            return Ok(());
        }
        Some(super::structure::LendAt::Lends { function, mutability, structured }) => {
            super::structure::reach_lend(cx, decision, function, mutability, structured);
            return Ok(());
        }
        None => {}
    }
    let required = matches!(decision.callee, Callee::Required(_));
    if !required && let Some(own) = super::structure::own_at(cx, &decision, reading) {
        return super::structure::reach_own(cx, decision, own);
    }
    let chosen = match choose(cx, &offers, &read) {
        Ok(chosen) if chosen.r#impl == Some(Impl::Structural) => {
            if let Some(lift) = super::structure::lift_at(cx, &decision, reading) {
                return super::structure::reach_lifted(cx, decision, lift, reading);
            }
            match super::structure::lift_open(cx, &decision, reading) {
                // The type the lift stands at is not known yet.
                true if reading == Reading::Closed => return Err(decision),
                true => return never_known(cx, decision),
                // No lifted impl at the type it stands at (types §1).
                false => None,
            }
        }
        Ok(chosen) => Some(chosen),
        Err(NoReach::None) => None,
        Err(NoReach::Several(left)) => return several(cx, decision, &offers, &left, &read, reading),
    };
    let chosen = match chosen {
        Some(chosen) => chosen,
        None if required && let Some(own) = super::structure::own_at(cx, &decision, reading) => {
            return super::structure::reach_own(cx, decision, own);
        }
        // The generic impl comes only where no concrete or lifted one is
        // left (D62).
        None => match choose(cx, &generic, &read) {
            Ok(chosen) => chosen,
            Err(NoReach::None) => {
                let name = callee_name(cx, &decision.callee);
                refusal::push(cx, RefusalKind::NoDeclaration { name }, decision.span);
                return Ok(());
            }
            Err(NoReach::Several(left)) => {
                return several(cx, decision, &generic, &left, &read, reading);
            }
        },
    };
    reached(cx, decision, chosen);
    Ok(())
}

/// What a reach read: the closed prefix of its positions, whether every
/// position is closed, and the head its result is taken at, where known.
struct Read<'p> {
    positions: &'p [Position],
    chains: &'p [Vec<Head>],
    complete: bool,
    result: Option<Head>,
}

/// The head a call's result is taken at, once closed (types §6). A `!`
/// that no use gave it, only the default of a variable nothing constrains
/// (types §5), is no type the result is taken at.
fn result_head(cx: &Ctxt<'_>, result: Term, reading: Reading) -> Option<Head> {
    closed_for(cx, result, reading).map(|closed| closed.head).filter(|head| *head != Head::Base(Base::Never))
}

/// A reach whose type never became known (types §6).
fn never_known(cx: &mut Ctxt<'_>, decision: Reach) -> Result<(), Reach> {
    let open = callee_name(cx, &decision.callee);
    refusal::push(cx, RefusalKind::NeverKnown { open }, decision.span);
    Ok(())
}

/// Several offers left: a call whose arguments and result are all read is
/// refused; one with a position or its result still open waits for it,
/// and, where a cycle is broken, one whose result would have decided and
/// is never known is refused as such (types §6).
fn several(
    cx: &mut Ctxt<'_>,
    decision: Reach,
    offers: &[Offer],
    left: &[usize],
    read: &Read<'_>,
    reading: Reading,
) -> Result<(), Reach> {
    let known = read.complete && read.result.is_some();
    if !known && reading == Reading::Closed {
        return Err(decision);
    }
    // Where the ones left differ in their results, the result's type would
    // have decided, and it never became known.
    let results: Vec<Match> = left.iter().map(|at| result_match(cx, &offers[*at])).collect();
    if read.complete && read.result.is_none() && results.iter().any(|result| *result != results[0]) {
        return never_known(cx, decision);
    }
    let name = callee_name(cx, &decision.callee);
    let left = left.iter().map(|at| offers[*at].declaration).collect();
    refusal::push(cx, RefusalKind::SeveralDeclarations { name, left }, decision.span);
    Ok(())
}

/// The one offer the closed prefix of the positions reaches, the result
/// deciding last (types §6).
fn choose(cx: &Ctxt<'_>, offers: &[Offer], read: &Read<'_>) -> Result<Offer, NoReach> {
    let rows: Vec<(Vec<Match>, Match)> = offers
        .iter()
        .map(|offer| {
            let row = read.positions[..read.chains.len()]
                .iter()
                .map(|position| param_match(cx, offer, &position.path))
                .collect();
            (row, result_match(cx, offer))
        })
        .collect();
    let candidates: Vec<Candidate<'_>> = rows.iter().map(|(params, result)| Candidate { params, result: *result }).collect();
    reach::reach(&candidates, read.chains, read.result).map(|at| offers[at])
}

/// What `offer`'s parameter at the position `path` matches. A lifted
/// candidate's every parameter takes its variable (`decl::lift`), so a
/// whole argument matches `Structured`, taken as the lift takes it there.
pub(super) fn param_match(cx: &Ctxt<'_>, offer: &Offer, path: &[usize]) -> Match {
    let types = cx.decls.types();
    match (offer.r#impl, path) {
        (Some(Impl::Structural), [at]) => {
            let lift = cx.decls.lift(offer.declaration).expect("a lifted offer is of a function with a lift");
            Match::Structured(lift.shape().params[*at])
        }
        _ => matched(param_head_at(types, params_of(types, offer.ty), path)),
    }
}

/// What `offer`'s result matches: a lifted candidate's is `Structured`
/// where it rebuilds.
fn result_match(cx: &Ctxt<'_>, offer: &Offer) -> Match {
    match offer.r#impl {
        Some(Impl::Structural) if matches!(cx.decls.lift(offer.declaration), Some(crate::decl::Lift::Rebuild(_))) => {
            Match::Structured(crate::decl::Taking::Value)
        }
        _ => matched(poly_head(cx.decls.types(), ret_of(cx.decls.types(), offer.ty))),
    }
}

/// A declared head as a candidate matches it: `None`, a type variable or a
/// reference to one, matches any.
pub(super) fn matched(head: Option<Head>) -> Match {
    head.map_or(Match::Any, Match::Head)
}

/// # Panics
/// `ty` is not a function type: verify refuses a signature or impl that is
/// not one (`NotAFunction`).
fn ret_of(types: &Types<Poly>, ty: TyId<Poly>) -> TyId<Poly> {
    match types.kind(ty) {
        TyKind::Fn { ret, .. } => *ret,
        _ => panic!("a verified signature is a function type (NotAFunction)"),
    }
}

/// Every offer a callee may reach: the concrete impls and the declarations
/// without impls, and apart, the generic impls, which a call reaches only
/// where no concrete one is left (`Impls`: the concrete ones in order, then
/// the generic one).
/// A type variable parameter matches at the exact step, so the generic
/// impl would tie with every exact concrete one; the concrete impls come
/// first (D62).
pub(super) struct Offers {
    pub concrete: Vec<Offer>,
    pub generic: Vec<Offer>,
}

pub(super) fn offers(cx: &Ctxt<'_>, callee: &Callee) -> Offers {
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
        Callee::Required(required) | Callee::Part(required) => {
            let law = required.fact;
            let states = move |facts: &crate::decl::DeclaredFacts| law.is_none_or(|law| law.stated_by(facts));
            if cx.decls.scheme(required.generic_function).is_some() {
                offer_of(required.generic_function, &states);
            }
        }
    }
    Offers { concrete, generic }
}

pub(super) fn core_name(cx: &Ctxt<'_>, sig: CoreSig) -> QualifiedRef {
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
        CoreSig::Unwrap => core.unwrap,
    }
}

/// The name a refusal of a reach names: what the program wrote.
pub(super) fn callee_name(cx: &Ctxt<'_>, callee: &Callee) -> Astr {
    match callee {
        Callee::Name(name) | Callee::Method(name) => *name,
        Callee::Declaration(declaration) => declaration.name,
        Callee::Core(sig) => core_name(cx, *sig).name,
        Callee::Required(required) | Callee::Part(required) => required.generic_function.name,
    }
}

/// # Panics
/// `ty` is not a function type: verify refuses a signature or impl that is
/// not one (`NotAFunction`).
pub(super) fn params_of(types: &Types<Poly>, ty: TyId<Poly>) -> &[TyId<Poly>] {
    match types.kind(ty) {
        TyKind::Fn { params, .. } => params,
        _ => panic!("a verified signature is a function type (NotAFunction)"),
    }
}

/// The arguments' positions front to back, stopping after the first that
/// is not closed: a later one is not read before it (types §6).
fn positions(cx: &Ctxt<'_>, args: &[(AstId, Term)], reading: Reading) -> Vec<Position> {
    let mut out = Vec::new();
    for (at, (_, arg)) in args.iter().enumerate() {
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
    let TyKind::Tuple(items, _) = cx.infer.types().kind(closed.term) else {
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
pub(super) fn chain_up(cx: &Ctxt<'_>, head: Head) -> Vec<Head> {
    let mut chain = vec![head];
    let mut at = head;
    while let Some(step) = cx.decls.casts().step_up(at) {
        at = match step {
            TreeStep::Cast(cast) => poly_head(cx.decls.types(), cast.rule().to)
                .expect("a tree step joins two heads: verify refuses one without (CastWithoutHead)"),
            TreeStep::Unmark => at.unmarked(),
        };
        chain.push(at);
    }
    chain
}

/// A declared parameter's head at `path`, `None` where any head matches:
/// a type variable, a reference to one, or a part below a head that is not
/// a tuple, which its own position already decided.
pub(super) fn param_head_at(types: &Types<Poly>, params: &[TyId<Poly>], path: &[usize]) -> Option<Head> {
    let (first, parts) = path.split_first().expect("a position names its argument");
    let mut at = params[*first];
    for part in parts {
        match types.kind(at) {
            TyKind::Tuple(items, _) => at = items[*part],
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
pub(super) fn poly_head(types: &Types<Poly>, ty: TyId<Poly>) -> Option<Head> {
    match types.kind(ty) {
        TyKind::Ref(mutability, target) => poly_base(types.kind(*target)).map(|base| Head::Ref(*mutability, base)),
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
        TyKind::Tuple(items, _) => Base::Tuple(items.len()),
        TyKind::Option(.., _) => Base::Option,
        TyKind::Result(..) => Base::Result,
        TyKind::Object(ObjectTy::Structural(..)) => Base::Object,
        TyKind::UserDefined { id, marked, .. } => Base::Named { id: *id, marked: *marked },
        TyKind::Enum { name, .. } => Base::Enum(*name),
        TyKind::Fn { .. } => Base::Fn,
        TyKind::Ref(..) => panic!("a declared reference's target is never a reference (ReferenceToReference)"),
        TyKind::Order | TyKind::Handle(_) | TyKind::Closures(_) => {
            panic!("no declaration writes the order token, a started call or a closure's signatures (InternalType)")
        }
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
    let Instantiated { ty, requirements, .. } = instantiate(cx, offer, span);
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
            cx.join(required.pattern, ty, crate::ty::Reach::Named, span);
            // A unit's instantiation is compiled at the call's types, where
            // its own calls reach their impls, so what a requirement of a
            // unit call chose is checked here and passed nowhere.
            match cx.out.reached.get_mut(&site) {
                Some(parent) => {
                    let mut all = parent.required.to_vec();
                    all.push(chosen);
                    parent.required = all.into();
                }
                None => assert!(
                    cx.out.units.contains_key(&site),
                    "a requirement is reached after the call that states it"
                ),
            }
        }
        Callee::Part(part) => {
            cx.join(part.pattern, ty, crate::ty::Reach::Named, span);
            super::effect::join_into(cx, &effect, &declared_effect, span);
            cx.out.structures.insert(site, super::output::StructureNode::Reached);
            cx.out.reached.insert(site, chosen);
            cx.out.callees.insert(site, ty);
        }
        _ => {
            for ((arg_site, arg), param) in args.iter().zip(params.iter()) {
                cx.enter_in(effect.clone(), *arg_site, *arg, *param, crate::ty::Reach::Named, span);
            }
            cx.eq(result, ret, span);
            super::effect::join_into(cx, &effect, &declared_effect, span);
            cx.out.reached.insert(site, chosen);
            cx.out.callees.insert(site, ty);
        }
    }
    settle_bounded(cx, ty, span);
    for required in requirements {
        push_required(cx, site, span, required, &effect);
    }
}

/// One coercion step's `via` at `node`: a declared cast's or conversion's
/// function (types §3, §7), or for a view (operators §5) its concrete impl
/// `view`, in the body whose effect is `body`.
pub(super) struct Step<'b> {
    pub node: AstId,
    pub span: Span,
    pub via: QualifiedRef,
    pub view: Option<usize>,
    pub body: &'b EffectTerm<InferPhase>,
}

/// A coercion step from `from` to `to`, performed by its `via` as a call
/// is: its type is instantiated with `from` its parameter and `to` its
/// result, its effect joins the body's (effects §1), and each of its
/// requirements is a reach of its own (extern-fn §6). What it chose is
/// recorded in `reached` at the step's node; its function type there is
/// returned.
///
/// # Panics
/// `via` names no function of one parameter: verify refuses a cast or a
/// conversion whose `via` is not its function (`ViaNotItsFunction`), and a
/// view is the compiler's, built as a function of one parameter
/// (`Core::generic_functions`).
pub(super) fn reach_step(cx: &mut Ctxt<'_>, step: Step<'_>, from: Term, to: Term) -> Term {
    let Step {
        node,
        span,
        via,
        view,
        body,
    } = step;
    let scheme = cx
        .decls
        .scheme(via)
        .expect("a step's `via` is declared: verify refuses one that is not (ViaNotItsFunction)");
    let offer = match view {
        Some(at) => Offer {
            declaration: via,
            r#impl: Some(Impl::Concrete(at)),
            ty: scheme.impls.as_ref().expect("a view step is an impl of its view").concrete[at].ty,
        },
        None => Offer {
            declaration: via,
            r#impl: None,
            ty: scheme.ty,
        },
    };
    let Instantiated { ty, requirements, .. } = instantiate(cx, offer, span);
    let TyKind::Fn { params, ret, effect, .. } = cx.infer.types().kind(cx.infer.shallow(ty)).clone() else {
        panic!("a step's `via` is a function type (ViaNotItsFunction, NotAFunction)");
    };
    let [param] = &params[..] else {
        panic!("a step's `via` takes one parameter (ViaNotItsFunction, operators §2)");
    };
    cx.eq(*param, from, span);
    cx.eq(ret, to, span);
    super::effect::flows_into(cx, &effect, body, span);
    cx.out.reached.insert(
        node,
        Chosen {
            declaration: via,
            r#impl: offer.r#impl,
            required: Box::new([]),
        },
    );
    settle_bounded(cx, ty, span);
    for required in requirements {
        push_required(cx, node, span, required, body);
    }
    ty
}

/// A requirement a reached declaration states, as a reach of its own over
/// the required generic function's impls, by the pattern's parameters
/// (extern-fn §6).
pub(super) fn push_required(cx: &mut Ctxt<'_>, site: AstId, span: Span, required: RequiredAt, serves: &EffectTerm<InferPhase>) {
    let TyKind::Fn { params, .. } = cx.infer.types().kind(cx.infer.shallow(required.pattern)).clone() else {
        // An impl is a function (extern-fn §1), so a pattern of another type
        // is answered by none.
        let name = required.generic_function.name;
        refusal::push(cx, RefusalKind::NoDeclaration { name }, span);
        return;
    };
    let args = params.iter().map(|param| (site, *param)).collect();
    let result = cx.fresh();
    cx.decisions.push(Decision::Reach {
        site,
        span,
        callee: Callee::Required(required),
        args,
        result,
        effect: serves.clone(),
    });
}

/// `offer`'s type at fresh variables with the generic function's bounds,
/// and every requirement the generic function and the impl state.
fn instantiate(cx: &mut Ctxt<'_>, offer: Offer, span: Span) -> Instantiated {
    let scheme = cx
        .decls
        .scheme(offer.declaration)
        .expect("an offer names a declared function");
    let generic = instantiate_scheme(cx, scheme, span);
    let Some(Impl::Concrete(at)) = offer.r#impl else {
        return generic;
    };
    let sig = &scheme.impls.as_ref().expect("a concrete impl's scheme has impls").concrete[at];
    let own = instantiate_scheme(cx, &impl_scheme(sig), span);
    cx.eq(generic.ty, own.ty, span);
    let mut requirements = generic.requirements;
    requirements.extend(own.requirements);
    let mut flows = generic.flows;
    flows.extend(own.flows);
    Instantiated { ty: own.ty, requirements, flows }
}

/// `scheme` at fresh variables, each flow into its call's effect refused,
/// where it is, at `span` (extern-fn §2).
pub(super) fn instantiate_scheme(cx: &mut Ctxt<'_>, scheme: &Scheme, span: Span) -> Instantiated {
    let instantiated = cx.infer.instantiate(cx.decls, scheme);
    for flow in &instantiated.flows {
        cx.out.effect_flows.insert(*flow, span);
    }
    instantiated
}

/// A concrete impl as a scheme of its own, each of its type variables at
/// the impl's own bound; the generic function's bounds reach them too,
/// through `eq` with the generic function's instantiation.
fn impl_scheme(sig: &ImplSig) -> Scheme {
    Scheme {
        ty: sig.ty,
        param_names: sig.param_names.clone(),
        bounds: sig.bounds.clone(),
        impls: None,
        requires: sig.requires.clone(),
        facts: sig.facts.clone(),
        runs: sig.runs.clone(),
    }
}

/// How many type variables `written` holds: one past the largest index.
pub(super) fn variables_of(types: &Types<Poly>, written: &[TyId<Poly>]) -> usize {
    let mut largest: Option<u32> = None;
    let mut seen = FxHashSet::default();
    let mut stack = written.to_vec();
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
    largest.map_or(0, |largest| largest as usize + 1)
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

/// At a stall, before any default, a form whose head is still open is
/// decided by its own use where that use decides it (types §5: defaults
/// apply only where nothing more can be decided; D62 Q7). Whether any was.
/// - A value entering a closed slot takes the slot's type.
/// - A value lent where the reference is already closed is that
///   reference's target.
/// - A field read of an open object makes it an object with that field,
///   and a pattern destructuring an open scrutinee makes it the pattern's
///   shape. A reference is never made of an open term: only a value's own
///   type makes one (types §4).
///
/// NOTE: the last two extend Q7's rule, which names only the slot, to
/// the field and the pattern that decide their operand's shape. Without
/// them the ownership §4 example (`let a; a.x = 0;`) would leave `a` open
/// and default it to `!`.
fn uses_decide(cx: &mut Ctxt<'_>) -> bool {
    uses_decide_owned(cx, &|_, _| true)
}

/// `uses_decide` over the decisions `owned` admits.
fn uses_decide_owned(cx: &mut Ctxt<'_>, owned: &dyn Fn(&Decision, Option<AstId>) -> bool) -> bool {
    let mut decided = false;
    let mut at = 0;
    while at < cx.decisions.pending.len() {
        let decides = owned(&cx.decisions.pending[at], cx.decisions.owners[at]) && decided_by_use(cx, &cx.decisions.pending[at]);
        if !decides {
            at += 1;
            continue;
        }
        let (decision, owner) = cx.decisions.remove(at);
        let enclosing = std::mem::replace(&mut cx.decisions.owner, owner);
        match decision {
            Decision::In { instance, decision } => super::generalize::in_instance(cx, instance, |cx| decide_by_use(cx, *decision)),
            decision => decide_by_use(cx, decision),
        }
        cx.decisions.owner = enclosing;
        decided = true;
    }
    decided
}

/// Whether `decision` is one its own use decides at a stall.
fn decided_by_use(cx: &Ctxt<'_>, decision: &Decision) -> bool {
    let open = |term: Term| cx.infer.closed(cx.decls.casts(), &NothingOpen, term).is_none();
    match decision {
        Decision::Enter { value, slot, .. } | Decision::Lend { value, result: slot, .. } => open(*value) && !open(*slot),
        Decision::Field { object, .. } => open(*object),
        Decision::Destructure { scrutinee, .. } => open(*scrutinee),
        Decision::In { decision, .. } => decided_by_use(cx, decision),
        _ => false,
    }
}

fn decide_by_use(cx: &mut Ctxt<'_>, decision: Decision) {
    match decision {
        Decision::Enter { value, slot, span, .. } => cx.eq(value, slot, span),
        Decision::Lend { span, value, lent, result } => super::place::fire_lend(cx, span, None, lent, result, value),
        Decision::Field { span, object, field, part } => {
            let reach = reach_of(cx, object, crate::ty::Reach::Named);
            super::place::fire_field(cx, span, (object, reach), None, field, part);
        }
        Decision::Destructure { shape, scrutinee, lent } => {
            let reach = reach_of(cx, scrutinee, crate::ty::Reach::Named);
            super::pat::fire_destructure(cx, *shape, (scrutinee, reach), None, lent);
        }
        other => panic!("uses_decide: only the forms `decided_by_use` admits are taken, not {other:?}"),
    }
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
        | Decision::Cast { span, .. }
        | Decision::Lend { span, .. }
        | Decision::Field { span, .. }
        | Decision::Index { span, .. }
        | Decision::Deref { span, .. }
        | Decision::Compare { span, .. }
        | Decision::Operand { span, .. }
        | Decision::BlockEnd { span, .. } => refusal::push(cx, RefusalKind::Undecided, span),
        Decision::Adjust(adjust) => refusal::push(cx, RefusalKind::Undecided, adjust.span),
        Decision::Destructure { shape, .. } => refusal::push(cx, RefusalKind::Undecided, shape.span()),
        Decision::Instantiate { span, .. } => refusal::push(cx, RefusalKind::Undecided, span),
        Decision::In { decision, .. } => refuse_left_open(cx, *decision),
    }
}
