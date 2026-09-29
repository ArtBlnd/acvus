//! G, let-bound function values (functions §2). A binding a `let` makes is
//! generalized where every function value entering it has its body in this
//! body: a lambda, a name of such a binding, or a join of them. Its scheme is
//! settled from every value that entered it, the `let`'s own and each
//! assignment's, and two values joined into it require what either
//! requires. Each use of the name instantiates the scheme, and each of its
//! requirements is made there, as a declaration's are reached
//! (extern-fn §6). A value whose body is elsewhere, such as a call's
//! result, is not generalized: its instance was fixed where it was made.
//!
//! A lambda is checked as a module's `fn` is (programs §3): its body's
//! decisions are its own (`Decisions` records each one's owner), decided as
//! far as they can be before its binding is generalized, and a decision
//! still open over a generalized variable is a requirement, carrying the
//! site it arose at.
//!
//! What the enclosing scope or the captured environment holds is not
//! generalized: it belongs to where it was written (types §1), and a
//! decision over it stays the enclosing body's.
//!
//! The walk ends before any binding is generalized, so a binding's scheme
//! sees every value that entered it; `generalize_all` runs between the walk
//! and `decision::solve`, innermost binding first.


use acvus_ast::{AstId, Span};
use rustc_hash::{FxHashMap, FxHashSet};

use super::ctxt::Ctxt;
use super::decision::{Decision, MapTerms, Reading, fire_owned_round};
use super::output::{Adjustment, AsCast, Chosen, CoercionNode, Converted, Entered, Iteration, StructureNode, TryExit};
use crate::infer::{AtInstance, CopiedFlow, Term, VarMarks, VarsOf};
use crate::ty::{EffectTerm, Infer as InferPhase, TyVarId};

/// A binding whose values are generalized, by its binder's site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct SchemeId(pub u32);

/// One use of a generalized binding: which, where, and the instance of an
/// enclosing binding it was made inside, where it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceId(pub u32);

/// A generalized binding while checking.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LetScheme {
    pub binder: AstId,
    /// The binding's term: its function type, over the generalized
    /// variables once `generalize_all` ran.
    pub term: Term,
    /// The lambdas whose values entered the binding: their bodies'
    /// decisions are what its requirements come from.
    pub lambdas: Vec<AstId>,
    /// `None` until `generalize_all` settles it.
    pub settled: Option<Settled>,
}

/// A scheme settled: what it generalizes, and what it requires.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Settled {
    /// Every variable of every kind made while checking the values that
    /// entered the binding and reached from no term the enclosing scope or
    /// a capture holds.
    pub generalized: Generalized,
    /// Each decision still open over a generalized variable, made at each
    /// use (functions §2).
    pub requirements: Box<[Decision]>,
}

/// The generalized variables, by kind: every kind a type states
/// (functions §2). What a function value's captures hold, how often it
/// may be called and its flows are decided where it is written, not per
/// instance.
pub(super) type Generalized = VarsOf;

/// One instance of a scheme at a use.
#[derive(Debug, Clone)]
pub(super) struct Instance {
    pub scheme: SchemeId,
    pub use_site: AstId,
    pub span: Span,
    /// The instance of the binding this use was checked inside, where the
    /// use is in a generalized binding's own value.
    pub within: Option<InstanceId>,
    /// Each generalized variable's copy at this instance, of every kind.
    pub at: AtInstance,
    /// What its requirements recorded at the lambda bodies' sites while
    /// they fired, swapped in for `Output`'s own tables (`in_instance`).
    pub recorded: Recorded,
}

/// The `Output` tables a decision records into at the sites it decides:
/// the run's, or an instance's while one of its requirements fires.
#[derive(Debug, Clone, Default)]
pub(super) struct Recorded {
    pub types: FxHashMap<AstId, Term>,
    pub reached: FxHashMap<AstId, Chosen>,
    pub callees: FxHashMap<AstId, Term>,
    pub iteration: FxHashMap<AstId, Iteration>,
    pub tries: FxHashMap<AstId, TryExit<Term>>,
    pub conversions: FxHashMap<AstId, Converted>,
    pub coercions: FxHashMap<AstId, CoercionNode>,
    pub entered: Vec<Entered>,
    pub casts: FxHashMap<AstId, AsCast>,
    pub adjustments: FxHashMap<AstId, Adjustment>,
    pub structures: FxHashMap<AstId, StructureNode>,
}

/// A walk whose decisions and variables are one owner's (S7): a lambda's
/// body, or a value given a binding by its `let` or an assignment, by the
/// lambda's or the binder's site. `vars` are the variables of every kind
/// each walk made, from its first mark to its last.
#[derive(Debug, Clone, Default)]
pub(super) struct Walk {
    pub parent: Option<AstId>,
    pub vars: Vec<(VarMarks, VarMarks)>,
}

/// Which variables, of every kind, a binding's own walks made.
struct Own(Vec<(VarMarks, VarMarks)>);

impl Own {
    fn made(&self, kind: impl Fn(&VarMarks) -> u32, index: u32) -> bool {
        self.0.iter().any(|(start, end)| (kind(start)..kind(end)).contains(&index))
    }
}

/// `walk` run as `owner`'s: the decisions it makes and the variables it
/// makes are `owner`'s, nested in the owner enclosing it.
pub(super) fn walk<R>(cx: &mut Ctxt<'_>, owner: AstId, walk: impl FnOnce(&mut Ctxt<'_>) -> R) -> R {
    let parent = cx.decisions.owner.replace(owner);
    let start = cx.infer.var_marks();
    let walked = walk(cx);
    let end = cx.infer.var_marks();
    cx.out.walks.entry(owner).or_insert_with(|| Walk { parent, vars: Vec::new() }).vars.push((start, end));
    cx.decisions.owner = parent;
    walked
}

/// Whether a `let` whose value is `value` makes a generalized binding: every
/// function value that can enter it has its body in this body (functions §2).
pub(super) fn generalizes(cx: &Ctxt<'_>, value: Term) -> bool {
    sources(cx, value, &mut FxHashSet::default(), &mut Vec::new())
}

/// Whether every value that can enter `term` is a lambda written here, a
/// use of a generalized binding, or a join of them; each such lambda is
/// pushed onto `lambdas`. A value of any other origin, such as a call's
/// result, is not.
fn sources(cx: &Ctxt<'_>, term: Term, seen: &mut FxHashSet<Term>, lambdas: &mut Vec<AstId>) -> bool {
    if !seen.insert(term) {
        return true;
    }
    if let Some(lambda) = cx.out.lambdas.iter().find(|lambda| cx.out.types.get(&lambda.site) == Some(&term)) {
        lambdas.push(lambda.site);
        return true;
    }
    if instance_of(cx, term) {
        return true;
    }
    let entering: Vec<Term> = cx
        .decisions
        .iter()
        .filter_map(|(decision, _)| match decision {
            Decision::Enter { value, slot, .. } if *slot == term => Some(*value),
            _ => None,
        })
        .collect();
    !entering.is_empty() && entering.into_iter().all(|value| sources(cx, value, seen, lambdas))
}

/// Whether `term` is a use of a generalized binding: an instance's term.
fn instance_of(cx: &Ctxt<'_>, term: Term) -> bool {
    cx.decisions.iter().any(|(decision, _)| matches!(decision, Decision::Instantiate { result, .. } if *result == term))
}

/// A generalized binding at `binder`, of term `term`, registered with no
/// value yet: each value entering it adds its lambdas (`generalizes`
/// finds them), and `generalize_all` settles it.
pub(super) fn register(cx: &mut Ctxt<'_>, binder: AstId, term: Term) -> SchemeId {
    let id = SchemeId(u32::try_from(cx.out.schemes.len()).expect("register: u32::MAX schemes"));
    cx.out.schemes.push(LetScheme { binder, term, lambdas: Vec::new(), settled: None });
    id
}

/// Settles every scheme, innermost first: each binding's lambdas' decisions
/// are decided as far as they can be without a default reaching a
/// generalizable variable (types §5 applies at the use, not here), and the
/// scheme takes what is left. A binding registers once its value is
/// walked, so one written in another's value registers, and settles,
/// first.
pub(super) fn generalize_all(cx: &mut Ctxt<'_>) {
    for at in 0..cx.out.schemes.len() {
        settle(cx, SchemeId(u32::try_from(at).expect("u32 schemes")));
    }
}

/// Whether `owner`'s walk is one of `roots`' or nested in one.
fn nested_in(walks: &FxHashMap<AstId, Walk>, roots: &FxHashSet<AstId>, owner: Option<AstId>) -> bool {
    let mut at = owner;
    while let Some(site) = at {
        if roots.contains(&site) {
            return true;
        }
        at = walks.get(&site).and_then(|walk| walk.parent);
    }
    false
}

fn settle(cx: &mut Ctxt<'_>, id: SchemeId) {
    let LetScheme { binder, term, .. } = cx.out.schemes[id.0 as usize].clone();
    let mut lambdas = Vec::new();
    let local = sources(cx, term, &mut FxHashSet::default(), &mut lambdas);
    cx.out.schemes[id.0 as usize].lambdas = lambdas.clone();
    if !local {
        // An assignment gave the binding a value whose body is elsewhere:
        // it is not generalized, and each use is its one term.
        cx.out.schemes[id.0 as usize].settled = Some(Settled { generalized: Generalized::default(), requirements: Box::new([]) });
        return;
    }
    let roots: FxHashSet<AstId> = lambdas.iter().copied().chain([binder]).collect();
    let walks = cx.out.walks.clone();
    let own_decision = |decision: &Decision, owner: Option<AstId>| {
        nested_in(&walks, &roots, owner) || matches!(decision, Decision::Enter { slot, .. } if *slot == term)
    };

    // The local fixpoint fires what its own terms decide, and nothing a
    // stall does: no default, no form decided by its own use, and no cycle
    // broken, since a use of the binding may still decide what they read
    // (types §5).
    let start = cx.infer.var_marks();
    while fire_owned_round(cx, &own_decision, Reading::Closed) {}
    let end = cx.infer.var_marks();

    let mut spans: Vec<(VarMarks, VarMarks)> = roots.iter().filter_map(|root| walks.get(root)).flat_map(|walk| walk.vars.clone()).collect();
    spans.push((start, end));
    let own = Own(spans);

    // What outside holds: every variable made elsewhere, and every one its
    // types reach.
    let marks = cx.infer.var_marks();
    let elsewhere: Vec<Term> = (0..marks.types)
        .filter(|index| !own.made(|marks| marks.types, *index))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|index| cx.infer.var_term(TyVarId(index)))
        .collect();
    // A body's effect is where its calls' effects flow (effects §1): the
    // run's, and each lambda's outside the binding's own, stay one, whichever
    // walk first asked for them.
    let bodies: Vec<EffectTerm<InferPhase>> = cx
        .out
        .body_effect
        .iter()
        .cloned()
        .chain(cx.out.lambdas.iter().filter(|lambda| !nested_in(&walks, &roots, Some(lambda.site))).map(|lambda| lambda.effect.clone()))
        .collect();
    let held = cx.infer.reached(elsewhere, bodies);

    // Every variable the values made that nothing outside holds: those the
    // binding's type reaches, and those its still open decisions read,
    // which each use must make anew too.
    let owned = cx.decisions.take_owned(&|owner| nested_in(&walks, &roots, owner));
    let types: Vec<Term> = owned.iter().flat_map(|(decision, _)| decision.terms()).chain([term]).collect();
    let effects: Vec<_> = owned.iter().flat_map(|(decision, _)| decision.effects()).collect();
    let reached = cx.infer.reached(types, effects);
    let generalized = Generalized {
        types: sorted(reached.types.iter().copied().filter(|var| cx.infer.is_unbound(*var) && own.made(|marks| marks.types, var.0) && !held.types.contains(var)), |var| var.0),
        effects: sorted(reached.effects.iter().copied().filter(|var| own.made(|marks| marks.effects, var.0) && !held.effects.contains(var)), |var| var.0),
        lengths: sorted(reached.lengths.iter().copied().filter(|var| cx.infer.is_open_length(*var) && own.made(|marks| marks.lengths, var.0) && !held.lengths.contains(var)), |var| var.0),
        identities: sorted(reached.identities.iter().copied().filter(|var| cx.infer.is_open_identity(*var) && own.made(|marks| marks.identities, var.0) && !held.identities.contains(var)), |var| var.0),
        marks: sorted(reached.marks.iter().copied().filter(|var| cx.infer.is_open_mark(*var) && own.made(|marks| marks.marks, var.0) && !held.marks.contains(var)), |var| var.0),
    };

    let mut requirements = Vec::new();
    for (decision, owner) in owned {
        let reads = cx.infer.reached(decision.terms(), decision.effects());
        let generalizes = reads.types.iter().any(|var| generalized.types.contains(var))
            || reads.effects.iter().any(|var| generalized.effects.contains(var))
            || reads.lengths.iter().any(|var| generalized.lengths.contains(var))
            || reads.identities.iter().any(|var| generalized.identities.contains(var))
            || reads.marks.iter().any(|var| generalized.marks.contains(var));
        match generalizes {
            true => requirements.push(decision),
            false => cx.decisions.restore(decision, owner),
        }
    }
    cx.out.schemes[id.0 as usize].settled = Some(Settled { generalized, requirements: requirements.into() });
}

fn sorted<T>(vars: impl Iterator<Item = T>, key: impl Fn(&T) -> u32) -> Box<[T]> {
    let mut vars: Vec<T> = vars.collect();
    vars.sort_by_key(key);
    vars.into()
}

/// A use of the generalized binding `scheme` at `use_site`: a term of the
/// binding's type at fresh variables, and each requirement made there as a
/// `Decision::In` of the new instance. It waits as a decision until the
/// scheme is settled.
pub(super) fn instantiate(cx: &mut Ctxt<'_>, scheme: SchemeId, use_site: AstId, span: Span) -> Term {
    let result = cx.fresh();
    cx.decisions.push(Decision::Instantiate { scheme, site: use_site, span, result });
    result
}

/// `Decision::Instantiate` fired; `Err` while its scheme is not settled.
pub(super) fn fire_instantiate(cx: &mut Ctxt<'_>, scheme: SchemeId, use_site: AstId, span: Span, result: Term) -> Result<(), ()> {
    let LetScheme { term, settled, .. } = cx.out.schemes[scheme.0 as usize].clone();
    let Some(Settled { generalized, requirements }) = settled else {
        return Err(());
    };
    let (at, flows) = cx.infer.instance_at(&generalized);
    copied_flows(cx, &flows);
    let within = cx.out.firing_in;
    let instance = new_instance(cx, Instance { scheme, use_site, span, within, at: at.clone(), recorded: Recorded::default() });
    let mut memo = FxHashMap::default();
    let copied = cx.infer.copy_at(term, &at, &mut memo);
    cx.eq(result, copied, span);
    let mut children: FxHashMap<InstanceId, InstanceId> = FxHashMap::default();
    for requirement in requirements.iter() {
        let (at_instance, decision) = match requirement {
            // A use inside the value is an instance of its own binding
            // within this one: made anew here, its variables at this
            // instance's (functions §2).
            Decision::In { instance: inner, decision } => {
                let child = match children.get(inner) {
                    Some(child) => *child,
                    None => {
                        let of = cx.out.instances[inner.0 as usize].clone();
                        let child_at = AtInstance {
                            types: of.at.types.iter().map(|(var, term)| (*var, cx.infer.copy_at(*term, &at, &mut memo))).collect(),
                            effects: of.at.effects.iter().map(|(var, copy)| (*var, at.effects.get(copy).copied().unwrap_or(*copy))).collect(),
                            lengths: of.at.lengths.iter().map(|(var, copy)| (*var, at.lengths.get(copy).copied().unwrap_or(*copy))).collect(),
                            identities: of.at.identities.iter().map(|(var, copy)| (*var, at.identities.get(copy).copied().unwrap_or(*copy))).collect(),
                            marks: of.at.marks.iter().map(|(var, copy)| (*var, at.marks.get(copy).copied().unwrap_or(*copy))).collect(),
                        };
                        let child = new_instance(cx, Instance { within: Some(instance), at: child_at, recorded: Recorded::default(), ..of });
                        children.insert(*inner, child);
                        child
                    }
                };
                (child, decision.as_ref())
            }
            decision => (instance, decision),
        };
        let decision = decision.map_terms(&mut Copying { infer: &mut cx.infer, at: &at, memo: &mut memo });
        cx.decisions.push(Decision::In { instance: at_instance, decision: Box::new(decision) });
    }
    Ok(())
}

/// Each copied effect flow refused, where it is, at its original's span:
/// the flow is the same constraint at the instance (effects §1).
fn copied_flows(cx: &mut Ctxt<'_>, flows: &[CopiedFlow]) {
    for flow in flows {
        if let Some(span) = cx.out.effect_flows.get(&flow.of).copied() {
            cx.out.effect_flows.insert(flow.copy, span);
        }
    }
}

/// A decision's terms and effects copied at an instance.
struct Copying<'c> {
    infer: &'c mut crate::infer::Infer,
    at: &'c AtInstance,
    memo: &'c mut FxHashMap<Term, Term>,
}

impl MapTerms for Copying<'_> {
    fn ty(&mut self, term: Term) -> Term {
        self.infer.copy_at(term, self.at, self.memo)
    }

    fn effect(&mut self, effect: &EffectTerm<InferPhase>) -> EffectTerm<InferPhase> {
        self.infer.copy_effect(effect, self.at)
    }
}

fn new_instance(cx: &mut Ctxt<'_>, instance: Instance) -> InstanceId {
    let id = InstanceId(u32::try_from(cx.out.instances.len()).expect("new_instance: u32::MAX instances"));
    cx.out.instances.push(instance);
    id
}

/// `fire` run at `instance`: what it records goes to the instance's
/// tables, and what it pushes is made at the instance too.
pub(super) fn in_instance<R>(cx: &mut Ctxt<'_>, instance: InstanceId, fire: impl FnOnce(&mut Ctxt<'_>) -> R) -> R {
    if cx.out.firing_in == Some(instance) {
        return fire(cx);
    }
    let outer = cx.out.firing_in.replace(instance);
    let within = cx.decisions.within.replace(instance);
    swap_recorded(cx, instance);
    let fired = fire(cx);
    swap_recorded(cx, instance);
    cx.out.firing_in = outer;
    cx.decisions.within = within;
    fired
}

fn swap_recorded(cx: &mut Ctxt<'_>, instance: InstanceId) {
    let out = &mut cx.out;
    let recorded = &mut out.instances[instance.0 as usize].recorded;
    std::mem::swap(&mut out.types, &mut recorded.types);
    std::mem::swap(&mut out.reached, &mut recorded.reached);
    std::mem::swap(&mut out.callees, &mut recorded.callees);
    std::mem::swap(&mut out.iteration, &mut recorded.iteration);
    std::mem::swap(&mut out.tries, &mut recorded.tries);
    std::mem::swap(&mut out.conversions, &mut recorded.conversions);
    std::mem::swap(&mut out.coercions, &mut recorded.coercions);
    std::mem::swap(&mut out.entered, &mut recorded.entered);
    std::mem::swap(&mut out.casts, &mut recorded.casts);
    std::mem::swap(&mut out.adjustments, &mut recorded.adjustments);
    std::mem::swap(&mut out.structures, &mut recorded.structures);
}

/// The sites the walks of an instance's binding's values recorded, those
/// of the walks nested in them included, and its variables' copies there.
pub(super) fn instance_sites(cx: &Ctxt<'_>, instance: InstanceId) -> (Vec<AstId>, AtInstance) {
    let Instance { scheme, at, .. } = &cx.out.instances[instance.0 as usize];
    let LetScheme { binder, lambdas, .. } = &cx.out.schemes[scheme.0 as usize];
    let roots: FxHashSet<AstId> = lambdas.iter().copied().chain([*binder]).collect();
    let sites = cx
        .out
        .site_owners
        .iter()
        .filter(|(_, owner)| nested_in(&cx.out.walks, &roots, Some(**owner)))
        .map(|(site, _)| *site)
        .collect();
    (sites, at.clone())
}
