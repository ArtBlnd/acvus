//! Every term to its concrete type, once (compilation); the lambdas' claimed
//! flows into their function types.

use acvus_ast::AstId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::ctxt::Ctxt;
use super::generalize::InstanceId;
use super::output::{Closure, Coercion, CoercionNode, Conversion, Entered, InstanceDecided, StructureNode, StructureTree, Typeck};
use crate::decl::{Base, DeclaredFacts, Head, Scheme, TyVarBound};
use crate::infer::{FreezeMemo, ShapeCoercion, Term, Unsettled};
use crate::ty::structural::PartKey;
use crate::ty::{
    Alignment, Calls, CallsTerm, Concrete, Effect, EffectTerm, Flow, FlowEnd, FlowTerm, Flows, Holds, HoldsTerm, Infer as InferPhase,
    Mutability, Ty, TyKind, Types,
};

/// Freezes every decision, `result` being the run's result term.
pub(super) fn finish(mut cx: Ctxt<'_>, types: &mut Types<Concrete>, result: Term) -> Typeck {
    let lambdas: Vec<Term> = cx
        .out
        .closures
        .keys()
        .filter_map(|lambda| cx.out.types.get(lambda).copied())
        .collect();
    for lambda in lambdas {
        claim_flows(&mut cx, lambda);
    }

    let mut memo = FreezeMemo::default();
    // Coercions first: each step's function is reached as a call is, and
    // its effect joins its body's, before any type holding that effect is
    // frozen (effects §1).
    let coercions = coercions(&mut cx, types, &mut memo);
    let sites: Vec<_> = cx.out.types.iter().map(|(site, term)| (*site, *term)).collect();
    let mut frozen = FxHashMap::default();
    for (site, term) in sites {
        if let Some(ty) = freeze(&mut cx, types, &mut memo, term) {
            frozen.insert(site, ty);
        }
    }
    let tries = std::mem::take(&mut cx.out.tries)
        .into_iter()
        .filter_map(|(site, exit)| {
            let exit = match exit {
                super::output::TryExit::Option(term) => super::output::TryExit::Option(freeze(&mut cx, types, &mut memo, term)?),
                super::output::TryExit::Result(term) => super::output::TryExit::Result(freeze(&mut cx, types, &mut memo, term)?),
            };
            Some((site, exit))
        })
        .collect();
    let structure_trees = structure_trees(&mut cx, types, &mut memo);
    let instances = instances(&mut cx, types, &mut memo);
    let value_sites = value_sites(&cx);
    let generalized = cx
        .out
        .schemes
        .iter()
        .filter(|scheme| scheme.generalizes)
        .map(|scheme| (scheme.binder, scheme.lambdas.clone().into()))
        .collect();
    let result = freeze(&mut cx, types, &mut memo, result);
    let callees = std::mem::take(&mut cx.out.callees)
        .into_iter()
        .filter_map(|(site, callee)| Some((site, freeze(&mut cx, types, &mut memo, callee)?)))
        .collect();
    let body_effect = match cx.out.body_effect.take() {
        Some(effect) => effect_of(&mut cx, effect),
        // A run that calls nothing has no effect to order (effects §1).
        None => Effect::pure(),
    };
    let closures: FxHashMap<_, Closure> = std::mem::take(&mut cx.out.closures);

    Typeck {
        types: frozen,
        names: std::mem::take(&mut cx.out.names),
        reached: std::mem::take(&mut cx.out.reached),
        callees,
        iteration: std::mem::take(&mut cx.out.iteration),
        tries,
        coercions,
        casts: std::mem::take(&mut cx.out.casts),
        closures,
        adjustments: std::mem::take(&mut cx.out.adjustments),
        structure_trees,
        instances,
        generalized,
        value_sites,
        body_effect,
        result,
        refusals: std::mem::take(&mut cx.out.refusals),
    }
}

/// `term` frozen. `None` only where a refusal already left a variable
/// open; with no refusal that is the checker's defect.
///
/// # Panics
/// A variable is open and no refusal was made: solving binds every type
/// variable and, at its last stall, every effect, mark, flow set, holds
/// and calls no lambda claims (types §5); each lambda's claim sets its own
/// flows, and settling its captures its own holds and calls; and every
/// length and identity is fixed by the term that made it.
fn freeze(cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo, term: Term) -> Option<Ty> {
    match cx.infer.freeze(types, memo, term) {
        Ok(ty) => Some(ty),
        Err(open) => {
            assert!(
                !cx.out.refusals.is_empty(),
                "{open:?} is open after solving, and no refusal explains it"
            );
            None
        }
    }
}

/// Each lambda's flow set, which its claim decides and no default does.
pub(super) fn lambda_flows(cx: &Ctxt<'_>) -> Vec<FlowTerm<InferPhase>> {
    cx.out
        .closures
        .keys()
        .filter_map(|lambda| cx.out.types.get(lambda))
        .filter_map(|term| match cx.infer.types().kind(cx.infer.shallow(*term)) {
            TyKind::Fn { flows, .. } => Some(flows.clone()),
            _ => None,
        })
        .collect()
}

/// `flows` set to `to`.
fn set_flows(cx: &mut Ctxt<'_>, flows: FlowTerm<InferPhase>, to: Flows) {
    let effect = EffectTerm::Known(Effect::pure());
    let open = probe(cx, effect.clone(), flows);
    let set = probe(cx, effect, FlowTerm::Known(to));
    cx.infer.eq(open, set).expect("an open flow set takes any flows");
}

/// A function type of no parameter carrying `effect` and `flows`: the
/// term through which B relates an effect or a flow variable.
fn probe(cx: &mut Ctxt<'_>, effect: EffectTerm<InferPhase>, flows: FlowTerm<InferPhase>) -> Term {
    let ret = cx.unit();
    let values = cx.infer.fresh_values();
    cx.node(TyKind::Fn {
        params: Box::new([]),
        ret,
        effect,
        flows,
        calls: CallsTerm::Known(Calls::Many),
        holds: HoldsTerm::Known(Holds::Nothing),
        values,
    })
}

/// What `effect` settled to.
///
/// # Panics
/// `effect` is open: `solve` ends only at a stall where no effect was left
/// to default, so every effect is known.
fn effect_of(cx: &mut Ctxt<'_>, effect: EffectTerm<InferPhase>) -> Effect {
    let probe = probe(cx, effect, FlowTerm::Known(Flows::none()));
    let mut types = Types::new();
    let frozen = cx
        .infer
        .freeze(&mut types, &mut FreezeMemo::default(), probe)
        .expect("solve defaults every open effect at its last stall (types §5)");
    match types.kind(frozen) {
        TyKind::Fn {
            effect: EffectTerm::Known(effect),
            ..
        } => effect.clone(),
        other => panic!("effect_of: a frozen probe is a function type of a known effect, not {other:?}"),
    }
}

/// A lambda's flows, claimed at the type level (ownership §5): every
/// output it has takes from every input it has, flow-insensitively. Where
/// a slot it entered fixed its flows already, those stand, and the borrow
/// check verifies the lambda against them.
fn claim_flows(cx: &mut Ctxt<'_>, lambda: Term) {
    let TyKind::Fn {
        params,
        ret,
        flows,
        holds,
        ..
    } = cx.infer.types().kind(cx.infer.shallow(lambda)).clone()
    else {
        assert!(
            !cx.out.refusals.is_empty(),
            "a lambda's term is a function type unless a refused relation left it otherwise"
        );
        return;
    };
    let flows_probe = probe(cx, EffectTerm::Known(Effect::pure()), flows.clone());
    match cx.infer.freeze(&mut Types::new(), &mut FreezeMemo::default(), flows_probe) {
        Err(Unsettled::Flow(_)) => {}
        Ok(_) => return,
        Err(other) => panic!("a probe of a known effect and no parameter has only its flows open, not {other:?}"),
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (at, param) in params.iter().enumerate() {
        let Some(held) = positions(cx, *param) else {
            return;
        };
        if held.any {
            inputs.push(FlowEnd::Param(at));
        }
        if held.exclusive {
            outputs.push(FlowEnd::Param(at));
        }
    }
    let Some(result) = positions(cx, ret) else {
        return;
    };
    if result.any {
        outputs.push(FlowEnd::Result);
    }
    let Some(holds) = cx.infer.holds(holds) else {
        assert!(
            !cx.out.refusals.is_empty(),
            "settling binds every lambda's holds, unless a refusal left its captures' types open"
        );
        return;
    };
    if holds != Holds::Nothing {
        inputs.push(FlowEnd::Captures);
    }
    if holds == Holds::Mut {
        outputs.push(FlowEnd::Captures);
    }
    let claimed = outputs.iter().flat_map(|to| {
        inputs.iter().filter(move |from| *from != to).map(move |from| Flow {
            to: *to,
            from: *from,
            alignment: Alignment::Any,
        })
    });
    let claimed = Flows::of(claimed.collect::<Vec<_>>());
    set_flows(cx, flows, claimed);
}

/// What positions a value of a term holds (ownership §5).
struct Held {
    any: bool,
    /// A `&mut`, or a function value whose captures hold one: what a callee
    /// may write through.
    exclusive: bool,
}

/// `None` where the term holds a variable still open, which a refusal
/// left: a type variable, or a function type's `holds` no settling bound.
fn positions(cx: &Ctxt<'_>, term: Term) -> Option<Held> {
    let mut held = Held {
        any: false,
        exclusive: false,
    };
    let mut stack = vec![term];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(at) = stack.pop() {
        let at = cx.infer.shallow(at);
        if !seen.insert(at) {
            continue;
        }
        match cx.infer.types().kind(at) {
            TyKind::Var(_) => return None,
            TyKind::Ref(mutability, target) => {
                held.any = true;
                held.exclusive |= *mutability == Mutability::Mut;
                stack.push(target.ty());
            }
            TyKind::Fn { holds, .. } => {
                held.any = true;
                held.exclusive |= cx.infer.holds(*holds)? == Holds::Mut;
            }
            TyKind::UserDefined { id, type_args, .. } => {
                held.any |= crate::ty::LifetimesOf::lifetimes_of(cx.decls, *id) > 0;
                stack.extend(type_args.iter().map(|arg| arg.ty()));
            }
            kind => kind.for_each_child(|child| stack.push(child)),
        }
    }
    Some(held)
}

/// Every coercion, each at the site of the value coerced (types §3, §7):
/// a structured value's by its `CoercionNode` and each part's own, a
/// concrete value's by `conversions`' steps. A structured coercion that
/// keeps its shape and every part as it is changes nothing, and is left
/// out; so is one some refusal left open.
fn coercions(cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo) -> FxHashMap<AstId, Coercion> {
    let mut steps = conversions(cx, types, memo);
    let nodes = std::mem::take(&mut cx.out.coercions);
    let parts: FxHashSet<AstId> = nodes.values().flat_map(|node| node.parts.iter().map(|(_, site)| *site)).collect();
    let mut out = FxHashMap::default();
    for root in nodes.keys().copied().filter(|site| !parts.contains(site)) {
        if let Some(Some(coercion)) = structured(cx, types, memo, &nodes, &mut steps, root) {
            out.insert(root, coercion);
        }
    }
    out.extend(steps.into_iter().map(|(site, steps)| (site, Coercion::Steps(steps))));
    out
}

/// The coercion of the structured value at `site`: `Some(None)` where it
/// changes nothing, and `None` where a refusal left its slot open.
fn structured(
    cx: &mut Ctxt<'_>,
    types: &mut Types<Concrete>,
    memo: &mut FreezeMemo,
    nodes: &FxHashMap<AstId, CoercionNode>,
    steps: &mut FxHashMap<AstId, Box<[Conversion<Ty>]>>,
    site: AstId,
) -> Option<Option<Coercion>> {
    let node = &nodes[&site];
    let mut parts = Vec::with_capacity(node.parts.len());
    for (key, part_site) in node.parts.iter() {
        let part = match nodes.contains_key(part_site) {
            true => structured(cx, types, memo, nodes, steps, *part_site)?,
            false => steps.remove(part_site).map(Coercion::Steps),
        };
        parts.push((*key, part));
    }
    let to = freeze(cx, types, memo, node.slot)?;
    let changes = node.shape != ShapeCoercion::Kept || parts.iter().any(|(_, part)| part.is_some());
    Some(changes.then(|| Coercion::Structured { shape: node.shape.clone(), through: node.through, to, parts: parts.into() }))
}

/// Every coercion, each at its site with its steps in order: a declared
/// conversion (types §7) as it was applied, and a value folded into a slot
/// as the steps up the cast tree from the value's final type to the slot's
/// (types §3, operators §5), read now since the slot may have risen after
/// the value entered it.
///
/// # Panics
/// Two different coercions at one site: every value enters its slot at its
/// own site, and the sites that two entries share (a branch with no tail
/// and a missing `else`, both `()`) coerce nothing.
fn conversions(cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo) -> FxHashMap<AstId, Box<[Conversion<Ty>]>> {
    let mut out: FxHashMap<AstId, Box<[Conversion<Ty>]>> = FxHashMap::default();
    let mut record = |site: AstId, steps: Vec<Conversion<Ty>>| {
        if steps.is_empty() {
            return;
        }
        let steps: Box<[Conversion<Ty>]> = steps.into();
        if let Some(earlier) = out.get(&site) {
            assert_eq!(earlier, &steps, "one site coerces one way");
            return;
        }
        out.insert(site, steps);
    };
    let mut applied: Vec<(AstId, Vec<Step>)> = std::mem::take(&mut cx.out.conversions)
        .into_iter()
        .map(|(site, converted)| {
            let step = Step {
                node: converted.node,
                via_ty: converted.via_ty,
                to: converted.to,
            };
            (site, vec![step])
        })
        .collect();
    for entered in std::mem::take(&mut cx.out.entered) {
        let site = entered.site;
        if let Some(steps) = climb(cx, entered) {
            applied.push((site, steps));
        }
    }
    // Each step's requirements are reaches of their own, fired now that
    // every type they read is known.
    super::decision::solve(cx);
    for (site, steps) in applied {
        let mut frozen = Vec::with_capacity(steps.len());
        let mut left_open = false;
        for Step { node, via_ty, to } in steps {
            let via = cx
                .out
                .reached
                .remove(&node)
                .expect("a coercion step is recorded in `reached` at its node");
            match (freeze(cx, types, memo, via_ty), freeze(cx, types, memo, to)) {
                (Some(via_ty), Some(to)) => frozen.push(Conversion { via, via_ty, to }),
                _ => left_open = true,
            }
        }
        if !left_open {
            record(site, frozen);
        }
    }
    out
}

/// One coercion step, its `via` reached at `node` (`decision::reach_step`),
/// before its requirements are reached and its types frozen.
struct Step {
    node: AstId,
    via_ty: Term,
    to: Term,
}

/// The steps up the cast tree from `value`'s head to `slot`'s, each with
/// the type it reaches. `None` where a refusal left either open.
///
/// # Panics
/// With no refusal made, the slot's head is not up the tree from the
/// value's: `enter` folded the value into it, so it is.
fn climb(cx: &mut Ctxt<'_>, entered: Entered) -> Option<Vec<Step>> {
    let Entered {
        span, value, slot, body, ..
    } = entered;
    let (Some(mut at), Some(to)) = (cx.closed(value), cx.closed(slot)) else {
        return None;
    };
    if at.head == Head::Base(Base::Never) {
        return Some(Vec::new());
    }
    let mut steps = Vec::new();
    while at.head != to.head {
        let Some(step) = cx.decls.casts().step_up(at.head) else {
            assert!(
                !cx.out.refusals.is_empty(),
                "a folded value's slot is up the cast tree from it, and no refusal explains otherwise"
            );
            return None;
        };
        let rule = step.rule().clone();
        let view_impl = step.view_impl();
        let reached = match super::decision::poly_head(cx.decls.types(), rule.to) == Some(to.head) {
            true => slot,
            false => {
                let scheme = Scheme {
                    ty: rule.to,
                    param_names: Box::new([]),
                    bounds: vec![TyVarBound::Any; super::decision::variables_of(cx.decls.types(), &[rule.to])].into(),
                    impls: None,
                    requires: Box::new([]),
                    facts: DeclaredFacts::default(),
                };
                let above = cx.infer.instantiate(cx.decls, &scheme).ty;
                if let Err(mismatch) = cx.infer.enter(cx.decls, at, above, crate::ty::Reach::Named) {
                    panic!("a step the tree admits takes its value up: {mismatch:?}");
                }
                above
            }
        };
        let node = AstId::alloc();
        let step = super::decision::Step {
            node,
            span,
            via: rule.via,
            view: view_impl,
            body: &body,
        };
        let via_ty = super::decision::reach_step(cx, step, at.term, reached);
        steps.push(Step { node, via_ty, to: reached });
        at = cx.closed(reached)?;
    }
    Some(steps)
}

/// Every structure tree, from each root down to the parts that reach an
/// impl: the comparisons' (operators §4), at the operator's site, and the
/// lifted impls' (types §1), at the site of the call that reached one. A
/// part node of a lifted impl that is itself lifted keeps no reach of its
/// own: its tree is its parts'. A tree some refusal left open is left out.
fn structure_trees(cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo) -> FxHashMap<AstId, StructureTree> {
    let nodes = std::mem::take(&mut cx.out.structures);
    let trees = Trees { nodes: &nodes };
    let mut out = FxHashMap::default();
    for root in trees.roots() {
        if let Some(tree) = trees.tree(cx, types, memo, root, true) {
            out.insert(root, tree);
        }
    }
    out
}

/// The structure-tree nodes while checking.
struct Trees<'n> {
    nodes: &'n FxHashMap<AstId, StructureNode>,
}

impl Trees<'_> {
    fn roots(&self) -> Vec<AstId> {
        let children: FxHashSet<AstId> = self
            .nodes
            .values()
            .flat_map(|node| match node {
                StructureNode::Parts(parts) => parts.iter().map(|(_, part)| *part).collect(),
                StructureNode::Reached | StructureNode::Own(_) => Vec::new(),
            })
            .collect();
        self.nodes.keys().copied().filter(|node| !children.contains(node)).collect()
    }

    /// The tree at `node`, a root where `root`.
    fn tree(&self, cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo, node: AstId, root: bool) -> Option<StructureTree> {
        match self.nodes.get(&node)? {
            StructureNode::Parts(parts) => {
                if !root {
                    cx.out.reached.remove(&node);
                    cx.out.callees.remove(&node);
                }
                let parts: Option<Vec<(PartKey, StructureTree)>> = parts
                    .iter()
                    .map(|(key, part)| Some((*key, self.tree(cx, types, memo, *part, false)?)))
                    .collect();
                Some(StructureTree::Parts(parts?.into()))
            }
            StructureNode::Own(own) => Some(StructureTree::Own(*own)),
            StructureNode::Reached => {
                let chosen = cx.out.reached.remove(&node);
                let callee = cx.out.callees.remove(&node);
                let callee = freeze(cx, types, memo, callee?)?;
                Some(StructureTree::Reached { chosen: chosen?, callee })
            }
        }
    }
}

/// Each instance of a generalized binding (functions §2), frozen: each
/// site of its values' walks at its type under the instance's terms, and
/// what its requirements recorded there while they fired.
fn instances(cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo) -> FxHashMap<InstanceId, InstanceDecided> {
    let mut out = FxHashMap::default();
    for at in 0..cx.out.instances.len() {
        if cx.out.instances[at].template {
            continue;
        }
        let id = InstanceId(u32::try_from(at).expect("u32 instances"));
        let (mut sites, at_instance_vars) = super::generalize::instance_sites(cx, id);
        // The binding's own type at the instance: the signature its code is
        // made at (functions §2).
        let binder = cx.out.schemes[cx.out.instances[at].scheme.0 as usize].binder;
        sites.push(binder);
        let mut copies = FxHashMap::default();
        let at_instance: Vec<(AstId, Term)> = sites
            .into_iter()
            .filter_map(|site| cx.out.types.get(&site).map(|term| (site, *term)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(site, term)| (site, cx.infer.copy_at(term, &at_instance_vars, &mut copies)))
            .collect();
        let super::generalize::Instance { use_site, within, .. } = cx.out.instances[at].clone();
        let decided = super::generalize::in_instance(cx, id, |cx| {
            let coercions = coercions(cx, types, memo);
            let mut site_types = FxHashMap::default();
            let recorded: Vec<(AstId, Term)> = cx.out.types.iter().map(|(site, term)| (*site, *term)).collect();
            for (site, term) in at_instance.into_iter().chain(recorded) {
                if let Some(ty) = freeze(cx, types, memo, term) {
                    site_types.insert(site, ty);
                }
            }
            let tries = std::mem::take(&mut cx.out.tries)
                .into_iter()
                .filter_map(|(site, exit)| {
                    let exit = match exit {
                        super::output::TryExit::Option(term) => super::output::TryExit::Option(freeze(cx, types, memo, term)?),
                        super::output::TryExit::Result(term) => super::output::TryExit::Result(freeze(cx, types, memo, term)?),
                    };
                    Some((site, exit))
                })
                .collect();
            let structure_trees = structure_trees(cx, types, memo);
            let callees = std::mem::take(&mut cx.out.callees)
                .into_iter()
                .filter_map(|(site, callee)| Some((site, freeze(cx, types, memo, callee)?)))
                .collect();
            InstanceDecided {
                binder: Some(binder),
                use_site: Some(use_site),
                within,
                types: site_types,
                reached: std::mem::take(&mut cx.out.reached),
                callees,
                iteration: std::mem::take(&mut cx.out.iteration),
                tries,
                coercions,
                casts: std::mem::take(&mut cx.out.casts),
                adjustments: std::mem::take(&mut cx.out.adjustments),
                structure_trees,
            }
        });
        out.insert(id, decided);
    }
    out
}

/// Each site a generalized binding's value passes through where it is
/// written: recorded by one of the binding's value walks, the `let`'s or an
/// assignment's, at a term one with the binding's own, and the binder.
fn value_sites(cx: &Ctxt<'_>) -> FxHashMap<AstId, AstId> {
    let mut out = FxHashMap::default();
    for scheme in cx.out.schemes.iter().filter(|scheme| scheme.generalizes) {
        let own = cx.infer.shallow(scheme.term);
        out.insert(scheme.binder, scheme.binder);
        for (site, owner) in &cx.out.site_owners {
            let carries = *owner == scheme.binder && cx.out.types.get(site).is_some_and(|term| cx.infer.shallow(*term) == own);
            if carries {
                out.insert(*site, scheme.binder);
            }
        }
    }
    out
}
