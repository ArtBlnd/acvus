//! Every term to its concrete type, once (compilation); the lambdas' claimed
//! flows into their function types.

use acvus_ast::AstId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::ctxt::Ctxt;
use super::output::{Chosen, Closure, Comparing, Comparison, Conversion, Impl, Typeck};
use crate::decl::{Base, Head, Scheme, TyVarBound};
use crate::infer::{FreezeMemo, Term, Unsettled};
use crate::ty::{
    Alignment, Calls, Concrete, Effect, EffectTerm, Flow, FlowEnd, FlowTerm, Flows, Holds, Infer as InferPhase,
    Mutability, Ty, TyKind, Types,
};

pub(super) fn finish(mut cx: Ctxt<'_>, types: &mut Types<Concrete>) -> Typeck {
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
    let callees = std::mem::take(&mut cx.out.callees)
        .into_iter()
        .filter_map(|(site, callee)| Some((site, freeze(&mut cx, types, &mut memo, callee)?)))
        .collect();
    let conversions = conversions(&mut cx, types, &mut memo);
    let comparisons = comparisons(&mut cx);
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
        conversions,
        casts: std::mem::take(&mut cx.out.casts),
        closures,
        comparisons,
        body_effect,
        refusals: std::mem::take(&mut cx.out.refusals),
    }
}

/// `term` frozen, each open flow set first. `None` only where a refusal
/// already left a variable open; with no refusal that is the checker's
/// defect.
///
/// # Panics
/// A variable other than a flow set is open and no refusal was made:
/// solving binds every type variable and, at its last stall, every effect
/// and mark (types §5), and every length and identity is fixed by the term
/// that made it.
fn freeze(cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo, term: Term) -> Option<Ty> {
    loop {
        match cx.infer.freeze(types, memo, term) {
            Ok(ty) => return Some(ty),
            Err(Unsettled::Flow(var)) => set_flows(cx, FlowTerm::Var(var), unclaimed_flows()),
            Err(open) => {
                assert!(
                    !cx.out.refusals.is_empty(),
                    "{open:?} is open after solving, and no refusal explains it"
                );
                return None;
            }
        }
    }
}

/// The flows of a function value nothing ever gave, such as a parameter
/// of a lambda never called: no lambda claimed them and no declaration
/// states them.
///
/// PARKED(P9): types §5 states no default for flows. Provisionally every
/// output takes from every input, what a declaration stating nothing gets
/// (extern-fn §4), which is sound; this function is the one place that
/// decides it.
fn unclaimed_flows() -> Flows {
    Flows::Every
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
    cx.node(TyKind::Fn {
        params: Box::new([]),
        ret,
        effect,
        flows,
        calls: Calls::Many,
        holds: Holds::Nothing,
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
/// left.
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
                held.exclusive |= *holds == Holds::Mut;
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
    for (site, conversion) in std::mem::take(&mut cx.out.conversions) {
        if let Some(to) = freeze(cx, types, memo, conversion.to) {
            record(site, vec![Conversion { via: conversion.via, to }]);
        }
    }
    for entered in std::mem::take(&mut cx.out.entered) {
        if let Some(steps) = climb(cx, types, memo, entered.value, entered.slot) {
            record(entered.site, steps);
        }
    }
    out
}

/// The steps up the cast tree from `value`'s head to `slot`'s, each with
/// the type it reaches. `None` where a refusal left either open.
///
/// # Panics
/// With no refusal made, the slot's head is not up the tree from the
/// value's: `enter` folded the value into it, so it is.
fn climb(cx: &mut Ctxt<'_>, types: &mut Types<Concrete>, memo: &mut FreezeMemo, value: Term, slot: Term) -> Option<Vec<Conversion<Ty>>> {
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
        let r#impl = step.view_impl().map(Impl::Concrete);
        let reached = match super::decision::poly_head(cx.decls.types(), rule.to) == Some(to.head) {
            true => slot,
            false => {
                let scheme = Scheme {
                    ty: rule.to,
                    param_names: Box::new([]),
                    bounds: vec![TyVarBound::Any; super::decision::variables_of(cx.decls.types(), &[rule.to])].into(),
                    impls: None,
                    requires: Box::new([]),
                };
                let above = cx.infer.instantiate(cx.decls, &scheme).ty;
                if let Err(mismatch) = cx.infer.enter(cx.decls, at, above) {
                    panic!("a step the tree admits takes its value up: {mismatch:?}");
                }
                above
            }
        };
        let via = Chosen {
            declaration: rule.via,
            r#impl,
            required: Box::new([]),
        };
        steps.push(Conversion {
            via,
            to: freeze(cx, types, memo, reached)?,
        });
        at = cx.closed(reached)?;
    }
    Some(steps)
}

/// Every comparison tree, from each root the program wrote down to the
/// parts that reach `core::eq` or `core::cmp` (operators §4). A tree some
/// refusal left open is left out.
fn comparisons(cx: &mut Ctxt<'_>) -> FxHashMap<AstId, Comparison> {
    let nodes = std::mem::take(&mut cx.out.comparisons);
    let children: FxHashSet<AstId> = nodes
        .values()
        .flat_map(|node| match node {
            Comparing::Parts(parts) => parts.to_vec(),
            Comparing::Reached => Vec::new(),
        })
        .collect();
    let roots: Vec<AstId> = nodes.keys().copied().filter(|node| !children.contains(node)).collect();
    let mut out = FxHashMap::default();
    for root in roots {
        if let Some(tree) = comparison(cx, &nodes, root) {
            out.insert(root, tree);
        }
    }
    out
}

fn comparison(cx: &mut Ctxt<'_>, nodes: &FxHashMap<AstId, Comparing>, node: AstId) -> Option<Comparison> {
    match nodes.get(&node)? {
        Comparing::Parts(parts) => {
            let parts: Option<Vec<Comparison>> = parts.iter().map(|part| comparison(cx, nodes, *part)).collect();
            Some(Comparison::Parts(parts?.into()))
        }
        Comparing::Reached => cx.out.reached.remove(&node).map(Comparison::Reached),
    }
}
