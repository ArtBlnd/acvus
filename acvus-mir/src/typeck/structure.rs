//! S, the structure rule (types §1, §3): a transparent structured type
//! taken apart by its kind's shape down to the parts that are not one, and
//! a rule applied there. Two rules reach it from typeck: a value coerced to
//! a slot written elsewhere (`fire_coerced`), and a generic function's
//! impl a structure has by its lift (`lift_at`, `reach_lifted`). Each part
//! is a node of its own, decided once its head is closed, as a comparison's
//! parts are (`op::fire_compare`).

use acvus_ast::{AstId, Span};
use acvus_utils::QualifiedRef;

use super::ctxt::Ctxt;
use super::decision::{Callee, Decision, Reach, Reading, closed_for, core_name, push_required, settle_bounded};
use super::output::{Chosen, CoercionNode, Impl, Own, StructureNode};
use super::refusal::{self, RefusalKind, Unimplemented};
use crate::decl::{Base, Head, Lift, LiftShape, NamedLaw, Taking};
use crate::infer::{Closed, Coerced, Instantiated, RequiredAt, Term};
use crate::ty::structural::{PartKey, Structure, structure};
use crate::ty::{Effect, EffectTerm, Infer as InferPhase, TyKind, copies_in};

/// A structured value coerced at `site` into `slot` (types §3): its node
/// records the shape and a node per part it keeps, and each part enters
/// the slot's part at that node, in the body whose effect is `body`.
pub(super) fn fire_coerced(cx: &mut Ctxt<'_>, site: AstId, span: Span, slot: Term, coerced: Coerced, body: EffectTerm<InferPhase>) {
    let Coerced { shape, through, parts } = coerced;
    let sites: Vec<(PartKey, AstId)> = parts.iter().map(|part| (part.key, AstId::alloc())).collect();
    cx.out.coercions.insert(site, CoercionNode { shape, through, slot, parts: sites.clone().into() });
    for (part, (_, part_site)) in parts.iter().zip(sites) {
        cx.enter_in(body.clone(), part_site, part.value, part.slot, span);
    }
}

/// Where a reach's callee has a lift and the lift's type variable stands
/// at a transparent structured type, closed as `reading` admits: the
/// generic function, its lift, and that type.
pub(super) struct LiftAt {
    pub function: QualifiedRef,
    pub lift: Lift,
    pub structured: Term,
}

/// The lift that answers `decision`, where no concrete impl did (types §1):
/// its callee's generic function has one, and the arguments at the lift's
/// type variable are closed at one transparent structured type.
pub(super) fn lift_at(cx: &Ctxt<'_>, decision: &Reach, reading: Reading) -> Option<LiftAt> {
    let CalleeLift { function, lift } = lift_of_callee(cx, &decision.callee, decision.args.len())?;
    let at = closed_for(cx, variable_position(cx, decision, lift.shape(), reading)?, reading)?;
    let structured = structure(cx.infer.types(), at.term)?;
    lifts_at(lift, &structured).then(|| LiftAt { function, lift: lift.clone(), structured: at.term })
}

/// Whether `lift` gives an impl at a structure of `structured`'s kind. An
/// enum, `Option` and `Result` hold one variant, so a rebuilt one's
/// variant must be its one argument's (types §1); and they have no order,
/// so no total order combines at one (operators §4).
fn lifts_at(lift: &Lift, structured: &Structure<'_, InferPhase>) -> bool {
    let sum = matches!(structured, Structure::Enum(..) | Structure::Option(..) | Structure::Result { .. });
    match lift {
        Lift::Rebuild(shape) => !sum || shape.params.len() == 1,
        Lift::Combine { law: NamedLaw::TotalOrder, .. } => !sum,
        Lift::Combine { law: NamedLaw::Equivalence, .. } => true,
    }
}

/// The language's own answer, where it defines one for `decision`'s
/// generic function at its closed type (ownership §2, operators §2): a
/// `clone` of a value that copies is its copy, and `Option`'s `default` is
/// `None`. It answers before any impl.
pub(super) struct OwnAt {
    pub function: QualifiedRef,
    pub own: Own,
    pub at: Term,
}

pub(super) fn own_at(cx: &Ctxt<'_>, decision: &Reach, reading: Reading) -> Option<OwnAt> {
    let CalleeLift { function, lift } = lift_of_callee(cx, &decision.callee, decision.args.len())?;
    let core = cx.decls.core();
    let at = closed_for(cx, variable_position(cx, decision, lift.shape(), reading)?, reading)?.term;
    let own = if function == core.clone && copies_in(&cx.infer, at) == Some(true) {
        Own::Copy
    } else if function == core.default && matches!(cx.infer.types().kind(at), TyKind::Option(..)) {
        Own::OptionNone
    } else {
        return None;
    };
    Some(OwnAt { function, own, at })
}

/// `decision` answered by the language's own answer `at`: the call is
/// reached as a lifted one, and its node records the answer.
pub(super) fn reach_own(cx: &mut Ctxt<'_>, decision: Reach, at: OwnAt) -> Result<(), Reach> {
    let OwnAt { function, own, at } = at;
    let shape = cx.decls.lift(function).expect("an own answer is of a function with a lift").shape().clone();
    if let Some(Structural { node, .. }) = reach_structural(cx, decision, function, &shape, at) {
        cx.out.structures.insert(node, StructureNode::Own(own));
    }
    Ok(())
}

/// Whether `decision` waits for the argument at its callee's lift's type
/// variable before a generic impl may answer it (types §6): its callee has
/// a lift that could answer, and that argument is not closed yet.
pub(super) fn lift_waits(cx: &Ctxt<'_>, decision: &Reach) -> bool {
    let Some(CalleeLift { lift, .. }) = lift_of_callee(cx, &decision.callee, decision.args.len()) else {
        return false;
    };
    let taking = match lift.shape().params.is_empty() {
        true => decision.result,
        false => decision.args[0].1,
    };
    cx.closed(taking).is_none()
}

/// The head a method's receiver takes where `callee`'s lift or the
/// language's own answer answers a call of `arity` arguments whose
/// receiver is `receiver`: the lift's first parameter at the receiver's
/// transparent structured type, or at a type whose value copies where the
/// callee is `clone` (ownership §2), which the receiver adjustment reads
/// as any candidate's (types §6).
pub(super) fn lifted_receiver(cx: &Ctxt<'_>, callee: &Callee, arity: usize, receiver: Closed) -> Option<Head> {
    let CalleeLift { function, lift } = lift_of_callee(cx, callee, arity)?;
    let (Head::Base(base) | Head::Ref(_, base)) = receiver.head;
    let value = match receiver.head {
        Head::Base(_) => receiver.term,
        Head::Ref(..) => cx.referent(receiver.term),
    };
    let structured = matches!(base, Base::Tuple(_) | Base::Array | Base::Option | Base::Result | Base::Object | Base::Enum(_));
    let copied = function == cx.decls.core().clone && copies_in(&cx.infer, value) == Some(true);
    match lift.shape().params.first() {
        Some(Taking::Value) if structured || copied => Some(Head::Base(base)),
        Some(&Taking::Ref(mutability)) if structured || copied => Some(Head::Ref(mutability, base)),
        Some(_) | None => None,
    }
}

/// The generic function a reach's callee names, and its lift.
struct CalleeLift<'d> {
    function: QualifiedRef,
    lift: &'d Lift,
}

/// The generic function `callee` names that has a lift, where the lift can
/// answer a call of `arity` arguments: the arity is the lift's, and a law
/// the callee names is the one the lift states.
fn lift_of_callee<'d>(cx: &Ctxt<'d>, callee: &Callee, arity: usize) -> Option<CalleeLift<'d>> {
    let (function, named_law) = match callee {
        Callee::Core(sig) => (core_name(cx, *sig), None),
        Callee::Declaration(declaration) => (*declaration, None),
        Callee::Required(required) | Callee::Part(required) => (required.generic_function, required.law),
        Callee::Name(name) | Callee::Method(name) => {
            let lifted: Vec<QualifiedRef> =
                cx.decls.overloads(*name).iter().copied().filter(|declaration| cx.decls.lift(*declaration).is_some()).collect();
            let [function] = lifted[..] else {
                return None;
            };
            (function, None)
        }
    };
    let lift = cx.decls.lift(function)?;
    if lift.shape().params.len() != arity {
        return None;
    }
    let states_named = match (named_law, lift) {
        (None, _) => true,
        (Some(named), Lift::Combine { law, .. }) => named == *law,
        (Some(_), Lift::Rebuild(_)) => false,
    };
    states_named.then_some(CalleeLift { function, lift })
}

/// The term the lift's type variable stands at in `decision`: the first
/// argument, through the reference it is where it takes the variable by
/// one; the result where no parameter takes it, as a `Lift::Rebuild` of no
/// parameter. `None` where that argument is not closed yet, or is no
/// reference where one is taken.
fn variable_position(cx: &Ctxt<'_>, decision: &Reach, shape: &LiftShape, reading: Reading) -> Option<Term> {
    let Some(taking) = shape.params.first() else {
        return Some(decision.result);
    };
    let (_, arg) = decision.args[0];
    match taking {
        Taking::Value => Some(arg),
        Taking::Ref(_) => {
            let closed = closed_for(cx, arg, reading)?;
            matches!(cx.infer.types().kind(closed.term), TyKind::Ref(..)).then(|| cx.referent(closed.term))
        }
    }
}

/// `decision` answered by the lifted impl `at` (types §1): the call reaches
/// `Impl::Structural`, its arguments enter the generic function's
/// parameters at the structured type, and each part is a `Callee::Part`
/// reach at a node of its own, stating the lift's law where it combines,
/// recorded as a `StructureNode` at the call's site. The lifted impl's
/// effect is its parts' impls' joined.
///
/// Each part's type is read to decide whether it copies (ownership §2)
/// and what its pattern is, so the reach waits until every part is closed
/// (types §6), unless a cycle is being broken.
pub(super) fn reach_lifted(cx: &mut Ctxt<'_>, decision: Reach, at: LiftAt, reading: Reading) -> Result<(), Reach> {
    let span = decision.span;
    let LiftAt { function, lift, structured } = at;
    let parts = structure(cx.infer.types(), structured)
        .expect("a lift answers only at a transparent structured type")
        .parts();
    if reading == Reading::Closed && parts.iter().any(|part| cx.closed(part.ty).is_none()) {
        return Err(decision);
    }
    let Some(Structural { node, effect }) = reach_structural(cx, decision, function, lift.shape(), structured) else {
        return Ok(());
    };
    let law = match &lift {
        Lift::Rebuild(_) => None,
        Lift::Combine { law, .. } => Some(*law),
    };
    let nodes: Vec<(PartKey, AstId)> = parts.iter().map(|part| (part.key, AstId::alloc())).collect();
    cx.out.structures.insert(node, StructureNode::Parts(nodes.clone().into()));
    for (part, (_, part_node)) in parts.into_iter().zip(nodes) {
        let reach = PartReach { node: part_node, span, function, shape: lift.shape(), law, part: part.ty, effect: &effect };
        reach_part(cx, reach);
    }
    Ok(())
}

/// A call answered at `Impl::Structural`: the node its structure is
/// recorded at, and the effect its parts' impls join.
struct Structural {
    node: AstId,
    effect: EffectTerm<InferPhase>,
}

/// `decision` reached at `Impl::Structural` of `function` at `at`: its
/// arguments enter the parameters, or, for a part, the pattern joins, and
/// the call's effect is the answer's. A requirement answered so is
/// refused as unimplemented: no declared function stands for it.
fn reach_structural(cx: &mut Ctxt<'_>, decision: Reach, function: QualifiedRef, shape: &LiftShape, at: Term) -> Option<Structural> {
    let Reach { site, span, callee, args, result, effect } = decision;
    if let Callee::Required(_) = callee {
        let feature = Unimplemented::LiftedImplAsRequirement;
        refusal::push(cx, RefusalKind::Unimplemented { feature }, span);
        return None;
    }
    let Instantiated { ty, requirements } = at_type(cx, function, shape, at, span)?;
    let TyKind::Fn { params, ret, effect: own_effect, .. } = cx.infer.types().kind(cx.infer.shallow(ty)).clone() else {
        panic!("an instantiated signature is a function type (NotAFunction)");
    };
    let own_effect = super::effect::join_into(cx, &own_effect, &EffectTerm::Known(Effect::pure()), span);
    match &callee {
        Callee::Part(part) => cx.join(part.pattern, ty, span),
        Callee::Required(_) => unreachable!("a requirement is refused above"),
        Callee::Name(_) | Callee::Declaration(_) | Callee::Method(_) | Callee::Core(_) => {
            for ((arg_site, arg), param) in args.iter().zip(params.iter()) {
                cx.enter_in(effect.clone(), *arg_site, *arg, *param, span);
            }
            cx.eq(result, ret, span);
        }
    }
    super::effect::join_into(cx, &effect, &own_effect, span);
    let chosen = Chosen { declaration: function, r#impl: Some(Impl::Structural), required: Box::new([]) };
    cx.out.reached.insert(site, chosen);
    cx.out.callees.insert(site, ty);
    settle_bounded(cx, ty, span);
    for required in requirements {
        push_required(cx, site, span, required, &effect);
    }
    Some(Structural { node: site, effect: own_effect })
}

/// One part's reach of a lifted impl: `function` at the part's type, at
/// `node`, stating `law` where named; its impl's effect joins `effect`.
struct PartReach<'l> {
    node: AstId,
    span: Span,
    function: QualifiedRef,
    shape: &'l LiftShape,
    law: Option<NamedLaw>,
    part: Term,
    effect: &'l EffectTerm<InferPhase>,
}

fn reach_part(cx: &mut Ctxt<'_>, reach: PartReach<'_>) {
    let PartReach { node, span, function, shape, law, part, effect } = reach;
    if function == cx.decls.core().clone && copies_in(&cx.infer, part) == Some(true) {
        cx.out.structures.insert(node, StructureNode::Own(Own::Copy));
        return;
    }
    let Some(Instantiated { ty: pattern, .. }) = at_type(cx, function, shape, part, span) else {
        return;
    };
    let TyKind::Fn { params, ret, .. } = cx.infer.types().kind(cx.infer.shallow(pattern)).clone() else {
        panic!("an instantiated signature is a function type (NotAFunction)");
    };
    let args = params.iter().map(|param| (node, *param)).collect();
    // The part's answer is the pattern's: where no parameter takes the
    // part, as `default`'s, the result is where its type is read.
    let result = ret;
    let required = RequiredAt { generic_function: function, pattern, law };
    cx.decisions.push(Decision::Reach { site: node, span, callee: Callee::Part(required), args, result, effect: effect.clone() });
}

/// `function`'s signature at fresh variables, its lift's type variable made
/// `ty`, with the requirements it states. `None` where `ty` cannot stand
/// there, which is refused: a reference where a parameter takes the
/// variable by reference (types §4).
fn at_type(cx: &mut Ctxt<'_>, function: QualifiedRef, shape: &LiftShape, ty: Term, span: Span) -> Option<Instantiated> {
    let scheme = cx.decls.scheme(function).expect("a lifted function is declared");
    let instantiated = cx.infer.instantiate(cx.decls, scheme);
    let TyKind::Fn { params, ret, .. } = cx.infer.types().kind(cx.infer.shallow(instantiated.ty)).clone() else {
        panic!("an instantiated signature is a function type (NotAFunction)");
    };
    let variable = match shape.params.first() {
        None => ret,
        Some(Taking::Value) => params[0],
        Some(Taking::Ref(_)) => cx.referent(params[0]),
    };
    if let Err(mismatch) = cx.infer.eq(variable, ty) {
        refusal::mismatch(cx, mismatch, span);
        return None;
    }
    Some(instantiated)
}
