//! Effects (effects §1): a reached call's effect is its declaration's, and
//! the body's is their join (types §2).

use acvus_ast::Span;

use super::ctxt::Ctxt;
use super::refusal::{self, Exceeding, RefusalKind};
use crate::infer::{EffectRefused, Exceeds};
use crate::ty::{Effect, EffectTerm, Infer};

/// A call's effect flows into the innermost body's: a lambda's, or the
/// run's (effects §1). The flow is one way: what the call's effect is, or
/// later rises to, raises the body's, and the body's never reaches the
/// callee. A lambda called in a body so keeps its own effect.
pub(super) fn note_call_effect(cx: &mut Ctxt<'_>, effect: &EffectTerm<Infer>, span: Span) {
    let body = body_effect(cx);
    flows_into(cx, effect, &body, span);
}

/// `into ⊒ from` (effects §1), its refusal, if any, made at `span` or at
/// the lambda whose effect `into` is (`refuse_effects_exceeded`).
pub(super) fn flows_into(cx: &mut Ctxt<'_>, from: &EffectTerm<Infer>, into: &EffectTerm<Infer>, span: Span) {
    let flow = cx.infer.flows_into(from, into);
    cx.out.effect_flows.insert(flow, span);
}

/// Every effect flow B refused since the last call, refused here (D94):
/// at the lambda whose effect would grow past what a slot fixed, or, where
/// no lambda's would, at the call whose effect flows.
pub(super) fn refuse_effects_exceeded(cx: &mut Ctxt<'_>) {
    for EffectRefused { flow, into, exceeds } in cx.infer.take_effects_refused() {
        let call = *cx.out.effect_flows.get(&flow).expect("every effect flow is recorded with its span");
        refuse_exceeds(cx, &into, exceeds, call);
    }
}

/// `effect` joined into the call effect `call` (types §2), refused where
/// `call` is fixed and would grow (D94).
pub(super) fn join_into(cx: &mut Ctxt<'_>, call: &EffectTerm<Infer>, effect: &EffectTerm<Infer>, span: Span) -> EffectTerm<Infer> {
    match cx.infer.join_effects(call, effect) {
        Ok(joined) => joined,
        Err(exceeds) => {
            refuse_exceeds(cx, call, exceeds, span);
            call.clone()
        }
    }
}

fn refuse_exceeds(cx: &mut Ctxt<'_>, grown: &EffectTerm<Infer>, Exceeds { fixed, needs }: Exceeds<Effect>, span: Span) {
    let at = cx
        .out
        .lambdas
        .iter()
        .find(|lambda| cx.infer.one_effect_variable(&lambda.effect, grown))
        .map_or(span, |lambda| lambda.span);
    refusal::push(cx, RefusalKind::ClosureExceedsSlot(Exceeding::Effect { needs, slot: fixed }), at);
}

/// The innermost body's effect as a term a later call can join into: a
/// body that has made no call yet is given one, pure until a call joins it
/// (effects §1). A decision that reaches a declaration only once it fires
/// carries this, so the call joins the body it was written in.
pub(super) fn body_effect(cx: &mut Ctxt<'_>) -> EffectTerm<Infer> {
    let current = match cx.scope.lambda_effect() {
        Some(lambda) => lambda.clone(),
        None => cx.out.body_effect.clone(),
    };
    if let Some(current) = current {
        return current;
    }
    let fresh = EffectTerm::Var(cx.infer.fresh_effect());
    let pure = cx
        .infer
        .join_effects(&EffectTerm::Known(Effect::pure()), &fresh)
        .expect("a fresh effect variable is fixed by no `eq`");
    match cx.scope.lambda_effect() {
        Some(lambda) => *lambda = Some(pure.clone()),
        None => cx.out.body_effect = Some(pure.clone()),
    }
    pure
}
