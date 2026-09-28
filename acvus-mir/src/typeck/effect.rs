//! Effects (effects §1): a reached call's effect is its declaration's, and
//! the body's is their join (types §2).

use super::ctxt::Ctxt;
use crate::ty::{Effect, EffectTerm, Infer};

/// Joins a call's effect into the innermost body's: a lambda's, or the
/// run's.
///
/// B joins two effect variables by linking them, so a lambda called
/// in a body shares that body's effect variable from then on: the
/// lambda's type carries the body's other calls' effects too. That is
/// sound, since a larger effect only keeps more order (effects), but it is
/// coarser than the lambda's own.
pub(super) fn note_call_effect(cx: &mut Ctxt<'_>, effect: &EffectTerm<Infer>) {
    let body = match cx.scope.lambda_effect() {
        Some(lambda) => lambda.clone(),
        None => cx.out.body_effect.clone(),
    };
    let joined = match body {
        Some(body) => cx.infer.join_effects(&body, effect),
        None => effect.clone(),
    };
    match cx.scope.lambda_effect() {
        Some(lambda) => *lambda = Some(joined),
        None => cx.out.body_effect = Some(joined),
    }
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
    let pure = cx.infer.join_effects(&EffectTerm::Known(Effect::pure()), &fresh);
    match cx.scope.lambda_effect() {
        Some(lambda) => *lambda = Some(pure.clone()),
        None => cx.out.body_effect = Some(pure.clone()),
    }
    pure
}
