//! Effects (effects §1): a reached call's effect is its declaration's, and
//! the body's is their join (types §2).

use super::ctxt::Ctxt;
use crate::ty::{EffectTerm, Infer};

/// Joins a call's effect into the innermost body's: a lambda's, or the
/// run's.
///
/// NOTE: B joins two effect variables by linking them, so a lambda called
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
