//! Effects (effects §1): a reached call's effect is its declaration's, and
//! the body's is their join (types §2).

use super::ctxt::Ctxt;
use crate::ty::{EffectTerm, Infer};

/// Joins a call's effect into the body's.
pub(super) fn note_call_effect(cx: &mut Ctxt<'_>, effect: &EffectTerm<Infer>) {
    todo!("typeck::effect::note_call_effect")
}
