//! Effect positions in ExternFn signatures.
//!
//! A `Term<kind::Effect>` names an effect: a known level, or the K-th
//! effect variable of a declaration as `Nth<kind::Effect, K>`.

use crate::ty_arg::{PolyVars, Term, Var, kind};
use acvus_mir::ty::{Effect, EffectTerm, Poly};

/// The runtime carries no effect: an effect variable is settled before the
/// handler runs, so the runtime fills it with nothing.
impl Var<kind::Effect> for () {}

pub struct Pure;
pub struct Idempotent;
pub struct Opaque;

/// A known level fills the effect position of a declaration's type where a
/// variable of the effect kind would stand. `Idempotent` has no such impl:
/// no declaration stands one there, and the build says so.
impl Var<kind::Effect> for Pure {}
impl Var<kind::Effect> for Opaque {}

impl Term<kind::Effect> for Pure {
    fn poly(_: &PolyVars) -> EffectTerm<Poly> {
        EffectTerm::Known(Effect::PURE)
    }
}

impl Term<kind::Effect> for Idempotent {
    fn poly(_: &PolyVars) -> EffectTerm<Poly> {
        EffectTerm::Known(Effect::IDEMPOTENT)
    }
}

impl Term<kind::Effect> for Opaque {
    fn poly(_: &PolyVars) -> EffectTerm<Poly> {
        EffectTerm::Known(Effect::OPAQUE)
    }
}
