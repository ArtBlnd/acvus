//! Effect positions in ExternFn signatures.
//!
//! A `Term<kind::Effect>` names an effect: a known level, or the K-th
//! effect variable of a declaration as `Nth<kind::Effect, K>`.

use crate::canonical::Canonical;
use crate::ty_arg::{PolyVars, Term, Var, kind};
use acvus_mir::ty::{Effect, EffectTerm, Poly};

/// The runtime carries no effect: an effect variable is settled before the
/// handler runs, so the runtime fills it with nothing.
impl Var<kind::Effect> for () {}

// SAFETY: an effect holds no `Erased`.
unsafe impl Canonical<kind::Effect> for () {
    type Canon = Self;
}

/// The bound of an effect variable whose task is `Async` or `Heavy`
/// (RFC-0011 rule 5). `#[extern_fn]` declares it as the variable's
/// `EffectVarBound::Suspends`; the checker verifies it when the variable
/// freezes, and the Rust type filling the variable carries nothing to check.
pub trait Suspends: Var<kind::Effect> {}

impl Suspends for () {}

pub struct Pure;
pub struct Idempotent;
pub struct Opaque;

/// A known level fills the effect position of a declaration's type where a
/// variable of the effect kind would stand, and the three levels stand
/// alike: a declaration or a closure parameter may name any of them.
impl Var<kind::Effect> for Pure {}
impl Var<kind::Effect> for Idempotent {}
impl Var<kind::Effect> for Opaque {}

// SAFETY: an effect holds no `Erased`.
unsafe impl Canonical<kind::Effect> for Pure {
    type Canon = Self;
}
// SAFETY: as above.
unsafe impl Canonical<kind::Effect> for Idempotent {
    type Canon = Self;
}
// SAFETY: as above.
unsafe impl Canonical<kind::Effect> for Opaque {
    type Canon = Self;
}

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
