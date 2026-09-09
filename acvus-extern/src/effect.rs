//! Effect positions in ExternFn signatures.
//!
//! `EffectArg` names an effect: a known level or the K-th effect variable.
//! `EffectVar` is the bound of a generic parameter that is an effect
//! variable; `()` fills it at runtime.

use crate::ty_arg::PolyVars;
use acvus_mir::ty::{Effect, EffectTerm, Poly};

pub trait EffectArg: Send + Sync + 'static {
    fn poly_effect(vars: &PolyVars) -> EffectTerm<Poly>;
}

pub trait EffectVar: Send + Sync + 'static {}

impl<E> EffectVar for E where E: EffectArg {}
impl EffectVar for () {}

/// The K-th effect variable of a declaration. Uninhabited.
pub enum Eff<const K: usize> {}

impl<const K: usize> EffectArg for Eff<K> {
    fn poly_effect(vars: &PolyVars) -> EffectTerm<Poly> {
        vars.effects[K]
    }
}

pub struct Pure;
pub struct Idempotent;
pub struct Opaque;

impl EffectArg for Pure {
    fn poly_effect(_: &PolyVars) -> EffectTerm<Poly> {
        EffectTerm::Known(Effect::Pure)
    }
}

impl EffectArg for Idempotent {
    fn poly_effect(_: &PolyVars) -> EffectTerm<Poly> {
        EffectTerm::Known(Effect::Idempotent)
    }
}

impl EffectArg for Opaque {
    fn poly_effect(_: &PolyVars) -> EffectTerm<Poly> {
        EffectTerm::Known(Effect::Opaque)
    }
}
