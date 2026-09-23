//! Effect positions in ExternFn signatures.
//!
//! A `Term<kind::Effect>` names an effect: a known level, or the K-th
//! effect variable of a declaration as `Nth<kind::Effect, K>`.

use crate::ty_arg::{PolyVars, Term, Var, kind};
use acvus_mir::ty::{Effect, EffectTerm, Poly};

/// The runtime carries no effect: an effect variable is settled before the
/// handler runs, so the runtime fills it with nothing.
impl Var<kind::Effect> for () {}

/// The bound of an effect variable whose task is `Async` or `Heavy`
/// (RFC-0011 rule 5). `#[extern_fn]` declares it as the variable's
/// `EffectVarBound::Suspends`; the checker verifies it when the variable
/// freezes, and the Rust type filling the variable carries nothing to check.
pub trait Suspends: Var<kind::Effect> {}

impl Suspends for () {}

/// An effect variable's run-time instantiation, `()`, at which a borrowed
/// container naming it is read in place. The glue instantiates every effect
/// variable at `()`, so a map or a set a script holds is a box of the Rust
/// type at `()`, and a borrow at a known level reads a box of another type.
/// This is `InPlaceElement`'s counterpart for the effect kind; a length and
/// an identity need none, because the stand-ins `Nth` and `()` are the only
/// Rust types of either kind, so a declaration names them only as variables.
#[diagnostic::on_unimplemented(
    message = "a borrowed container at the effect `{Self}` has no storage of that type: the runtime keeps it at its effect variable's run-time instantiation",
    label = "this parameter borrows a container at a known effect",
    note = "borrow it at the declaration's own effect variable, as `&HashMap<K, V, E, Rt>` with `E: Var<kind::Effect>`: every effect variable is `()` when the handler runs."
)]
pub trait InPlaceEffect: Var<kind::Effect> + crate::obj::sealed::Sealed {}

impl crate::obj::sealed::Sealed for () {}

impl InPlaceEffect for () {}

pub struct Pure;
pub struct Idempotent;
pub struct Opaque;

/// A known level fills the effect position of a declaration's type where a
/// variable of the effect kind would stand, and the three levels stand
/// alike: a declaration or a closure parameter may name any of them.
impl Var<kind::Effect> for Pure {}
impl Var<kind::Effect> for Idempotent {}
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
