//! An instance carries what it requires in a field of its own payload,
//! laid at construction (RFC-0067 Decision 1), so it takes no `Instance`
//! parameter. The refused item is never emitted, so every name it uses
//! is unused.
#![allow(unused_imports)]
use std::ops::DerefMut;

use acvus_extern::{Ctx, Instance, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "g", fn advance<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

#[extern_fn(instance_of = advance, effect = pure)]
fn advance_twice<I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut I,
    inner: Instance<advance<I, Rt>, I, Rt>,
) -> i64
where
    I: Var<kind::Type> + acvus_extern::Borrowable<Rt> + DerefMut<Target = Rt::Value>,
    Rt: Runtime,
{
    // SAFETY: the instance stands at the type the reference names, which
    // is what `Externs::combine` would have met the requirement with.
    unsafe { inner.call(ctx, it, ()) }
}

fn main() {}
