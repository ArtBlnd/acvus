//! An `Instance` calls the receiver it owns and no other: its call takes
//! no receiver argument, so a handler holding another value of any type,
//! its own variable's included, has no way to hand it to the word the
//! checker chose for the owned one (RFC-0067 rule 4).
use std::ops::Deref;

use acvus_extern::{Ctx, Instance, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "g", fn advance<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

#[extern_fn(effect = pure)]
fn drive<I, J, Rt>(ctx: &mut Ctx<'_, Rt>, it: Instance<'_, advance<I, Rt>, I, Rt>, other: J) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    J: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut it = it;
    let mut other = other;
    it.call(ctx, &mut other, ())
}

fn main() {}
