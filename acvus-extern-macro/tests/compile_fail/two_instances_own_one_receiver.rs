//! Two `Instance`s cannot own one receiver: a declaration that requires two
//! signatures stepping the value at `I` has one value to give (RFC-0067
//! rule 1).
use std::ops::Deref;

use acvus_extern::{Ctx, Instance, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "g", fn advance<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }
extern_signature! { ns: "g", fn rewind<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

#[extern_fn(effect = pure)]
fn shuttle<I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    forward: Instance<'_, advance<I, Rt>, I, Rt>,
    back: Instance<'_, rewind<I, Rt>, I, Rt>,
) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let (mut forward, mut back) = (forward, back);
    forward.call(ctx, ()) + back.call(ctx, ())
}

fn main() {}
