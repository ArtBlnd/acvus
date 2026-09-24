//! The one ground for `Value -> T` is the crossing the glue writes at the
//! type the checker settled. A handler holding a receiver's word cannot read
//! it back at a type of its choosing: `OneValue::materialize` takes the
//! glue's `Crossing`, and even inside `unsafe` a handler has only `ctx.rt`
//! (RFC-0068 rules 1 and 4).
use std::ops::Deref;

use acvus_extern::{Ctx, OneValue, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = opaque)]
fn read_back<I, Rt>(ctx: &mut Ctx<'_, Rt>, it: &I) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    unsafe { <i64 as OneValue<Rt>>::materialize(ctx.rt, **it) }
}

fn main() {}
