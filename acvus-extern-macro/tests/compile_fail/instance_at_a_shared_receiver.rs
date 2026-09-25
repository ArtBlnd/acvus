//! A signature whose receiver is `&I` is a function of its type's values:
//! it is required by an `InstanceOf`, and an `Instance`, which owns the
//! receiver of a call that steps it, is refused there (RFC-0067 rule 1).
use std::ops::Deref;

use acvus_extern::{Ctx, Instance, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "g", fn eq<T>(a: &T, b: &T) -> bool where T: Var<kind::Type>; }

#[extern_fn(effect = pure)]
fn same<T, Rt>(ctx: &mut Ctx<'_, Rt>, a: Instance<'_, eq<T, Rt>, T, Rt>, b: T) -> bool
where
    T: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let _ = (ctx, b);
    let _ = a.into_inner();
    true
}

fn main() {}
