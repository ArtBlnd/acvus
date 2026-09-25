//! A signature whose receiver is `&mut I` steps its receiver: it is
//! required by the `Instance` that owns that receiver, and an
//! `InstanceOf`, which stands at the type and is handed any value of it,
//! is refused there (RFC-0067 rule 1).
use std::ops::Deref;

use acvus_extern::{Ctx, InstanceOf, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "g", fn advance<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

#[extern_fn(effect = pure)]
fn drive<I, Rt>(ctx: &mut Ctx<'_, Rt>, it: I, step: InstanceOf<'_, advance<I, Rt>, I, Rt>) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let _ = (ctx, it, step);
    0
}

fn main() {}
