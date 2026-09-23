//! `Instance<S, I, Rt>` is paired with the variable it was declared for,
//! and Rust is what checks the pairing (RFC-0067 rule 4): a receiver
//! of another variable does not compile, and the refusal is the type
//! error, not a sentence the macro writes.
use std::ops::DerefMut;

use acvus_extern::{Ctx, Instance, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "g", fn advance<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

#[extern_fn(effect = pure)]
fn drive<I, J, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: I,
    other: J,
    step: Instance<advance<I, Rt>, I, Rt>,
) -> i64
where
    I: Var<kind::Type> + DerefMut<Target = Rt::Value>,
    J: Var<kind::Type> + DerefMut<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut it = it;
    let mut other = other;
    let first = step.call(ctx, &mut it, ());
    first + step.call(ctx, &mut other, ())
}

fn main() {}
