//! A declaration requires `core::eq` at `T` and names the `InstanceOf` at
//! its other variable `U`: the instance the checker chose for `T` would
//! read a `U`. The signature's receiver type is `T`, and an `InstanceOf`'s
//! `I` is that type by a bound of the type (RFC-0067 rule 1).
use std::ops::Deref;

use acvus_extern::{
    Borrowable, Ctx, InstanceOf, Runtime, TransparentOver, Var, core, extern_fn, kind,
};

#[extern_fn(effect = pure)]
fn same<T, U, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &T,
    b: &U,
    eq: InstanceOf<'_, core::eq<T, Rt>, U, Rt>,
) -> bool
where
    T: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt> + Deref<Target = Rt::Value>,
    U: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    eq.call(ctx, b, (a,))
}

fn main() {}
