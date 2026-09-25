//! An `InstanceOf` stands at its signature's receiver type: `I` is
//! `S::This` by a bound of the type, so an instance of `core::eq` at `T`
//! is not named at another type `U`, and no call hands it a `U` to read
//! (RFC-0067 rule 1).
use std::ops::Deref;

use acvus_extern::{Borrowable, Ctx, InstanceOf, Runtime, TransparentOver, Var, core, kind};

fn compare_at_another<T, U, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    eq: InstanceOf<'_, core::eq<T, Rt>, U, Rt>,
    a: &U,
    b: &T,
) -> bool
where
    T: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt> + Deref<Target = Rt::Value>,
    U: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    eq.call(ctx, a, (b,))
}

fn main() {}
