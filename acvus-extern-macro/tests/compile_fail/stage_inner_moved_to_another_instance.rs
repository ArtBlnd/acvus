//! Two stages over pipelines of one Rust type: the glue fills every
//! variable with `Owned<Rt>`, so the type alone does not tell the two
//! pipelines apart. Moving one stage's pipeline into the other's instance
//! still does not compile, because an `Instance` takes no receiver once it
//! is built (RFC-0067 rule 4).
use std::ops::Deref;

use acvus_extern::{Instance, Later, Pure, Runtime, Var, kind};

acvus_extern::extern_signature! {
    ns: "g",
    effect = E,
    fn next<I, T, E, Rt>(it: &mut I) -> Option<T>
    where
        I: Var<kind::Type>,
        T: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

struct Stage<'a, I, Rt>
where
    I: Var<kind::Type>,
    Rt: Runtime,
{
    inner: Instance<'a, next<I, i64, Pure, Rt>, I, Rt, Later>,
}

fn rewire<'a, I, Rt>(first: Stage<'a, I, Rt>, second: Stage<'a, I, Rt>) -> Stage<'a, I, Rt>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let moved: I = first.inner.into_inner();
    let mut inner = second.inner;
    inner.recv = moved;
    Stage { inner }
}

fn main() {}
