//! A step calls closures and registered externs, and `n` is neither (RFC-0099 rule 1).
use acvus_extern::{Closure, Instance, Later, Runtime, Stored, Cross, PassedByValue, Var, extern_fn, kind};
use acvus_ext::iter_sig::next;

#[extern_fn(effect = pure, step(yield n(x)))]
fn applies<I, T, E, Rt>(it: Instance<'_, next<I, T, E, Rt>, I, Rt, Later>, n: i64) -> i64
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    drop((it, n));
    unreachable!("a refused step is never called")
}

fn main() {}
