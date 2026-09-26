//! A value parameter is moved once, and a per-element step runs many times (RFC-0099 rule 1).
use acvus_extern::{Closure, Instance, Later, Runtime, Stored, Cross, PassedByValue, Var, extern_fn, kind};
use acvus_ext::iter_sig::next;

#[extern_fn(effect = pure, step(state s = 0; s = s +% n; finish s))]
fn adds<I, T, E, Rt>(it: Instance<'_, next<I, T, E, Rt>, I, Rt, Later>, n: i64) -> i64
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
