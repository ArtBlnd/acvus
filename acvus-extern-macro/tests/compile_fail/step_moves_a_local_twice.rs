//! A local is moved once on each path; a comparison reads it (RFC-0099 rule 1).
use acvus_extern::{Closure, Instance, Later, Runtime, Stored, Cross, PassedByValue, Var, extern_fn, kind};
use acvus_ext::iter_sig::next;

#[extern_fn(effect = pure, step(state s = None; let k = f(&x); s = Some({ value: x, key: k, again: k }); finish s))]
fn keyed<'a, I, T, E, Rt>(
    it: Instance<'a, next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'a, (&'a T,), i64, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    drop((it, f));
    unreachable!("a refused step is never called")
}

fn main() {}
