//! A `match` term states both arms (RFC-0099 rule 1).
use acvus_extern::{Closure, Instance, Later, Runtime, Stored, Cross, PassedByValue, Var, extern_fn, kind};
use acvus_ext::iter_sig::next;

#[extern_fn(effect = pure, step(state s = None; s = Some(x); finish match s { Some(b) => Some(b) }))]
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
