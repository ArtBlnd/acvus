//! An adaptor keeps no state: `state` is a consumer's (RFC-0099 rule 1).
use acvus_extern::{Closure, Instance, Later, Runtime, Stored, Cross, PassedByValue, Var, extern_fn, kind};
use acvus_ext::iter_sig::next;

#[extern_fn(effect = pure, step(state s = 0; yield x))]
fn counted<'a, I, T, E, Rt>(it: Instance<'a, next<I, T, E, Rt>, I, Rt, Later>) -> i64
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    drop(it);
    unreachable!("a refused step is never called")
}

fn main() {}
