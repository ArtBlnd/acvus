//! An adaptor's flow ends in `yield`, `skip`, `done` or `nest` (RFC-0099 rule 1).
use acvus_extern::{Closure, Instance, Later, Runtime, Stored, Cross, PassedByValue, Var, extern_fn, kind};
use acvus_ext::iter_sig::next;

#[extern_fn(effect = pure, step(if f(&x) { yield x } else { drop }))]
fn keep<'a, I, T, E, Rt>(
    it: Instance<'a, next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'a, (acvus_extern::Ref<'a, T, acvus_extern::Shared, Rt>,), bool, E, Rt>,
) -> i64
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
