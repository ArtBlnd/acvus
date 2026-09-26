//! A term names parameters, and an `Args` takes its members by value where
//! no term can name them (RFC-0104 rule 1, RFC-0097 rule 1).
use acvus_extern::{Args, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure, means(*args))]
fn counted<A, R>(args: Args<'_, (A,), R>) -> u64
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.len() as u64
}

fn main() {}
