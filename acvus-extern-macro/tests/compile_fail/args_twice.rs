//! One `Args` holds every position a declaration names only by a variable
//! (RFC-0097 rule 1).
use acvus_extern::{Args, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
fn count<A, B, R>(first: Args<'_, (A,), R>, second: Args<'_, (B,), R>) -> u64
where
    A: Var<kind::Type>,
    B: Var<kind::Type>,
    R: Runtime,
{
    (first.len() + second.len()) as u64
}

fn main() {}
