//! A view converts a value of the type its one parameter names, and an
//! `Args` member names none (RFC-0023 rule 8, RFC-0097 rule 1).
use acvus_extern::{Args, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
#[extern_view]
fn counted<A, R>(args: Args<'_, (A,), R>) -> u64
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.len() as u64
}

fn main() {}
