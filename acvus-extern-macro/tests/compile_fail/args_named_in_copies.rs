//! `copies(x)` names a reference parameter whose target the result equals,
//! and an `Args` takes its members by value (RFC-0082 rule 10, RFC-0097
//! rule 1).
use acvus_extern::{Args, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure, copies(args))]
fn counted<A, R>(args: Args<'_, (A,), R>) -> u64
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.len() as u64
}

fn main() {}
