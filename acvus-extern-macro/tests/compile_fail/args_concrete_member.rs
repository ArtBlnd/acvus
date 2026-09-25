//! A member of `Args` is one of the declaration's own type variables: a
//! position whose type Rust names is an ordinary parameter (RFC-0097 rule 1).
use acvus_extern::{Args, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
fn count<A, R>(args: Args<'_, (A, i64), R>) -> u64
where
    A: Var<kind::Type>,
    R: Runtime,
{
    args.len() as u64
}

fn main() {}
