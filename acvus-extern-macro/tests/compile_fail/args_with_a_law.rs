//! A law is stated over a declaration's typed parameters, and the shape
//! `f(a: T, b: T) -> T` would otherwise admit two `Args` members of one
//! variable (RFC-0082, RFC-0097 rule 1).
use acvus_extern::{Args, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure, law(associative))]
fn first<A, R>(args: Args<'_, (A, A), R>) -> A
where
    A: Var<kind::Type>,
    R: Runtime,
{
    drop(args);
    unreachable!("a law's shape is refused before any call")
}

fn main() {}
