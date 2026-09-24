//! A lent type variable is an assertion the handler keeps none of its
//! values past the call, so it is written `unsafe(lent(T))`; `lent(T)`
//! alone is refused (RFC-0079 rule 8, RFC-0080 rule 1).
use acvus_extern::{Var, extern_fn, kind};

#[extern_fn(effect = pure, lent(T))]
fn pass<T>(x: T) -> T
where
    T: Var<kind::Type>,
{
    x
}

fn main() {}
