//! A `RustFn` parameter is a concrete type: a call of a function value
//! settles no type for `Args` to check against (RFC-0097 rule 2).
use acvus_extern::{Runtime, RustFn, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
fn counter<A, R>(k: i64) -> RustFn<(A,), i64, R>
where
    A: Var<kind::Type>,
    R: Runtime,
{
    RustFn::new(move |_, args| args.len() as i64 + k)
}

fn main() {}
