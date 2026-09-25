//! A `dynamic` declaration returns the `Finished` its `Output` builds
//! (RFC-0097 rule 3).
use acvus_extern::{Output, Runtime, Var, extern_fn, kind};

#[extern_fn(dynamic, effect = pure)]
fn parse<'c, T, R>(out: Output<'c, T, R>) -> Option<i64>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    drop(out);
    None
}

fn main() {}
