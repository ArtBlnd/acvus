//! A `Finished` result is typed by its call site, which only a `dynamic`
//! declaration asks for (RFC-0097 rule 3).
use acvus_extern::{Finished, Output, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
fn parse<'c, T, R>(out: Output<'c, T, R>) -> Finished<'c, T, R>
where
    T: Var<kind::Type>,
    R: Runtime,
{
    out.finish()
}

fn main() {}
