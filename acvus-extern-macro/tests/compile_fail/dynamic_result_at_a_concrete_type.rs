//! A `dynamic` result is one of the declaration's own type variables, which
//! each call site settles (RFC-0097 rule 3).
use acvus_extern::{Finished, Output, Runtime, extern_fn};

#[extern_fn(dynamic, effect = pure)]
fn parse<'c, R>(out: Output<'c, i64, R>) -> Finished<'c, i64, R>
where
    R: Runtime,
{
    out.finish()
}

fn main() {}
