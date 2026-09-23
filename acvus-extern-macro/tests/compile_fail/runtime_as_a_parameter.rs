//! The runtime and the window cross as one `ctx`, so `&Rt` at a parameter
//! is the runtime's own type where a value of the call belongs (RFC-0023
//! rule 2).
use acvus_extern::{Runtime, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
fn twice<T, Rt>(rt: &Rt, n: i64) -> i64
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let _ = rt;
    n * 2
}

fn main() {}
