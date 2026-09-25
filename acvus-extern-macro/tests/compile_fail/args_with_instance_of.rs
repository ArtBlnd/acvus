//! An instance of a shared signature is reached through its mono glue,
//! which builds no `Args` (RFC-0067 rule 8, RFC-0097 rule 1).
use acvus_extern::{Args, Runtime, Var, extern_fn, kind};

#[extern_fn(instance_of = acvus_extern::core::display, effect = pure)]
fn shown<T, R>(args: Args<'_, (T,), R>) -> String
where
    T: Var<kind::Type>,
    R: Runtime,
{
    args.len().to_string()
}

fn main() {}
