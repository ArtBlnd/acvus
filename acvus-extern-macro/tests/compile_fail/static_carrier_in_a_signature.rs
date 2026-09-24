//! A carrier written at `'static` in a signature is refused by the macro,
//! at the parameter and inside a closure's parameters (RFC-0079 rule 6).
#![forbid(unsafe_code)]
use acvus_extern::{Closure, Ref, Runtime, Shared, Var, extern_fn, kind};

#[extern_fn(effect = opaque)]
fn keep<Rt>(r: Ref<'static, String, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    let _ = r;
    0
}

#[extern_fn(effect = E)]
fn count<T, E, Rt>(keep: Closure<'_, (Ref<'static, T, Shared, Rt>,), bool, E, Rt>) -> i64
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let _ = keep;
    0
}

fn main() {}
