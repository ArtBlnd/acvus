//! Rust refuses this keep: a call hands the handler what it names at the
//! call (RFC-0079 rules 6 and 8).
#![forbid(unsafe_code)]
use acvus_extern::{Ref, Runtime, Shared, Var, extern_fn, kind};

#[extern_fn(effect = opaque)]
fn keep<T, Rt>(x: Ref<'_, T, Shared, Rt>) -> i64
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    std::thread::spawn(move || drop(x));
    0
}

fn main() {}
