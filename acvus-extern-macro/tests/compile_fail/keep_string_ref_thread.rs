//! Rust refuses this keep: a call hands the handler what it names at the
//! call (RFC-0079 rules 6 and 8).
#![forbid(unsafe_code)]
use acvus_extern::{Ref, Runtime, Shared, extern_fn};

#[extern_fn(effect = opaque)]
fn keep<Rt>(x: Ref<'_, String, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    std::thread::spawn(move || drop(x));
    0
}

fn main() {}
