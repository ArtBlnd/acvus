//! Rust refuses this keep: a call hands the handler what it names at the
//! call (RFC-0079 rules 6 and 8).
#![forbid(unsafe_code)]
use std::any::Any;

use acvus_extern::{Ref, Runtime, Shared, Var, extern_fn, kind};

#[extern_fn(effect = opaque)]
fn keep<T, Rt>(x: Ref<'_, T, Shared, Rt>) -> i64
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let kept: Box<dyn Any + Send + Sync> = Box::new(x);
    std::mem::forget(kept);
    0
}

fn main() {}
