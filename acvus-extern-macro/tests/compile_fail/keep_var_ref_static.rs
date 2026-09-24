//! Rust refuses this keep: a call hands the handler what it names at the
//! call (RFC-0079 rules 6 and 8).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Ref, Runtime, Shared, Var, extern_fn, kind};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep<T, Rt>(x: Ref<'_, T, Shared, Rt>) -> i64
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(x));
    0
}

fn main() {}
