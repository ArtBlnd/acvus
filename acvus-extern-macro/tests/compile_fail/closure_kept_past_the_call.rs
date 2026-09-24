//! As `ref_kept_past_the_call`, for a `Closure`: what it captured is the
//! caller's for the call, and the call's lifetime is its brand (RFC-0079
//! rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Closure, Runtime, Var, extern_fn, kind};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep<E, Rt>(f: Closure<'_, (), i64, E, Rt>) -> i64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(f));
    0
}

fn main() {}
