//! Probe p4's handler: a `Ref` put into a static outlives the call it was
//! handed to, and a read through it after the run names freed storage. The
//! call's lifetime is the `Ref`'s brand, so keeping it does not compile
//! (RFC-0079 rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Ref, Runtime, Shared, extern_fn};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep<Rt>(r: Ref<'_, String, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(r));
    0
}

fn main() {}
