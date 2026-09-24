//! As `ref_kept_past_the_call`, for a `Slice`: the run it names belongs to
//! the call's argument, and the call's lifetime is its brand (RFC-0079
//! rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Erased, Runtime, Shared, Slice, extern_fn};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep<Rt>(s: Slice<'_, Erased<Rt, i64>, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(s));
    0
}

fn main() {}
