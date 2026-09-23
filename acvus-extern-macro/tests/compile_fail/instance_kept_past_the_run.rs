//! An `Instance` names an entry the prepared program keeps for its run. The
//! run's lifetime is its brand, the lifetime of the `Ctx` the handler is
//! called with, so keeping it past the run does not compile (RFC-0079 rule
//! 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Instance, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "g", fn advance<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep<I, Rt>(it: I, step: Instance<advance<I, Rt>, I, Rt>) -> i64
where
    I: Var<kind::Type>,
    Rt: Runtime,
{
    let _ = it;
    *KEPT.lock().unwrap() = Some(Box::new(step));
    0
}

fn main() {}
