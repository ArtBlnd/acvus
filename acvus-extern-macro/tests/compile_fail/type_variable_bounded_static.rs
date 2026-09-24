//! A type variable takes no lifetime bound: `T: 'static`, which rustc's
//! help suggests for a keep, would let the handler keep a `T` past the
//! call (RFC-0079 rule 8).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Var, extern_fn, kind};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep<T>(x: T) -> i64
where
    T: Var<kind::Type> + 'static,
{
    *KEPT.lock().unwrap() = Some(Box::new(x));
    0
}

fn main() {}
