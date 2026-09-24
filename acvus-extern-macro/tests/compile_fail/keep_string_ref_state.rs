//! Rust refuses this keep: a call hands the handler what it names at the
//! call (RFC-0079 rules 6 and 8).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Ref, Runtime, Shared, extern_fn};

#[derive(Default)]
pub struct Stash(Mutex<Vec<Box<dyn Any + Send + Sync>>>);

#[extern_fn(effect = opaque)]
fn keep<Rt>(#[state] stash: &Stash, x: Ref<'_, String, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    stash.0.lock().unwrap().push(Box::new(x));
    0
}

fn main() {}
