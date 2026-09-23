//! As `ref_in_a_container_kept_past_the_call`, through `Deque`, whose impls
//! are written by hand: its brand is its fields', and it holds its elements
//! by value (RFC-0079 rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_ext::Deque;
use acvus_extern::{Ref, Runtime, Shared, extern_fn};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep_deque<Rt>(d: Deque<Ref<'static, String, Shared, Rt>>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(d));
    0
}

fn main() {}
