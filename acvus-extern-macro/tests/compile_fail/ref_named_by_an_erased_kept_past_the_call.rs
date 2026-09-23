//! A `Ref` at `'static` as the type an `Erased` was erased from. `Erased`'s
//! brand keeps `T` (RFC-0076 rule 1), so the value it holds would outlive
//! the call; its declared type is refused instead (RFC-0079 rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Erased, Ref, Runtime, Shared, extern_fn};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[extern_fn(effect = opaque)]
fn keep_erased<Rt>(e: Erased<Rt, Ref<'static, String, Shared, Rt>>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(e));
    0
}

fn main() {}
