//! p4 through a derived container that holds its type argument through
//! `Erased`: the handler names the `Ref` at `'static` as that argument and
//! puts the container into a static. `Erased` has a brand only where the
//! type it was erased from is `Unbranded` (RFC-0076 rule 1's exception), and
//! a `Ref` is not, so the container's payload has no brand and the handler
//! is refused (RFC-0079 rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Mutex;

use acvus_extern::{Erased, ExternType, Ref, Runtime, Shared, Var, extern_fn, kind};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[derive(ExternType)]
#[extern_type(name = "Box9")]
#[repr(transparent)]
struct Box9<T, Rt>(Vec<Erased<Rt, T>>, PhantomData<T>)
where
    T: Var<kind::Type>,
    Rt: Runtime;

#[extern_fn(effect = opaque)]
fn keep_box<Rt>(b: Box9<Ref<'static, String, Shared, Rt>, Rt>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(b));
    0
}

fn main() {}
