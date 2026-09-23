//! p4 through a derived container: the handler names the `Ref` at `'static`
//! inside the container's type argument and puts the container into a
//! static. The container's brand is its payload's, and the payload holds the
//! `Ref` by value, so the handler is handed it at the call's brand and the
//! `'static` it names does not match (RFC-0079 rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{ExternType, Ref, Runtime, Shared, Var, extern_fn, kind};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

#[derive(ExternType)]
#[extern_type(name = "Box8")]
#[repr(transparent)]
struct Box8<T>(Vec<T>)
where
    T: Var<kind::Type>;

#[extern_fn(effect = opaque)]
fn keep_box<Rt>(b: Box8<Ref<'static, String, Shared, Rt>>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(b));
    0
}

fn main() {}
