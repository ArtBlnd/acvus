//! As `ref_in_a_container_kept_past_the_call`, with the `Ref` one level
//! deeper, inside an option the container holds (RFC-0079 rule 6).
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
fn keep_box<Rt>(b: Box8<Option<Ref<'static, String, Shared, Rt>>>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(b));
    0
}

fn main() {}
