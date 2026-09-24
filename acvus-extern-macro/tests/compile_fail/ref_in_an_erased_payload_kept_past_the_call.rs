//! p4 through a derived container that holds its type argument through
//! `Erased`: the handler names the `Ref` at `'static`, behind an alias, as
//! that argument and puts the container into a static. `Erased` is `Within`
//! only where the type it was erased from holds no carrier (RFC-0076 rule 1's
//! exception), and a `Ref` is a carrier, so the glue refuses the handler
//! (RFC-0079 rule 6).
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

/// `'static` behind an alias, which the macro does not see through.
type Kept<Rt> = Ref<'static, String, Shared, Rt>;

#[extern_fn(effect = opaque)]
fn keep_box<Rt>(b: Box9<Kept<Rt>, Rt>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(b));
    0
}

fn main() {}
