//! A `Ref` at `'static`, behind an alias, as the type an `Erased` was erased
//! from. `Erased` is `Within` only where that type holds no carrier
//! (RFC-0076 rule 1), so the glue refuses its declared type (RFC-0079
//! rule 6).
#![forbid(unsafe_code)]
use std::any::Any;
use std::sync::Mutex;

use acvus_extern::{Erased, Ref, Runtime, Shared, extern_fn};

static KEPT: Mutex<Option<Box<dyn Any + Send + Sync>>> = Mutex::new(None);

/// `'static` behind an alias, which the macro does not see through.
type Kept<Rt> = Ref<'static, String, Shared, Rt>;

#[extern_fn(effect = opaque)]
fn keep_erased<Rt>(e: Erased<Rt, Kept<Rt>>) -> i64
where
    Rt: Runtime,
{
    *KEPT.lock().unwrap() = Some(Box::new(e));
    0
}

fn main() {}
