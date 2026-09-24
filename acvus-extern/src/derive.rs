//! The library half of what `acvus-extern-macro` expands to.
//!
//! Nothing here is exported from the crate root, and that is a decision. Each
//! item has one caller — a derive or a crossing macro — which names it by
//! path. At the root the same name would read as a step a hand-written
//! crossing is invited to take, and a hand-written crossing that took one
//! would be writing half a derived type's layout by hand.

pub mod canonical;
pub mod object;
pub mod transparent;
pub mod variant;

use crate::obj::OneValue;
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::uniform::UniformPayload;

/// The obligation `#[derive(ExternType)]` writes on its payload, at a
/// marker `M` only its check names.
pub fn uniform_payload<P, M>()
where
    P: UniformPayload<M>,
{
}

pub fn erase_field<T, Rt>(rt: &Rt, value: T) -> Owned<Rt>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    Owned::erased(rt, value)
}

/// # Safety
/// The value held at that field was erased from a `T`.
pub unsafe fn materialize_field<T, Rt>(rt: &Rt, field: Owned<Rt>) -> T
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { T::materialize(rt, field.into_value()) }
}

pub fn take_payload<V>(payload: Option<V>, tag: &str) -> V {
    let Some(payload) = payload else {
        panic!(
            "variant `{tag}` has no payload: the checker admits only variants of the declared enum"
        )
    };
    payload
}

/// # Safety
/// The payload of variant `tag` was erased from a `T`.
pub unsafe fn materialize_payload<T, Rt>(rt: &Rt, payload: Option<Owned<Rt>>, tag: &str) -> T
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { T::materialize(rt, take_payload(payload, tag).into_value()) }
}
