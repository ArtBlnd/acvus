//! The variant a derived enum crosses as (RFC-0048 §7).
//!
//! As in `object`, the layout — today a `Variant<Owned<Rt>>` whose tag is a
//! symbol and whose payload is boxed — lives in this file alone.

use crate::obj::Variant;
use crate::owned::Owned;
use crate::runtime::Runtime;

/// The variant a value holds: which of the declared tags it is, by position,
/// and the payload that tag carried.
pub struct Opened<Rt>
where
    Rt: Runtime,
{
    pub at: usize,
    pub payload: Option<Owned<Rt>>,
}

pub fn erase<Rt>(rt: &Rt, tag: &str, payload: Option<Owned<Rt>>) -> Rt::Value
where
    Rt: Runtime,
{
    let variant = Variant {
        tag: rt.symbol(tag),
        payload: payload.map(Box::new),
    };
    // SAFETY: the language's variant is `Variant<Owned<Rt>>`, and `opened` is
    // the only reader.
    unsafe { rt.erase::<Variant<Owned<Rt>>>(variant) }
}

/// # Safety
/// `value` is what `erase` wrote.
///
/// # Panics
/// The tag is none of `tags`: the checker admits only variants of the
/// declared enum.
pub unsafe fn opened<Rt>(rt: &Rt, value: Rt::Value, name: &str, tags: &[&str]) -> Opened<Rt>
where
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    let Variant { tag, payload } = unsafe { rt.materialize::<Variant<Owned<Rt>>>(value) };
    let at = tags
        .iter()
        .position(|declared| rt.symbol(declared) == tag)
        .unwrap_or_else(|| {
            panic!(
                "a variant not in enum `{name}`: the checker admits only variants of the \
                 declared enum"
            )
        });
    Opened {
        at,
        payload: payload.map(|boxed| *boxed),
    }
}
