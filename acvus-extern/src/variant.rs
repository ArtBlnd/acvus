//! The variant a derived enum crosses as (RFC-0048 §7).
//!
//! As in `object`, the layout — RFC-0050 rule 8's flat `[tag, payload]` — lives
//! in this file alone.

use crate::obj::Variant;
use crate::owned::Owned;
use crate::runtime::Runtime;

/// `at` indexes the `tags` its `opened` was given, which is
/// `acvus-extern-macro`'s field table in the Rust enum's declaration order.
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
    let payload = payload.unwrap_or_else(|| Owned::from_value(rt.undef()));
    let variant = Variant::of(Owned::from_value(rt.variant_tag(tag)), payload);
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
    let variant = unsafe { rt.materialize::<Variant<Owned<Rt>>>(value) };
    // SAFETY: the same contract — `erase` wrote the tag register.
    let tag = unsafe { rt.tag_symbol(variant.tag()) };
    let at = tags
        .iter()
        .position(|declared| rt.symbol(declared) == tag)
        .unwrap_or_else(|| {
            panic!(
                "a variant not in enum `{name}`: the checker admits only variants of the \
                 declared enum"
            )
        });
    let payload = variant.into_payload();
    let carried = match rt.is_undef(&payload) {
        true => None,
        false => Some(payload),
    };
    Opened {
        at,
        payload: carried,
    }
}
