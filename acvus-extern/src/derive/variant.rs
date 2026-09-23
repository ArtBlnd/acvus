//! The variant a derived enum crosses as (RFC-0039 rule 4, RFC-0048 rule 7).
//!
//! As in `object`, the layout — RFC-0050 rule 8's flat `[tag, payload]` — lives
//! in this file alone.

use acvus_utils::Astr;

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

pub fn words<Rt>(rt: &Rt, tag: &str, payload: Option<Owned<Rt>>) -> Variant<Owned<Rt>>
where
    Rt: Runtime,
{
    let payload = payload.unwrap_or_else(|| Owned::from_value(rt.undef()));
    Variant::of(Owned::from_value(rt.variant_tag(tag)), payload)
}

pub fn erase<Rt>(rt: &Rt, tag: &str, payload: Option<Owned<Rt>>) -> Rt::Value
where
    Rt: Runtime,
{
    // SAFETY: the language's variant is `Variant<Owned<Rt>>`, and `opened` is
    // the only reader.
    unsafe { rt.erase::<Variant<Owned<Rt>>>(words(rt, tag, payload)) }
}

/// Decision not to resolve a name here. `opened`, the by-value crossing,
/// calls `Runtime::symbol` once per arm per call because it is handed the
/// declared names and no site; a projection is built per call site, so its
/// names are interned once at `prepare` and the call compares words.
///
/// # Panics
/// The tag is none of `tags`: the checker settles the declared enum's own
/// type on a projection's argument, so every variant reaching one is
/// declared.
pub fn arm_of<const K: usize>(tag: Astr, tags: &[u64; K], name: &str) -> usize {
    let bits = tag.bits();
    tags.iter()
        .position(|declared| *declared == bits)
        .unwrap_or_else(|| {
            panic!(
                "a variant not in enum `{name}`: the checker admits only variants of the \
                 declared enum"
            )
        })
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
