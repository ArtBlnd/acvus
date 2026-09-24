//! A runtime value as JSON.

use std::any::TypeId;

use acvus_ext::Deque;
use acvus_extern::Owned;
use acvus_interpreter::{AcvusRuntime, Composite, Kind, Value};
use acvus_mir::ty::IntTy;
use acvus_utils::Interner;
use serde_json::{Map, Value as Json};

/// A value read without a type: its `Kind` is the witness for a word, and
/// the vtable's `Composite` for an allocation.
///
/// RFC-0054: a host that declares `!` states no return type, so it has no
/// `Ty` for the value that comes back.
pub fn by_kind(interner: &Interner, value: &Value) -> Json {
    match value.kind() {
        Kind::Instance | Kind::InstanceAwait => Json::from("<instance>"),
        Kind::Code => Json::from("<fn>"),
        Kind::I8 => int(IntTy::I8, value),
        Kind::I16 => int(IntTy::I16, value),
        Kind::I32 => int(IntTy::I32, value),
        Kind::I64 => int(IntTy::I64, value),
        Kind::U8 => int(IntTy::U8, value),
        Kind::U16 => int(IntTy::U16, value),
        Kind::U32 => int(IntTy::U32, value),
        Kind::U64 => int(IntTy::U64, value),
        Kind::F64 => Json::from(value.as_float()),
        Kind::Char => Json::from(char_of(value).to_string()),
        Kind::Bool => Json::from(value.as_bool()),
        Kind::Unit | Kind::None => Json::Null,
        // SAFETY: a reference names a live value for as long as it lives.
        Kind::Ref => by_kind(interner, unsafe { value.target() }),
        Kind::Undef => Json::from("<undef>"),
        Kind::LargeRef => Json::from("<projection>"),
        Kind::Large => by_composite(interner, value),
    }
}

/// The scalar value a `Char` word spells. JSON has no character, so a
/// `char` reads out as the one-character string it is.
fn char_of(value: &Value) -> char {
    char::from_u32(value.as_char()).expect("as_char asserted a scalar value")
}

fn int(kind: IntTy, value: &Value) -> Json {
    let v = kind.read(value.bits());
    if kind.signed() {
        Json::from(v as i64)
    } else {
        Json::from(v as u64)
    }
}

fn by_composite(interner: &Interner, value: &Value) -> Json {
    // SAFETY, every arm: the vtable's `Composite` is the runtime's own
    // witness of the type behind the pointer.
    match value.composite() {
        Some(Composite::String) => Json::from(unsafe { value.as_str() }),
        Some(Composite::Array) => Json::Array(
            unsafe { value.as_array() }
                .iter()
                .map(|v| by_kind(interner, v))
                .collect(),
        ),
        Some(Composite::Tuple) => Json::Array(
            unsafe { value.as_tuple() }
                .iter()
                .map(|v| by_kind(interner, v))
                .collect(),
        ),
        Some(Composite::Object) => Json::Object(
            unsafe { value.as_shape() }
                .names()
                .iter()
                .zip(unsafe { value.as_object() })
                .map(|(k, v)| (interner.resolve(*k).to_string(), by_kind(interner, v)))
                .collect(),
        ),
        Some(Composite::Variant) => {
            let variant = unsafe { value.as_variant() };
            // SAFETY: the same witness — a variant's first register is its tag.
            let tag = interner
                .resolve(unsafe { variant.tag().as_tag() })
                .to_string();
            match variant.payload().kind() {
                Kind::Undef => Json::from(tag),
                _ => Json::Object(Map::from_iter([(
                    tag,
                    by_kind(interner, variant.payload()),
                )])),
            }
        }
        None => by_extension(interner, value),
        Some(Composite::Fn | Composite::Handle) => named(value),
    }
}

/// A result leaves a run through a uniform slot, where acvus-extern keys a
/// container's box at its element's canonical form, `Owned` over the runtime
/// (RFC-0039 rule 5, RFC-0076). A change to that key moves this type with it.
type ResultElement = Owned<AcvusRuntime>;

fn by_extension(interner: &Interner, value: &Value) -> Json {
    if let Some(items) = payload_of::<Vec<ResultElement>>(value) {
        return items_array(interner, items.iter());
    }
    if let Some(items) = payload_of::<Deque<ResultElement>>(value) {
        return items_array(interner, items.iter());
    }
    named(value)
}

fn payload_of<T>(value: &Value) -> Option<&T>
where
    T: 'static,
{
    (value.vtable().type_id == TypeId::of::<T>()).then(|| {
        // SAFETY: the header's `type_id` is the runtime's own witness that the
        // payload is a `T`.
        unsafe { value.peek::<T>() }
    })
}

fn items_array<'v, I>(interner: &Interner, items: I) -> Json
where
    I: Iterator<Item = &'v ResultElement>,
{
    Json::Array(items.map(|item| by_kind(interner, item)).collect())
}

fn named(value: &Value) -> Json {
    Json::from(format!("<{}>", (value.vtable().name)()))
}
