//! A runtime value as JSON.

use acvus_ext::{Deque, HashMap, HashSet};
use acvus_extern::{Owned, repr};
use acvus_interpreter::{AcvusRuntime, Array, Kind, Object, Tuple, Value, VariantValue, Vtable};
use acvus_mir::ty::IntTy;
use acvus_utils::Interner;
use serde_json::{Map, Value as Json};

/// A value read without a type: its `Kind` is the witness for a word, and
/// the vtable's type for an allocation (`Value::get`).
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
        Kind::Large => by_composite(
            interner,
            value,
            value.vtable().expect("a value of kind Large has a vtable"),
        ),
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

fn by_composite(interner: &Interner, value: &Value, vtable: &Vtable) -> Json {
    if let Some(text) = value.get::<String>() {
        return Json::from(text.as_str());
    }
    if let Some(items) = value.get::<Array>() {
        return items_array(interner, items.0.iter());
    }
    if let Some(items) = value.get::<Tuple>() {
        return items_array(interner, items.0.iter());
    }
    if let Some(object) = value.get::<Object>() {
        return Json::Object(
            object
                .shape
                .names()
                .iter()
                .zip(object.values.iter())
                .map(|(k, v)| (interner.resolve(*k).to_string(), by_kind(interner, v)))
                .collect(),
        );
    }
    if let Some(variant) = value.get::<VariantValue>() {
        let tag = interner
            .resolve(repr::tag_of_word(variant.tag().bits()))
            .to_string();
        return match variant.payload().kind() {
            Kind::Undef => Json::from(tag),
            _ => Json::Object(Map::from_iter([(
                tag,
                by_kind(interner, variant.payload()),
            )])),
        };
    }
    by_extension(interner, value, vtable)
}

/// A result leaves a run through a uniform slot, where acvus-extern keys a
/// container's box at its element's canonical form, `Owned` over the runtime
/// (RFC-0039 rule 5, RFC-0076). A change to that key moves this type with it.
type ResultElement = Owned<AcvusRuntime>;
type ResultKeying = Owned<AcvusRuntime>;
type ResultMap = HashMap<'static, ResultElement, ResultElement, ResultKeying, (), AcvusRuntime>;
type ResultSet = HashSet<'static, ResultElement, ResultKeying, (), AcvusRuntime>;

fn by_extension(interner: &Interner, value: &Value, vtable: &Vtable) -> Json {
    if let Some(items) = value.get::<Vec<ResultElement>>() {
        return items_array(interner, items.iter());
    }
    if let Some(items) = value.get::<Deque<ResultElement>>() {
        return items_array(interner, items.iter());
    }
    if let Some(map) = value.get::<ResultMap>() {
        return Json::Array(
            map.iter()
                .map(|(key, item)| {
                    Json::Array(vec![by_kind(interner, key), by_kind(interner, item)])
                })
                .collect(),
        );
    }
    if let Some(keys) = value.get::<ResultSet>() {
        return items_array(interner, keys.iter());
    }
    Json::from(format!("<{}>", vtable.name()))
}

fn items_array<'v, I>(interner: &Interner, items: I) -> Json
where
    I: Iterator<Item = &'v ResultElement>,
{
    Json::Array(items.map(|item| by_kind(interner, item)).collect())
}
