//! A runtime value as JSON.

use acvus_interpreter::{Composite, Kind, Value};
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;
use serde_json::{Map, Value as Json};

pub fn of(interner: &Interner, ty: &Ty, value: &Value) -> Json {
    match ty {
        Ty::Int(k) => {
            let v = k.read(value.bits());
            if k.signed() {
                Json::from(v as i64)
            } else {
                Json::from(v as u64)
            }
        }
        Ty::Float => Json::from(value.as_float()),
        Ty::Char => Json::from(char_of(value).to_string()),
        Ty::Bool => Json::from(value.as_bool()),
        Ty::Unit => Json::Null,
        // SAFETY: the type is the runtime's own witness of the value's shape.
        Ty::String => Json::from(unsafe { value.as_str() }),
        Ty::Array(elem, _) => Json::Array(
            unsafe { value.as_array() }
                .iter()
                .map(|v| of(interner, elem, v))
                .collect(),
        ),
        Ty::Tuple(elems) => Json::Array(
            unsafe { value.as_tuple() }
                .iter()
                .zip(elems)
                .map(|(v, t)| of(interner, t, v))
                .collect(),
        ),
        Ty::Object(fields) => {
            let mut laid: Vec<_> = fields.iter().collect();
            laid.sort_by(|(a, _), (b, _)| interner.resolve(**a).cmp(interner.resolve(**b)));
            let values = unsafe { value.as_object() };
            Json::Object(
                laid.iter()
                    .zip(values)
                    .map(|((k, t), v)| (interner.resolve(**k).to_string(), of(interner, t, v)))
                    .collect(),
            )
        }
        Ty::Option(inner) => match value.option_payload() {
            Some(v) => of(interner, inner, &v),
            None => Json::Null,
        },
        Ty::Result(ok, err) => {
            let (tag, ty, v) = match unsafe { value.as_result() } {
                Ok(v) => ("Ok", ok, v),
                Err(e) => ("Err", err, e),
            };
            Json::Object(Map::from_iter([(tag.to_owned(), of(interner, ty, v))]))
        }
        Ty::Enum { variants, .. } => {
            let variant = unsafe { value.as_variant() };
            // SAFETY: the same witness — a variant's first register is its tag.
            let tag = unsafe { variant.tag().as_tag() };
            let name = interner.resolve(tag).to_string();
            match variants.get(&tag) {
                Some(Some(t)) => {
                    let mut out = Map::new();
                    out.insert(name, of(interner, t, variant.payload()));
                    Json::Object(out)
                }
                _ => Json::from(name),
            }
        }
        other => Json::from(format!("<{}>", other.display(interner))),
    }
}

/// A value read without a type: its `Kind` is the witness for a word, and
/// the vtable's `Composite` for an allocation.
///
/// RFC-0054: a host that declares `!` states no return type, so it has no
/// `Ty` for the value that comes back.
pub fn by_kind(interner: &Interner, value: &Value) -> Json {
    match value.kind() {
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
        Some(Composite::Result) => {
            let (tag, v) = match unsafe { value.as_result() } {
                Ok(v) => ("Ok", v),
                Err(e) => ("Err", e),
            };
            Json::Object(Map::from_iter([(tag.to_owned(), by_kind(interner, v))]))
        }
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
        Some(Composite::Fn | Composite::Handle) | None => {
            Json::from(format!("<{}>", (value.vtable().name)()))
        }
    }
}
