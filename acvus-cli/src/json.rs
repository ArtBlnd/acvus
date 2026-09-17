//! A runtime value as JSON, read by its type.

use acvus_interpreter::Value;
use acvus_mir::ty::Ty;
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
            let values = unsafe { value.as_object() };
            let mut out = Map::new();
            let mut keys: Vec<_> = fields.iter().collect();
            keys.sort_by_key(|(k, _)| interner.resolve(**k).to_string());
            for (k, t) in keys {
                let v = values
                    .get(k)
                    .expect("an object value holds every field of its type");
                out.insert(interner.resolve(*k).to_string(), of(interner, t, v));
            }
            Json::Object(out)
        }
        Ty::Option(inner) => match unsafe { value.as_option() } {
            Some(v) => of(interner, inner, v),
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
            let tag = interner.resolve(variant.tag).to_string();
            match (&variant.payload, variants.get(&variant.tag)) {
                (Some(payload), Some(Some(t))) => {
                    let mut out = Map::new();
                    out.insert(tag, of(interner, t, payload));
                    Json::Object(out)
                }
                _ => Json::from(tag),
            }
        }
        other => Json::from(format!("<{}>", other.display(interner))),
    }
}
