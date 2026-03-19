use acvus_ast::Literal;
use acvus_interpreter::{LazyValue, PureValue, TypedValue, Value};
use acvus_mir::analysis::reachable_context::KnownValue;
use acvus_utils::Interner;

pub fn json_to_value(interner: &Interner, v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::unit(),
        serde_json::Value::Bool(b) => Value::bool_(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::int(i)
            } else if let Some(u) = n.as_u64() {
                // u64 values > i64::MAX: store as float (lossy for very large u64,
                // but no fabrication — the value originated from the JSON source).
                Value::float(u as f64)
            } else {
                // Must be an f64.
                Value::float(n.as_f64().expect("serde_json Number must be representable as i64, u64, or f64"))
            }
        }
        serde_json::Value::String(s) => Value::string(s.clone()),
        serde_json::Value::Array(arr) => {
            Value::list(arr.iter().map(|v| json_to_value(interner, v)).collect())
        }
        serde_json::Value::Object(obj) => Value::object(
            obj.iter()
                .map(|(k, v)| (interner.intern(k), json_to_value(interner, v)))
                .collect(),
        ),
    }
}

pub fn value_to_literal(value: &TypedValue) -> Option<Literal> {
    match value.value() {
        Value::Pure(PureValue::String(s)) => Some(Literal::String(s.clone())),
        Value::Pure(PureValue::Bool(b)) => Some(Literal::Bool(*b)),
        Value::Pure(PureValue::Int(i)) => Some(Literal::Int(*i)),
        Value::Pure(PureValue::Float(f)) => Some(Literal::Float(*f)),
        _ => None,
    }
}

pub fn value_to_known(value: &TypedValue) -> Option<KnownValue> {
    value_to_known_inner(value.value())
}

fn value_to_known_inner(value: &Value) -> Option<KnownValue> {
    match value {
        Value::Pure(PureValue::String(s)) => Some(KnownValue::Literal(Literal::String(s.clone()))),
        Value::Pure(PureValue::Bool(b)) => Some(KnownValue::Literal(Literal::Bool(*b))),
        Value::Pure(PureValue::Int(i)) => Some(KnownValue::Literal(Literal::Int(*i))),
        Value::Pure(PureValue::Float(f)) => Some(KnownValue::Literal(Literal::Float(*f))),
        Value::Lazy(LazyValue::Variant { tag, payload }) => {
            let payload = match payload {
                Some(p) => Some(Box::new(value_to_known_inner(p)?)),
                None => None,
            };
            Some(KnownValue::Variant { tag: *tag, payload })
        }
        _ => None,
    }
}
