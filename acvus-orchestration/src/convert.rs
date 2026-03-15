use acvus_ast::Literal;
use acvus_interpreter::{LazyValue, PureValue, Value};
use acvus_mir_pass::analysis::reachable_context::KnownValue;
use acvus_utils::Interner;

pub fn json_to_value(interner: &Interner, v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::unit(),
        serde_json::Value::Bool(b) => Value::bool_(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::int(i)
            } else {
                Value::float(n.as_f64().unwrap_or(0.0))
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

pub fn value_to_literal(value: &Value) -> Option<Literal> {
    match value {
        Value::Pure(PureValue::String(s)) => Some(Literal::String(s.clone())),
        Value::Pure(PureValue::Bool(b)) => Some(Literal::Bool(*b)),
        Value::Pure(PureValue::Int(i)) => Some(Literal::Int(*i)),
        Value::Pure(PureValue::Float(f)) => Some(Literal::Float(*f)),
        _ => None,
    }
}

pub fn value_to_known(value: &Value) -> Option<KnownValue> {
    match value {
        Value::Pure(PureValue::String(s)) => Some(KnownValue::Literal(Literal::String(s.clone()))),
        Value::Pure(PureValue::Bool(b)) => Some(KnownValue::Literal(Literal::Bool(*b))),
        Value::Pure(PureValue::Int(i)) => Some(KnownValue::Literal(Literal::Int(*i))),
        Value::Pure(PureValue::Float(f)) => Some(KnownValue::Literal(Literal::Float(*f))),
        Value::Lazy(LazyValue::Variant { tag, payload }) => {
            let payload = match payload {
                Some(p) => Some(Box::new(value_to_known(p)?)),
                None => None,
            };
            Some(KnownValue::Variant { tag: *tag, payload })
        }
        _ => None,
    }
}
