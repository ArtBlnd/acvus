use acvus_ast::Literal;
use acvus_interpreter::Value;

pub fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        ),
    }
}

pub fn value_to_literal(value: &Value) -> Option<Literal> {
    match value {
        Value::String(s) => Some(Literal::String(s.clone())),
        Value::Bool(b) => Some(Literal::Bool(*b)),
        Value::Int(i) => Some(Literal::Int(*i)),
        Value::Float(f) => Some(Literal::Float(*f)),
        _ => None,
    }
}
