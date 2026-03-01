use crate::value::Value;

pub const BUILTIN_NAMES: &[&str] = &[
    "to_string", "to_int", "to_float", "filter", "map", "pmap",
    "find", "reduce", "fold", "any", "all", "len", "reverse", "join", "contains",
];

pub fn is_builtin(name: &str) -> bool {
    BUILTIN_NAMES.contains(&name)
}

/// Dispatch a pure (non-HOF) builtin.
pub fn call_pure(name: &str, args: Vec<Value>) -> Value {
    match name {
        "to_string" => call_to_string(args.into_iter().next().unwrap()),
        "to_int" => call_to_int(args.into_iter().next().unwrap()),
        "to_float" => call_to_float(args.into_iter().next().unwrap()),
        "len" => call_len(args),
        "reverse" => call_reverse(args),
        "join" => call_join(args),
        "contains" => call_contains(args),
        _ => panic!("not a pure builtin: {name}"),
    }
}

// -- pure builtins ------------------------------------------------------------

fn call_to_string(arg: Value) -> Value {
    Value::String(value_to_string(arg))
}

fn call_to_int(arg: Value) -> Value {
    match arg {
        Value::Float(f) => Value::Int(f as i64),
        _ => panic!("to_int: expected Float, got {arg:?}"),
    }
}

fn call_to_float(arg: Value) -> Value {
    match arg {
        Value::Int(n) => Value::Float(n as f64),
        _ => panic!("to_float: expected Int, got {arg:?}"),
    }
}

fn call_len(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::List(items) => Value::Int(items.len() as i64),
        v => panic!("len: expected List, got {v:?}"),
    }
}

fn call_reverse(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::List(mut items) => {
            items.reverse();
            Value::List(items)
        }
        v => panic!("reverse: expected List, got {v:?}"),
    }
}

fn call_join(args: Vec<Value>) -> Value {
    let mut it = args.into_iter();
    let list = it.next().unwrap();
    let sep = it.next().unwrap();
    match (list, sep) {
        (Value::List(items), Value::String(sep)) => {
            let strs: Vec<String> = items
                .into_iter()
                .map(|v| match v {
                    Value::String(s) => s,
                    v => panic!("join: expected List<String>, got element {v:?}"),
                })
                .collect();
            Value::String(strs.join(&sep))
        }
        (l, s) => panic!("join: expected (List<String>, String), got ({l:?}, {s:?})"),
    }
}

fn call_contains(args: Vec<Value>) -> Value {
    let mut it = args.into_iter();
    let list = it.next().unwrap();
    let target = it.next().unwrap();
    match list {
        Value::List(items) => Value::Bool(items.iter().any(|v| values_equal(v, &target))),
        v => panic!("contains: expected List, got {v:?}"),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

// -- display ------------------------------------------------------------------

fn value_to_string(v: Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s,
        Value::Bool(b) => b.to_string(),
        Value::Unit => "()".to_string(),
        Value::Range {
            start,
            end,
            inclusive,
        } => {
            if inclusive {
                format!("{start}..={end}")
            } else {
                format!("{start}..{end}")
            }
        }
        Value::List(items) => {
            let inner: Vec<String> = items.into_iter().map(value_to_string).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(fields) => {
            let inner: Vec<String> = fields
                .into_iter()
                .map(|(k, v)| format!("{k}: {}", value_to_string(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Tuple(elems) => {
            let inner: Vec<String> = elems.into_iter().map(value_to_string).collect();
            format!("({})", inner.join(", "))
        }
        Value::Fn(_) => panic!("cannot convert Fn to string"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn to_string_int() {
        assert!(matches!(call_to_string(Value::Int(42)), Value::String(s) if s == "42"));
    }

    #[test]
    fn to_string_float() {
        assert!(matches!(call_to_string(Value::Float(3.14)), Value::String(s) if s == "3.14"));
    }

    #[test]
    fn to_string_bool() {
        assert!(matches!(call_to_string(Value::Bool(true)), Value::String(s) if s == "true"));
    }

    #[test]
    fn to_string_string() {
        assert!(
            matches!(call_to_string(Value::String("hi".into())), Value::String(s) if s == "hi")
        );
    }

    #[test]
    fn to_string_list() {
        let v = Value::List(vec![Value::Int(1), Value::Int(2)]);
        assert!(matches!(call_to_string(v), Value::String(s) if s == "[1, 2]"));
    }

    #[test]
    fn to_string_object() {
        let v = Value::Object(BTreeMap::from([("a".into(), Value::Int(1))]));
        assert!(matches!(call_to_string(v), Value::String(s) if s == "{a: 1}"));
    }

    #[test]
    fn to_int_float() {
        assert!(matches!(call_to_int(Value::Float(3.7)), Value::Int(3)));
    }

    #[test]
    fn to_float_int() {
        assert!(matches!(call_to_float(Value::Int(5)), Value::Float(f) if f == 5.0));
    }
}
