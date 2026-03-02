use crate::value::Value;

pub const BUILTIN_NAMES: &[&str] = &[
    "to_string", "to_int", "to_float", "char_to_int", "int_to_char", "filter", "map", "pmap",
    "find", "reduce", "fold", "any", "all", "len", "reverse", "join", "contains", "substring",
    "len_str", "to_bytes", "to_utf8", "to_utf8_lossy",
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
        "char_to_int" => call_char_to_int(args.into_iter().next().unwrap()),
        "int_to_char" => call_int_to_char(args.into_iter().next().unwrap()),
        "len" => call_len(args),
        "reverse" => call_reverse(args),
        "join" => call_join(args),
        "contains" => call_contains(args),
        "substring" => call_substring(args),
        "len_str" => call_len_str(args),
        "to_bytes" => call_to_bytes(args),
        "to_utf8" => call_to_utf8(args),
        "to_utf8_lossy" => call_to_utf8_lossy(args),
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
        Value::Byte(b) => Value::Int(b as i64),
        _ => unreachable!("to_int: expected Float or Byte, got {arg:?}"),
    }
}

fn call_to_float(arg: Value) -> Value {
    match arg {
        Value::Int(n) => Value::Float(n as f64),
        _ => unreachable!("to_float: expected Int, got {arg:?}"),
    }
}

fn call_char_to_int(arg: Value) -> Value {
    match arg {
        Value::String(s) => {
            let ch = s.chars().next().expect("char_to_int: empty string");
            Value::Int(ch as i64)
        }
        _ => unreachable!("char_to_int: expected String, got {arg:?}"),
    }
}

fn call_int_to_char(arg: Value) -> Value {
    match arg {
        Value::Int(n) => {
            let ch = char::from_u32(n as u32).expect("int_to_char: invalid codepoint");
            Value::String(ch.to_string())
        }
        _ => unreachable!("int_to_char: expected Int, got {arg:?}"),
    }
}

fn call_len(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::List(items) => Value::Int(items.len() as i64),
        v => unreachable!("len: expected List, got {v:?}"),
    }
}

fn call_reverse(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::List(mut items) => {
            items.reverse();
            Value::List(items)
        }
        v => unreachable!("reverse: expected List, got {v:?}"),
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
                    v => unreachable!("join: expected List<String>, got element {v:?}"),
                })
                .collect();
            Value::String(strs.join(&sep))
        }
        (l, s) => unreachable!("join: expected (List<String>, String), got ({l:?}, {s:?})"),
    }
}

fn call_contains(args: Vec<Value>) -> Value {
    let mut it = args.into_iter();
    let list = it.next().unwrap();
    let target = it.next().unwrap();
    match list {
        Value::List(items) => Value::Bool(items.iter().any(|v| values_equal(v, &target))),
        v => unreachable!("contains: expected List, got {v:?}"),
    }
}

fn call_substring(args: Vec<Value>) -> Value {
    let mut it = args.into_iter();
    let s = it.next().unwrap();
    let start = it.next().unwrap();
    let len = it.next().unwrap();
    match (s, start, len) {
        (Value::String(s), Value::Int(start), Value::Int(len)) => {
            let start = start.max(0) as usize;
            let len = len.max(0) as usize;
            let result: String = s.chars().skip(start).take(len).collect();
            Value::String(result)
        }
        (s, start, len) => unreachable!("substring: expected (String, Int, Int), got ({s:?}, {start:?}, {len:?})"),
    }
}

fn call_len_str(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::String(s) => Value::Int(s.chars().count() as i64),
        v => unreachable!("len_str: expected String, got {v:?}"),
    }
}

fn call_to_bytes(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::String(s) => Value::List(
            s.into_bytes().into_iter().map(Value::Byte).collect(),
        ),
        v => unreachable!("to_bytes: expected String, got {v:?}"),
    }
}

fn call_to_utf8(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::List(items) => {
            let bytes: Vec<u8> = items
                .into_iter()
                .map(|v| match v {
                    Value::Byte(b) => b,
                    v => unreachable!("to_utf8: expected List<Byte>, got element {v:?}"),
                })
                .collect();
            Value::String(String::from_utf8(bytes).unwrap())
        }
        v => unreachable!("to_utf8: expected List<Byte>, got {v:?}"),
    }
}

fn call_to_utf8_lossy(args: Vec<Value>) -> Value {
    match args.into_iter().next().unwrap() {
        Value::List(items) => {
            let bytes: Vec<u8> = items
                .into_iter()
                .map(|v| match v {
                    Value::Byte(b) => b,
                    v => unreachable!("to_utf8_lossy: expected List<Byte>, got element {v:?}"),
                })
                .collect();
            Value::String(String::from_utf8_lossy(&bytes).into_owned())
        }
        v => unreachable!("to_utf8_lossy: expected List<Byte>, got {v:?}"),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Byte(a), Value::Byte(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        _ => unreachable!("values_equal: unsupported types ({a:?}, {b:?})"),
    }
}

// -- display ------------------------------------------------------------------

fn value_to_string(v: Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s,
        Value::Bool(b) => b.to_string(),
        Value::Byte(b) => b.to_string(),
        Value::Unit => "()".to_string(),
        _ => unreachable!("to_string: expected scalar or Unit, got {v:?}"),
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
    #[should_panic(expected = "to_string: expected scalar or Unit, got")]
    fn to_string_list_panics() {
        call_to_string(Value::List(vec![Value::Int(1), Value::Int(2)]));
    }

    #[test]
    #[should_panic(expected = "to_string: expected scalar or Unit, got")]
    fn to_string_object_panics() {
        call_to_string(Value::Object(BTreeMap::from([("a".into(), Value::Int(1))])));
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
