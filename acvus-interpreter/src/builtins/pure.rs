use crate::value::Value;

pub(crate) fn builtin_to_string(v: Value) -> Value {
    Value::String(value_to_string(v))
}

pub(crate) fn builtin_to_int(v: Value) -> Value {
    match v {
        Value::Float(f) => Value::Int(f as i64),
        Value::Byte(b) => Value::Int(b as i64),
        _ => unreachable!("to_int: expected Float or Byte, got {v:?}"),
    }
}

pub(crate) fn builtin_to_float(n: i64) -> f64 {
    n as f64
}

pub(crate) fn builtin_char_to_int(s: String) -> i64 {
    s.chars().next().expect("char_to_int: empty string") as i64
}

pub(crate) fn builtin_int_to_char(n: i64) -> String {
    char::from_u32(n as u32)
        .expect("int_to_char: invalid codepoint")
        .to_string()
}

pub(crate) fn builtin_len(items: Vec<Value>) -> i64 {
    items.len() as i64
}

pub(crate) fn builtin_reverse(items: Vec<Value>) -> Value {
    let mut items = items;
    items.reverse();
    Value::List(items)
}

pub(crate) fn builtin_flatten(items: Vec<Value>) -> Value {
    let mut out = Vec::new();
    for item in items {
        match item {
            Value::List(inner) => out.extend(inner),
            other => out.push(other),
        }
    }
    Value::List(out)
}

pub(crate) fn builtin_join(items: Vec<String>, sep: String) -> String {
    items.join(&sep)
}

pub(crate) fn builtin_contains(items: Vec<Value>, target: Value) -> bool {
    items.iter().any(|v| values_equal(v, &target))
}

pub(crate) fn builtin_contains_str(haystack: String, needle: String) -> bool {
    haystack.contains(&needle)
}

pub(crate) fn builtin_substring(s: String, start: i64, len: i64) -> String {
    s.chars()
        .skip(start.max(0) as usize)
        .take(len.max(0) as usize)
        .collect()
}

pub(crate) fn builtin_len_str(s: String) -> i64 {
    s.chars().count() as i64
}

pub(crate) fn builtin_to_bytes(s: String) -> Value {
    Value::List(s.into_bytes().into_iter().map(Value::Byte).collect())
}

pub(crate) fn builtin_to_utf8(bytes: Vec<u8>) -> Option<String> {
    String::from_utf8(bytes).ok()
}

pub(crate) fn builtin_to_utf8_lossy(bytes: Vec<u8>) -> String {
    String::from_utf8_lossy(&bytes).into_owned()
}

pub(crate) fn builtin_trim(s: String) -> String {
    s.trim().to_string()
}

pub(crate) fn builtin_trim_start(s: String) -> String {
    s.trim_start().to_string()
}

pub(crate) fn builtin_trim_end(s: String) -> String {
    s.trim_end().to_string()
}

pub(crate) fn builtin_upper(s: String) -> String {
    s.to_uppercase()
}

pub(crate) fn builtin_lower(s: String) -> String {
    s.to_lowercase()
}

pub(crate) fn builtin_replace_str(s: String, from: String, to: String) -> String {
    s.replace(&from, &to)
}

pub(crate) fn builtin_split_str(s: String, sep: String) -> Value {
    Value::List(
        s.split(&sep)
            .map(|p| Value::String(p.to_string()))
            .collect(),
    )
}

pub(crate) fn builtin_starts_with_str(s: String, prefix: String) -> bool {
    s.starts_with(&prefix)
}

pub(crate) fn builtin_ends_with_str(s: String, suffix: String) -> bool {
    s.ends_with(&suffix)
}

pub(crate) fn builtin_repeat_str(s: String, n: i64) -> String {
    s.repeat(n.max(0) as usize)
}

pub(crate) fn builtin_first(items: Vec<Value>) -> Option<Value> {
    items.into_iter().next()
}

pub(crate) fn builtin_last(items: Vec<Value>) -> Option<Value> {
    items.into_iter().next_back()
}

pub(crate) fn builtin_unwrap_or(v: Option<Value>, default: Value) -> Value {
    v.unwrap_or(default)
}

pub(crate) fn builtin_unwrap(v: Value) -> Value {
    let interner =
        crate::interner_ctx::get_interner().expect("builtin_unwrap: requires interner context");
    let some_tag = interner.intern("Some");
    let none_tag = interner.intern("None");
    match v {
        Value::Variant {
            tag,
            payload: Some(inner),
        } if tag == some_tag => *inner,
        Value::Variant { tag, .. } if tag == none_tag => {
            panic!("unwrap: called on None")
        }
        _ => panic!("unwrap: expected Option variant, got {v:?}"),
    }
}

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
