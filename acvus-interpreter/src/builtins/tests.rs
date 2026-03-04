use std::collections::{BTreeMap, HashSet};

use acvus_mir::builtins::BuiltinId;

use super::*;

#[test]
fn to_string_int() {
    assert!(matches!(call_pure(BuiltinId::ToString, vec![Value::Int(42)]), Value::String(s) if s == "42"));
}

#[test]
fn to_string_float() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::Float(3.14)]), Value::String(s) if s == "3.14")
    );
}

#[test]
fn to_string_bool() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::Bool(true)]), Value::String(s) if s == "true")
    );
}

#[test]
fn to_string_string() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::String("hi".into())]), Value::String(s) if s == "hi")
    );
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_list_panics() {
    call_pure(
        BuiltinId::ToString,
        vec![Value::List(vec![Value::Int(1), Value::Int(2)])],
    );
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_object_panics() {
    call_pure(
        BuiltinId::ToString,
        vec![Value::Object(BTreeMap::from([("a".into(), Value::Int(1))]))],
    );
}

#[test]
fn to_int_float() {
    assert!(matches!(
        call_pure(BuiltinId::ToInt, vec![Value::Float(3.7)]),
        Value::Int(3)
    ));
}

#[test]
fn to_float_int() {
    assert!(matches!(call_pure(BuiltinId::ToFloat, vec![Value::Int(5)]), Value::Float(f) if f == 5.0));
}

/// Names of pure (non-HOF) builtins dispatched by `call_pure`.
const PURE_NAMES: &[&str] = &[
    "to_string",
    "to_int",
    "to_float",
    "char_to_int",
    "int_to_char",
    "len",
    "reverse",
    "flatten",
    "join",
    "contains",
    "contains_str",
    "substring",
    "len_str",
    "to_bytes",
    "to_utf8",
    "to_utf8_lossy",
    "trim",
    "trim_start",
    "trim_end",
    "upper",
    "lower",
    "replace_str",
    "split_str",
    "starts_with_str",
    "ends_with_str",
    "repeat_str",
    "unwrap",
];

/// Names of higher-order function builtins dispatched by `exec_builtin`.
const HOF_NAMES: &[&str] = &[
    "filter", "map", "pmap", "find", "reduce", "fold", "any", "all",
];

#[test]
fn all_mir_builtins_handled() {
    let mir_names: HashSet<&str> = acvus_mir::builtins::builtins()
        .iter()
        .map(|(_, b)| b.name())
        .collect();
    let handled: HashSet<&str> = PURE_NAMES.iter().chain(HOF_NAMES.iter()).copied().collect();
    let missing: Vec<&&str> = mir_names.difference(&handled).collect();
    assert!(
        missing.is_empty(),
        "builtins registered in MIR but not handled in interpreter: {missing:?}",
    );
    let extra: Vec<&&str> = handled.difference(&mir_names).collect();
    assert!(
        extra.is_empty(),
        "builtins handled in interpreter but not registered in MIR: {extra:?}",
    );
}
