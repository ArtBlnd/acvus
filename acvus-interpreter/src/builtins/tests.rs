use acvus_mir::builtins::BuiltinId;
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

#[test]
fn to_string_int() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::Int(42)]).unwrap(), Value::String(s) if s == "42")
    );
}

#[test]
fn to_string_float() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::Float(3.14)]).unwrap(), Value::String(s) if s == "3.14")
    );
}

#[test]
fn to_string_bool() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::Bool(true)]).unwrap(), Value::String(s) if s == "true")
    );
}

#[test]
fn to_string_string() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::String("hi".into())]).unwrap(), Value::String(s) if s == "hi")
    );
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_list_panics() {
    call_pure(
        BuiltinId::ToString,
        vec![Value::List(vec![Value::Int(1), Value::Int(2)])],
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_object_panics() {
    let interner = Interner::new();
    call_pure(
        BuiltinId::ToString,
        vec![Value::Object(FxHashMap::from_iter([(
            interner.intern("a"),
            Value::Int(1),
        )]))],
    )
    .unwrap();
}

#[test]
fn to_int_float() {
    assert!(matches!(
        call_pure(BuiltinId::ToInt, vec![Value::Float(3.7)]).unwrap(),
        Value::Int(3)
    ));
}

#[test]
fn to_float_int() {
    assert!(
        matches!(call_pure(BuiltinId::ToFloat, vec![Value::Int(5)]).unwrap(), Value::Float(f) if f == 5.0)
    );
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
    "first",
    "last",
    "unwrap_or",
];

/// Names of higher-order function / iterator builtins dispatched by `exec_builtin`.
const HOF_NAMES: &[&str] = &[
    "filter", "map", "pmap", "find", "reduce", "fold", "any", "all",
    "iter", "rev_iter", "collect", "take", "skip", "chain",
    "flat_map",
];

#[test]
fn all_mir_builtins_handled() {
    let registry = acvus_mir::builtins::registry();
    let handled: FxHashSet<&str> = PURE_NAMES.iter().chain(HOF_NAMES.iter()).copied().collect();
    // Every name the interpreter handles should exist in the registry.
    for name in &handled {
        assert!(
            !registry.candidates(name).is_empty(),
            "interpreter handles `{name}` but it is not registered in MIR",
        );
    }
}
