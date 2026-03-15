use acvus_mir::builtins::BuiltinId;
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use super::*;
use crate::value::{LazyValue, PureValue};

#[test]
fn to_string_int() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::int(42)]).unwrap(), Value::Pure(PureValue::String(s)) if s == "42")
    );
}

#[test]
fn to_string_float() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::float(3.14)]).unwrap(), Value::Pure(PureValue::String(s)) if s == "3.14")
    );
}

#[test]
fn to_string_bool() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::bool_(true)]).unwrap(), Value::Pure(PureValue::String(s)) if s == "true")
    );
}

#[test]
fn to_string_string() {
    assert!(
        matches!(call_pure(BuiltinId::ToString, vec![Value::string("hi".into())]).unwrap(), Value::Pure(PureValue::String(s)) if s == "hi")
    );
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_list_panics() {
    call_pure(
        BuiltinId::ToString,
        vec![Value::list(vec![Value::int(1), Value::int(2)])],
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_object_panics() {
    let interner = Interner::new();
    call_pure(
        BuiltinId::ToString,
        vec![Value::object(FxHashMap::from_iter([(
            interner.intern("a"),
            Value::int(1),
        )]))],
    )
    .unwrap();
}

#[test]
fn to_int_float() {
    assert!(matches!(
        call_pure(BuiltinId::ToInt, vec![Value::float(3.7)]).unwrap(),
        Value::Pure(PureValue::Int(3))
    ));
}

#[test]
fn to_float_int() {
    assert!(
        matches!(call_pure(BuiltinId::ToFloat, vec![Value::int(5)]).unwrap(), Value::Pure(PureValue::Float(f)) if f == 5.0)
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
    "reverse", "iter", "rev_iter", "collect", "take", "skip", "chain",
    "flat_map", "flatten", "join",
    "append", "extend", "consume",
];

#[test]
fn all_mir_builtins_handled() {
    let registry = acvus_mir::builtins::registry();
    let handled: FxHashSet<&str> = PURE_NAMES.iter().chain(HOF_NAMES.iter()).copied().collect();

    // No name should appear in both lists.
    let pure_set: FxHashSet<&str> = PURE_NAMES.iter().copied().collect();
    let hof_set: FxHashSet<&str> = HOF_NAMES.iter().copied().collect();
    let overlap: Vec<_> = pure_set.intersection(&hof_set).collect();
    assert!(overlap.is_empty(), "names appear in both PURE and HOF: {overlap:?}");

    // Every name the interpreter handles should exist in the registry.
    for name in &handled {
        assert!(
            !registry.candidates(name).is_empty(),
            "interpreter handles `{name}` but it is not registered in MIR",
        );
    }

    // Every registered builtin must appear in exactly one of the two lists.
    let all_ids: &[BuiltinId] = &[
        BuiltinId::Filter,
        BuiltinId::Map,
        BuiltinId::Pmap,
        BuiltinId::ToString,
        BuiltinId::ToFloat,
        BuiltinId::ToInt,
        BuiltinId::Find,
        BuiltinId::Reduce,
        BuiltinId::Fold,
        BuiltinId::Any,
        BuiltinId::All,
        BuiltinId::Len,
        BuiltinId::Reverse,
        BuiltinId::Flatten,
        BuiltinId::Join,
        BuiltinId::CharToInt,
        BuiltinId::IntToChar,
        BuiltinId::Contains,
        BuiltinId::ContainsStr,
        BuiltinId::Substring,
        BuiltinId::LenStr,
        BuiltinId::ToBytes,
        BuiltinId::ToUtf8,
        BuiltinId::ToUtf8Lossy,
        BuiltinId::Trim,
        BuiltinId::TrimStart,
        BuiltinId::TrimEnd,
        BuiltinId::Upper,
        BuiltinId::Lower,
        BuiltinId::ReplaceStr,
        BuiltinId::SplitStr,
        BuiltinId::StartsWithStr,
        BuiltinId::EndsWithStr,
        BuiltinId::RepeatStr,
        BuiltinId::Unwrap,
        BuiltinId::First,
        BuiltinId::Last,
        BuiltinId::UnwrapOr,
        BuiltinId::Iter,
        BuiltinId::RevIter,
        BuiltinId::Collect,
        BuiltinId::Take,
        BuiltinId::Skip,
        BuiltinId::Chain,
        BuiltinId::Append,
        BuiltinId::Extend,
        BuiltinId::Consume,
        BuiltinId::FlattenIter,
        BuiltinId::FlatMap,
        BuiltinId::FlatMapIter,
        BuiltinId::JoinIter,
        BuiltinId::ContainsIter,
        BuiltinId::FirstIter,
        BuiltinId::LastIter,
    ];
    for &id in all_ids {
        let name = id.name();
        assert!(
            handled.contains(name),
            "MIR builtin `{name}` ({id:?}) is not covered by PURE_NAMES or HOF_NAMES",
        );
    }
}
