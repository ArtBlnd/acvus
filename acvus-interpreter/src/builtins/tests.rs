use acvus_mir::builtins::BuiltinId;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::value::{PureValue, TypedValue, Value};
use super::get_builtin_impl;

/// Helper: call a sync builtin via the ImplRegistry.
fn call(id: BuiltinId, args: Vec<Value>) -> Value {
    let exec_fn = get_builtin_impl(id)
        .unwrap_or_else(|| panic!("builtin {id:?} not in ImplRegistry"));
    let typed_args: Vec<TypedValue> = args.into_iter()
        .map(|v| TypedValue::new(v, Ty::error()))
        .collect();
    exec_fn(typed_args).expect("builtin call failed").into_inner()
}

#[test]
fn to_string_int() {
    assert!(matches!(call(BuiltinId::ToString, vec![Value::int(42)]), Value::Pure(PureValue::String(s)) if s == "42"));
}

#[test]
fn to_string_float() {
    assert!(matches!(call(BuiltinId::ToString, vec![Value::float(3.14)]), Value::Pure(PureValue::String(s)) if s == "3.14"));
}

#[test]
fn to_string_bool() {
    assert!(matches!(call(BuiltinId::ToString, vec![Value::bool_(true)]), Value::Pure(PureValue::String(s)) if s == "true"));
}

#[test]
fn to_string_string() {
    assert!(matches!(call(BuiltinId::ToString, vec![Value::string("hi")]), Value::Pure(PureValue::String(s)) if s == "hi"));
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit")]
fn to_string_list_panics() {
    call(BuiltinId::ToString, vec![Value::list(vec![Value::int(1), Value::int(2)])]);
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit")]
fn to_string_object_panics() {
    let interner = Interner::new();
    call(BuiltinId::ToString, vec![Value::object(FxHashMap::from_iter([(interner.intern("a"), Value::int(1))]))]);
}

#[test]
fn to_int_float() {
    assert!(matches!(call(BuiltinId::ToInt, vec![Value::float(3.7)]), Value::Pure(PureValue::Int(3))));
}

#[test]
fn to_float_int() {
    assert!(matches!(call(BuiltinId::ToFloat, vec![Value::int(5)]), Value::Pure(PureValue::Float(f)) if f == 5.0));
}

/// Verify that every BuiltinId is handled — either in the ImplRegistry (sync)
/// or in exec_builtin's async match.
#[test]
fn all_builtins_covered() {
    // Async builtins — handled by exec_builtin's match, NOT in ImplRegistry.
    let async_ids: &[BuiltinId] = &[
        BuiltinId::First,
        BuiltinId::Last,
        BuiltinId::Contains,
        BuiltinId::Next,
        BuiltinId::NextSeq,
        BuiltinId::Collect,
        BuiltinId::Join,
        BuiltinId::Find,
        BuiltinId::Reduce,
        BuiltinId::Fold,
        BuiltinId::Any,
        BuiltinId::All,
        BuiltinId::Extend,
    ];

    // All BuiltinId variants.
    let all_ids: &[BuiltinId] = &[
        BuiltinId::Filter, BuiltinId::Map, BuiltinId::Pmap,
        BuiltinId::ToString, BuiltinId::ToFloat, BuiltinId::ToInt,
        BuiltinId::Find, BuiltinId::Reduce, BuiltinId::Fold,
        BuiltinId::Any, BuiltinId::All,
        BuiltinId::Len, BuiltinId::Reverse, BuiltinId::Flatten, BuiltinId::Join,
        BuiltinId::CharToInt, BuiltinId::IntToChar,
        BuiltinId::Contains, BuiltinId::ContainsStr, BuiltinId::Substring,
        BuiltinId::LenStr, BuiltinId::ToBytes, BuiltinId::ToUtf8, BuiltinId::ToUtf8Lossy,
        BuiltinId::Trim, BuiltinId::TrimStart, BuiltinId::TrimEnd,
        BuiltinId::Upper, BuiltinId::Lower,
        BuiltinId::ReplaceStr, BuiltinId::SplitStr,
        BuiltinId::StartsWithStr, BuiltinId::EndsWithStr, BuiltinId::RepeatStr,
        BuiltinId::Unwrap, BuiltinId::First, BuiltinId::Last, BuiltinId::UnwrapOr,
        BuiltinId::Iter, BuiltinId::RevIter, BuiltinId::Collect,
        BuiltinId::Take, BuiltinId::Skip, BuiltinId::Chain,
        BuiltinId::Append, BuiltinId::Extend, BuiltinId::Consume,
        BuiltinId::FlatMap,
        BuiltinId::TakeSeq, BuiltinId::SkipSeq, BuiltinId::ChainSeq,
        BuiltinId::Next, BuiltinId::NextSeq,
    ];

    let async_set: std::collections::HashSet<BuiltinId> = async_ids.iter().copied().collect();

    for &id in all_ids {
        let in_registry = get_builtin_impl(id).is_some();
        let is_async = async_set.contains(&id);
        assert!(
            in_registry || is_async,
            "BuiltinId::{:?} is neither in ImplRegistry nor in async list",
            id,
        );
        // Should not be in both
        assert!(
            !(in_registry && is_async),
            "BuiltinId::{:?} is in BOTH ImplRegistry and async list — pick one",
            id,
        );
    }
}
