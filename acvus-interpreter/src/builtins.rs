mod handler;
mod pure;

use std::collections::HashSet;
use std::sync::LazyLock;

use handler::PureBuiltin;
use pure::*;

use crate::value::Value;

static BUILTIN_NAMES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    acvus_mir::builtins::builtins()
        .iter()
        .map(|b| b.name())
        .collect()
});

pub fn is_builtin(name: &str) -> bool {
    BUILTIN_NAMES.contains(name)
}

/// Dispatch a pure (non-HOF) builtin.
pub fn call_pure(name: &str, args: Vec<Value>) -> Value {
    match name {
        "to_string" => PureBuiltin::call(builtin_to_string, args),
        "to_int" => PureBuiltin::call(builtin_to_int, args),
        "to_float" => PureBuiltin::call(builtin_to_float, args),
        "char_to_int" => PureBuiltin::call(builtin_char_to_int, args),
        "int_to_char" => PureBuiltin::call(builtin_int_to_char, args),
        "len" => PureBuiltin::call(builtin_len, args),
        "reverse" => PureBuiltin::call(builtin_reverse, args),
        "flatten" => PureBuiltin::call(builtin_flatten, args),
        "join" => PureBuiltin::call(builtin_join, args),
        "contains" => PureBuiltin::call(builtin_contains, args),
        "contains_str" => PureBuiltin::call(builtin_contains_str, args),
        "substring" => PureBuiltin::call(builtin_substring, args),
        "len_str" => PureBuiltin::call(builtin_len_str, args),
        "to_bytes" => PureBuiltin::call(builtin_to_bytes, args),
        "to_utf8" => PureBuiltin::call(builtin_to_utf8, args),
        "to_utf8_lossy" => PureBuiltin::call(builtin_to_utf8_lossy, args),
        _ => panic!("not a pure builtin: {name}"),
    }
}

#[cfg(test)]
mod tests;
