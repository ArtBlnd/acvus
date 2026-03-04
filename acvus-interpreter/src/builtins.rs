mod handler;
mod pure;

pub use handler::{FromValue, IntoValue};

use handler::PureBuiltin;
use pure::*;

use acvus_mir::builtins::BuiltinId;
use crate::value::Value;

/// Dispatch a pure (non-HOF) builtin by ID.
pub fn call_pure(id: BuiltinId, args: Vec<Value>) -> Value {
    match id {
        BuiltinId::ToString => PureBuiltin::call(builtin_to_string, args),
        BuiltinId::ToInt => PureBuiltin::call(builtin_to_int, args),
        BuiltinId::ToFloat => PureBuiltin::call(builtin_to_float, args),
        BuiltinId::CharToInt => PureBuiltin::call(builtin_char_to_int, args),
        BuiltinId::IntToChar => PureBuiltin::call(builtin_int_to_char, args),
        BuiltinId::Len => PureBuiltin::call(builtin_len, args),
        BuiltinId::Reverse => PureBuiltin::call(builtin_reverse, args),
        BuiltinId::Flatten => PureBuiltin::call(builtin_flatten, args),
        BuiltinId::Join => PureBuiltin::call(builtin_join, args),
        BuiltinId::Contains => PureBuiltin::call(builtin_contains, args),
        BuiltinId::ContainsStr => PureBuiltin::call(builtin_contains_str, args),
        BuiltinId::Substring => PureBuiltin::call(builtin_substring, args),
        BuiltinId::LenStr => PureBuiltin::call(builtin_len_str, args),
        BuiltinId::ToBytes => PureBuiltin::call(builtin_to_bytes, args),
        BuiltinId::ToUtf8 => PureBuiltin::call(builtin_to_utf8, args),
        BuiltinId::ToUtf8Lossy => PureBuiltin::call(builtin_to_utf8_lossy, args),
        BuiltinId::Trim => PureBuiltin::call(builtin_trim, args),
        BuiltinId::TrimStart => PureBuiltin::call(builtin_trim_start, args),
        BuiltinId::TrimEnd => PureBuiltin::call(builtin_trim_end, args),
        BuiltinId::Upper => PureBuiltin::call(builtin_upper, args),
        BuiltinId::Lower => PureBuiltin::call(builtin_lower, args),
        BuiltinId::ReplaceStr => PureBuiltin::call(builtin_replace_str, args),
        BuiltinId::SplitStr => PureBuiltin::call(builtin_split_str, args),
        BuiltinId::StartsWithStr => PureBuiltin::call(builtin_starts_with_str, args),
        BuiltinId::EndsWithStr => PureBuiltin::call(builtin_ends_with_str, args),
        BuiltinId::RepeatStr => PureBuiltin::call(builtin_repeat_str, args),
        BuiltinId::Unwrap => PureBuiltin::call(builtin_unwrap, args),
        other => panic!("not a pure builtin: {}", other.name()),
    }
}

#[cfg(test)]
mod tests;
