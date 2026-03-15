mod handler;
mod pure;

pub use handler::{FromValue, IntoValue};

use handler::PureBuiltin;
use pure::*;

use crate::error::RuntimeError;
use crate::value::Value;
use acvus_mir::builtins::BuiltinId;

/// Dispatch a pure (non-HOF) builtin by ID.
pub fn call_pure(id: BuiltinId, args: Vec<Value>) -> Result<Value, RuntimeError> {
    match id {
        BuiltinId::ToString => PureBuiltin::call(builtin_to_string, args),
        BuiltinId::ToInt => PureBuiltin::call(builtin_to_int, args),
        BuiltinId::ToFloat => PureBuiltin::call(builtin_to_float, args),
        BuiltinId::CharToInt => PureBuiltin::call(builtin_char_to_int, args),
        BuiltinId::IntToChar => PureBuiltin::call(builtin_int_to_char, args),
        BuiltinId::Len => PureBuiltin::call(builtin_len, args),
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
        BuiltinId::First => PureBuiltin::call(builtin_first, args),
        BuiltinId::Last => PureBuiltin::call(builtin_last, args),
        BuiltinId::UnwrapOr => PureBuiltin::call(builtin_unwrap_or, args),
        BuiltinId::Filter
        | BuiltinId::Map
        | BuiltinId::Pmap
        | BuiltinId::Find
        | BuiltinId::Reduce
        | BuiltinId::Fold
        | BuiltinId::Any
        | BuiltinId::All
        | BuiltinId::Reverse
        | BuiltinId::Iter
        | BuiltinId::RevIter
        | BuiltinId::Collect
        | BuiltinId::Take
        | BuiltinId::Skip
        | BuiltinId::Chain
        | BuiltinId::Flatten
        | BuiltinId::FlattenIter
        | BuiltinId::FlatMap
        | BuiltinId::FlatMapIter
        | BuiltinId::Join
        | BuiltinId::JoinIter
        | BuiltinId::ContainsIter
        | BuiltinId::FirstIter
        | BuiltinId::LastIter
        | BuiltinId::Append
        | BuiltinId::Extend
        | BuiltinId::Consume
        | BuiltinId::MapSeq
        | BuiltinId::PmapSeq
        | BuiltinId::FilterSeq
        | BuiltinId::TakeSeq
        | BuiltinId::SkipSeq
        | BuiltinId::ChainSeq
        | BuiltinId::FlattenSeq
        | BuiltinId::FlatMapSeq
        | BuiltinId::FlatMapIterSeq
        | BuiltinId::CollectSeq
        | BuiltinId::RevSeq
        | BuiltinId::Next
        | BuiltinId::NextSeq => Err(RuntimeError::other(format!(
            "not a pure builtin: {}",
            id.name()
        ))),
    }
}

#[cfg(test)]
mod tests;
