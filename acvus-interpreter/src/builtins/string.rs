//! String operations cluster.
//!
//! All string builtins are pure synchronous functions: `String → String` or similar.

use acvus_mir::builtins::BuiltinId;

use super::handler::{BuiltinExecute, sync};

// ── Implementations ────────────────────────────────────────────────

fn trim(s: String) -> String { s.trim().to_string() }
fn trim_start(s: String) -> String { s.trim_start().to_string() }
fn trim_end(s: String) -> String { s.trim_end().to_string() }
fn upper(s: String) -> String { s.to_uppercase() }
fn lower(s: String) -> String { s.to_lowercase() }
fn replace_str(s: String, from: String, to: String) -> String { s.replace(&from, &to) }
fn starts_with_str(s: String, prefix: String) -> bool { s.starts_with(&prefix) }
fn ends_with_str(s: String, suffix: String) -> bool { s.ends_with(&suffix) }
fn repeat_str(s: String, n: i64) -> String { s.repeat(n.max(0) as usize) }
fn contains_str(haystack: String, needle: String) -> bool { haystack.contains(&needle) }
fn len_str(s: String) -> i64 { s.chars().count() as i64 }

fn substring(s: String, start: i64, len: i64) -> String {
    s.chars()
        .skip(start.max(0) as usize)
        .take(len.max(0) as usize)
        .collect()
}

fn split_str(s: String, sep: String) -> Vec<String> {
    s.split(&sep).map(|p| p.to_string()).collect()
}

// ── Registration ───────────────────────────────────────────────────

/// Returns (BuiltinId, execute) pairs for all string builtins.
pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        (BuiltinId::Trim,          sync(trim)),
        (BuiltinId::TrimStart,     sync(trim_start)),
        (BuiltinId::TrimEnd,       sync(trim_end)),
        (BuiltinId::Upper,         sync(upper)),
        (BuiltinId::Lower,         sync(lower)),
        (BuiltinId::ReplaceStr,    sync(replace_str)),
        (BuiltinId::StartsWithStr, sync(starts_with_str)),
        (BuiltinId::EndsWithStr,   sync(ends_with_str)),
        (BuiltinId::RepeatStr,     sync(repeat_str)),
        (BuiltinId::ContainsStr,   sync(contains_str)),
        (BuiltinId::LenStr,        sync(len_str)),
        (BuiltinId::Substring,     sync(substring)),
        (BuiltinId::SplitStr,      sync(split_str)),
    ]
}
