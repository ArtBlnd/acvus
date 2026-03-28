//! Pomollu-specific builtin type definitions.
//!
//! These types are pomollu application concepts — NOT generic LSP types.
//! The LSP layer (acvus-lsp) knows nothing about these.

use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

/// Tag entry: `{key: String, value: String}`.
fn tag_entry_type(interner: &Interner) -> Ty {
    let mut fields = FxHashMap::default();
    fields.insert(interner.intern("key"), Ty::String);
    fields.insert(interner.intern("value"), Ty::String);
    Ty::Object(fields)
}

/// Context entry: `{name, description, tags, content, content_type}`.
fn context_entry_type(interner: &Interner) -> Ty {
    let mut fields = FxHashMap::default();
    fields.insert(interner.intern("name"), Ty::String);
    fields.insert(interner.intern("description"), Ty::String);
    fields.insert(
        interner.intern("tags"),
        Ty::List(Box::new(tag_entry_type(interner))),
    );
    fields.insert(interner.intern("content"), Ty::String);
    fields.insert(interner.intern("content_type"), Ty::String);
    Ty::Object(fields)
}

/// Custom context entry: same as context entry + `type` field.
fn context_custom_entry_type(interner: &Interner) -> Ty {
    let mut fields = FxHashMap::default();
    fields.insert(interner.intern("name"), Ty::String);
    fields.insert(interner.intern("description"), Ty::String);
    fields.insert(
        interner.intern("tags"),
        Ty::List(Box::new(tag_entry_type(interner))),
    );
    fields.insert(interner.intern("content"), Ty::String);
    fields.insert(interner.intern("content_type"), Ty::String);
    fields.insert(interner.intern("type"), Ty::String);
    Ty::Object(fields)
}

/// Type of `@context` — the canonical definition.
///
/// ```text
/// {
///     system: List<ContextEntry>,
///     character: List<ContextEntry>,
///     world_info: List<ContextEntry>,
///     lorebook: List<ContextEntry>,
///     memory: List<ContextEntry>,
///     custom: List<ContextCustomEntry>,
///     bot_name: String,
/// }
/// ```
pub fn context_type(interner: &Interner) -> Ty {
    let entry = context_entry_type(interner);
    let custom_entry = context_custom_entry_type(interner);
    let mut fields = FxHashMap::default();
    fields.insert(interner.intern("system"), Ty::List(Box::new(entry.clone())));
    fields.insert(
        interner.intern("character"),
        Ty::List(Box::new(entry.clone())),
    );
    fields.insert(
        interner.intern("world_info"),
        Ty::List(Box::new(entry.clone())),
    );
    fields.insert(
        interner.intern("lorebook"),
        Ty::List(Box::new(entry.clone())),
    );
    fields.insert(interner.intern("memory"), Ty::List(Box::new(entry)));
    fields.insert(interner.intern("custom"), Ty::List(Box::new(custom_entry)));
    fields.insert(interner.intern("bot_name"), Ty::String);
    Ty::Object(fields)
}

/// Type of `@history` — `List<{content, content_type, role}>`.
pub fn history_entry_type(interner: &Interner) -> Ty {
    let mut fields = FxHashMap::default();
    fields.insert(interner.intern("content"), Ty::String);
    fields.insert(interner.intern("content_type"), Ty::String);
    fields.insert(interner.intern("role"), Ty::String);
    Ty::List(Box::new(Ty::Object(fields)))
}

/// Builtin context refs — automatically injected, NOT user-defined params.
///
/// - `turn_index`: engine internal
/// - `raw`/`self`/`content`: node-internal variables
/// - `context`: @context object
/// - `item`/`index`: iterator loop variables
pub fn builtin_context_refs(interner: &Interner) -> FxHashSet<Astr> {
    [
        "turn_index",
        "raw",
        "self",
        "content",
        "context",
        "item",
        "index",
    ]
    .iter()
    .map(|n| interner.intern(n))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_type_has_all_required_fields() {
        let interner = Interner::new();
        let ty = context_type(&interner);
        let Ty::Object(fields) = &ty else {
            panic!("expected Object");
        };
        for name in [
            "system",
            "character",
            "world_info",
            "lorebook",
            "memory",
            "custom",
            "bot_name",
        ] {
            assert!(
                fields.contains_key(&interner.intern(name)),
                "missing field: {name}"
            );
        }
        assert_eq!(fields.len(), 7);
    }

    #[test]
    fn context_type_custom_entry_has_type_field() {
        let interner = Interner::new();
        let ty = context_type(&interner);
        let Ty::Object(fields) = &ty else { panic!() };
        let Ty::List(elem) = fields.get(&interner.intern("custom")).unwrap() else {
            panic!()
        };
        let Ty::Object(custom_fields) = elem.as_ref() else {
            panic!()
        };
        assert!(custom_fields.contains_key(&interner.intern("type")));
    }

    #[test]
    fn history_entry_type_structure() {
        let interner = Interner::new();
        let ty = history_entry_type(&interner);
        let Ty::List(elem) = &ty else { panic!() };
        let Ty::Object(fields) = elem.as_ref() else {
            panic!()
        };
        assert!(fields.contains_key(&interner.intern("content")));
        assert!(fields.contains_key(&interner.intern("content_type")));
        assert!(fields.contains_key(&interner.intern("role")));
        assert_eq!(fields.len(), 3);
    }

    #[test]
    fn builtin_refs_complete() {
        let interner = Interner::new();
        let refs = builtin_context_refs(&interner);
        for name in [
            "turn_index",
            "raw",
            "self",
            "content",
            "context",
            "item",
            "index",
        ] {
            assert!(refs.contains(&interner.intern(name)), "missing: {name}");
        }
        assert_eq!(refs.len(), 7);
    }
}
