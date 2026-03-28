//! Journal traits + ContextOverlay — context storage for the interpreter.
//!
//! The orchestration layer *implements* `EntryRef` (backed by a tree-shaped
//! journal). The interpreter uses `ContextOverlay` which wraps a readonly
//! `Arc<dyn EntryRef>` with a COW write layer.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_utils::{Interner, OwnedDequeDiff, TrackedDeque};
use rustc_hash::FxHashMap;

use crate::value::Value;

// ── Entry traits ─────────────────────────────────────────────────────

/// Read-only handle to a journal entry (one turn's context state).
pub trait EntryRef: Send + Sync {
    fn get(&self, key: &str) -> Option<&Value>;
}

/// Mutable handle to a journal entry (one turn).
pub trait EntryMut: EntryRef {
    fn apply_field(&mut self, key: &str, path: &[&str], value: Value);
    fn apply_diff(&mut self, key: &str, working: TrackedDeque<Value>);
}

/// Journal lifecycle — advancing turns and forking branches.
pub trait EntryLifecycle: EntryMut + Sized {
    fn next(self) -> Self;
    fn fork(self) -> Self;
}

// ── ContextWrite ─────────────────────────────────────────────────────

/// A single context mutation recorded during execution.
#[derive(Debug)]
pub enum ContextWrite {
    /// Whole-value replacement (scalar, list, etc.)
    Set { key: String, value: Value },
    /// Nested object field patch.
    FieldPatch {
        key: String,
        path: Vec<String>,
        value: Value,
    },
    /// Deque/Sequence diff (checksum-verified by journal).
    DequeDiff {
        key: String,
        diff: OwnedDequeDiff<Value>,
    },
}

// ── ContextOverlay ───────────────────────────────────────────────────

/// COW overlay on top of a readonly base.
///
/// - `base`: shared, immutable journal snapshot (`Arc<dyn EntryRef>`)
/// - `cache`: materialized values for read-after-write correctness
/// - `patches`: accumulated diffs for journal writeback
///
/// `fork()` creates an independent copy — same base, same cache snapshot,
/// empty patches. Used for spawn.
pub struct ContextOverlay {
    base: Arc<dyn EntryRef>,
    patches: Vec<ContextWrite>,
    cache: HashMap<String, Value>,
    interner: Interner,
}

impl ContextOverlay {
    pub fn new(base: Arc<dyn EntryRef>, interner: Interner) -> Self {
        Self {
            base,
            patches: Vec::new(),
            cache: HashMap::new(),
            interner,
        }
    }

    /// Fork for spawn — same base, current cache snapshot, empty patches.
    pub fn spawn_fork(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            patches: Vec::new(),
            cache: self.cache.clone(),
            interner: self.interner.clone(),
        }
    }

    /// Take accumulated patches (consumes overlay).
    pub fn into_patches(self) -> Vec<ContextWrite> {
        self.patches
    }

    /// Borrow accumulated patches.
    pub fn patches(&self) -> &[ContextWrite] {
        &self.patches
    }

    /// Merge patches from a child (after eval).
    pub fn merge_patches(&mut self, child_patches: Vec<ContextWrite>) {
        for patch in child_patches {
            // Apply to cache for read-after-write correctness.
            match &patch {
                ContextWrite::Set { key, value } => {
                    self.cache.insert(key.clone(), value.clone());
                }
                ContextWrite::FieldPatch { key, path, value } => {
                    let existing = self
                        .cache
                        .get(key.as_str())
                        .or_else(|| self.base.get(key))
                        .cloned()
                        .unwrap_or(Value::Unit);
                    let path_strs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
                    let updated =
                        deep_set_field(&self.interner, existing, &path_strs, value.clone());
                    self.cache.insert(key.clone(), updated);
                }
                ContextWrite::DequeDiff { key, .. } => {
                    // Deque diffs don't update cache (read original + diff).
                    let _ = key;
                }
            }
            self.patches.push(patch);
        }
    }
}

impl EntryRef for ContextOverlay {
    fn get(&self, key: &str) -> Option<&Value> {
        // Overlay first, then base.
        self.cache.get(key).or_else(|| self.base.get(key))
    }
}

impl EntryMut for ContextOverlay {
    fn apply_field(&mut self, key: &str, path: &[&str], value: Value) {
        if path.is_empty() {
            self.cache.insert(key.to_string(), value.clone());
            self.patches.push(ContextWrite::Set {
                key: key.to_string(),
                value,
            });
        } else {
            // Deep-set: read existing object, update nested field, write back.
            let existing = self
                .cache
                .get(key)
                .or_else(|| self.base.get(key))
                .cloned()
                .unwrap_or(Value::Unit);
            let updated = deep_set_field(&self.interner, existing, path, value.clone());
            self.cache.insert(key.to_string(), updated);
            self.patches.push(ContextWrite::FieldPatch {
                key: key.to_string(),
                path: path.iter().map(|s| s.to_string()).collect(),
                value,
            });
        }
    }

    fn apply_diff(&mut self, key: &str, working: TrackedDeque<Value>) {
        // Compute diff against base.
        let base_val = self.base.get(key);
        let origin = base_val.and_then(|v| match v {
            Value::Deque(d) => Some(d.as_ref()),
            _ => None,
        });

        let diff = match origin {
            Some(origin_deque) => {
                let (squashed, diff) = working.into_diff(origin_deque);
                self.cache.insert(key.to_string(), Value::deque(squashed));
                diff
            }
            None => {
                // First write — entire content is "pushed".
                let items = working.as_slice().to_vec();
                let diff = OwnedDequeDiff {
                    consumed: 0,
                    removed_back: 0,
                    pushed: items,
                };
                self.cache.insert(key.to_string(), Value::deque(working));
                diff
            }
        };

        self.patches.push(ContextWrite::DequeDiff {
            key: key.to_string(),
            diff,
        });
    }
}

/// Deep-set a nested field on a Value. Clones the object at each level.
/// path must be non-empty.
fn deep_set_field(interner: &Interner, root: Value, path: &[&str], value: Value) -> Value {
    debug_assert!(!path.is_empty());

    if path.len() == 1 {
        // Base case: set field directly on root object.
        if let Value::Object(ref arc_map) = root {
            let mut map = (**arc_map).clone();
            let field_key = interner.intern(path[0]);
            map.insert(field_key, value);
            return Value::object(map);
        }
        // Not an object — replace entirely (best-effort).
        return value;
    }

    // Recursive case: navigate into nested object.
    if let Value::Object(ref arc_map) = root {
        let mut map = (**arc_map).clone();
        let field_key = interner.intern(path[0]);
        let child = map.get(&field_key).cloned().unwrap_or(Value::Unit);
        let updated_child = deep_set_field(interner, child, &path[1..], value);
        map.insert(field_key, updated_child);
        return Value::object(map);
    }

    // Not navigable — replace entirely.
    value
}

impl EntryLifecycle for ContextOverlay {
    fn next(self) -> Self {
        // Next turn: keep base + cache (accumulated state), clear patches.
        Self {
            base: self.base,
            patches: Vec::new(),
            cache: self.cache,
            interner: self.interner,
        }
    }

    fn fork(self) -> Self {
        // Fork: same base, snapshot of cache, empty patches.
        Self {
            base: Arc::clone(&self.base),
            patches: Vec::new(),
            cache: self.cache.clone(),
            interner: self.interner.clone(),
        }
    }
}

// ── Journal trait ────────────────────────────────────────────────────

pub trait Journal {
    type Ref<'a>: EntryRef
    where
        Self: 'a;
    type Mut<'a>: EntryMut
    where
        Self: 'a;

    fn entry_ref(&self, id: uuid::Uuid) -> Self::Ref<'_>;
    fn entry_mut(&mut self, id: uuid::Uuid) -> Self::Mut<'_>;
    fn contains(&self, id: uuid::Uuid) -> bool;
}

// ── Simple in-memory EntryRef (testing) ──────────────────────────────

impl EntryRef for HashMap<String, Value> {
    fn get(&self, key: &str) -> Option<&Value> {
        HashMap::get(self, key)
    }
}
