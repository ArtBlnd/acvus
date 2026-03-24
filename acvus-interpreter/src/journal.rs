//! Journal traits + ContextOverlay — context storage for the interpreter.
//!
//! The orchestration layer *implements* `EntryRef` (backed by a tree-shaped
//! journal). The interpreter uses `ContextOverlay` which wraps a readonly
//! `Arc<dyn EntryRef>` with a COW write layer.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_utils::{OwnedDequeDiff, TrackedDeque};

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
}

impl ContextOverlay {
    pub fn new(base: Arc<dyn EntryRef>) -> Self {
        Self {
            base,
            patches: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// Fork for spawn — same base, current cache snapshot, empty patches.
    pub fn spawn_fork(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            patches: Vec::new(),
            cache: self.cache.clone(),
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
                ContextWrite::FieldPatch {
                    key,
                    path: _,
                    value,
                } => {
                    // TODO: nested field patching
                    self.cache.insert(key.clone(), value.clone());
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
            // TODO: nested field patching on cache
            self.cache.insert(key.to_string(), value.clone());
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

impl EntryLifecycle for ContextOverlay {
    fn next(self) -> Self {
        // Next turn: keep base + cache (accumulated state), clear patches.
        Self {
            base: self.base,
            patches: Vec::new(),
            cache: self.cache,
        }
    }

    fn fork(self) -> Self {
        // Fork: same base, snapshot of cache, empty patches.
        Self {
            base: Arc::clone(&self.base),
            patches: Vec::new(),
            cache: self.cache.clone(),
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
