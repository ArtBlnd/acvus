use std::sync::Arc;

use acvus_interpreter::{LazyValue, TypedValue, Value};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, OwnedDequeDiff, TrackedDeque};
use rustc_hash::FxHashMap;
use uuid::Uuid;

// ── Patch types ─────────────────────────────────────────────────────

/// Storage patch — describes how to update a stored value.
///
/// All values must be **eager** (already collected) before reaching storage.
/// Sequence collect happens inside the interpreter's coroutine (FuturesUnordered),
/// not in storage — no async leakage.
#[derive(Debug, Clone)]
pub enum StoragePatch {
    /// Overwrite the stored value entirely.
    Snapshot(TypedValue),
    /// Sequence mode: squashed TrackedDeque + diff from origin.
    /// Produced by Resolver after collect_seq + into_diff.
    Sequence {
        squashed: TrackedDeque<TypedValue>,
        diff: OwnedDequeDiff<TypedValue>,
    },
    /// Apply field-level patches to an existing value.
    Patch(PatchDiff),
}

/// Recursive deep-patch.  Each field is either atomically replaced,
/// recursively patched (if both old and new are Object), or removed.
#[derive(Debug, Clone)]
pub struct PatchDiff {
    pub updates: FxHashMap<Astr, PatchOp>,
    pub removals: Vec<Astr>,
}

/// A single field operation inside a [`PatchDiff`].
#[derive(Debug, Clone)]
pub enum PatchOp {
    /// Atomic replacement — the field value is entirely overwritten.
    Set(Value),
    /// Recursive merge — both old and new values are Object.
    Nested(PatchDiff),
}

impl PatchDiff {
    /// Compute a recursive diff between two Values.
    ///
    /// Returns `Some(diff)` if the values differ, `None` if they are identical.
    /// Only Object×Object pairs produce `Nested` ops; everything else is atomic `Set`.
    pub fn compute(old: &Value, new: &Value) -> Option<Self> {
        let (old_fields, new_fields) = match (old, new) {
            (
                Value::Lazy(LazyValue::Object(old_f)),
                Value::Lazy(LazyValue::Object(new_f)),
            ) => (old_f, new_f),
            _ => {
                // Non-Object: atomic compare.
                return if old == new { None } else {
                    // Caller should use Set directly; we can't produce a meaningful PatchDiff.
                    None
                };
            }
        };

        let mut updates = FxHashMap::default();
        let mut removals = Vec::new();

        // Fields present in new
        for (key, new_val) in new_fields {
            match old_fields.get(key) {
                Some(old_val) if old_val == new_val => {} // identical → skip
                Some(old_val) => {
                    // Both Object → recurse; otherwise atomic Set
                    match Self::compute(old_val, new_val) {
                        Some(nested) => { updates.insert(*key, PatchOp::Nested(nested)); }
                        None => {
                            // Either identical after deep compare, or non-Object mismatch
                            if old_val != new_val {
                                updates.insert(*key, PatchOp::Set(new_val.clone()));
                            }
                        }
                    }
                }
                None => {
                    updates.insert(*key, PatchOp::Set(new_val.clone()));
                }
            }
        }

        // Fields removed in new
        for key in old_fields.keys() {
            if !new_fields.contains_key(key) {
                removals.push(*key);
            }
        }

        if updates.is_empty() && removals.is_empty() {
            None
        } else {
            Some(PatchDiff { updates, removals })
        }
    }

    /// Apply this patch to an existing field map (in-place, recursive).
    pub fn apply_to(self, fields: &mut FxHashMap<Astr, Value>) {
        for (k, op) in self.updates {
            match op {
                PatchOp::Set(v) => { fields.insert(k, v); }
                PatchOp::Nested(nested) => {
                    let entry = fields.entry(k)
                        .or_insert_with(|| Value::object(FxHashMap::default()));
                    if let Value::Lazy(LazyValue::Object(inner)) = entry {
                        nested.apply_to(inner);
                    } else {
                        // Existing value not Object but got Nested → shouldn't happen
                        // if PatchDiff::compute was used correctly.  Fallback: skip.
                    }
                }
            }
        }
        for k in self.removals {
            fields.remove(&k);
        }
    }
}

/// How to prune a node.
#[derive(Debug, Clone, Copy)]
pub enum Prune {
    /// Remove only this leaf (must have no children).
    Leaf,
    /// Remove the entire subtree rooted at this node.
    Subtree,
}

// ── Traits ──────────────────────────────────────────────────────────

/// Read-only handle to a single storage entry.
pub trait EntryRef<'a> {
    fn get(&self, key: &str) -> Option<Arc<TypedValue>>;
    fn depth(&self) -> usize;
    fn uuid(&self) -> Uuid;
}

/// Mutable handle to a single storage entry.
///
/// `next`, `fork`, and `prune` consume self to prevent dangling references.
#[trait_variant::make(Send)]
pub trait EntryMut<'a>: Sized {
    type Ref<'x>: EntryRef<'x>
    where
        'a: 'x,
        Self: 'x;

    fn get(&self, key: &str) -> Option<Arc<TypedValue>>;
    fn apply(&mut self, key: &str, patch: StoragePatch);
    async fn next(self) -> Self;
    async fn fork(self) -> Self;
    fn prune(self, mode: Prune);
    fn depth(&self) -> usize;
    fn uuid(&self) -> Uuid;
    fn as_ref(&self) -> Self::Ref<'_>;
}

/// Tree-shaped storage backend.
///
/// Each entry represents one turn. Parent-child edges form a COW overlay:
/// - `accumulated`: squashed state from all ancestors (shared via `Arc`)
/// - `turn_diff`: changes made during this turn
#[trait_variant::make(Send)]
pub trait Journal {
    type Ref<'a>: EntryRef<'a>
    where
        Self: 'a;
    type Mut<'a>: EntryMut<'a>
    where
        Self: 'a;

    async fn entry(&self, id: Uuid) -> Self::Ref<'_>;
    async fn entry_mut(&mut self, id: Uuid) -> Self::Mut<'_>;
    fn parent_of(&self, id: Uuid) -> Option<Uuid>;
    fn contains(&self, id: Uuid) -> bool;
}

// ── Tree export types ──────────────────────────────────────────────

/// Exported representation of a single tree node.
pub struct TreeNodeExport {
    pub uuid: Uuid,
    pub parent: Option<Uuid>,
    pub depth: usize,
    /// Only turn_diff entries (not accumulated — that's derived).
    pub turn_diff: FxHashMap<String, Arc<TypedValue>>,
}

/// Full tree export — enough to reconstruct the entire TreeJournal.
pub struct TreeExport {
    /// Nodes in topological order (parents before children).
    pub nodes: Vec<TreeNodeExport>,
}

// ── History query types ─────────────────────────────────────────────

/// Info about a single node in the history tree.
pub struct HistoryEntry {
    pub uuid: Uuid,
    pub depth: usize,
    pub changed_keys: Vec<String>,
    pub child_count: usize,
}

// ── TreeJournal (concrete impl) ─────────────────────────────────────

struct TreeNode {
    parent: Option<usize>,
    children: Vec<usize>,
    /// Squashed state from all ancestors (shared via Arc for COW).
    accumulated: Arc<FxHashMap<String, Arc<TypedValue>>>,
    /// Changes made during this turn.
    turn_diff: FxHashMap<String, Arc<TypedValue>>,
    depth: usize,
    uuid: Uuid,
}

impl std::fmt::Debug for TreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeNode")
            .field("parent", &self.parent)
            .field("children", &self.children)
            .field("depth", &self.depth)
            .field("turn_diff_keys", &self.turn_diff.keys().collect::<Vec<_>>())
            .finish()
    }
}

pub(crate) struct TreeJournalInner {
    nodes: Vec<TreeNode>,
    uuid_to_idx: FxHashMap<Uuid, usize>,
}

impl std::fmt::Debug for TreeJournalInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeJournalInner")
            .field("node_count", &self.nodes.len())
            .finish()
    }
}

/// Simple in-memory tree-based storage.
#[derive(Debug)]
pub struct TreeJournal {
    pub(crate) inner: TreeJournalInner,
}

impl TreeJournal {
    /// Create a new journal with a single root entry (depth 0, empty state).
    /// Returns the journal and the root entry's UUID.
    pub fn new() -> (Self, Uuid) {
        let root_uuid = Uuid::new_v4();
        let root = TreeNode {
            parent: None,
            children: Vec::new(),
            accumulated: Arc::new(FxHashMap::default()),
            turn_diff: FxHashMap::default(),
            depth: 0,
            uuid: root_uuid,
        };
        let mut uuid_to_idx = FxHashMap::default();
        uuid_to_idx.insert(root_uuid, 0);
        (
            Self {
                inner: TreeJournalInner {
                    nodes: vec![root],
                    uuid_to_idx,
                },
            },
            root_uuid,
        )
    }
}

impl TreeJournal {
    /// Export the full tree structure in topological (BFS) order.
    pub fn export_tree(&self) -> TreeExport {
        let inner = &self.inner;
        let mut nodes = Vec::with_capacity(inner.nodes.len());
        let mut queue = std::collections::VecDeque::new();

        // Find root(s) — nodes with no parent that are still in uuid_to_idx.
        for (idx, node) in inner.nodes.iter().enumerate() {
            if node.parent.is_none() && inner.uuid_to_idx.contains_key(&node.uuid) {
                queue.push_back(idx);
            }
        }

        while let Some(idx) = queue.pop_front() {
            let node = &inner.nodes[idx];
            if !inner.uuid_to_idx.contains_key(&node.uuid) {
                continue;
            }
            let parent_uuid = node.parent.map(|pidx| inner.nodes[pidx].uuid);
            nodes.push(TreeNodeExport {
                uuid: node.uuid,
                parent: parent_uuid,
                depth: node.depth,
                turn_diff: node.turn_diff.clone(),
            });
            for &child_idx in &node.children {
                queue.push_back(child_idx);
            }
        }

        TreeExport { nodes }
    }

    /// Reconstruct a TreeJournal from an export.
    ///
    /// Nodes must be in topological order (parents before children).
    /// `accumulated` is recomputed from parent state + parent turn_diff.
    pub fn import_tree(export: TreeExport) -> Self {
        assert!(!export.nodes.is_empty(), "cannot import empty tree");

        let mut nodes: Vec<TreeNode> = Vec::with_capacity(export.nodes.len());
        let mut uuid_to_idx: FxHashMap<Uuid, usize> = FxHashMap::default();

        for export_node in &export.nodes {
            let idx = nodes.len();

            let (parent_idx, accumulated) = match export_node.parent {
                None => {
                    // Root node — empty accumulated.
                    (None, Arc::new(FxHashMap::default()))
                }
                Some(parent_uuid) => {
                    let pidx = uuid_to_idx[&parent_uuid];
                    let parent = &nodes[pidx];
                    // Merge parent's accumulated + parent's turn_diff.
                    let mut merged = (*parent.accumulated).clone();
                    for (k, v) in &parent.turn_diff {
                        merged.insert(k.clone(), Arc::clone(v));
                    }
                    (Some(pidx), Arc::new(merged))
                }
            };

            // Register this node as child of parent.
            if let Some(pidx) = parent_idx {
                nodes[pidx].children.push(idx);
            }

            uuid_to_idx.insert(export_node.uuid, idx);
            nodes.push(TreeNode {
                parent: parent_idx,
                children: Vec::new(),
                accumulated,
                turn_diff: export_node.turn_diff.clone(),
                depth: export_node.depth,
                uuid: export_node.uuid,
            });
        }

        Self {
            inner: TreeJournalInner {
                nodes,
                uuid_to_idx,
            },
        }
    }
}

// ── History query methods ────────────────────────────────────────────

impl TreeJournal {
    /// Returns `true` if the tree contains a live node with the given UUID.
    pub fn contains(&self, id: Uuid) -> bool {
        self.inner.uuid_to_idx.contains_key(&id)
    }

    /// Get the path from root to the given cursor (inclusive).
    ///
    /// Returns nodes ordered root-first. Panics if `cursor` is not in the tree.
    pub fn path_to(&self, cursor: Uuid) -> Vec<HistoryEntry> {
        let inner = &self.inner;
        let mut idx = inner.uuid_to_idx[&cursor];
        let mut path = Vec::new();
        loop {
            let node = &inner.nodes[idx];
            path.push(HistoryEntry {
                uuid: node.uuid,
                depth: node.depth,
                changed_keys: node.turn_diff.keys().cloned().collect(),
                child_count: node.children.len(),
            });
            match node.parent {
                Some(parent_idx) => idx = parent_idx,
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Get all branch points in the tree (nodes with more than one child).
    ///
    /// Returns `(parent_uuid, children_uuids)` for each branch point.
    pub fn branch_points(&self) -> Vec<(Uuid, Vec<Uuid>)> {
        let inner = &self.inner;
        let mut result = Vec::new();
        for node in &inner.nodes {
            if node.children.len() > 1 && inner.uuid_to_idx.contains_key(&node.uuid) {
                let children: Vec<Uuid> = node
                    .children
                    .iter()
                    .filter_map(|&child_idx| {
                        let child = &inner.nodes[child_idx];
                        inner.uuid_to_idx.contains_key(&child.uuid).then_some(child.uuid)
                    })
                    .collect();
                if children.len() > 1 {
                    result.push((node.uuid, children));
                }
            }
        }
        result
    }

    /// Get the parent UUID of a given entry, or `None` if it is the root.
    ///
    /// Panics if `id` is not in the tree.
    pub fn parent_of(&self, id: Uuid) -> Option<Uuid> {
        let inner = &self.inner;
        let idx = inner.uuid_to_idx[&id];
        inner.nodes[idx].parent.map(|pidx| inner.nodes[pidx].uuid)
    }

    /// Get the children UUIDs of a given entry.
    ///
    /// Panics if `id` is not in the tree.
    pub fn children_of(&self, id: Uuid) -> Vec<Uuid> {
        let inner = &self.inner;
        let idx = inner.uuid_to_idx[&id];
        inner.nodes[idx]
            .children
            .iter()
            .map(|&cidx| inner.nodes[cidx].uuid)
            .collect()
    }
}

impl Journal for TreeJournal {
    type Ref<'a> = TreeEntryRef<'a> where Self: 'a;
    type Mut<'a> = TreeEntryMut<'a> where Self: 'a;

    async fn entry(&self, id: Uuid) -> TreeEntryRef<'_> {
        let idx = self.inner.uuid_to_idx[&id];
        TreeEntryRef {
            inner: &self.inner,
            idx,
        }
    }

    async fn entry_mut(&mut self, id: Uuid) -> TreeEntryMut<'_> {
        let idx = self.inner.uuid_to_idx[&id];
        TreeEntryMut {
            inner: &mut self.inner,
            idx,
        }
    }

    fn parent_of(&self, id: Uuid) -> Option<Uuid> {
        let idx = self.inner.uuid_to_idx[&id];
        self.inner.nodes[idx].parent.map(|pidx| self.inner.nodes[pidx].uuid)
    }

    fn contains(&self, id: Uuid) -> bool {
        self.inner.uuid_to_idx.contains_key(&id)
    }
}

/// Read-only handle to a tree journal entry.
pub struct TreeEntryRef<'a> {
    pub(crate) inner: &'a TreeJournalInner,
    idx: usize,
}

/// Mutable handle to a tree journal entry.
pub struct TreeEntryMut<'a> {
    pub(crate) inner: &'a mut TreeJournalInner,
    idx: usize,
}

impl<'a> TreeEntryRef<'a> {
    /// Return all key-value pairs visible from this entry (accumulated + turn_diff merged).
    pub fn entries(&self) -> FxHashMap<String, Arc<TypedValue>> {
        let node = &self.inner.nodes[self.idx];
        let mut result = (*node.accumulated).clone();
        for (k, v) in &node.turn_diff {
            result.insert(k.clone(), Arc::clone(v));
        }
        result
    }
}

impl<'a> EntryRef<'a> for TreeEntryRef<'a> {
    fn get(&self, key: &str) -> Option<Arc<TypedValue>> {
        let node = &self.inner.nodes[self.idx];
        if let Some(val) = node.turn_diff.get(key) {
            return Some(Arc::clone(val));
        }
        node.accumulated.get(key).cloned()
    }

    fn depth(&self) -> usize {
        self.inner.nodes[self.idx].depth
    }

    fn uuid(&self) -> Uuid {
        self.inner.nodes[self.idx].uuid
    }
}

impl<'a> EntryMut<'a> for TreeEntryMut<'a> {
    type Ref<'x> = TreeEntryRef<'x> where 'a: 'x;

    fn get(&self, key: &str) -> Option<Arc<TypedValue>> {
        let node = &self.inner.nodes[self.idx];
        if let Some(val) = node.turn_diff.get(key) {
            return Some(Arc::clone(val));
        }
        node.accumulated.get(key).cloned()
    }

    fn apply(&mut self, key: &str, patch: StoragePatch) {
        let idx = self.idx;
        debug_assert!(
            self.inner.nodes[idx].children.is_empty(),
            "apply on non-leaf"
        );

        match patch {
            StoragePatch::Snapshot(v) => {
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), Arc::new(v));
            }
            StoragePatch::Sequence { squashed, .. } => {
                let value_deque = TrackedDeque::from_vec(
                    squashed.into_vec().into_iter().map(|tv| Arc::unwrap_or_clone(tv.into_value())).collect(),
                );
                let stored = TypedValue::new(
                    Arc::new(Value::Lazy(LazyValue::Deque(value_deque))),
                    Ty::Infer,
                );
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), Arc::new(stored));
            }
            StoragePatch::Patch(patch_diff) => {
                let mut fields = self
                    .get(key)
                    .and_then(|arc| match arc.value() {
                        Value::Lazy(LazyValue::Object(fields)) => Some(fields.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                patch_diff.apply_to(&mut fields);
                let stored = TypedValue::new(
                    Arc::new(Value::Lazy(LazyValue::Object(fields))),
                    Ty::Infer,
                );
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), Arc::new(stored));
            }
        }
    }

    async fn next(self) -> Self {
        let parent_idx = self.idx;
        let inner = self.inner;

        let parent_node = &inner.nodes[parent_idx];
        let mut merged = (*parent_node.accumulated).clone();
        for (k, v) in &parent_node.turn_diff {
            merged.insert(k.clone(), Arc::clone(v));
        }
        let depth = parent_node.depth + 1;
        let new_uuid = Uuid::new_v4();
        let new_idx = inner.nodes.len();

        let child = TreeNode {
            parent: Some(parent_idx),
            children: Vec::new(),
            accumulated: Arc::new(merged),
            turn_diff: FxHashMap::default(),
            depth,
            uuid: new_uuid,
        };
        inner.nodes.push(child);
        inner.nodes[parent_idx].children.push(new_idx);
        inner.uuid_to_idx.insert(new_uuid, new_idx);

        TreeEntryMut { inner, idx: new_idx }
    }

    async fn fork(self) -> Self {
        let sibling_idx = self.idx;
        let inner = self.inner;
        let parent_idx = inner.nodes[sibling_idx]
            .parent
            .expect("cannot fork root");

        let parent_node = &inner.nodes[parent_idx];
        let mut merged = (*parent_node.accumulated).clone();
        for (k, v) in &parent_node.turn_diff {
            merged.insert(k.clone(), Arc::clone(v));
        }
        let depth = parent_node.depth + 1;
        let new_uuid = Uuid::new_v4();
        let new_idx = inner.nodes.len();

        let child = TreeNode {
            parent: Some(parent_idx),
            children: Vec::new(),
            accumulated: Arc::new(merged),
            turn_diff: FxHashMap::default(),
            depth,
            uuid: new_uuid,
        };
        inner.nodes.push(child);
        inner.nodes[parent_idx].children.push(new_idx);
        inner.uuid_to_idx.insert(new_uuid, new_idx);

        TreeEntryMut { inner, idx: new_idx }
    }

    fn prune(self, mode: Prune) {
        let idx = self.idx;
        let inner = self.inner;

        match mode {
            Prune::Leaf => {
                debug_assert!(
                    inner.nodes[idx].children.is_empty(),
                    "prune Leaf on non-leaf node"
                );
                if let Some(parent_idx) = inner.nodes[idx].parent {
                    inner.nodes[parent_idx]
                        .children
                        .retain(|&child| child != idx);
                }
                let uuid = inner.nodes[idx].uuid;
                inner.uuid_to_idx.remove(&uuid);
                inner.nodes[idx].accumulated = Arc::new(FxHashMap::default());
                inner.nodes[idx].turn_diff.clear();
            }
            Prune::Subtree => {
                let mut stack = vec![idx];
                let mut to_clear = Vec::new();
                while let Some(current) = stack.pop() {
                    to_clear.push(current);
                    let children: Vec<usize> = inner.nodes[current].children.clone();
                    stack.extend(children);
                }
                if let Some(parent_idx) = inner.nodes[idx].parent {
                    inner.nodes[parent_idx]
                        .children
                        .retain(|&child| child != idx);
                }
                for node_idx in to_clear {
                    let uuid = inner.nodes[node_idx].uuid;
                    inner.uuid_to_idx.remove(&uuid);
                    inner.nodes[node_idx].accumulated = Arc::new(FxHashMap::default());
                    inner.nodes[node_idx].turn_diff.clear();
                    inner.nodes[node_idx].children.clear();
                }
            }
        }
    }

    fn depth(&self) -> usize {
        self.inner.nodes[self.idx].depth
    }

    fn uuid(&self) -> Uuid {
        self.inner.nodes[self.idx].uuid
    }

    fn as_ref(&self) -> TreeEntryRef<'_> {
        TreeEntryRef {
            inner: self.inner,
            idx: self.idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use acvus_interpreter::{LazyValue, PureValue, TypedValue, Value};
    use acvus_mir::ty::Ty;
    use acvus_utils::Interner;

    use super::*;

    // --- Basic get/apply tests ---

    #[tokio::test]
    async fn apply_and_get() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("x", StoragePatch::Snapshot(TypedValue::string("hello".to_string())));
        assert!(matches!(
            e.get("x").unwrap().value(),
            Value::Pure(PureValue::String(v)) if v == "hello"
        ));
        assert!(e.get("y").is_none());
    }

    #[tokio::test]
    async fn overwrite() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        e.apply(
            "x",
            StoragePatch::Snapshot(TypedValue::string("first")),
        );
        e.apply(
            "x",
            StoragePatch::Snapshot(TypedValue::new(
                Arc::new(Value::object(FxHashMap::from_iter([(
                    interner.intern("v"),
                    Value::int(2),
                )]))),
                Ty::Infer,
            )),
        );
        assert!(matches!(e.get("x").unwrap().value(), Value::Lazy(LazyValue::Object(_))));
    }

    #[tokio::test]
    async fn deque_stores_squashed() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(2)]);
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![TypedValue::int(1), TypedValue::int(2)],
        };
        e.apply(
            "q",
            StoragePatch::Sequence {
                squashed,
                diff,
            },
        );
        let val = e.get("q").unwrap();
        let Value::Lazy(LazyValue::Deque(d)) = val.value() else {
            panic!("expected Deque");
        };
        assert_eq!(d.as_slice(), &[Value::int(1), Value::int(2)]);
    }

    #[tokio::test]
    async fn deque_checksum_preserved() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1)]);
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![TypedValue::int(1)],
        };
        e.apply("q", StoragePatch::Sequence { squashed, diff });
        let val = e.get("q").unwrap();
        let Value::Lazy(LazyValue::Deque(stored)) = val.value() else {
            panic!("expected Deque")
        };
        // After conversion through apply, checksum is regenerated (not preserved).
        assert_eq!(stored.as_slice(), &[Value::int(1)]);
    }

    #[tokio::test]
    async fn object_diff_updates_and_removals() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");
        e.apply(
            "obj",
            StoragePatch::Snapshot(TypedValue::new(
                Arc::new(Value::object(FxHashMap::from_iter([
                    (a, Value::int(1)),
                    (b, Value::int(2)),
                ]))),
                Ty::Infer,
            )),
        );
        let diff = PatchDiff {
            updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(100))), (c, PatchOp::Set(Value::int(3)))]),
            removals: vec![b],
        };
        e.apply("obj", StoragePatch::Patch(diff));
        let val = e.get("obj").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = val.value() else {
            panic!("expected Object")
        };
        assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(100))));
        assert_eq!(fields.get(&b), None);
        assert_eq!(fields.get(&c), Some(&Value::Pure(PureValue::Int(3))));
    }

    #[tokio::test]
    async fn object_diff_on_missing_key() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let a = interner.intern("a");
        let diff = PatchDiff {
            updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(42)))]),
            removals: vec![],
        };
        e.apply("obj", StoragePatch::Patch(diff));
        let val = e.get("obj").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = val.value() else {
            panic!("expected Object")
        };
        assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(42))));
    }

    // --- Tree structure tests ---

    #[tokio::test]
    async fn depth() {
        let (mut j, root) = TreeJournal::new();

        {
            let e = j.entry(root).await;
            assert_eq!(e.depth(), 0);
        }

        let n1;
        {
            let e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            assert_eq!(e.depth(), 1);
        }

        {
            let e = j.entry_mut(n1).await.next().await;
            assert_eq!(e.depth(), 2);
        }
    }

    #[tokio::test]
    async fn next_squashes_parent() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(1)));
            e.apply("y", StoragePatch::Snapshot(TypedValue::int(2)));
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            // n2 should see parent's values via accumulated
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            assert!(matches!(e.get("y").unwrap().value(), Value::Pure(PureValue::Int(2))));

            // Modifying n2 doesn't affect n1
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(99)));
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        }

        // n1 still has original value
        {
            let e = j.entry(n1).await;
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        }
        // n2 has the override
        {
            let e = j.entry(n2).await;
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        }
    }

    #[tokio::test]
    async fn fork_creates_sibling() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(1)));
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(2)));
        }

        // Fork from n2 — creates sibling of n2 (child of n1)
        {
            let e = j.entry_mut(n2).await.fork().await;
            assert_eq!(e.depth(), 2);
            // Sees n1's accumulated (x=1), not n2's turn_diff (x=2)
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        }
    }

    #[tokio::test]
    async fn prune_leaf() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(1)));
        }

        j.entry_mut(n2).await.prune(Prune::Leaf);

        // n1 still accessible
        {
            let e = j.entry(n1).await;
            assert_eq!(e.depth(), 1);
        }
    }

    #[tokio::test]
    async fn prune_subtree() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
        }
        {
            let e = j.entry_mut(n1).await.next().await;
            let _n2 = e.uuid();
            // n3 = child of n2 — all will be pruned
        }

        j.entry_mut(n1).await.prune(Prune::Subtree);

        // root still accessible
        {
            let e = j.entry(root).await;
            assert_eq!(e.depth(), 0);
        }
    }

    #[tokio::test]
    async fn cow_sharing() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(1)));
        }

        let n2;
        {
            let e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
        }
        let n3;
        {
            let e = j.entry_mut(n1).await.next().await;
            n3 = e.uuid();
        }

        // Both see parent's data
        assert!(matches!(j.entry(n2).await.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        assert!(matches!(j.entry(n3).await.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));

        // Modifying one doesn't affect the other
        {
            let mut e = j.entry_mut(n2).await;
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(99)));
        }
        assert!(matches!(j.entry(n2).await.get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        assert!(matches!(j.entry(n3).await.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
    }

    #[tokio::test]
    async fn get_prefers_turn_diff() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(1)));
        }

        {
            let mut e = j.entry_mut(n1).await.next().await;
            // x=1 is in accumulated
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            // Now override in turn_diff
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(2)));
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(2))));
        }
    }

    // =========================================================================
    // PatchDiff::compute — unit tests
    // =========================================================================

    fn obj_i(interner: &Interner, fields: &[(&str, Value)]) -> Value {
        Value::object(fields.iter().map(|(k, v)| (interner.intern(k), v.clone())).collect())
    }

    #[test]
    fn compute_identical_objects_returns_none() {
        let i = Interner::new();
        let v = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(2))]);
        assert!(PatchDiff::compute(&v, &v).is_none());
    }

    #[test]
    fn compute_different_field_value() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1))]);
        let new = obj_i(&i, &[("a", Value::int(2))]);
        let diff = PatchDiff::compute(&old, &new).expect("should have diff");
        let a = i.intern("a");
        assert!(matches!(diff.updates.get(&a), Some(PatchOp::Set(Value::Pure(PureValue::Int(2))))));
        assert!(diff.removals.is_empty());
    }

    #[test]
    fn compute_added_field() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1))]);
        let new = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(2))]);
        let diff = PatchDiff::compute(&old, &new).expect("should have diff");
        let b = i.intern("b");
        assert!(matches!(diff.updates.get(&b), Some(PatchOp::Set(Value::Pure(PureValue::Int(2))))));
        assert!(diff.removals.is_empty());
    }

    #[test]
    fn compute_removed_field() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(2))]);
        let new = obj_i(&i, &[("a", Value::int(1))]);
        let diff = PatchDiff::compute(&old, &new).expect("should have diff");
        let b = i.intern("b");
        assert!(diff.updates.is_empty());
        assert!(diff.removals.contains(&b));
    }

    #[test]
    fn compute_nested_object_recursive() {
        let i = Interner::new();
        let old = obj_i(&i, &[
            ("x", Value::int(1)),
            ("nested", obj_i(&i, &[("a", Value::int(10)), ("b", Value::int(20))])),
        ]);
        let new = obj_i(&i, &[
            ("x", Value::int(1)),
            ("nested", obj_i(&i, &[("a", Value::int(10)), ("b", Value::int(99))])),
        ]);
        let diff = PatchDiff::compute(&old, &new).expect("should have diff");
        let nested = i.intern("nested");
        // x is identical → not in updates
        assert!(!diff.updates.contains_key(&i.intern("x")));
        // nested should be Nested, not Set
        match diff.updates.get(&nested) {
            Some(PatchOp::Nested(inner)) => {
                let b = i.intern("b");
                assert!(matches!(inner.updates.get(&b), Some(PatchOp::Set(Value::Pure(PureValue::Int(99))))));
                assert!(!inner.updates.contains_key(&i.intern("a"))); // a unchanged
            }
            other => panic!("expected Nested, got {:?}", other),
        }
    }

    #[test]
    fn compute_deeply_nested_3_levels() {
        let i = Interner::new();
        let old = obj_i(&i, &[
            ("l1", obj_i(&i, &[
                ("l2", obj_i(&i, &[("l3", Value::int(1))])),
            ])),
        ]);
        let new = obj_i(&i, &[
            ("l1", obj_i(&i, &[
                ("l2", obj_i(&i, &[("l3", Value::int(2))])),
            ])),
        ]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        // l1 → Nested → l2 → Nested → l3 → Set(2)
        let l1 = i.intern("l1");
        let l2 = i.intern("l2");
        let l3 = i.intern("l3");
        let PatchOp::Nested(d1) = diff.updates.get(&l1).unwrap() else { panic!("l1 not Nested") };
        let PatchOp::Nested(d2) = d1.updates.get(&l2).unwrap() else { panic!("l2 not Nested") };
        assert!(matches!(d2.updates.get(&l3), Some(PatchOp::Set(Value::Pure(PureValue::Int(2))))));
    }

    #[test]
    fn compute_non_object_returns_none() {
        // Non-Object values: compute returns None (caller uses Set directly)
        let old = Value::int(1);
        let new = Value::int(2);
        assert!(PatchDiff::compute(&old, &new).is_none());
    }

    #[test]
    fn compute_identical_non_object_returns_none() {
        let old = Value::int(1);
        let new = Value::int(1);
        assert!(PatchDiff::compute(&old, &new).is_none());
    }

    #[test]
    fn compute_object_to_non_object_field() {
        let i = Interner::new();
        // nested was Object, now it's Int → atomic Set
        let old = obj_i(&i, &[("x", obj_i(&i, &[("a", Value::int(1))]))]);
        let new = obj_i(&i, &[("x", Value::int(42))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        let x = i.intern("x");
        assert!(matches!(diff.updates.get(&x), Some(PatchOp::Set(Value::Pure(PureValue::Int(42))))));
    }

    #[test]
    fn compute_non_object_to_object_field() {
        let i = Interner::new();
        let old = obj_i(&i, &[("x", Value::int(1))]);
        let new = obj_i(&i, &[("x", obj_i(&i, &[("a", Value::int(2))]))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        let x = i.intern("x");
        // Int → Object: can't recurse, must be Set
        match diff.updates.get(&x) {
            Some(PatchOp::Set(Value::Lazy(LazyValue::Object(_)))) => {}
            other => panic!("expected Set(Object), got {:?}", other),
        }
    }

    #[test]
    fn compute_empty_objects_returns_none() {
        let i = Interner::new();
        let v = obj_i(&i, &[]);
        assert!(PatchDiff::compute(&v, &v).is_none());
    }

    #[test]
    fn compute_empty_to_nonempty() {
        let i = Interner::new();
        let old = obj_i(&i, &[]);
        let new = obj_i(&i, &[("a", Value::int(1))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        assert_eq!(diff.updates.len(), 1);
    }

    #[test]
    fn compute_nonempty_to_empty() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1))]);
        let new = obj_i(&i, &[]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        assert!(diff.updates.is_empty());
        assert_eq!(diff.removals.len(), 1);
    }

    #[test]
    fn compute_string_field_change() {
        let i = Interner::new();
        let old = obj_i(&i, &[("name", Value::string("alice".to_string()))]);
        let new = obj_i(&i, &[("name", Value::string("bob".to_string()))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        let name = i.intern("name");
        match diff.updates.get(&name) {
            Some(PatchOp::Set(Value::Pure(PureValue::String(s)))) => assert_eq!(s, "bob"),
            other => panic!("expected Set(String), got {:?}", other),
        }
    }

    #[test]
    fn compute_mixed_add_remove_update() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(2)), ("c", Value::int(3))]);
        let new = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(99)), ("d", Value::int(4))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        // a: unchanged
        assert!(!diff.updates.contains_key(&i.intern("a")));
        // b: updated
        assert!(matches!(diff.updates.get(&i.intern("b")), Some(PatchOp::Set(Value::Pure(PureValue::Int(99))))));
        // c: removed
        assert!(diff.removals.contains(&i.intern("c")));
        // d: added
        assert!(matches!(diff.updates.get(&i.intern("d")), Some(PatchOp::Set(Value::Pure(PureValue::Int(4))))));
    }

    // =========================================================================
    // PatchDiff::apply_to — unit tests
    // =========================================================================

    #[test]
    fn apply_set_overwrites() {
        let i = Interner::new();
        let a = i.intern("a");
        let mut fields = FxHashMap::from_iter([(a, Value::int(1))]);
        let diff = PatchDiff {
            updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(2)))]),
            removals: vec![],
        };
        diff.apply_to(&mut fields);
        assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(2))));
    }

    #[test]
    fn apply_nested_merges_recursively() {
        let i = Interner::new();
        let a = i.intern("a");
        let b = i.intern("b");
        let nested_key = i.intern("nested");
        let mut fields = FxHashMap::from_iter([
            (nested_key, Value::object(FxHashMap::from_iter([
                (a, Value::int(1)),
                (b, Value::int(2)),
            ]))),
        ]);
        let diff = PatchDiff {
            updates: FxHashMap::from_iter([(nested_key, PatchOp::Nested(PatchDiff {
                updates: FxHashMap::from_iter([(b, PatchOp::Set(Value::int(99)))]),
                removals: vec![],
            }))]),
            removals: vec![],
        };
        diff.apply_to(&mut fields);
        let Value::Lazy(LazyValue::Object(inner)) = fields.get(&nested_key).unwrap() else { panic!() };
        assert_eq!(inner.get(&a), Some(&Value::Pure(PureValue::Int(1)))); // unchanged
        assert_eq!(inner.get(&b), Some(&Value::Pure(PureValue::Int(99)))); // updated
    }

    #[test]
    fn apply_removal() {
        let i = Interner::new();
        let a = i.intern("a");
        let b = i.intern("b");
        let mut fields = FxHashMap::from_iter([(a, Value::int(1)), (b, Value::int(2))]);
        let diff = PatchDiff {
            updates: FxHashMap::default(),
            removals: vec![b],
        };
        diff.apply_to(&mut fields);
        assert_eq!(fields.len(), 1);
        assert!(fields.contains_key(&a));
    }

    #[test]
    fn apply_nested_on_missing_creates_object() {
        let i = Interner::new();
        let x = i.intern("x");
        let a = i.intern("a");
        let mut fields = FxHashMap::default();
        let diff = PatchDiff {
            updates: FxHashMap::from_iter([(x, PatchOp::Nested(PatchDiff {
                updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(1)))]),
                removals: vec![],
            }))]),
            removals: vec![],
        };
        diff.apply_to(&mut fields);
        let Value::Lazy(LazyValue::Object(inner)) = fields.get(&x).unwrap() else { panic!() };
        assert_eq!(inner.get(&a), Some(&Value::Pure(PureValue::Int(1))));
    }

    // =========================================================================
    // Snapshot — storage tests
    // =========================================================================

    #[tokio::test]
    async fn snapshot_overwrites_entirely() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("x", StoragePatch::Snapshot(TypedValue::int(1)));
        assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        e.apply("x", StoragePatch::Snapshot(TypedValue::int(2)));
        assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(2))));
    }

    #[tokio::test]
    async fn snapshot_no_history_across_turns() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(1)));
        }
        {
            let mut e = j.entry_mut(n1).await.next().await;
            e.apply("x", StoragePatch::Snapshot(TypedValue::int(2)));
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(2))));
        }
        // Going back to n1 → should see value 1, not 2
        {
            let e = j.entry(n1).await;
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        }
    }

    #[tokio::test]
    async fn snapshot_string_value() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("msg", StoragePatch::Snapshot(TypedValue::new(
            Arc::new(Value::string("hello".to_string())),
            Ty::String,
        )));
        match e.get("msg").unwrap().value() {
            Value::Pure(PureValue::String(s)) => assert_eq!(s, "hello"),
            other => panic!("expected String, got {:?}", other),
        }
    }

    // =========================================================================
    // Sequence — storage tests
    // =========================================================================

    #[tokio::test]
    async fn sequence_stores_deque() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(2)]);
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![TypedValue::int(1), TypedValue::int(2)],
        };
        e.apply("seq", StoragePatch::Sequence { squashed, diff });
        let val = e.get("seq").unwrap();
        assert!(matches!(val.value(), Value::Lazy(LazyValue::Deque(_))));
    }

    // =========================================================================
    // Patch — storage integration tests
    // =========================================================================

    #[tokio::test]
    async fn patch_recursive_nested_object() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let a = interner.intern("a");
        let b = interner.intern("b");
        let inner_key = interner.intern("inner");

        // Initial: {a: 1, inner: {b: 10}}
        let initial = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)),
            (inner_key, Value::object(FxHashMap::from_iter([(b, Value::int(10))]))),
        ]));
        e.apply("state", StoragePatch::Snapshot(TypedValue::new(Arc::new(initial), Ty::Infer)));

        // Patch: inner.b = 99 (recursive)
        let diff = PatchDiff {
            updates: FxHashMap::from_iter([(inner_key, PatchOp::Nested(PatchDiff {
                updates: FxHashMap::from_iter([(b, PatchOp::Set(Value::int(99)))]),
                removals: vec![],
            }))]),
            removals: vec![],
        };
        e.apply("state", StoragePatch::Patch(diff));

        let val = e.get("state").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!("expected Object") };
        // a unchanged
        assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(1))));
        // inner.b = 99
        let Value::Lazy(LazyValue::Object(inner)) = fields.get(&inner_key).unwrap() else { panic!("expected inner Object") };
        assert_eq!(inner.get(&b), Some(&Value::Pure(PureValue::Int(99))));
    }

    #[tokio::test]
    async fn patch_history_accumulates_across_turns() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let b = interner.intern("b");

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            let initial = Value::object(FxHashMap::from_iter([
                (a, Value::int(1)),
                (b, Value::int(2)),
            ]));
            e.apply("state", StoragePatch::Snapshot(TypedValue::new(Arc::new(initial), Ty::Infer)));
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            // Patch: a = 10
            let diff = PatchDiff {
                updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(10)))]),
                removals: vec![],
            };
            e.apply("state", StoragePatch::Patch(diff));
        }

        // At n2: a=10, b=2
        {
            let e = j.entry(n2).await;
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
            assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(10))));
            assert_eq!(fields.get(&b), Some(&Value::Pure(PureValue::Int(2))));
        }

        // At n1: a=1, b=2 (history preserved!)
        {
            let e = j.entry(n1).await;
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
            assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(fields.get(&b), Some(&Value::Pure(PureValue::Int(2))));
        }
    }

    #[tokio::test]
    async fn patch_multiple_patches_same_turn() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");
        let mut e = j.entry_mut(root).await.next().await;

        let initial = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)),
            (b, Value::int(2)),
        ]));
        e.apply("state", StoragePatch::Snapshot(TypedValue::new(Arc::new(initial), Ty::Infer)));

        // First patch: update a
        e.apply("state", StoragePatch::Patch(PatchDiff {
            updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(10)))]),
            removals: vec![],
        }));

        // Second patch: add c, remove b
        e.apply("state", StoragePatch::Patch(PatchDiff {
            updates: FxHashMap::from_iter([(c, PatchOp::Set(Value::int(3)))]),
            removals: vec![b],
        }));

        let val = e.get("state").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
        assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(10))));
        assert_eq!(fields.get(&b), None);
        assert_eq!(fields.get(&c), Some(&Value::Pure(PureValue::Int(3))));
    }

    #[tokio::test]
    async fn patch_on_nonexistent_key_creates_object() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let mut e = j.entry_mut(root).await.next().await;

        e.apply("new_obj", StoragePatch::Patch(PatchDiff {
            updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(42)))]),
            removals: vec![],
        }));

        let val = e.get("new_obj").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
        assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(42))));
    }

    #[tokio::test]
    async fn patch_compute_and_apply_roundtrip() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");
        let mut e = j.entry_mut(root).await.next().await;

        let old_val = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)),
            (b, Value::int(2)),
            (c, Value::int(3)),
        ]));
        let new_val = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)),    // unchanged
            (b, Value::int(99)),   // changed
            // c removed
        ]));

        e.apply("state", StoragePatch::Snapshot(TypedValue::new(Arc::new(old_val.clone()), Ty::Infer)));

        let diff = PatchDiff::compute(&old_val, &new_val).expect("should have diff");
        e.apply("state", StoragePatch::Patch(diff));

        let result = e.get("state").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = result.value() else { panic!() };
        assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(1))));
        assert_eq!(fields.get(&b), Some(&Value::Pure(PureValue::Int(99))));
        assert_eq!(fields.get(&c), None);
    }

    #[tokio::test]
    async fn patch_compute_nested_roundtrip() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let x = interner.intern("x");
        let y = interner.intern("y");
        let mut e = j.entry_mut(root).await.next().await;

        let old_val = Value::object(FxHashMap::from_iter([
            (a, Value::object(FxHashMap::from_iter([
                (x, Value::int(1)),
                (y, Value::int(2)),
            ]))),
        ]));
        let new_val = Value::object(FxHashMap::from_iter([
            (a, Value::object(FxHashMap::from_iter([
                (x, Value::int(1)),   // unchanged
                (y, Value::int(99)),  // changed
            ]))),
        ]));

        e.apply("state", StoragePatch::Snapshot(TypedValue::new(Arc::new(old_val.clone()), Ty::Infer)));

        let diff = PatchDiff::compute(&old_val, &new_val).expect("should have diff");
        // The diff should be Nested, not Set for the whole object
        assert!(matches!(diff.updates.get(&a), Some(PatchOp::Nested(_))));
        e.apply("state", StoragePatch::Patch(diff));

        let result = e.get("state").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = result.value() else { panic!() };
        let Value::Lazy(LazyValue::Object(inner)) = fields.get(&a).unwrap() else { panic!() };
        assert_eq!(inner.get(&x), Some(&Value::Pure(PureValue::Int(1))));
        assert_eq!(inner.get(&y), Some(&Value::Pure(PureValue::Int(99))));
    }

    // =========================================================================
    // Sequence — diff-only storage, history accumulation, undo
    // =========================================================================

    fn seq_patch(items: Vec<TypedValue>) -> StoragePatch {
        let squashed = TrackedDeque::from_vec(items.clone());
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: items,
        };
        StoragePatch::Sequence { squashed, diff }
    }

    fn seq_patch_with_diff(
        squashed_items: Vec<TypedValue>,
        consumed: usize,
        removed_back: usize,
        pushed: Vec<TypedValue>,
    ) -> StoragePatch {
        let squashed = TrackedDeque::from_vec(squashed_items);
        let diff = OwnedDequeDiff { consumed, removed_back, pushed };
        StoragePatch::Sequence { squashed, diff }
    }

    fn get_deque_values(val: &TypedValue) -> Vec<Value> {
        let Value::Lazy(LazyValue::Deque(d)) = val.value() else {
            panic!("expected Deque");
        };
        d.as_slice().to_vec()
    }

    fn get_obj_fields(val: &TypedValue) -> FxHashMap<Astr, Value> {
        let Value::Lazy(LazyValue::Object(f)) = val.value() else {
            panic!("expected Object");
        };
        f.clone()
    }

    #[tokio::test]
    async fn sequence_append_only_diff() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            // Initial: [1, 2]
            e.apply("q", seq_patch(vec![TypedValue::int(1), TypedValue::int(2)]));
        }
        {
            let mut e = j.entry_mut(n1).await.next().await;
            // Append 3: squashed=[1,2,3], diff={pushed=[3]}
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(3)],
                0, 0,
                vec![TypedValue::int(3)],
            ));
            let val = e.get("q").unwrap();
            let vals = get_deque_values(&val);
            assert_eq!(vals, vec![Value::int(1), Value::int(2), Value::int(3)]);
        }
        // Go back to n1: should see [1, 2]
        {
            let e = j.entry(n1).await;
            let val = e.get("q").unwrap();
            let vals = get_deque_values(&val);
            assert_eq!(vals, vec![Value::int(1), Value::int(2)]);
        }
    }

    #[tokio::test]
    async fn sequence_consume_diff() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(3)]));
        }
        {
            let mut e = j.entry_mut(n1).await.next().await;
            // Consumed 1 from front: squashed=[2,3], diff={consumed=1}
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(2), TypedValue::int(3)],
                1, 0,
                vec![],
            ));
            let val = e.get("q").unwrap();
            let vals = get_deque_values(&val);
            assert_eq!(vals, vec![Value::int(2), Value::int(3)]);
        }
        // Go back to n1: should see [1, 2, 3]
        {
            let e = j.entry(n1).await;
            let val = e.get("q").unwrap();
            let vals = get_deque_values(&val);
            assert_eq!(vals, vec![Value::int(1), Value::int(2), Value::int(3)]);
        }
    }

    #[tokio::test]
    async fn sequence_consume_and_append_diff() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(10), TypedValue::int(20)]));
        }
        {
            let mut e = j.entry_mut(n1).await.next().await;
            // Consume 1 + append 30: squashed=[20,30], diff={consumed=1, pushed=[30]}
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(20), TypedValue::int(30)],
                1, 0,
                vec![TypedValue::int(30)],
            ));
            let val = e.get("q").unwrap();
            let vals = get_deque_values(&val);
            assert_eq!(vals, vec![Value::int(20), Value::int(30)]);
        }
        {
            let e = j.entry(n1).await;
            let val = e.get("q").unwrap();
            let vals = get_deque_values(&val);
            assert_eq!(vals, vec![Value::int(10), Value::int(20)]);
        }
    }

    #[tokio::test]
    async fn sequence_multi_turn_accumulation() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(1)]));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2)],
                0, 0, vec![TypedValue::int(2)],
            ));
        }
        let n3;
        {
            let mut e = j.entry_mut(n2).await.next().await;
            n3 = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(3)],
                0, 0, vec![TypedValue::int(3)],
            ));
        }
        // n3: [1, 2, 3]
        {
            let e = j.entry(n3).await;
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2), Value::int(3)]);
        }
        // n2: [1, 2]
        {
            let e = j.entry(n2).await;
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2)]);
        }
        // n1: [1]
        {
            let e = j.entry(n1).await;
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1)]);
        }
    }

    #[tokio::test]
    async fn sequence_empty_initial() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![]));
        }
        {
            let mut e = j.entry_mut(n1).await.next().await;
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1)],
                0, 0, vec![TypedValue::int(1)],
            ));
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1)]);
        }
        {
            let e = j.entry(n1).await;
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), Vec::<Value>::new());
        }
    }

    #[tokio::test]
    async fn sequence_fork_preserves_history() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(1), TypedValue::int(2)]));
        }
        // Fork: two children from n1
        let branch_a;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            branch_a = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(100)],
                0, 0, vec![TypedValue::int(100)],
            ));
        }
        let branch_b;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            branch_b = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(200)],
                0, 0, vec![TypedValue::int(200)],
            ));
        }
        // branch_a: [1, 2, 100]
        {
            let e = j.entry(branch_a).await;
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2), Value::int(100)]);
        }
        // branch_b: [1, 2, 200]
        {
            let e = j.entry(branch_b).await;
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2), Value::int(200)]);
        }
        // n1: still [1, 2]
        {
            let e = j.entry(n1).await;
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2)]);
        }
    }

    // =========================================================================
    // Mixed — Snapshot + Sequence + Patch on the same journal
    // =========================================================================

    #[tokio::test]
    async fn mixed_snapshot_and_sequence_coexist() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("counter", StoragePatch::Snapshot(TypedValue::int(0)));
            e.apply("log", seq_patch(vec![TypedValue::int(1)]));
        }
        {
            let mut e = j.entry_mut(n1).await.next().await;
            e.apply("counter", StoragePatch::Snapshot(TypedValue::int(1)));
            e.apply("log", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2)],
                0, 0, vec![TypedValue::int(2)],
            ));
            // counter=1, log=[1,2]
            assert!(matches!(e.get("counter").unwrap().value(), Value::Pure(PureValue::Int(1))));
            let val = e.get("log").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2)]);
        }
        // Back to n1: counter=0, log=[1]
        {
            let e = j.entry(n1).await;
            assert!(matches!(e.get("counter").unwrap().value(), Value::Pure(PureValue::Int(0))));
            let val = e.get("log").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1)]);
        }
    }

    #[tokio::test]
    async fn mixed_patch_and_sequence_coexist() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            let state = Value::object(FxHashMap::from_iter([(a, Value::int(0))]));
            e.apply("state", StoragePatch::Snapshot(TypedValue::new(Arc::new(state), Ty::Infer)));
            e.apply("log", seq_patch(vec![]));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            // Patch state.a = 1
            e.apply("state", StoragePatch::Patch(PatchDiff {
                updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(1)))]),
                removals: vec![],
            }));
            // Append to log
            e.apply("log", seq_patch_with_diff(
                vec![TypedValue::int(100)],
                0, 0, vec![TypedValue::int(100)],
            ));
        }
        // n2: state.a=1, log=[100]
        {
            let e = j.entry(n2).await;
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
            assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            let val = e.get("log").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(100)]);
        }
        // n1: state.a=0, log=[]
        {
            let e = j.entry(n1).await;
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
            assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(0))));
            let val = e.get("log").unwrap();
            assert_eq!(get_deque_values(&val), Vec::<Value>::new());
        }
    }

    #[tokio::test]
    async fn mixed_all_three_persistencies() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let x = interner.intern("x");
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("snap", StoragePatch::Snapshot(TypedValue::int(0)));
            e.apply("seq", seq_patch(vec![TypedValue::int(10)]));
            let obj = Value::object(FxHashMap::from_iter([(x, Value::int(100))]));
            e.apply("patch_obj", StoragePatch::Snapshot(TypedValue::new(Arc::new(obj), Ty::Infer)));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("snap", StoragePatch::Snapshot(TypedValue::int(1)));
            e.apply("seq", seq_patch_with_diff(
                vec![TypedValue::int(10), TypedValue::int(20)],
                0, 0, vec![TypedValue::int(20)],
            ));
            e.apply("patch_obj", StoragePatch::Patch(PatchDiff {
                updates: FxHashMap::from_iter([(x, PatchOp::Set(Value::int(200)))]),
                removals: vec![],
            }));
        }
        let n3;
        {
            let mut e = j.entry_mut(n2).await.next().await;
            n3 = e.uuid();
            e.apply("snap", StoragePatch::Snapshot(TypedValue::int(2)));
            e.apply("seq", seq_patch_with_diff(
                vec![TypedValue::int(10), TypedValue::int(20), TypedValue::int(30)],
                0, 0, vec![TypedValue::int(30)],
            ));
            e.apply("patch_obj", StoragePatch::Patch(PatchDiff {
                updates: FxHashMap::from_iter([(x, PatchOp::Set(Value::int(300)))]),
                removals: vec![],
            }));
        }
        // n3: snap=2, seq=[10,20,30], patch_obj.x=300
        {
            let e = j.entry(n3).await;
            assert!(matches!(e.get("snap").unwrap().value(), Value::Pure(PureValue::Int(2))));
            let val = e.get("seq").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(10), Value::int(20), Value::int(30)]);
            let val = e.get("patch_obj").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&x), Some(&Value::Pure(PureValue::Int(300))));
        }
        // n1: snap=0, seq=[10], patch_obj.x=100
        {
            let e = j.entry(n1).await;
            assert!(matches!(e.get("snap").unwrap().value(), Value::Pure(PureValue::Int(0))));
            let val = e.get("seq").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(10)]);
            let val = e.get("patch_obj").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&x), Some(&Value::Pure(PureValue::Int(100))));
        }
    }

    // =========================================================================
    // Patch — history across turns (undo)
    // =========================================================================

    #[tokio::test]
    async fn patch_3_turn_undo() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let b = interner.intern("b");

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(0)), (b, Value::int(0))]));
            e.apply("s", StoragePatch::Snapshot(TypedValue::new(Arc::new(obj), Ty::Infer)));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("s", StoragePatch::Patch(PatchDiff {
                updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(1)))]),
                removals: vec![],
            }));
        }
        let n3;
        {
            let mut e = j.entry_mut(n2).await.next().await;
            n3 = e.uuid();
            e.apply("s", StoragePatch::Patch(PatchDiff {
                updates: FxHashMap::from_iter([(b, PatchOp::Set(Value::int(2)))]),
                removals: vec![],
            }));
        }
        // n3: a=1, b=2
        {
            let e = j.entry(n3).await;
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(2))));
        }
        // n2: a=1, b=0
        {
            let e = j.entry(n2).await;
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(0))));
        }
        // n1: a=0, b=0
        {
            let e = j.entry(n1).await;
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(0))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(0))));
        }
    }

    #[tokio::test]
    async fn patch_fork_independent_branches() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(0))]));
            e.apply("s", StoragePatch::Snapshot(TypedValue::new(Arc::new(obj), Ty::Infer)));
        }
        let ba;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            ba = e.uuid();
            e.apply("s", StoragePatch::Patch(PatchDiff {
                updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(100)))]),
                removals: vec![],
            }));
        }
        let bb;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            bb = e.uuid();
            e.apply("s", StoragePatch::Patch(PatchDiff {
                updates: FxHashMap::from_iter([(a, PatchOp::Set(Value::int(200)))]),
                removals: vec![],
            }));
        }
        // ba.a=100, bb.a=200, n1.a=0
        {
            let e = j.entry(ba).await;
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(100))));
        }
        {
            let e = j.entry(bb).await;
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(200))));
        }
        {
            let e = j.entry(n1).await;
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(0))));
        }
    }
}
