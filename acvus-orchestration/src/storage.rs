use std::sync::Arc;

use acvus_interpreter::{LazyValue, TypedValue, Value};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, OwnedDequeDiff, TrackedDeque};
use rustc_hash::FxHashMap;
use uuid::Uuid;

// NOTE: Arc<TypedValue> has been removed from the public API.
// TypedValue already contains Arc<Value> internally, so wrapping
// it in another Arc was unnecessary double-indirection. TypedValue::clone
// is cheap (Arc::clone on the inner value + Ty::clone).

// ── JournalError ───────────────────────────────────────────────────

/// Errors from journal storage operations (serialization, blob access, etc.).
#[derive(Debug, Clone)]
pub enum JournalError {
    /// Serialization of entries or tree metadata failed.
    Serialization(String),
    /// Deserialization of entries or tree metadata failed.
    Deserialization(String),
    /// A referenced blob was not found in the store.
    MissingBlob(String),
    /// Stored data is structurally invalid (e.g. bad UUID).
    CorruptedData(String),
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JournalError::Serialization(detail) => write!(f, "journal serialization: {detail}"),
            JournalError::Deserialization(detail) => write!(f, "journal deserialization: {detail}"),
            JournalError::MissingBlob(detail) => write!(f, "journal missing blob: {detail}"),
            JournalError::CorruptedData(detail) => write!(f, "journal corrupted data: {detail}"),
        }
    }
}

impl std::error::Error for JournalError {}

/// Storage operation — a complete instruction for what to store and at what type.
///
/// Every variant carries `ty: Ty` — the output type of the stored value.
/// Storage preserves this type so that loaded values have correct types.
#[derive(Debug, Clone)]
pub enum StorageOps {
    /// Sequence mode: squashed TrackedDeque + diff from origin.
    /// Produced by Resolver after collect_seq + into_diff.
    Sequence {
        squashed: TrackedDeque<TypedValue>,
        diff: OwnedDequeDiff<TypedValue>,
        ty: Ty,
    },
    /// Recursive value patch. Works for any value type:
    /// - `Set`: atomic replacement
    /// - `Rec`: recursive key-value diff
    Patch {
        diff: PatchDiff,
        ty: Ty,
    },
}

/// Recursive value diff. Works for any Value type.
///
/// - `Set`: atomic replacement — the entire value is overwritten.
/// - `Rec`: recursive key-value patch — updates and removes specific keys.
///
/// `compute` produces minimal diffs: identical values yield `None`,
/// non-keyed value changes yield `Set`, keyed value changes yield `Rec`
/// with only changed keys.
#[derive(Debug, Clone)]
pub enum PatchDiff {
    /// Atomic replacement — the entire value is overwritten.
    Set(Value),
    /// Recursive key-value patch: update specific keys, remove others.
    Rec {
        updates: FxHashMap<Astr, PatchDiff>,
        removals: Vec<Astr>,
    },
}

impl PatchDiff {
    /// Create a full-replacement patch from a TypedValue.
    pub fn set(tv: TypedValue) -> Self {
        PatchDiff::Set(tv.into_inner())
    }

    /// Compute a recursive diff between two Values.
    ///
    /// Returns `Some(diff)` if the values differ, `None` if they are identical.
    /// Keyed values produce `Rec` diffs; all other values produce `Set`.
    pub fn compute(old: &Value, new: &Value) -> Option<Self> {
        match (old, new) {
            (
                Value::Lazy(LazyValue::Object(old_f)),
                Value::Lazy(LazyValue::Object(new_f)),
            ) => {
                let mut updates = FxHashMap::default();
                let mut removals = Vec::new();

                for (key, new_val) in new_f {
                    match old_f.get(key) {
                        Some(old_val) if old_val == new_val => {}
                        Some(old_val) => {
                            match Self::compute(old_val, new_val) {
                                Some(nested) => { updates.insert(*key, nested); }
                                None => {
                                    if old_val != new_val {
                                        updates.insert(*key, PatchDiff::Set(new_val.clone()));
                                    }
                                }
                            }
                        }
                        None => {
                            updates.insert(*key, PatchDiff::Set(new_val.clone()));
                        }
                    }
                }

                for key in old_f.keys() {
                    if !new_f.contains_key(key) {
                        removals.push(*key);
                    }
                }

                if updates.is_empty() && removals.is_empty() {
                    None
                } else {
                    Some(PatchDiff::Rec { updates, removals })
                }
            }
            _ => {
                if old == new { None } else { Some(PatchDiff::Set(new.clone())) }
            }
        }
    }

    /// Apply this patch to an existing value, producing a new TypedValue.
    ///
    /// - `Set`: replaces the value entirely.
    /// - `Rec`: patches keyed fields in place. If the existing value has no
    ///   keyed fields, starts from an empty field set.
    ///
    /// `ty` is the output type for the resulting TypedValue. The caller must
    /// provide the correct type (typically from CompiledNode::output_ty).
    pub fn apply(self, existing: Option<&TypedValue>, ty: Ty) -> TypedValue {
        let old_value = existing.map(|e| e.value().clone());
        let new_value = self.apply_value(old_value.as_ref());
        TypedValue::new(new_value, ty)
    }

    fn apply_value(self, existing: Option<&Value>) -> Value {
        match self {
            PatchDiff::Set(v) => v,
            PatchDiff::Rec { updates, removals } => {
                let mut fields = existing
                    .and_then(|v| match v {
                        Value::Lazy(LazyValue::Object(f)) => Some(f.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                for (k, diff) in updates {
                    let old = fields.get(&k).cloned();
                    fields.insert(k, diff.apply_value(old.as_ref()));
                }
                for k in removals {
                    fields.remove(&k);
                }
                Value::object(fields)
            }
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
    fn get(&self, key: &str) -> Option<TypedValue>;
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

    fn get(&self, key: &str) -> Option<TypedValue>;
    fn apply(&mut self, key: &str, patch: StorageOps);
    async fn next(self) -> Result<Self, JournalError>;
    async fn fork(self) -> Result<Self, JournalError>;
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

    async fn entry(&self, id: Uuid) -> Result<Self::Ref<'_>, JournalError>;
    async fn entry_mut(&mut self, id: Uuid) -> Result<Self::Mut<'_>, JournalError>;
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
    pub turn_diff: FxHashMap<String, TypedValue>,
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
    accumulated: Arc<FxHashMap<String, TypedValue>>,
    /// Changes made during this turn.
    turn_diff: FxHashMap<String, TypedValue>,
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
                        merged.insert(k.clone(), v.clone());
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

    async fn entry(&self, id: Uuid) -> Result<TreeEntryRef<'_>, JournalError> {
        let idx = self.inner.uuid_to_idx[&id];
        Ok(TreeEntryRef {
            inner: &self.inner,
            idx,
        })
    }

    async fn entry_mut(&mut self, id: Uuid) -> Result<TreeEntryMut<'_>, JournalError> {
        let idx = self.inner.uuid_to_idx[&id];
        Ok(TreeEntryMut {
            inner: &mut self.inner,
            idx,
        })
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
    pub fn entries(&self) -> FxHashMap<String, TypedValue> {
        let node = &self.inner.nodes[self.idx];
        let mut result = (*node.accumulated).clone();
        for (k, v) in &node.turn_diff {
            result.insert(k.clone(), v.clone());
        }
        result
    }
}

impl<'a> EntryRef<'a> for TreeEntryRef<'a> {
    fn get(&self, key: &str) -> Option<TypedValue> {
        let node = &self.inner.nodes[self.idx];
        if let Some(val) = node.turn_diff.get(key) {
            return Some(val.clone());
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

    fn get(&self, key: &str) -> Option<TypedValue> {
        let node = &self.inner.nodes[self.idx];
        if let Some(val) = node.turn_diff.get(key) {
            return Some(val.clone());
        }
        node.accumulated.get(key).cloned()
    }

    fn apply(&mut self, key: &str, patch: StorageOps) {
        let idx = self.idx;
        debug_assert!(
            self.inner.nodes[idx].children.is_empty(),
            "apply on non-leaf"
        );

        match patch {
            StorageOps::Sequence { squashed, ty, .. } => {
                let value_deque = TrackedDeque::from_vec(
                    squashed.into_vec().into_iter().map(|tv| tv.into_inner()).collect(),
                );
                let stored = TypedValue::new(
                    Value::Lazy(LazyValue::Deque(value_deque)),
                    ty,
                );
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), stored);
            }
            StorageOps::Patch { diff, ty } => {
                let existing = self.get(key);
                let new_val = diff.apply(existing.as_ref(), ty);
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), new_val);
            }
        }
    }

    async fn next(self) -> Result<Self, JournalError> {
        let parent_idx = self.idx;
        let inner = self.inner;

        let parent_node = &inner.nodes[parent_idx];
        let mut merged = (*parent_node.accumulated).clone();
        for (k, v) in &parent_node.turn_diff {
            merged.insert(k.clone(), v.clone());
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

        Ok(TreeEntryMut { inner, idx: new_idx })
    }

    async fn fork(self) -> Result<Self, JournalError> {
        let sibling_idx = self.idx;
        let inner = self.inner;
        let parent_idx = inner.nodes[sibling_idx]
            .parent
            .expect("cannot fork root");

        let parent_node = &inner.nodes[parent_idx];
        let mut merged = (*parent_node.accumulated).clone();
        for (k, v) in &parent_node.turn_diff {
            merged.insert(k.clone(), v.clone());
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

        Ok(TreeEntryMut { inner, idx: new_idx })
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::string("hello".to_string())), ty: Ty::error() });
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.apply(
            "x",
            StorageOps::Patch { diff: PatchDiff::set(TypedValue::string("first")), ty: Ty::error() },
        );
        e.apply(
            "x",
            StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(
                Value::object(FxHashMap::from_iter([(
                    interner.intern("v"),
                    Value::int(2),
                )])),
                Ty::error(),
            )), ty: Ty::error() },
        );
        assert!(matches!(e.get("x").unwrap().value(), Value::Lazy(LazyValue::Object(_))));
    }

    #[tokio::test]
    async fn deque_stores_squashed() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(2)]);
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![TypedValue::int(1), TypedValue::int(2)],
        };
        e.apply(
            "q",
            StorageOps::Sequence {
                squashed,
                diff,
                ty: Ty::error(),
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1)]);
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![TypedValue::int(1)],
        };
        e.apply("q", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");
        e.apply(
            "obj",
            StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(
                Value::object(FxHashMap::from_iter([
                    (a, Value::int(1)),
                    (b, Value::int(2)),
                ])),
                Ty::error(),
            )), ty: Ty::error() },
        );
        let diff = PatchDiff::Rec {
            updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(100))), (c, PatchDiff::Set(Value::int(3)))]),
            removals: vec![b],
        };
        e.apply("obj", StorageOps::Patch { diff, ty: Ty::error() });
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let a = interner.intern("a");
        let diff = PatchDiff::Rec {
            updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(42)))]),
            removals: vec![],
        };
        e.apply("obj", StorageOps::Patch { diff, ty: Ty::error() });
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
            let e = j.entry(root).await.unwrap();
            assert_eq!(e.depth(), 0);
        }

        let n1;
        {
            let e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            assert_eq!(e.depth(), 1);
        }

        {
            let e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            assert_eq!(e.depth(), 2);
        }
    }

    #[tokio::test]
    async fn next_squashes_parent() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
            e.apply("y", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // n2 should see parent's values via accumulated
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            assert!(matches!(e.get("y").unwrap().value(), Value::Pure(PureValue::Int(2))));

            // Modifying n2 doesn't affect n1
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(99)), ty: Ty::error() });
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        }

        // n1 still has original value
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        }
        // n2 has the override
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        }
    }

    #[tokio::test]
    async fn fork_creates_sibling() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
        }

        // Fork from n2 — creates sibling of n2 (child of n1)
        {
            let e = j.entry_mut(n2).await.unwrap().fork().await.unwrap();
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
            let e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }

        j.entry_mut(n2).await.unwrap().prune(Prune::Leaf);

        // n1 still accessible
        {
            let e = j.entry(n1).await.unwrap();
            assert_eq!(e.depth(), 1);
        }
    }

    #[tokio::test]
    async fn prune_subtree() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
        }
        {
            let e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            let _n2 = e.uuid();
            // n3 = child of n2 — all will be pruned
        }

        j.entry_mut(n1).await.unwrap().prune(Prune::Subtree);

        // root still accessible
        {
            let e = j.entry(root).await.unwrap();
            assert_eq!(e.depth(), 0);
        }
    }

    #[tokio::test]
    async fn cow_sharing() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }

        let n2;
        {
            let e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
        }
        let n3;
        {
            let e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
        }

        // Both see parent's data
        assert!(matches!(j.entry(n2).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        assert!(matches!(j.entry(n3).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));

        // Modifying one doesn't affect the other
        {
            let mut e = j.entry_mut(n2).await.unwrap();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(99)), ty: Ty::error() });
        }
        assert!(matches!(j.entry(n2).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        assert!(matches!(j.entry(n3).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
    }

    #[tokio::test]
    async fn get_prefers_turn_diff() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }

        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            // x=1 is in accumulated
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            // Now override in turn_diff
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
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
        let PatchDiff::Rec { ref updates, ref removals } = diff else { panic!("expected Rec") };
        assert!(matches!(updates.get(&a), Some(PatchDiff::Set(Value::Pure(PureValue::Int(2))))));
        assert!(removals.is_empty());
    }

    #[test]
    fn compute_added_field() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1))]);
        let new = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(2))]);
        let diff = PatchDiff::compute(&old, &new).expect("should have diff");
        let b = i.intern("b");
        let PatchDiff::Rec { ref updates, ref removals } = diff else { panic!("expected Rec") };
        assert!(matches!(updates.get(&b), Some(PatchDiff::Set(Value::Pure(PureValue::Int(2))))));
        assert!(removals.is_empty());
    }

    #[test]
    fn compute_removed_field() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(2))]);
        let new = obj_i(&i, &[("a", Value::int(1))]);
        let diff = PatchDiff::compute(&old, &new).expect("should have diff");
        let b = i.intern("b");
        let PatchDiff::Rec { ref updates, ref removals } = diff else { panic!("expected Rec") };
        assert!(updates.is_empty());
        assert!(removals.contains(&b));
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
        let PatchDiff::Rec { ref updates, .. } = diff else { panic!("expected Rec") };
        // x is identical → not in updates
        assert!(!updates.contains_key(&i.intern("x")));
        // nested should be Rec, not Set
        let nested_diff = updates.get(&nested).expect("missing nested");
        let PatchDiff::Rec { updates: inner_updates, .. } = nested_diff else { panic!("expected Rec") };
        let b = i.intern("b");
        assert!(matches!(inner_updates.get(&b), Some(PatchDiff::Set(Value::Pure(PureValue::Int(99))))));
        assert!(!inner_updates.contains_key(&i.intern("a"))); // a unchanged
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
        // l1 → Rec → l2 → Rec → l3 → Set(2)
        let l1 = i.intern("l1");
        let l2 = i.intern("l2");
        let l3 = i.intern("l3");
        let PatchDiff::Rec { updates, .. } = &diff else { panic!("expected Rec") };
        let PatchDiff::Rec { updates: u1, .. } = updates.get(&l1).unwrap() else { panic!("l1 not Rec") };
        let PatchDiff::Rec { updates: u2, .. } = u1.get(&l2).unwrap() else { panic!("l2 not Rec") };
        assert!(matches!(u2.get(&l3), Some(PatchDiff::Set(Value::Pure(PureValue::Int(2))))));
    }

    #[test]
    fn compute_non_object_returns_set() {
        // Non-Object different values return Some(PatchDiff::Set(...))
        let old = Value::int(1);
        let new = Value::int(2);
        assert!(matches!(PatchDiff::compute(&old, &new), Some(PatchDiff::Set(Value::Pure(PureValue::Int(2))))));
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
        let PatchDiff::Rec { ref updates, .. } = diff else { panic!("expected Rec") };
        assert!(matches!(updates.get(&x), Some(PatchDiff::Set(Value::Pure(PureValue::Int(42))))));
    }

    #[test]
    fn compute_non_object_to_object_field() {
        let i = Interner::new();
        let old = obj_i(&i, &[("x", Value::int(1))]);
        let new = obj_i(&i, &[("x", obj_i(&i, &[("a", Value::int(2))]))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        let x = i.intern("x");
        let PatchDiff::Rec { ref updates, .. } = diff else { panic!("expected Rec") };
        // Int → Object: can't recurse, must be Set
        match updates.get(&x) {
            Some(PatchDiff::Set(Value::Lazy(LazyValue::Object(_)))) => {}
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
        let PatchDiff::Rec { ref updates, .. } = diff else { panic!("expected Rec") };
        assert_eq!(updates.len(), 1);
    }

    #[test]
    fn compute_nonempty_to_empty() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1))]);
        let new = obj_i(&i, &[]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        let PatchDiff::Rec { ref updates, ref removals } = diff else { panic!("expected Rec") };
        assert!(updates.is_empty());
        assert_eq!(removals.len(), 1);
    }

    #[test]
    fn compute_string_field_change() {
        let i = Interner::new();
        let old = obj_i(&i, &[("name", Value::string("alice".to_string()))]);
        let new = obj_i(&i, &[("name", Value::string("bob".to_string()))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        let name = i.intern("name");
        let PatchDiff::Rec { ref updates, .. } = diff else { panic!("expected Rec") };
        match updates.get(&name) {
            Some(PatchDiff::Set(Value::Pure(PureValue::String(s)))) => assert_eq!(s, "bob"),
            other => panic!("expected Set(String), got {:?}", other),
        }
    }

    #[test]
    fn compute_mixed_add_remove_update() {
        let i = Interner::new();
        let old = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(2)), ("c", Value::int(3))]);
        let new = obj_i(&i, &[("a", Value::int(1)), ("b", Value::int(99)), ("d", Value::int(4))]);
        let diff = PatchDiff::compute(&old, &new).expect("should diff");
        let PatchDiff::Rec { ref updates, ref removals } = diff else { panic!("expected Rec") };
        // a: unchanged
        assert!(!updates.contains_key(&i.intern("a")));
        // b: updated
        assert!(matches!(updates.get(&i.intern("b")), Some(PatchDiff::Set(Value::Pure(PureValue::Int(99))))));
        // c: removed
        assert!(removals.contains(&i.intern("c")));
        // d: added
        assert!(matches!(updates.get(&i.intern("d")), Some(PatchDiff::Set(Value::Pure(PureValue::Int(4))))));
    }

    // =========================================================================
    // PatchDiff::apply — unit tests
    // =========================================================================

    #[test]
    fn apply_set_overwrites() {
        let i = Interner::new();
        let a = i.intern("a");
        let fields = FxHashMap::from_iter([(a, Value::int(1))]);
        let existing = TypedValue::new(Value::object(fields), Ty::error());
        let diff = PatchDiff::Rec {
            updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(2)))]),
            removals: vec![],
        };
        let result = diff.apply(Some(&existing), Ty::error());
        let result_fields = get_obj_fields(&result);
        assert_eq!(result_fields.get(&a), Some(&Value::Pure(PureValue::Int(2))));
    }

    #[test]
    fn apply_nested_merges_recursively() {
        let i = Interner::new();
        let a = i.intern("a");
        let b = i.intern("b");
        let nested_key = i.intern("nested");
        let fields = FxHashMap::from_iter([
            (nested_key, Value::object(FxHashMap::from_iter([
                (a, Value::int(1)),
                (b, Value::int(2)),
            ]))),
        ]);
        let existing = TypedValue::new(Value::object(fields), Ty::error());
        let diff = PatchDiff::Rec {
            updates: FxHashMap::from_iter([(nested_key, PatchDiff::Rec {
                updates: FxHashMap::from_iter([(b, PatchDiff::Set(Value::int(99)))]),
                removals: vec![],
            })]),
            removals: vec![],
        };
        let result = diff.apply(Some(&existing), Ty::error());
        let result_fields = get_obj_fields(&result);
        let Value::Lazy(LazyValue::Object(inner)) = result_fields.get(&nested_key).unwrap() else { panic!() };
        assert_eq!(inner.get(&a), Some(&Value::Pure(PureValue::Int(1)))); // unchanged
        assert_eq!(inner.get(&b), Some(&Value::Pure(PureValue::Int(99)))); // updated
    }

    #[test]
    fn apply_removal() {
        let i = Interner::new();
        let a = i.intern("a");
        let b = i.intern("b");
        let fields = FxHashMap::from_iter([(a, Value::int(1)), (b, Value::int(2))]);
        let existing = TypedValue::new(Value::object(fields), Ty::error());
        let diff = PatchDiff::Rec {
            updates: FxHashMap::default(),
            removals: vec![b],
        };
        let result = diff.apply(Some(&existing), Ty::error());
        let result_fields = get_obj_fields(&result);
        assert_eq!(result_fields.len(), 1);
        assert!(result_fields.contains_key(&a));
    }

    #[test]
    fn apply_nested_on_missing_creates_object() {
        let i = Interner::new();
        let x = i.intern("x");
        let a = i.intern("a");
        let diff = PatchDiff::Rec {
            updates: FxHashMap::from_iter([(x, PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(1)))]),
                removals: vec![],
            })]),
            removals: vec![],
        };
        let result = diff.apply(None, Ty::error());
        let result_fields = get_obj_fields(&result);
        let Value::Lazy(LazyValue::Object(inner)) = result_fields.get(&x).unwrap() else { panic!() };
        assert_eq!(inner.get(&a), Some(&Value::Pure(PureValue::Int(1))));
    }

    // =========================================================================
    // Patch(Set) — storage tests
    // =========================================================================

    #[tokio::test]
    async fn snapshot_overwrites_entirely() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
        assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(2))));
    }

    #[tokio::test]
    async fn snapshot_no_history_across_turns() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(2))));
        }
        // Going back to n1 → should see value 1, not 2
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        }
    }

    #[tokio::test]
    async fn snapshot_string_value() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.apply("msg", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(
            Value::string("hello".to_string()),
            Ty::String,
        )), ty: Ty::error() });
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(2)]);
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![TypedValue::int(1), TypedValue::int(2)],
        };
        e.apply("seq", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let inner_key = interner.intern("inner");

        // Initial: {a: 1, inner: {b: 10}}
        let initial = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)),
            (inner_key, Value::object(FxHashMap::from_iter([(b, Value::int(10))]))),
        ]));
        e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(initial, Ty::error())), ty: Ty::error() });

        // Patch: inner.b = 99 (recursive)
        let diff = PatchDiff::Rec {
            updates: FxHashMap::from_iter([(inner_key, PatchDiff::Rec {
                updates: FxHashMap::from_iter([(b, PatchDiff::Set(Value::int(99)))]),
                removals: vec![],
            })]),
            removals: vec![],
        };
        e.apply("state", StorageOps::Patch { diff, ty: Ty::error() });

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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let initial = Value::object(FxHashMap::from_iter([
                (a, Value::int(1)),
                (b, Value::int(2)),
            ]));
            e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(initial, Ty::error())), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // Patch: a = 10
            let diff = PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(10)))]),
                removals: vec![],
            };
            e.apply("state", StorageOps::Patch { diff, ty: Ty::error() });
        }

        // At n2: a=10, b=2
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
            assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(10))));
            assert_eq!(fields.get(&b), Some(&Value::Pure(PureValue::Int(2))));
        }

        // At n1: a=1, b=2 (history preserved!)
        {
            let e = j.entry(n1).await.unwrap();
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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();

        let initial = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)),
            (b, Value::int(2)),
        ]));
        e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(initial, Ty::error())), ty: Ty::error() });

        // First patch: update a
        e.apply("state", StorageOps::Patch { diff: PatchDiff::Rec {
            updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(10)))]),
            removals: vec![],
        }, ty: Ty::error() });

        // Second patch: add c, remove b
        e.apply("state", StorageOps::Patch { diff: PatchDiff::Rec {
            updates: FxHashMap::from_iter([(c, PatchDiff::Set(Value::int(3)))]),
            removals: vec![b],
        }, ty: Ty::error() });

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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();

        e.apply("new_obj", StorageOps::Patch { diff: PatchDiff::Rec {
            updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(42)))]),
            removals: vec![],
        }, ty: Ty::error() });

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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();

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

        e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(old_val.clone(), Ty::error())), ty: Ty::error() });

        let diff = PatchDiff::compute(&old_val, &new_val).expect("should have diff");
        e.apply("state", StorageOps::Patch { diff, ty: Ty::error() });

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
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();

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

        e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(old_val.clone(), Ty::error())), ty: Ty::error() });

        let diff = PatchDiff::compute(&old_val, &new_val).expect("should have diff");
        // The diff should be Rec, not Set for the whole object
        assert!(matches!(&diff, PatchDiff::Rec { updates, .. } if matches!(updates.get(&a), Some(PatchDiff::Rec { .. }))));
        e.apply("state", StorageOps::Patch { diff, ty: Ty::error() });

        let result = e.get("state").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = result.value() else { panic!() };
        let Value::Lazy(LazyValue::Object(inner)) = fields.get(&a).unwrap() else { panic!() };
        assert_eq!(inner.get(&x), Some(&Value::Pure(PureValue::Int(1))));
        assert_eq!(inner.get(&y), Some(&Value::Pure(PureValue::Int(99))));
    }

    // =========================================================================
    // Sequence — diff-only storage, history accumulation, undo
    // =========================================================================

    fn seq_patch(items: Vec<TypedValue>) -> StorageOps {
        let squashed = TrackedDeque::from_vec(items.clone());
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: items,
        };
        StorageOps::Sequence { squashed, diff, ty: Ty::error() }
    }

    fn seq_patch_with_diff(
        squashed_items: Vec<TypedValue>,
        consumed: usize,
        removed_back: usize,
        pushed: Vec<TypedValue>,
    ) -> StorageOps {
        let squashed = TrackedDeque::from_vec(squashed_items);
        let diff = OwnedDequeDiff { consumed, removed_back, pushed };
        StorageOps::Sequence { squashed, diff, ty: Ty::error() }
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // Initial: [1, 2]
            e.apply("q", seq_patch(vec![TypedValue::int(1), TypedValue::int(2)]));
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
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
            let e = j.entry(n1).await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(3)]));
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
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
            let e = j.entry(n1).await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(10), TypedValue::int(20)]));
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
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
            let e = j.entry(n1).await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(1)]));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2)],
                0, 0, vec![TypedValue::int(2)],
            ));
        }
        let n3;
        {
            let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(3)],
                0, 0, vec![TypedValue::int(3)],
            ));
        }
        // n3: [1, 2, 3]
        {
            let e = j.entry(n3).await.unwrap();
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2), Value::int(3)]);
        }
        // n2: [1, 2]
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2)]);
        }
        // n1: [1]
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1)]);
        }
    }

    #[tokio::test]
    async fn sequence_empty_initial() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![]));
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1)],
                0, 0, vec![TypedValue::int(1)],
            ));
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1)]);
        }
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), Vec::<Value>::new());
        }
    }

    #[tokio::test]
    async fn sequence_fork_preserves_history() {
        let (mut j, root) = TreeJournal::new();
        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("q", seq_patch(vec![TypedValue::int(1), TypedValue::int(2)]));
        }
        // Fork: two children from n1
        let branch_a;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            branch_a = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(100)],
                0, 0, vec![TypedValue::int(100)],
            ));
        }
        let branch_b;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            branch_b = e.uuid();
            e.apply("q", seq_patch_with_diff(
                vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(200)],
                0, 0, vec![TypedValue::int(200)],
            ));
        }
        // branch_a: [1, 2, 100]
        {
            let e = j.entry(branch_a).await.unwrap();
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2), Value::int(100)]);
        }
        // branch_b: [1, 2, 200]
        {
            let e = j.entry(branch_b).await.unwrap();
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2), Value::int(200)]);
        }
        // n1: still [1, 2]
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("q").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(1), Value::int(2)]);
        }
    }

    // =========================================================================
    // Mixed — Patch(Set) + Sequence + Patch(Rec) on the same journal
    // =========================================================================

    #[tokio::test]
    async fn mixed_snapshot_and_sequence_coexist() {
        let interner = Interner::new();
        let (mut j, root) = TreeJournal::new();
        let a = interner.intern("a");
        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("counter", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(0)), ty: Ty::error() });
            e.apply("log", seq_patch(vec![TypedValue::int(1)]));
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            e.apply("counter", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
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
            let e = j.entry(n1).await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let state = Value::object(FxHashMap::from_iter([(a, Value::int(0))]));
            e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(state, Ty::error())), ty: Ty::error() });
            e.apply("log", seq_patch(vec![]));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // Patch state.a = 1
            e.apply("state", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(1)))]),
                removals: vec![],
            }, ty: Ty::error() });
            // Append to log
            e.apply("log", seq_patch_with_diff(
                vec![TypedValue::int(100)],
                0, 0, vec![TypedValue::int(100)],
            ));
        }
        // n2: state.a=1, log=[100]
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(fields)) = val.value() else { panic!() };
            assert_eq!(fields.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            let val = e.get("log").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(100)]);
        }
        // n1: state.a=0, log=[]
        {
            let e = j.entry(n1).await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("snap", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(0)), ty: Ty::error() });
            e.apply("seq", seq_patch(vec![TypedValue::int(10)]));
            let obj = Value::object(FxHashMap::from_iter([(x, Value::int(100))]));
            e.apply("patch_obj", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(obj, Ty::error())), ty: Ty::error() });
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.apply("snap", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
            e.apply("seq", seq_patch_with_diff(
                vec![TypedValue::int(10), TypedValue::int(20)],
                0, 0, vec![TypedValue::int(20)],
            ));
            e.apply("patch_obj", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(x, PatchDiff::Set(Value::int(200)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }
        let n3;
        {
            let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
            e.apply("snap", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
            e.apply("seq", seq_patch_with_diff(
                vec![TypedValue::int(10), TypedValue::int(20), TypedValue::int(30)],
                0, 0, vec![TypedValue::int(30)],
            ));
            e.apply("patch_obj", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(x, PatchDiff::Set(Value::int(300)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }
        // n3: snap=2, seq=[10,20,30], patch_obj.x=300
        {
            let e = j.entry(n3).await.unwrap();
            assert!(matches!(e.get("snap").unwrap().value(), Value::Pure(PureValue::Int(2))));
            let val = e.get("seq").unwrap();
            assert_eq!(get_deque_values(&val), vec![Value::int(10), Value::int(20), Value::int(30)]);
            let val = e.get("patch_obj").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&x), Some(&Value::Pure(PureValue::Int(300))));
        }
        // n1: snap=0, seq=[10], patch_obj.x=100
        {
            let e = j.entry(n1).await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(0)), (b, Value::int(0))]));
            e.apply("s", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(obj, Ty::error())), ty: Ty::error() });
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.apply("s", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(1)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }
        let n3;
        {
            let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
            e.apply("s", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(b, PatchDiff::Set(Value::int(2)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }
        // n3: a=1, b=2
        {
            let e = j.entry(n3).await.unwrap();
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(2))));
        }
        // n2: a=1, b=0
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(0))));
        }
        // n1: a=0, b=0
        {
            let e = j.entry(n1).await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(0))]));
            e.apply("s", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(obj, Ty::error())), ty: Ty::error() });
        }
        let ba;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            ba = e.uuid();
            e.apply("s", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(100)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }
        let bb;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            bb = e.uuid();
            e.apply("s", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(200)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }
        // ba.a=100, bb.a=200, n1.a=0
        {
            let e = j.entry(ba).await.unwrap();
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(100))));
        }
        {
            let e = j.entry(bb).await.unwrap();
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(200))));
        }
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("s").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(0))));
        }
    }
}
