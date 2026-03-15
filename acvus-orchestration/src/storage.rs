use std::sync::Arc;

use acvus_interpreter::{LazyValue, Value};
use acvus_utils::{Astr, OwnedDequeDiff, TrackedDeque};
use rustc_hash::FxHashMap;
use uuid::Uuid;

// ── Diff types ──────────────────────────────────────────────────────

/// Diff-based storage update.
#[derive(Debug, Clone)]
pub enum StorageDiff {
    /// No-op — ephemeral nodes don't persist.
    Ephemeral,
    /// Overwrite the stored value entirely.
    Snapshot(Value),
    /// Deque diff + squashed deque with evolved checksum.
    Deque {
        squashed: TrackedDeque<Value>,
        diff: OwnedDequeDiff<Value>,
    },
    /// Apply field-level patches to an existing Object value.
    Diff(ObjectDiff),
}

/// Object field-level patch.
#[derive(Debug, Clone)]
pub struct ObjectDiff {
    pub updates: FxHashMap<Astr, Value>,
    pub removals: Vec<Astr>,
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
    fn get(&self, key: &str) -> Option<Arc<Value>>;
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

    fn get(&self, key: &str) -> Option<Arc<Value>>;
    fn apply(&mut self, key: &str, diff: StorageDiff);
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
    pub turn_diff: FxHashMap<String, Arc<Value>>,
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
    accumulated: Arc<FxHashMap<String, Arc<Value>>>,
    /// Changes made during this turn.
    turn_diff: FxHashMap<String, Arc<Value>>,
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
    pub fn entries(&self) -> FxHashMap<String, Arc<Value>> {
        let node = &self.inner.nodes[self.idx];
        let mut result = (*node.accumulated).clone();
        for (k, v) in &node.turn_diff {
            result.insert(k.clone(), Arc::clone(v));
        }
        result
    }
}

impl<'a> EntryRef<'a> for TreeEntryRef<'a> {
    fn get(&self, key: &str) -> Option<Arc<Value>> {
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

    fn get(&self, key: &str) -> Option<Arc<Value>> {
        let node = &self.inner.nodes[self.idx];
        if let Some(val) = node.turn_diff.get(key) {
            return Some(Arc::clone(val));
        }
        node.accumulated.get(key).cloned()
    }

    fn apply(&mut self, key: &str, diff: StorageDiff) {
        let idx = self.idx;
        debug_assert!(
            self.inner.nodes[idx].children.is_empty(),
            "apply on non-leaf"
        );

        match diff {
            StorageDiff::Ephemeral => {}
            StorageDiff::Snapshot(v) => {
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), Arc::new(v));
            }
            StorageDiff::Deque { squashed, .. } => {
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), Arc::new(Value::Lazy(LazyValue::Deque(squashed))));
            }
            StorageDiff::Diff(obj_diff) => {
                let mut fields = self
                    .get(key)
                    .and_then(|arc| match arc.as_ref() {
                        Value::Lazy(LazyValue::Object(fields)) => Some(fields.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                for (k, v) in obj_diff.updates {
                    fields.insert(k, v);
                }
                for k in &obj_diff.removals {
                    fields.remove(k);
                }
                self.inner.nodes[idx]
                    .turn_diff
                    .insert(key.to_string(), Arc::new(Value::Lazy(LazyValue::Object(fields))));
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
    use acvus_interpreter::PureValue;
    use acvus_utils::Interner;

    use super::*;

    // --- Basic get/apply tests ---

    #[tokio::test]
    async fn apply_and_get() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("x", StorageDiff::Snapshot(Value::string("hello".into())));
        assert!(matches!(
            &*e.get("x").unwrap(),
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
            StorageDiff::Snapshot(Value::string("first".into())),
        );
        e.apply(
            "x",
            StorageDiff::Snapshot(Value::object(FxHashMap::from_iter([(
                interner.intern("v"),
                Value::int(2),
            )]))),
        );
        assert!(matches!(&*e.get("x").unwrap(), Value::Lazy(LazyValue::Object(_))));
    }

    #[tokio::test]
    async fn ephemeral_is_noop() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("x", StorageDiff::Ephemeral);
        assert!(e.get("x").is_none());
    }

    #[tokio::test]
    async fn deque_stores_squashed() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let squashed = TrackedDeque::from_vec(vec![Value::int(1), Value::int(2)]);
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![Value::int(1), Value::int(2)],
        };
        e.apply(
            "q",
            StorageDiff::Deque {
                squashed: squashed.clone(),
                diff,
            },
        );
        let val = e.get("q").unwrap();
        let Value::Lazy(LazyValue::Deque(d)) = val.as_ref() else {
            panic!("expected Deque");
        };
        assert_eq!(d.as_slice(), squashed.as_slice());
    }

    #[tokio::test]
    async fn deque_checksum_preserved() {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let squashed = TrackedDeque::from_vec(vec![Value::int(1)]);
        let cs = squashed.checksum();
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![Value::int(1)],
        };
        e.apply("q", StorageDiff::Deque { squashed, diff });
        let val = e.get("q").unwrap();
        let Value::Lazy(LazyValue::Deque(stored)) = val.as_ref() else {
            panic!("expected Deque")
        };
        assert_eq!(stored.checksum(), cs);
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
            StorageDiff::Snapshot(Value::object(FxHashMap::from_iter([
                (a, Value::int(1)),
                (b, Value::int(2)),
            ]))),
        );
        let diff = ObjectDiff {
            updates: FxHashMap::from_iter([(a, Value::int(100)), (c, Value::int(3))]),
            removals: vec![b],
        };
        e.apply("obj", StorageDiff::Diff(diff));
        let val = e.get("obj").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = val.as_ref() else {
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
        let diff = ObjectDiff {
            updates: FxHashMap::from_iter([(a, Value::int(42))]),
            removals: vec![],
        };
        e.apply("obj", StorageDiff::Diff(diff));
        let val = e.get("obj").unwrap();
        let Value::Lazy(LazyValue::Object(fields)) = val.as_ref() else {
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
            e.apply("x", StorageDiff::Snapshot(Value::int(1)));
            e.apply("y", StorageDiff::Snapshot(Value::int(2)));
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            // n2 should see parent's values via accumulated
            assert!(matches!(&*e.get("x").unwrap(), Value::Pure(PureValue::Int(1))));
            assert!(matches!(&*e.get("y").unwrap(), Value::Pure(PureValue::Int(2))));

            // Modifying n2 doesn't affect n1
            e.apply("x", StorageDiff::Snapshot(Value::int(99)));
            assert!(matches!(&*e.get("x").unwrap(), Value::Pure(PureValue::Int(99))));
        }

        // n1 still has original value
        {
            let e = j.entry(n1).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Pure(PureValue::Int(1))));
        }
        // n2 has the override
        {
            let e = j.entry(n2).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Pure(PureValue::Int(99))));
        }
    }

    #[tokio::test]
    async fn fork_creates_sibling() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::int(1)));
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::int(2)));
        }

        // Fork from n2 — creates sibling of n2 (child of n1)
        {
            let e = j.entry_mut(n2).await.fork().await;
            assert_eq!(e.depth(), 2);
            // Sees n1's accumulated (x=1), not n2's turn_diff (x=2)
            assert!(matches!(&*e.get("x").unwrap(), Value::Pure(PureValue::Int(1))));
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
            e.apply("x", StorageDiff::Snapshot(Value::int(1)));
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
            e.apply("x", StorageDiff::Snapshot(Value::int(1)));
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
        assert!(matches!(&*j.entry(n2).await.get("x").unwrap(), Value::Pure(PureValue::Int(1))));
        assert!(matches!(&*j.entry(n3).await.get("x").unwrap(), Value::Pure(PureValue::Int(1))));

        // Modifying one doesn't affect the other
        {
            let mut e = j.entry_mut(n2).await;
            e.apply("x", StorageDiff::Snapshot(Value::int(99)));
        }
        assert!(matches!(&*j.entry(n2).await.get("x").unwrap(), Value::Pure(PureValue::Int(99))));
        assert!(matches!(&*j.entry(n3).await.get("x").unwrap(), Value::Pure(PureValue::Int(1))));
    }

    #[tokio::test]
    async fn get_prefers_turn_diff() {
        let (mut j, root) = TreeJournal::new();

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::int(1)));
        }

        {
            let mut e = j.entry_mut(n1).await.next().await;
            // x=1 is in accumulated
            assert!(matches!(&*e.get("x").unwrap(), Value::Pure(PureValue::Int(1))));
            // Now override in turn_diff
            e.apply("x", StorageDiff::Snapshot(Value::int(2)));
            assert!(matches!(&*e.get("x").unwrap(), Value::Pure(PureValue::Int(2))));
        }
    }
}
