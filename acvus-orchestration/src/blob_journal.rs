use std::sync::Arc;

use acvus_interpreter::{ConcreteValue, PureValue, Value};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::blob::{BlobHash, BlobStore};
use crate::storage::{EntryMut, EntryRef, Journal, Prune, StorageDiff};

// ── Serialization types ──────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "v")]
enum SerTreeMeta {
    #[serde(rename = "1")]
    V1 {
        /// Append-only node entries.
        #[serde(rename = "n")]
        nodes: Vec<SerNodeEntryV1>,
        /// Append-only tombstone set (pruned UUIDs).
        #[serde(rename = "t", skip_serializing_if = "Vec::is_empty", default)]
        tombstones: Vec<String>,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerNodeEntryV1 {
    #[serde(rename = "u")]
    uuid: String,
    #[serde(rename = "p", skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
    #[serde(rename = "d")]
    depth: usize,
    // children: derived from parent pointers on load. Not stored.
    #[serde(rename = "s", skip_serializing_if = "Option::is_none")]
    snapshot_hash: Option<[u8; 32]>,
    #[serde(rename = "f", skip_serializing_if = "Option::is_none")]
    diff_hash: Option<[u8; 32]>,
}

// ── Internal tree ───────────────────────────────────────────────────

struct NodeEntry {
    uuid: Uuid,
    parent: Option<usize>,
    children: Vec<usize>,
    depth: usize,
    snapshot_hash: Option<BlobHash>,
    diff_hash: Option<BlobHash>,
}

struct TreeMeta {
    nodes: Vec<NodeEntry>,
    uuid_to_idx: FxHashMap<Uuid, usize>,
    /// Pruned UUIDs. Append-only for CRDT merge.
    tombstones: std::collections::HashSet<Uuid>,
}

struct HotNode {
    idx: usize,
    /// Full accumulated state at this node.
    state: FxHashMap<String, Arc<Value>>,
    /// Keys changed during this node's turn (subset of `state`).
    turn_diff: FxHashMap<String, Arc<Value>>,
}

// ── Value serialization helpers ─────────────────────────────────────

fn serialize_entries(
    entries: &FxHashMap<String, Arc<Value>>,
    interner: &Interner,
) -> Vec<u8> {
    let concrete: Vec<(String, ConcreteValue)> = entries
        .iter()
        .map(|(k, v)| {
            let cv = v.as_ref().clone().into_pure().to_concrete(interner);
            (k.clone(), cv)
        })
        .collect();
    serde_json::to_vec(&concrete).expect("entry serialization cannot fail")
}

fn deserialize_entries(
    bytes: &[u8],
    interner: &Interner,
) -> FxHashMap<String, Arc<Value>> {
    let concrete: Vec<(String, ConcreteValue)> =
        serde_json::from_slice(bytes).expect("entry deserialization failed");
    concrete
        .into_iter()
        .map(|(k, cv)| {
            let pv = PureValue::from_concrete(&cv, interner);
            (k, Arc::new(Value::from_pure(pv)))
        })
        .collect()
}

// ── StorageDiff application ─────────────────────────────────────────

fn apply_diff(
    state: &mut FxHashMap<String, Arc<Value>>,
    turn_diff: &mut FxHashMap<String, Arc<Value>>,
    key: &str,
    diff: StorageDiff,
) {
    match diff {
        StorageDiff::Ephemeral => {}
        StorageDiff::Snapshot(v) => {
            let arc = Arc::new(v);
            state.insert(key.to_string(), Arc::clone(&arc));
            turn_diff.insert(key.to_string(), arc);
        }
        StorageDiff::Deque { squashed, .. } => {
            let arc = Arc::new(Value::Deque(squashed));
            state.insert(key.to_string(), Arc::clone(&arc));
            turn_diff.insert(key.to_string(), arc);
        }
        StorageDiff::Diff(obj_diff) => {
            let mut fields = state
                .get(key)
                .and_then(|arc| match arc.as_ref() {
                    Value::Object(fields) => Some(fields.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            for (k, v) in obj_diff.updates {
                fields.insert(k, v);
            }
            for k in &obj_diff.removals {
                fields.remove(k);
            }
            let arc = Arc::new(Value::Object(fields));
            state.insert(key.to_string(), Arc::clone(&arc));
            turn_diff.insert(key.to_string(), arc);
        }
    }
}

// ── BlobStoreJournal ────────────────────────────────────────────────

/// Journal implementation backed by a [`BlobStore`].
///
/// In-memory: current hot node state + tree metadata.
/// BlobStore: snapshots, diffs, tree metadata blob.
///
/// Snapshots are stored every `snapshot_interval` turns (by depth).
/// Intermediate nodes store diffs only. Reconstruction walks up to
/// the nearest snapshot ancestor and applies diffs forward.
pub struct BlobStoreJournal<S: BlobStore> {
    store: S,
    tree: TreeMeta,
    interner: Interner,
    hot: Option<HotNode>,
    /// Current "tree" ref hash for CAS.
    tree_ref: Option<BlobHash>,
    /// Store a full snapshot every N turns. Default: 128.
    /// Root (depth 0) always has a snapshot.
    snapshot_interval: usize,
}

impl<S: BlobStore> BlobStoreJournal<S> {
    /// Default snapshot interval.
    pub const DEFAULT_SNAPSHOT_INTERVAL: usize = 128;

    /// Create a new empty journal. Returns the root node's UUID.
    pub async fn new(store: S, interner: Interner) -> (Self, Uuid) {
        Self::with_snapshot_interval(store, interner, Self::DEFAULT_SNAPSHOT_INTERVAL).await
    }

    /// Create a new journal with a custom snapshot interval.
    pub async fn with_snapshot_interval(
        mut store: S,
        interner: Interner,
        snapshot_interval: usize,
    ) -> (Self, Uuid) {
        assert!(snapshot_interval > 0, "snapshot_interval must be > 0");
        let root_uuid = Uuid::new_v4();
        let root = NodeEntry {
            uuid: root_uuid,
            parent: None,
            children: Vec::new(),
            depth: 0,
            snapshot_hash: None,
            diff_hash: None,
        };
        let mut uuid_to_idx = FxHashMap::default();
        uuid_to_idx.insert(root_uuid, 0);

        // Root always has a snapshot.
        let empty = serialize_entries(&FxHashMap::default(), &interner);
        let hash = store.put(empty).await;

        let mut journal = Self {
            store,
            tree: TreeMeta {
                nodes: vec![root],
                uuid_to_idx,
                tombstones: std::collections::HashSet::new(),
            },
            interner,
            hot: None,
            tree_ref: None,
            snapshot_interval,
        };

        journal.tree.nodes[0].snapshot_hash = Some(hash);

        (journal, root_uuid)
    }

    /// Load an existing journal from the blob store.
    /// Returns `None` if no journal is stored (no "tree" ref).
    pub async fn open(store: S, interner: Interner) -> Option<Self> {
        Self::open_with_snapshot_interval(store, interner, Self::DEFAULT_SNAPSHOT_INTERVAL).await
    }

    /// Load with a custom snapshot interval.
    pub async fn open_with_snapshot_interval(
        store: S,
        interner: Interner,
        snapshot_interval: usize,
    ) -> Option<Self> {
        assert!(snapshot_interval > 0, "snapshot_interval must be > 0");
        let tree_hash = store.ref_get("tree").await?;
        let tree_bytes = store.get(&tree_hash).await?;
        let ser: SerTreeMeta =
            serde_json::from_slice(&tree_bytes).expect("tree metadata deserialization");
        let tree = Self::deser_tree(ser);
        Some(Self {
            store,
            tree,
            interner,
            hot: None,
            tree_ref: Some(tree_hash),
            snapshot_interval,
        })
    }

    /// Persist tree metadata to the blob store.
    ///
    /// On CAS conflict: loads the remote version, merges (union entries +
    /// union tombstones), and retries. Merge is always convergent.
    pub async fn flush_tree(&mut self) {
        self.persist_hot_node().await;
        let mut my_ser = self.ser_tree();
        let mut expected = self.tree_ref;

        loop {
            let bytes = serde_json::to_vec(&my_ser).expect("tree metadata serialization");
            let hash = self.store.put(bytes).await;

            match self.store.ref_cas("tree", expected, hash).await {
                Ok(()) => {
                    self.tree_ref = Some(hash);
                    // Reload in-memory tree from merged state.
                    self.tree = Self::deser_tree(my_ser);
                    return;
                }
                Err(actual) => {
                    // CAS conflict: merge with remote version and retry.
                    let remote_hash = actual.expect("tree ref disappeared during flush");
                    let remote_bytes = self.store.get(&remote_hash).await
                        .expect("remote tree blob missing");
                    let remote_ser: SerTreeMeta =
                        serde_json::from_slice(&remote_bytes)
                            .expect("remote tree metadata deserialization");
                    my_ser = Self::merge_ser(my_ser, remote_ser);
                    expected = actual;
                }
            }
        }
    }

    /// Garbage-collect unreferenced blobs.
    ///
    /// Walks all live tree nodes, collects their referenced blob hashes (live set),
    /// then removes any blobs NOT in the live set.
    ///
    /// Returns the number of blobs removed.
    pub async fn gc(&mut self, all_blob_hashes: &[BlobHash]) -> usize {
        // Collect live set from tree metadata.
        let mut live = std::collections::HashSet::new();
        for node in &self.tree.nodes {
            if !self.tree.uuid_to_idx.contains_key(&node.uuid) {
                continue;
            }
            if let Some(h) = node.snapshot_hash {
                live.insert(h);
            }
            if let Some(h) = node.diff_hash {
                live.insert(h);
            }
        }
        // Also keep the tree metadata blob itself.
        if let Some(h) = self.tree_ref {
            live.insert(h);
        }

        let garbage: Vec<BlobHash> = all_blob_hashes
            .iter()
            .copied()
            .filter(|h| !live.contains(h))
            .collect();
        let count = garbage.len();
        self.store.batch_remove(garbage).await;
        count
    }

    /// Access the underlying blob store.
    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Returns all live tree nodes as `(uuid, parent_uuid, depth)` tuples.
    pub fn tree_nodes(&self) -> Vec<(Uuid, Option<Uuid>, usize)> {
        self.tree
            .nodes
            .iter()
            .filter(|n| self.tree.uuid_to_idx.contains_key(&n.uuid))
            .map(|n| {
                let parent_uuid = n.parent.map(|pidx| self.tree.nodes[pidx].uuid);
                (n.uuid, parent_uuid, n.depth)
            })
            .collect()
    }
}

// ── Internal helpers ────────────────────────────────────────────────

impl<S: BlobStore> BlobStoreJournal<S> {
    /// Load the full accumulated state for a given node index.
    ///
    /// If the node has a snapshot → single blob load (O(1)).
    /// Otherwise → walk up to the nearest snapshot ancestor, apply diffs forward.
    async fn load_state(&self, idx: usize) -> FxHashMap<String, Arc<Value>> {
        // Check hot node first.
        if let Some(ref hot) = self.hot {
            if hot.idx == idx {
                return hot.state.clone();
            }
        }

        // Walk up to the nearest node with a snapshot.
        let mut path = Vec::new(); // nodes from target up to (exclusive of) snapshot node
        let mut current = idx;
        loop {
            if self.tree.nodes[current].snapshot_hash.is_some() {
                break;
            }
            path.push(current);
            current = self.tree.nodes[current]
                .parent
                .expect("root must have a snapshot");
        }

        // Load snapshot.
        let snap_hash = self.tree.nodes[current].snapshot_hash.unwrap();
        let snap_bytes = self
            .store
            .get(&snap_hash)
            .await
            .expect("snapshot blob missing from store");
        let mut state = deserialize_entries(&snap_bytes, &self.interner);

        // Apply diffs forward from snapshot descendant toward target.
        for &node_idx in path.iter().rev() {
            if let Some(diff_hash) = self.tree.nodes[node_idx].diff_hash {
                let diff_bytes = self
                    .store
                    .get(&diff_hash)
                    .await
                    .expect("diff blob missing from store");
                let diff = deserialize_entries(&diff_bytes, &self.interner);
                for (k, v) in diff {
                    state.insert(k, v);
                }
            }
        }

        state
    }

    /// Persist the current hot node's state to the blob store.
    ///
    /// - Diff: always stored if non-empty.
    /// - Snapshot: only at depth % snapshot_interval == 0 (root always qualifies).
    async fn persist_hot_node(&mut self) {
        let Some(ref hot) = self.hot else { return };
        let idx = hot.idx;
        let depth = self.tree.nodes[idx].depth;

        // Always store diff if non-empty.
        if !hot.turn_diff.is_empty() {
            let diff_bytes = serialize_entries(&hot.turn_diff, &self.interner);
            let diff_hash = self.store.put(diff_bytes).await;
            self.tree.nodes[idx].diff_hash = Some(diff_hash);
        }

        // Snapshot only at interval boundaries.
        if depth % self.snapshot_interval == 0 {
            let snap_bytes = serialize_entries(&hot.state, &self.interner);
            let snap_hash = self.store.put(snap_bytes).await;
            self.tree.nodes[idx].snapshot_hash = Some(snap_hash);
        }
    }

    /// Ensure the given node is the hot node.
    /// Persists the previous hot node if switching.
    async fn ensure_hot(&mut self, target_idx: usize) {
        if let Some(ref hot) = self.hot {
            if hot.idx == target_idx {
                return;
            }
        }
        self.persist_hot_node().await;
        let state = self.load_state(target_idx).await;
        self.hot = Some(HotNode {
            idx: target_idx,
            state,
            turn_diff: FxHashMap::default(),
        });
    }

    fn ser_tree(&self) -> SerTreeMeta {
        SerTreeMeta::V1 {
            nodes: self
                .tree
                .nodes
                .iter()
                .map(|n| SerNodeEntryV1 {
                    uuid: n.uuid.to_string(),
                    parent: n.parent.map(|pi| self.tree.nodes[pi].uuid.to_string()),
                    depth: n.depth,
                    snapshot_hash: n.snapshot_hash.map(|h| *h.as_bytes()),
                    diff_hash: n.diff_hash.map(|h| *h.as_bytes()),
                })
                .collect(),
            tombstones: self.tree.tombstones.iter().map(|u| u.to_string()).collect(),
        }
    }

    fn deser_tree(ser: SerTreeMeta) -> TreeMeta {
        match ser {
            SerTreeMeta::V1 { nodes: ser_nodes, tombstones: ser_tombstones } => {
                let tombstones: std::collections::HashSet<Uuid> = ser_tombstones
                    .iter()
                    .map(|s| Uuid::parse_str(s).expect("invalid tombstone uuid"))
                    .collect();

                let mut nodes = Vec::with_capacity(ser_nodes.len());
                let mut uuid_to_idx = FxHashMap::default();

                let uuids: Vec<Uuid> = ser_nodes
                    .iter()
                    .map(|n| Uuid::parse_str(&n.uuid).expect("invalid uuid"))
                    .collect();

                for (i, sn) in ser_nodes.iter().enumerate() {
                    if !tombstones.contains(&uuids[i]) {
                        uuid_to_idx.insert(uuids[i], i);
                    }
                    nodes.push(NodeEntry {
                        uuid: uuids[i],
                        parent: None,
                        children: vec![],
                        depth: sn.depth,
                        snapshot_hash: sn.snapshot_hash.map(BlobHash::from_bytes),
                        diff_hash: sn.diff_hash.map(BlobHash::from_bytes),
                    });
                }

                for (i, sn) in ser_nodes.iter().enumerate() {
                    if let Some(ref parent_str) = sn.parent {
                        let parent_uuid = Uuid::parse_str(parent_str).expect("invalid parent uuid");
                        if let Some(&pidx) = uuid_to_idx.get(&parent_uuid) {
                            nodes[i].parent = Some(pidx);
                            if uuid_to_idx.contains_key(&uuids[i]) {
                                nodes[pidx].children.push(i);
                            }
                        }
                    }
                }

                TreeMeta { nodes, uuid_to_idx, tombstones }
            }
        }
    }

    /// Merge two SerTreeMeta. Both must be the same version.
    fn merge_ser(a: SerTreeMeta, b: SerTreeMeta) -> SerTreeMeta {
        match (a, b) {
            (
                SerTreeMeta::V1 { nodes: a_nodes, tombstones: a_tombstones },
                SerTreeMeta::V1 { nodes: b_nodes, tombstones: b_tombstones },
            ) => {
                let mut seen = std::collections::HashSet::new();
                let mut nodes = Vec::new();
                for n in a_nodes.into_iter().chain(b_nodes) {
                    if seen.insert(n.uuid.clone()) {
                        nodes.push(n);
                    }
                }
                let mut tombstone_set = std::collections::HashSet::new();
                for t in a_tombstones.into_iter().chain(b_tombstones) {
                    tombstone_set.insert(t);
                }
                SerTreeMeta::V1 {
                    nodes,
                    tombstones: tombstone_set.into_iter().collect(),
                }
            }
        }
    }
}

// ── Journal trait ───────────────────────────────────────────────────

impl<S: BlobStore> Journal for BlobStoreJournal<S> {
    type Ref<'a> = BlobEntryRef<'a, S> where S: 'a;
    type Mut<'a> = BlobEntryMut<'a, S> where S: 'a;

    async fn entry(&self, id: Uuid) -> BlobEntryRef<'_, S> {
        let idx = self.tree.uuid_to_idx[&id];
        let state = self.load_state(idx).await;
        BlobEntryRef {
            _journal: self,
            state,
            depth: self.tree.nodes[idx].depth,
            uuid: self.tree.nodes[idx].uuid,
        }
    }

    async fn entry_mut(&mut self, id: Uuid) -> BlobEntryMut<'_, S> {
        let idx = self.tree.uuid_to_idx[&id];
        self.ensure_hot(idx).await;
        BlobEntryMut { journal: self, idx }
    }

    fn parent_of(&self, id: Uuid) -> Option<Uuid> {
        let idx = self.tree.uuid_to_idx[&id];
        self.tree.nodes[idx].parent.map(|pidx| self.tree.nodes[pidx].uuid)
    }

    fn contains(&self, id: Uuid) -> bool {
        self.tree.uuid_to_idx.contains_key(&id)
    }
}

// ── BlobEntryRef ────────────────────────────────────────────────────

pub struct BlobEntryRef<'a, S: BlobStore> {
    _journal: &'a BlobStoreJournal<S>,
    state: FxHashMap<String, Arc<Value>>,
    depth: usize,
    uuid: Uuid,
}

impl<'a, S: BlobStore> BlobEntryRef<'a, S> {
    /// All key-value pairs visible at this node.
    pub fn entries(&self) -> FxHashMap<String, Arc<Value>> {
        self.state.clone()
    }
}

impl<'a, S: BlobStore> EntryRef<'a> for BlobEntryRef<'a, S> {
    fn get(&self, key: &str) -> Option<Arc<Value>> {
        self.state.get(key).cloned()
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn uuid(&self) -> Uuid {
        self.uuid
    }
}

// ── BlobEntryMut ────────────────────────────────────────────────────

pub struct BlobEntryMut<'a, S: BlobStore> {
    journal: &'a mut BlobStoreJournal<S>,
    idx: usize,
}

impl<'a, S: BlobStore> EntryMut<'a> for BlobEntryMut<'a, S> {
    type Ref<'x> = BlobEntryRef<'x, S> where 'a: 'x, Self: 'x;

    fn get(&self, key: &str) -> Option<Arc<Value>> {
        let hot = self.journal.hot.as_ref().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        hot.state.get(key).cloned()
    }

    fn apply(&mut self, key: &str, diff: StorageDiff) {
        let hot = self.journal.hot.as_mut().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        debug_assert!(
            self.journal.tree.nodes[self.idx].children.is_empty(),
            "apply on non-leaf"
        );
        apply_diff(&mut hot.state, &mut hot.turn_diff, key, diff);
    }

    async fn next(self) -> Self {
        let journal = self.journal;
        let idx = self.idx;

        // Persist current node.
        journal.persist_hot_node().await;

        // Inherit state from current node.
        let parent_state = journal.hot.as_ref().unwrap().state.clone();

        // Create child node.
        let new_uuid = Uuid::new_v4();
        let new_idx = journal.tree.nodes.len();
        let depth = journal.tree.nodes[idx].depth + 1;

        journal.tree.nodes.push(NodeEntry {
            uuid: new_uuid,
            parent: Some(idx),
            children: vec![],
            depth,
            snapshot_hash: None,
            diff_hash: None,
        });
        journal.tree.nodes[idx].children.push(new_idx);
        journal.tree.uuid_to_idx.insert(new_uuid, new_idx);

        // New child becomes hot.
        journal.hot = Some(HotNode {
            idx: new_idx,
            state: parent_state,
            turn_diff: FxHashMap::default(),
        });

        BlobEntryMut {
            journal,
            idx: new_idx,
        }
    }

    async fn fork(self) -> Self {
        let journal = self.journal;
        let idx = self.idx;
        let parent_idx = journal.tree.nodes[idx]
            .parent
            .expect("cannot fork root");

        // Persist current hot node before switching.
        journal.persist_hot_node().await;

        // Load parent's state (parent was a leaf → has snapshot → O(1)).
        let parent_state = journal.load_state(parent_idx).await;

        // Create sibling node.
        let new_uuid = Uuid::new_v4();
        let new_idx = journal.tree.nodes.len();
        let depth = journal.tree.nodes[parent_idx].depth + 1;

        journal.tree.nodes.push(NodeEntry {
            uuid: new_uuid,
            parent: Some(parent_idx),
            children: vec![],
            depth,
            snapshot_hash: None,
            diff_hash: None,
        });
        journal.tree.nodes[parent_idx].children.push(new_idx);
        journal.tree.uuid_to_idx.insert(new_uuid, new_idx);

        // Sibling becomes hot.
        journal.hot = Some(HotNode {
            idx: new_idx,
            state: parent_state,
            turn_diff: FxHashMap::default(),
        });

        BlobEntryMut {
            journal,
            idx: new_idx,
        }
    }

    fn prune(self, mode: Prune) {
        let journal = self.journal;
        let idx = self.idx;

        // Prune only updates tree structure.
        // Blob cleanup is deferred to gc() — content-addressed blobs may be
        // shared across nodes, so eager deletion is unsound.
        match mode {
            Prune::Leaf => {
                debug_assert!(
                    journal.tree.nodes[idx].children.is_empty(),
                    "prune Leaf on non-leaf node"
                );
                if let Some(parent_idx) = journal.tree.nodes[idx].parent {
                    journal.tree.nodes[parent_idx]
                        .children
                        .retain(|&c| c != idx);
                }
                let uuid = journal.tree.nodes[idx].uuid;
                journal.tree.uuid_to_idx.remove(&uuid);
                journal.tree.tombstones.insert(uuid);
                journal.tree.nodes[idx].snapshot_hash = None;
                journal.tree.nodes[idx].diff_hash = None;
            }
            Prune::Subtree => {
                let mut stack = vec![idx];
                let mut to_clear = Vec::new();
                while let Some(current) = stack.pop() {
                    to_clear.push(current);
                    let children: Vec<usize> = journal.tree.nodes[current].children.clone();
                    stack.extend(children);
                }
                if let Some(parent_idx) = journal.tree.nodes[idx].parent {
                    journal.tree.nodes[parent_idx]
                        .children
                        .retain(|&c| c != idx);
                }
                for node_idx in to_clear {
                    let uuid = journal.tree.nodes[node_idx].uuid;
                    journal.tree.uuid_to_idx.remove(&uuid);
                    journal.tree.tombstones.insert(uuid);
                    journal.tree.nodes[node_idx].snapshot_hash = None;
                    journal.tree.nodes[node_idx].diff_hash = None;
                    journal.tree.nodes[node_idx].children.clear();
                }
            }
        }

        // Clear hot if pruned.
        if let Some(ref hot) = journal.hot {
            if !journal.tree.uuid_to_idx.contains_key(&journal.tree.nodes[hot.idx].uuid) {
                journal.hot = None;
            }
        }
    }

    fn depth(&self) -> usize {
        self.journal.tree.nodes[self.idx].depth
    }

    fn uuid(&self) -> Uuid {
        self.journal.tree.nodes[self.idx].uuid
    }

    fn as_ref(&self) -> BlobEntryRef<'_, S> {
        let hot = self.journal.hot.as_ref().unwrap();
        BlobEntryRef {
            _journal: self.journal,
            state: hot.state.clone(),
            depth: self.journal.tree.nodes[self.idx].depth,
            uuid: self.journal.tree.nodes[self.idx].uuid,
        }
    }
}


// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use acvus_utils::TrackedDeque;

    use super::*;
    use crate::blob::MemBlobStore;
    use crate::storage::ObjectDiff;

    /// Tests use interval=1 to match TreeJournal behavior (snapshot every turn).
    async fn new_journal() -> (BlobStoreJournal<MemBlobStore>, Uuid) {
        BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), Interner::new(), 1).await
    }

    // ── Basic get/apply ──

    #[tokio::test]
    async fn apply_and_get() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("x", StorageDiff::Snapshot(Value::String("hello".into())));
        assert!(matches!(
            &*e.get("x").unwrap(),
            Value::String(v) if v == "hello"
        ));
        assert!(e.get("y").is_none());
    }

    #[tokio::test]
    async fn overwrite() {
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::new(MemBlobStore::new(), interner.clone()).await;
        let mut e = j.entry_mut(root).await.next().await;
        e.apply(
            "x",
            StorageDiff::Snapshot(Value::String("first".into())),
        );
        e.apply(
            "x",
            StorageDiff::Snapshot(Value::Object(FxHashMap::from_iter([(
                interner.intern("v"),
                Value::Int(2),
            )]))),
        );
        assert!(matches!(&*e.get("x").unwrap(), Value::Object(_)));
    }

    #[tokio::test]
    async fn ephemeral_is_noop() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("x", StorageDiff::Ephemeral);
        assert!(e.get("x").is_none());
    }

    #[tokio::test]
    async fn deque_stores_squashed() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.next().await;
        let squashed = TrackedDeque::from_vec(vec![Value::Int(1), Value::Int(2)]);
        let diff = acvus_utils::OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![Value::Int(1), Value::Int(2)],
        };
        e.apply(
            "q",
            StorageDiff::Deque {
                squashed: squashed.clone(),
                diff,
            },
        );
        let val = e.get("q").unwrap();
        assert!(
            matches!(val.as_ref(), Value::Deque(d) if d.as_slice() == squashed.as_slice())
        );
    }

    #[tokio::test]
    async fn object_diff_updates_and_removals() {
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::new(MemBlobStore::new(), interner.clone()).await;
        let mut e = j.entry_mut(root).await.next().await;
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");
        e.apply(
            "obj",
            StorageDiff::Snapshot(Value::Object(FxHashMap::from_iter([
                (a, Value::Int(1)),
                (b, Value::Int(2)),
            ]))),
        );
        let diff = ObjectDiff {
            updates: FxHashMap::from_iter([(a, Value::Int(100)), (c, Value::Int(3))]),
            removals: vec![b],
        };
        e.apply("obj", StorageDiff::Diff(diff));
        let val = e.get("obj").unwrap();
        let Value::Object(fields) = val.as_ref() else {
            panic!("expected Object")
        };
        assert_eq!(fields.get(&a), Some(&Value::Int(100)));
        assert_eq!(fields.get(&b), None);
        assert_eq!(fields.get(&c), Some(&Value::Int(3)));
    }

    // ── Tree structure ──

    #[tokio::test]
    async fn depth() {
        let (mut j, root) = new_journal().await;
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
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(1)));
            e.apply("y", StorageDiff::Snapshot(Value::Int(2)));
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            // Child sees parent's values.
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(1)));
            assert!(matches!(&*e.get("y").unwrap(), Value::Int(2)));
            // Modifying child doesn't affect parent.
            e.apply("x", StorageDiff::Snapshot(Value::Int(99)));
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(99)));
        }

        // Parent unchanged.
        {
            let e = j.entry(n1).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(1)));
        }
        // Child has override.
        {
            let e = j.entry(n2).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(99)));
        }
    }

    #[tokio::test]
    async fn fork_creates_sibling() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(1)));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(2)));
        }

        // Fork from n2 → sibling of n2 (child of n1).
        {
            let e = j.entry_mut(n2).await.fork().await;
            assert_eq!(e.depth(), 2);
            // Sees n1's state (x=1), not n2's (x=2).
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(1)));
        }
    }

    #[tokio::test]
    async fn prune_leaf() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(1)));
        }

        j.entry_mut(n2).await.prune(Prune::Leaf);

        // n1 still accessible.
        {
            let e = j.entry(n1).await;
            assert_eq!(e.depth(), 1);
        }
    }

    #[tokio::test]
    async fn prune_subtree() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
        }
        {
            let _e = j.entry_mut(n1).await.next().await;
        }

        j.entry_mut(n1).await.prune(Prune::Subtree);

        // Root still accessible.
        {
            let e = j.entry(root).await;
            assert_eq!(e.depth(), 0);
        }
    }

    // ── COW / isolation ──

    #[tokio::test]
    async fn cow_sharing() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(1)));
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

        // Both children see parent's data.
        assert!(matches!(&*j.entry(n2).await.get("x").unwrap(), Value::Int(1)));
        assert!(matches!(&*j.entry(n3).await.get("x").unwrap(), Value::Int(1)));

        // Modifying one child doesn't affect the other.
        {
            let mut e = j.entry_mut(n2).await;
            e.apply("x", StorageDiff::Snapshot(Value::Int(99)));
        }
        assert!(matches!(&*j.entry(n2).await.get("x").unwrap(), Value::Int(99)));
        assert!(matches!(&*j.entry(n3).await.get("x").unwrap(), Value::Int(1)));
    }

    #[tokio::test]
    async fn get_prefers_turn_diff() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(1)));
        }
        {
            let mut e = j.entry_mut(n1).await.next().await;
            // x=1 from parent.
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(1)));
            // Override.
            e.apply("x", StorageDiff::Snapshot(Value::Int(2)));
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(2)));
        }
    }

    // ── BlobStore persistence ──

    #[tokio::test]
    async fn snapshot_persisted_after_next() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(42)));
        }

        // next() persists n1's snapshot.
        let _n2;
        {
            let e = j.entry_mut(n1).await.next().await;
            _n2 = e.uuid();
        }

        // n1 now has a snapshot in blob store.
        let n1_idx = j.tree.uuid_to_idx[&n1];
        assert!(j.tree.nodes[n1_idx].snapshot_hash.is_some());

        let snap_hash = j.tree.nodes[n1_idx].snapshot_hash.unwrap();
        assert!(j.store.get(&snap_hash).await.is_some());
    }

    #[tokio::test]
    async fn state_survives_hot_swap() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(1)));
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("y", StorageDiff::Snapshot(Value::Int(2)));
        }

        // Switch back to n1 — triggers hot swap.
        // n2's state should be persisted, n1's state loaded.
        {
            let e = j.entry(n1).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(1)));
            assert!(e.get("y").is_none()); // y was added on n2, not n1
        }

        // n2 still has its data.
        {
            let e = j.entry(n2).await;
            assert!(matches!(&*e.get("y").unwrap(), Value::Int(2)));
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(1))); // inherited
        }
    }

    #[tokio::test]
    async fn prune_does_not_remove_shared_blobs() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            // n1 has empty state — same snapshot blob as root.
        }

        // Persist n1 by creating child.
        let n2;
        {
            let e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
        }

        // root and n1 share the same snapshot blob (empty state).
        let root_snap = j.tree.nodes[j.tree.uuid_to_idx[&root]].snapshot_hash.unwrap();
        let n1_snap = j.tree.nodes[j.tree.uuid_to_idx[&n1]].snapshot_hash.unwrap();
        assert_eq!(root_snap, n1_snap); // content-addressed dedup

        // Prune n2 then n1 — blobs NOT removed (shared with root).
        j.entry_mut(n2).await.prune(Prune::Leaf);
        j.entry_mut(n1).await.prune(Prune::Leaf);

        // Root's snapshot blob still accessible.
        assert!(j.store.get(&root_snap).await.is_some());
    }

    #[tokio::test]
    async fn gc_removes_unreferenced_blobs() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(42)));
        }
        let n2;
        {
            let e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
        }

        // n1 has a unique snapshot (contains x=42).
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap = j.tree.nodes[n1_idx].snapshot_hash.unwrap();
        assert!(j.store.get(&n1_snap).await.is_some());

        // Prune n2, then n1.
        j.entry_mut(n2).await.prune(Prune::Leaf);
        j.entry_mut(n1).await.prune(Prune::Leaf);

        // Blob still in store (prune doesn't delete).
        assert!(j.store.get(&n1_snap).await.is_some());

        // Collect all blob hashes from store for gc.
        let all_hashes = j.store.blob_hashes();
        let removed = j.gc(&all_hashes).await;
        assert!(removed > 0);

        // n1's unique snapshot is now gone.
        assert!(j.store.get(&n1_snap).await.is_none());

        // Root's snapshot still alive.
        let root_snap = j.tree.nodes[j.tree.uuid_to_idx[&root]].snapshot_hash.unwrap();
        assert!(j.store.get(&root_snap).await.is_some());
    }

    // ── flush_tree / open round-trip ──

    #[tokio::test]
    async fn flush_and_open_round_trip() {
        let interner = Interner::new();

        let n1;
        let n2;
        let store;
        {
            let (mut j, root) =
                BlobStoreJournal::new(MemBlobStore::new(), interner.clone()).await;

            {
                let mut e = j.entry_mut(root).await.next().await;
                n1 = e.uuid();
                e.apply("x", StorageDiff::Snapshot(Value::Int(10)));
            }
            {
                let mut e = j.entry_mut(n1).await.next().await;
                n2 = e.uuid();
                e.apply("y", StorageDiff::Snapshot(Value::Int(20)));
            }

            j.flush_tree().await;
            store = j.store;
        }

        // Re-open from the same blob store.
        let j2 = BlobStoreJournal::open(store, interner).await.expect("should open");

        // Verify data is intact.
        {
            let e = j2.entry(n1).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(10)));
        }
        {
            let e = j2.entry(n2).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(10)));
            assert!(matches!(&*e.get("y").unwrap(), Value::Int(20)));
        }
    }

    // ── Deep chain ──

    #[tokio::test]
    async fn deep_chain() {
        let (mut j, root) = new_journal().await;
        let mut cursor = root;
        for i in 0..50 {
            let mut e = j.entry_mut(cursor).await.next().await;
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageDiff::Snapshot(Value::Int(i as i64)),
            );
        }

        // Leaf should see all ancestors' values.
        let e = j.entry(cursor).await;
        assert_eq!(e.depth(), 50);
        for i in 0..50 {
            assert!(matches!(
                &*e.get(&format!("k{i}")).unwrap(),
                Value::Int(v) if *v == i as i64
            ));
        }
    }

    // ── as_ref ──

    #[tokio::test]
    async fn as_ref_returns_current_state() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.next().await;
        e.apply("x", StorageDiff::Snapshot(Value::Int(7)));
        let r = e.as_ref();
        assert!(matches!(&*r.get("x").unwrap(), Value::Int(7)));
        assert_eq!(r.depth(), 1);
    }

    // ── entries() ──

    #[tokio::test]
    async fn entries_returns_all_keys() {
        let (mut j, root) = new_journal().await;
        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("a", StorageDiff::Snapshot(Value::Int(1)));
            e.apply("b", StorageDiff::Snapshot(Value::Int(2)));
        }

        let entries = j.entry(n1).await.entries();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key("a"));
        assert!(entries.contains_key("b"));
    }

    // ── Multiple entry_mut switches ──

    #[tokio::test]
    async fn entry_mut_switch_preserves_both() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.next().await;
            n1 = e.uuid();
            e.apply("x", StorageDiff::Snapshot(Value::Int(1)));
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.next().await;
            n2 = e.uuid();
            e.apply("y", StorageDiff::Snapshot(Value::Int(2)));
        }

        // Switch to n1 to mutate.
        // This persists n2, loads n1.
        // n1 already has children so apply would fail (non-leaf assert).
        // But we can read.
        {
            let e = j.entry(n1).await;
            assert!(matches!(&*e.get("x").unwrap(), Value::Int(1)));
            assert!(e.get("y").is_none());
        }

        // n2 data still intact.
        {
            let e = j.entry(n2).await;
            assert!(matches!(&*e.get("y").unwrap(), Value::Int(2)));
        }
    }

    // ── Snapshot interval ──

    #[tokio::test]
    async fn interval_only_snapshots_at_boundaries() {
        // Interval=4: snapshots at depth 0, 4, 8, ...
        let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            Interner::new(),
            4,
        ).await;

        let mut cursor = root;
        let mut uuids = vec![root];
        for i in 1..=8 {
            let mut e = j.entry_mut(cursor).await.next().await;
            cursor = e.uuid();
            uuids.push(cursor);
            e.apply(
                &format!("k{i}"),
                StorageDiff::Snapshot(Value::Int(i as i64)),
            );
        }
        // Force persist of last node.
        {
            let e = j.entry_mut(cursor).await.next().await;
            cursor = e.uuid();
            uuids.push(cursor);
        }

        // Check which nodes have snapshots.
        // depth 0 (root): snapshot ✓
        // depth 1,2,3: no snapshot
        // depth 4: snapshot ✓
        // depth 5,6,7: no snapshot
        // depth 8: snapshot ✓
        for (i, &uuid) in uuids.iter().enumerate() {
            let idx = j.tree.uuid_to_idx.get(&uuid);
            if idx.is_none() { continue; }
            let idx = *idx.unwrap();
            let has_snap = j.tree.nodes[idx].snapshot_hash.is_some();
            let expected = i % 4 == 0;
            assert_eq!(
                has_snap, expected,
                "depth {i}: expected snapshot={expected}, got {has_snap}"
            );
        }
    }

    #[tokio::test]
    async fn interval_reconstruction_correct() {
        // Interval=4: intermediate nodes have diffs only.
        let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            Interner::new(),
            4,
        ).await;

        let mut cursor = root;
        for i in 1..=7 {
            let mut e = j.entry_mut(cursor).await.next().await;
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageDiff::Snapshot(Value::Int(i as i64)),
            );
        }

        // depth 7: no snapshot, must reconstruct from depth 4 + diffs at 5,6,7.
        let e = j.entry(cursor).await;
        assert_eq!(e.depth(), 7);
        for i in 1..=7 {
            assert!(matches!(
                &*e.get(&format!("k{i}")).unwrap(),
                Value::Int(v) if *v == i as i64
            ));
        }
    }

    #[tokio::test]
    async fn interval_fork_reconstructs_parent() {
        // Interval=4: fork from depth 3 → parent at depth 2 has no snapshot.
        let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            Interner::new(),
            4,
        ).await;

        let mut cursor = root;
        for i in 1..=3 {
            let mut e = j.entry_mut(cursor).await.next().await;
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageDiff::Snapshot(Value::Int(i as i64)),
            );
        }

        // Fork from depth 3 → sibling at depth 3, parent at depth 2 (no snapshot).
        // Must reconstruct parent from root snapshot + diffs at depth 1, 2.
        let forked = j.entry_mut(cursor).await.fork().await;
        assert_eq!(forked.depth(), 3);
        // Sees state up to depth 2 (parent), not depth 3 changes.
        assert!(matches!(&*forked.get("k1").unwrap(), Value::Int(1)));
        assert!(matches!(&*forked.get("k2").unwrap(), Value::Int(2)));
        assert!(forked.get("k3").is_none());
    }

    #[tokio::test]
    async fn interval_deep_chain_no_data_loss() {
        // Interval=128 (default), 200 turns.
        let (mut j, root) = BlobStoreJournal::new(MemBlobStore::new(), Interner::new()).await;
        let mut cursor = root;
        for i in 1..=200 {
            let mut e = j.entry_mut(cursor).await.next().await;
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageDiff::Snapshot(Value::Int(i as i64)),
            );
        }

        // Leaf at depth 200 sees all values.
        let e = j.entry(cursor).await;
        assert_eq!(e.depth(), 200);
        for i in 1..=200 {
            assert!(
                matches!(
                    &*e.get(&format!("k{i}")).unwrap(),
                    Value::Int(v) if *v == i as i64
                ),
                "missing k{i} at depth 200"
            );
        }

        // Count snapshots: should be at depths 0, 128 = 2 snapshots only
        // (depth 200 is hot, not yet persisted)
        let snap_count = j.tree.nodes.iter()
            .filter(|n| j.tree.uuid_to_idx.contains_key(&n.uuid))
            .filter(|n| n.snapshot_hash.is_some())
            .count();
        assert_eq!(snap_count, 2, "only root and depth-128 should have snapshots");
    }

    #[tokio::test]
    async fn interval_blob_count_much_lower() {
        // Compare blob count: interval=1 vs interval=4 over 8 turns.
        // Each turn adds a NEW key, so snapshots accumulate and differ from diffs.
        let count_with_interval = |interval: usize| async move {
            let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
                MemBlobStore::new(),
                Interner::new(),
                interval,
            ).await;
            let mut cursor = root;
            for i in 1..=8 {
                let mut e = j.entry_mut(cursor).await.next().await;
                cursor = e.uuid();
                // Each turn adds a distinct key — snapshot grows, diff stays small.
                e.apply(
                    &format!("k{i}"),
                    StorageDiff::Snapshot(Value::Int(i as i64)),
                );
            }
            j.persist_hot_node().await;
            j.store.blob_count()
        };

        let count_1 = count_with_interval(1).await;
        let count_4 = count_with_interval(4).await;

        // interval=1: snapshot + diff per turn. Snapshots grow ({k1}, {k1,k2}, ...),
        //             all unique → many blobs.
        // interval=4: only depths 0,4,8 get snapshots → far fewer blobs.
        assert!(
            count_4 < count_1,
            "interval=4 ({count_4}) should use fewer blobs than interval=1 ({count_1})"
        );
    }
}
