use acvus_interpreter::{ConcreteValue, LazyValue, TypedValue, Value};
use acvus_utils::{Interner, TrackedDeque};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::blob::{BlobHash, BlobStore};
use crate::storage::{EntryMut, EntryRef, Journal, JournalError, Prune, StorageOps};

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
    state: FxHashMap<String, TypedValue>,
    /// Keys changed during this node's turn (subset of `state`).
    turn_diff: FxHashMap<String, TypedValue>,
}

// ── Value serialization helpers ─────────────────────────────────────

fn serialize_entries(
    entries: &FxHashMap<String, TypedValue>,
    interner: &Interner,
) -> Result<Vec<u8>, JournalError> {
    let concrete: Vec<(String, ConcreteValue, acvus_mir::ser_ty::SerTy)> = entries
        .iter()
        .map(|(k, v)| {
            let cv = v.to_concrete(interner);
            let ty = v.ty().to_ser(interner);
            (k.clone(), cv, ty)
        })
        .collect();
    serde_json::to_vec(&concrete).map_err(|e| JournalError::Serialization(e.to_string()))
}

fn deserialize_entries(
    bytes: &[u8],
    interner: &Interner,
) -> Result<FxHashMap<String, TypedValue>, JournalError> {
    let concrete: Vec<(String, ConcreteValue, acvus_mir::ser_ty::SerTy)> =
        serde_json::from_slice(bytes)
            .map_err(|e| JournalError::Deserialization(e.to_string()))?;
    Ok(concrete
        .into_iter()
        .map(|(k, cv, ser_ty)| {
            let ty = ser_ty.to_ty(interner);
            (k, TypedValue::from_concrete(&cv, interner, ty))
        })
        .collect())
}

// ── StoragePatch application ─────────────────────────────────────────

fn apply_diff(
    state: &mut FxHashMap<String, TypedValue>,
    turn_diff: &mut FxHashMap<String, TypedValue>,
    key: &str,
    diff: StorageOps,
) {
    match diff {
        StorageOps::Sequence { squashed, ty, .. } => {
            let values = TrackedDeque::from_vec(
                squashed.into_vec().into_iter().map(|tv| tv.into_inner()).collect(),
            );
            let stored = TypedValue::new(
                Value::Lazy(LazyValue::Deque(values)),
                ty,
            );
            let key_owned = key.to_string();
            state.insert(key_owned.clone(), stored.clone());
            turn_diff.insert(key_owned, stored);
        }
        StorageOps::Patch { diff, ty } => {
            let existing = state.get(key);
            let new_val = diff.apply(existing, ty);
            let key_owned = key.to_string();
            state.insert(key_owned.clone(), new_val.clone());
            turn_diff.insert(key_owned, new_val);
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
    pub async fn new(store: S, interner: Interner) -> Result<(Self, Uuid), JournalError> {
        Self::with_snapshot_interval(store, interner, Self::DEFAULT_SNAPSHOT_INTERVAL).await
    }

    /// Create a new journal with a custom snapshot interval.
    pub async fn with_snapshot_interval(
        mut store: S,
        interner: Interner,
        snapshot_interval: usize,
    ) -> Result<(Self, Uuid), JournalError> {
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
        let empty = serialize_entries(&FxHashMap::default(), &interner)?;
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

        Ok((journal, root_uuid))
    }

    /// Load an existing journal from the blob store.
    /// Returns `Ok(None)` if no journal is stored (no "tree" ref).
    /// Returns `Err` if the journal exists but is corrupted.
    pub async fn open(store: S, interner: Interner) -> Result<Option<Self>, JournalError> {
        Self::open_with_snapshot_interval(store, interner, Self::DEFAULT_SNAPSHOT_INTERVAL).await
    }

    /// Load with a custom snapshot interval.
    /// Returns `Ok(None)` if no journal is stored, `Err` if corrupted.
    pub async fn open_with_snapshot_interval(
        store: S,
        interner: Interner,
        snapshot_interval: usize,
    ) -> Result<Option<Self>, JournalError> {
        assert!(snapshot_interval > 0, "snapshot_interval must be > 0");
        let tree_hash = match store.ref_get("tree").await {
            Some(h) => h,
            None => return Ok(None),
        };
        let tree_bytes = match store.get(&tree_hash).await {
            Some(b) => b,
            None => return Ok(None),
        };
        let ser: SerTreeMeta =
            serde_json::from_slice(&tree_bytes)
                .map_err(|e| JournalError::Deserialization(e.to_string()))?;
        let tree = Self::deser_tree(ser)?;
        Ok(Some(Self {
            store,
            tree,
            interner,
            hot: None,
            tree_ref: Some(tree_hash),
            snapshot_interval,
        }))
    }

    /// Persist tree metadata to the blob store.
    ///
    /// On CAS conflict: loads the remote version, merges (union entries +
    /// union tombstones), and retries. Merge is always convergent.
    pub async fn flush_tree(&mut self) -> Result<(), JournalError> {
        self.persist_hot_node().await?;
        let mut my_ser = self.ser_tree();
        let mut expected = self.tree_ref;

        loop {
            let bytes = serde_json::to_vec(&my_ser)
                .map_err(|e| JournalError::Serialization(e.to_string()))?;
            let hash = self.store.put(bytes).await;

            match self.store.ref_cas("tree", expected, hash).await {
                Ok(()) => {
                    self.tree_ref = Some(hash);
                    // Reload in-memory tree from merged state.
                    self.tree = Self::deser_tree(my_ser)?;
                    return Ok(());
                }
                Err(actual) => {
                    // CAS conflict: merge with remote version and retry.
                    let remote_hash = actual
                        .ok_or_else(|| JournalError::MissingBlob("tree ref disappeared during flush".into()))?;
                    let remote_bytes = self.store.get(&remote_hash).await
                        .ok_or_else(|| JournalError::MissingBlob("remote tree blob".into()))?;
                    let remote_ser: SerTreeMeta =
                        serde_json::from_slice(&remote_bytes)
                            .map_err(|e| JournalError::Deserialization(e.to_string()))?;
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
    async fn load_state(&self, idx: usize) -> Result<FxHashMap<String, TypedValue>, JournalError> {
        // Check hot node first.
        if let Some(ref hot) = self.hot {
            if hot.idx == idx {
                return Ok(hot.state.clone());
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
            .ok_or_else(|| JournalError::MissingBlob("snapshot".into()))?;
        let mut state = deserialize_entries(&snap_bytes, &self.interner)?;

        // Apply diffs forward from snapshot descendant toward target.
        for &node_idx in path.iter().rev() {
            if let Some(diff_hash) = self.tree.nodes[node_idx].diff_hash {
                let diff_bytes = self
                    .store
                    .get(&diff_hash)
                    .await
                    .ok_or_else(|| JournalError::MissingBlob("diff".into()))?;
                let diff = deserialize_entries(&diff_bytes, &self.interner)?;
                for (k, v) in diff {
                    state.insert(k, v);
                }
            }
        }

        Ok(state)
    }

    /// Persist the current hot node's state to the blob store.
    ///
    /// - Patch: always stored if non-empty.
    /// - Snapshot: only at depth % snapshot_interval == 0 (root always qualifies).
    async fn persist_hot_node(&mut self) -> Result<(), JournalError> {
        let Some(ref hot) = self.hot else { return Ok(()) };
        let idx = hot.idx;
        let depth = self.tree.nodes[idx].depth;

        // Always store diff if non-empty.
        if !hot.turn_diff.is_empty() {
            let diff_bytes = serialize_entries(&hot.turn_diff, &self.interner)?;
            let diff_hash = self.store.put(diff_bytes).await;
            self.tree.nodes[idx].diff_hash = Some(diff_hash);
        }

        // Snapshot only at interval boundaries.
        if depth % self.snapshot_interval == 0 {
            let snap_bytes = serialize_entries(&hot.state, &self.interner)?;
            let snap_hash = self.store.put(snap_bytes).await;
            self.tree.nodes[idx].snapshot_hash = Some(snap_hash);
        }

        Ok(())
    }

    /// Ensure the given node is the hot node.
    /// Persists the previous hot node if switching.
    async fn ensure_hot(&mut self, target_idx: usize) -> Result<(), JournalError> {
        if let Some(ref hot) = self.hot {
            if hot.idx == target_idx {
                return Ok(());
            }
        }
        self.persist_hot_node().await?;
        let state = self.load_state(target_idx).await?;
        self.hot = Some(HotNode {
            idx: target_idx,
            state,
            turn_diff: FxHashMap::default(),
        });
        Ok(())
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

    fn deser_tree(ser: SerTreeMeta) -> Result<TreeMeta, JournalError> {
        match ser {
            SerTreeMeta::V1 { nodes: ser_nodes, tombstones: ser_tombstones } => {
                let tombstones: std::collections::HashSet<Uuid> = ser_tombstones
                    .iter()
                    .map(|s| Uuid::parse_str(s).map_err(|e| JournalError::CorruptedData(format!("invalid uuid: {e}"))))
                    .collect::<Result<_, _>>()?;

                let mut nodes = Vec::with_capacity(ser_nodes.len());
                let mut uuid_to_idx = FxHashMap::default();

                let uuids: Vec<Uuid> = ser_nodes
                    .iter()
                    .map(|n| Uuid::parse_str(&n.uuid).map_err(|e| JournalError::CorruptedData(format!("invalid uuid: {e}"))))
                    .collect::<Result<_, _>>()?;

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
                        let parent_uuid = Uuid::parse_str(parent_str)
                            .map_err(|e| JournalError::CorruptedData(format!("invalid uuid: {e}")))?;
                        if let Some(&pidx) = uuid_to_idx.get(&parent_uuid) {
                            nodes[i].parent = Some(pidx);
                            if uuid_to_idx.contains_key(&uuids[i]) {
                                nodes[pidx].children.push(i);
                            }
                        }
                    }
                }

                Ok(TreeMeta { nodes, uuid_to_idx, tombstones })
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

    async fn entry(&self, id: Uuid) -> Result<BlobEntryRef<'_, S>, JournalError> {
        let idx = self.tree.uuid_to_idx[&id];
        let state = self.load_state(idx).await?;
        Ok(BlobEntryRef {
            _journal: self,
            state,
            depth: self.tree.nodes[idx].depth,
            uuid: self.tree.nodes[idx].uuid,
        })
    }

    async fn entry_mut(&mut self, id: Uuid) -> Result<BlobEntryMut<'_, S>, JournalError> {
        let idx = self.tree.uuid_to_idx[&id];
        self.ensure_hot(idx).await?;
        Ok(BlobEntryMut { journal: self, idx })
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
    state: FxHashMap<String, TypedValue>,
    depth: usize,
    uuid: Uuid,
}

impl<'a, S: BlobStore> BlobEntryRef<'a, S> {
    /// All key-value pairs visible at this node.
    pub fn entries(&self) -> FxHashMap<String, TypedValue> {
        self.state.clone()
    }
}

impl<'a, S: BlobStore> EntryRef<'a> for BlobEntryRef<'a, S> {
    fn get(&self, key: &str) -> Option<TypedValue> {
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

    fn get(&self, key: &str) -> Option<TypedValue> {
        let hot = self.journal.hot.as_ref().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        hot.state.get(key).cloned()
    }

    fn apply(&mut self, key: &str, diff: StorageOps) {
        let hot = self.journal.hot.as_mut().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        debug_assert!(
            self.journal.tree.nodes[self.idx].children.is_empty(),
            "apply on non-leaf"
        );
        apply_diff(&mut hot.state, &mut hot.turn_diff, key, diff);
    }

    async fn next(self) -> Result<Self, JournalError> {
        let journal = self.journal;
        let idx = self.idx;

        // Persist current node.
        journal.persist_hot_node().await?;

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

        Ok(BlobEntryMut {
            journal,
            idx: new_idx,
        })
    }

    async fn fork(self) -> Result<Self, JournalError> {
        let journal = self.journal;
        let idx = self.idx;
        let parent_idx = journal.tree.nodes[idx]
            .parent
            .expect("cannot fork root");

        // Persist current hot node before switching.
        journal.persist_hot_node().await?;

        // Load parent's state (parent was a leaf → has snapshot → O(1)).
        let parent_state = journal.load_state(parent_idx).await?;

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

        Ok(BlobEntryMut {
            journal,
            idx: new_idx,
        })
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
    use acvus_interpreter::{LazyValue, PureValue, TypedValue};
    use acvus_mir::ty::Ty;
    use acvus_utils::{OwnedDequeDiff, TrackedDeque};

    use super::*;
    use crate::blob::MemBlobStore;
    use crate::storage::PatchDiff;

    /// Tests use interval=1 to match TreeJournal behavior (snapshot every turn).
    async fn new_journal() -> (BlobStoreJournal<MemBlobStore>, Uuid) {
        BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), Interner::new(), 1).await.unwrap()
    }

    // ── Basic get/apply ──

    #[tokio::test]
    async fn apply_and_get() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::string("hello")), ty: Ty::error() });
        assert!(matches!(
            e.get("x").unwrap().value(),
            Value::Pure(PureValue::String(v)) if v == "hello"
        ));
        assert!(e.get("y").is_none());
    }

    #[tokio::test]
    async fn overwrite() {
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::new(MemBlobStore::new(), interner.clone()).await.unwrap();
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
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(2)]);
        let diff = acvus_utils::OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![TypedValue::int(1), TypedValue::int(2)],
        };
        e.apply(
            "q",
            StorageOps::Sequence {
                squashed: squashed.clone(),
                diff,
                ty: Ty::error(),
            },
        );
        let val = e.get("q").unwrap();
        let Value::Lazy(LazyValue::Deque(d)) = val.value() else {
            panic!("expected Deque");
        };
        let expected: Vec<Value> = squashed.into_vec().into_iter().map(|tv| tv.value().clone()).collect();
        assert_eq!(d.as_slice(), &expected[..]);
    }

    #[tokio::test]
    async fn object_diff_updates_and_removals() {
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::new(MemBlobStore::new(), interner.clone()).await.unwrap();
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

    // ── Tree structure ──

    #[tokio::test]
    async fn depth() {
        let (mut j, root) = new_journal().await;
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
        let (mut j, root) = new_journal().await;

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
            // Child sees parent's values.
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            assert!(matches!(e.get("y").unwrap().value(), Value::Pure(PureValue::Int(2))));
            // Modifying child doesn't affect parent.
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(99)), ty: Ty::error() });
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        }

        // Parent unchanged.
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        }
        // Child has override.
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        }
    }

    #[tokio::test]
    async fn fork_creates_sibling() {
        let (mut j, root) = new_journal().await;

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

        // Fork from n2 → sibling of n2 (child of n1).
        {
            let e = j.entry_mut(n2).await.unwrap().fork().await.unwrap();
            assert_eq!(e.depth(), 2);
            // Sees n1's state (x=1), not n2's (x=2).
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        }
    }

    #[tokio::test]
    async fn prune_leaf() {
        let (mut j, root) = new_journal().await;

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

        // n1 still accessible.
        {
            let e = j.entry(n1).await.unwrap();
            assert_eq!(e.depth(), 1);
        }
    }

    #[tokio::test]
    async fn prune_subtree() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
        }
        {
            let _e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
        }

        j.entry_mut(n1).await.unwrap().prune(Prune::Subtree);

        // Root still accessible.
        {
            let e = j.entry(root).await.unwrap();
            assert_eq!(e.depth(), 0);
        }
    }

    // ── COW / isolation ──

    #[tokio::test]
    async fn cow_sharing() {
        let (mut j, root) = new_journal().await;

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

        // Both children see parent's data.
        assert!(matches!(j.entry(n2).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
        assert!(matches!(j.entry(n3).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));

        // Modifying one child doesn't affect the other.
        {
            let mut e = j.entry_mut(n2).await.unwrap();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(99)), ty: Ty::error() });
        }
        assert!(matches!(j.entry(n2).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(99))));
        assert!(matches!(j.entry(n3).await.unwrap().get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
    }

    #[tokio::test]
    async fn get_prefers_turn_diff() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            // x=1 from parent.
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            // Override.
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(2))));
        }
    }

    // ── BlobStore persistence ──

    #[tokio::test]
    async fn snapshot_persisted_after_next() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(42)), ty: Ty::error() });
        }

        // next() persists n1's snapshot.
        let _n2;
        {
            let e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.apply("y", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
        }

        // Switch back to n1 — triggers hot swap.
        // n2's state should be persisted, n1's state loaded.
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            assert!(e.get("y").is_none()); // y was added on n2, not n1
        }

        // n2 still has its data.
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(e.get("y").unwrap().value(), Value::Pure(PureValue::Int(2))));
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1)))); // inherited
        }
    }

    #[tokio::test]
    async fn prune_does_not_remove_shared_blobs() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // n1 has empty state — same snapshot blob as root.
        }

        // Persist n1 by creating child.
        let n2;
        {
            let e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
        }

        // root and n1 share the same snapshot blob (empty state).
        let root_snap = j.tree.nodes[j.tree.uuid_to_idx[&root]].snapshot_hash.unwrap();
        let n1_snap = j.tree.nodes[j.tree.uuid_to_idx[&n1]].snapshot_hash.unwrap();
        assert_eq!(root_snap, n1_snap); // content-addressed dedup

        // Prune n2 then n1 — blobs NOT removed (shared with root).
        j.entry_mut(n2).await.unwrap().prune(Prune::Leaf);
        j.entry_mut(n1).await.unwrap().prune(Prune::Leaf);

        // Root's snapshot blob still accessible.
        assert!(j.store.get(&root_snap).await.is_some());
    }

    #[tokio::test]
    async fn gc_removes_unreferenced_blobs() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(42)), ty: Ty::error() });
        }
        let n2;
        {
            let e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
        }

        // n1 has a unique snapshot (contains x=42).
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap = j.tree.nodes[n1_idx].snapshot_hash.unwrap();
        assert!(j.store.get(&n1_snap).await.is_some());

        // Prune n2, then n1.
        j.entry_mut(n2).await.unwrap().prune(Prune::Leaf);
        j.entry_mut(n1).await.unwrap().prune(Prune::Leaf);

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
                BlobStoreJournal::new(MemBlobStore::new(), interner.clone()).await.unwrap();

            {
                let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
                n1 = e.uuid();
                e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(10)), ty: Ty::error() });
            }
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                n2 = e.uuid();
                e.apply("y", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(20)), ty: Ty::error() });
            }

            j.flush_tree().await.unwrap();
            store = j.store;
        }

        // Re-open from the same blob store.
        let j2 = BlobStoreJournal::open(store, interner).await.expect("should open").unwrap();

        // Verify data is intact.
        {
            let e = j2.entry(n1).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(10))));
        }
        {
            let e = j2.entry(n2).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(10))));
            assert!(matches!(e.get("y").unwrap().value(), Value::Pure(PureValue::Int(20))));
        }
    }

    // ── Deep chain ──

    #[tokio::test]
    async fn deep_chain() {
        let (mut j, root) = new_journal().await;
        let mut cursor = root;
        for i in 0..50 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(i as i64)), ty: Ty::error() },
            );
        }

        // Leaf should see all ancestors' values.
        let e = j.entry(cursor).await.unwrap();
        assert_eq!(e.depth(), 50);
        for i in 0..50 {
            assert!(matches!(
                e.get(&format!("k{i}")).unwrap().value(),
                Value::Pure(PureValue::Int(v)) if *v == i as i64
            ));
        }
    }

    // ── as_ref ──

    #[tokio::test]
    async fn as_ref_returns_current_state() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(7)), ty: Ty::error() });
        let r = e.as_ref();
        assert!(matches!(r.get("x").unwrap().value(), Value::Pure(PureValue::Int(7))));
        assert_eq!(r.depth(), 1);
    }

    // ── entries() ──

    #[tokio::test]
    async fn entries_returns_all_keys() {
        let (mut j, root) = new_journal().await;
        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("a", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
            e.apply("b", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
        }

        let entries = j.entry(n1).await.unwrap().entries();
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
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("x", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.apply("y", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(2)), ty: Ty::error() });
        }

        // Switch to n1 to mutate.
        // This persists n2, loads n1.
        // n1 already has children so apply would fail (non-leaf assert).
        // But we can read.
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(e.get("x").unwrap().value(), Value::Pure(PureValue::Int(1))));
            assert!(e.get("y").is_none());
        }

        // n2 data still intact.
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(e.get("y").unwrap().value(), Value::Pure(PureValue::Int(2))));
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
        ).await.unwrap();

        let mut cursor = root;
        let mut uuids = vec![root];
        for i in 1..=8 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            uuids.push(cursor);
            e.apply(
                &format!("k{i}"),
                StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(i as i64)), ty: Ty::error() },
            );
        }
        // Force persist of last node.
        {
            let e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
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
        ).await.unwrap();

        let mut cursor = root;
        for i in 1..=7 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(i as i64)), ty: Ty::error() },
            );
        }

        // depth 7: no snapshot, must reconstruct from depth 4 + diffs at 5,6,7.
        let e = j.entry(cursor).await.unwrap();
        assert_eq!(e.depth(), 7);
        for i in 1..=7 {
            assert!(matches!(
                e.get(&format!("k{i}")).unwrap().value(),
                Value::Pure(PureValue::Int(v)) if *v == i as i64
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
        ).await.unwrap();

        let mut cursor = root;
        for i in 1..=3 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(i as i64)), ty: Ty::error() },
            );
        }

        // Fork from depth 3 → sibling at depth 3, parent at depth 2 (no snapshot).
        // Must reconstruct parent from root snapshot + diffs at depth 1, 2.
        let forked = j.entry_mut(cursor).await.unwrap().fork().await.unwrap();
        assert_eq!(forked.depth(), 3);
        // Sees state up to depth 2 (parent), not depth 3 changes.
        assert!(matches!(forked.get("k1").unwrap().value(), Value::Pure(PureValue::Int(1))));
        assert!(matches!(forked.get("k2").unwrap().value(), Value::Pure(PureValue::Int(2))));
        assert!(forked.get("k3").is_none());
    }

    #[tokio::test]
    async fn interval_deep_chain_no_data_loss() {
        // Interval=128 (default), 200 turns.
        let (mut j, root) = BlobStoreJournal::new(MemBlobStore::new(), Interner::new()).await.unwrap();
        let mut cursor = root;
        for i in 1..=200 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            e.apply(
                &format!("k{i}"),
                StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(i as i64)), ty: Ty::error() },
            );
        }

        // Leaf at depth 200 sees all values.
        let e = j.entry(cursor).await.unwrap();
        assert_eq!(e.depth(), 200);
        for i in 1..=200 {
            assert!(
                matches!(
                    e.get(&format!("k{i}")).unwrap().value(),
                    Value::Pure(PureValue::Int(v)) if *v == i as i64
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
            ).await.unwrap();
            let mut cursor = root;
            for i in 1..=8 {
                let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
                cursor = e.uuid();
                // Each turn adds a distinct key — snapshot grows, diff stays small.
                e.apply(
                    &format!("k{i}"),
                    StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(i as i64)), ty: Ty::error() },
                );
            }
            j.persist_hot_node().await.unwrap();
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

    /// Extract sequence items from a value (handles both Deque and List,
    /// since BlobStore serialization may convert Deque → List).
    fn get_seq_items(val: &TypedValue) -> Vec<Value> {
        match val.value() {
            Value::Lazy(LazyValue::Deque(d)) => d.as_slice().to_vec(),
            Value::Lazy(LazyValue::List(items)) => items.clone(),
            other => panic!("expected Deque or List, got {other:?}"),
        }
    }

    // =========================================================================
    // Sequence — diff-based history preservation via BlobStore
    // =========================================================================

    #[tokio::test]
    async fn sequence_history_preserved_across_persist() {
        // Sequence 값이 persist 후에도 각 턴의 상태가 정확히 복원되는지 검증.
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // Turn 1: seq = [1, 2]
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(2)]);
            let diff = OwnedDequeDiff { consumed: 0, removed_back: 0, pushed: vec![TypedValue::int(1), TypedValue::int(2)] };
            e.apply("seq", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // Turn 2: seq = [1, 2, 3] (appended 3)
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(2), TypedValue::int(3)]);
            let diff = OwnedDequeDiff { consumed: 0, removed_back: 0, pushed: vec![TypedValue::int(3)] };
            e.apply("seq", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
        }

        let n3;
        {
            let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
            // Turn 3: seq = [2, 3, 4] (consumed 1, appended 4)
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(2), TypedValue::int(3), TypedValue::int(4)]);
            let diff = OwnedDequeDiff { consumed: 1, removed_back: 0, pushed: vec![TypedValue::int(4)] };
            e.apply("seq", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
        }

        // Force persist by advancing
        { j.entry_mut(n3).await.unwrap().next().await.unwrap(); }

        // 각 시점의 값이 정확히 복원되는지 검증
        {
            let e = j.entry(n3).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(2), Value::int(3), Value::int(4)]);
        }
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(1), Value::int(2), Value::int(3)]);
        }
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(1), Value::int(2)]);
        }
    }

    #[tokio::test]
    async fn sequence_fork_independent_branches() {
        // Sequence가 fork 후 각 브랜치에서 독립적으로 동작하는지 검증.
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1)]);
            let diff = OwnedDequeDiff { consumed: 0, removed_back: 0, pushed: vec![TypedValue::int(1)] };
            e.apply("seq", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
        }

        // Branch A: append 100
        let ba;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            ba = e.uuid();
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(100)]);
            let diff = OwnedDequeDiff { consumed: 0, removed_back: 0, pushed: vec![TypedValue::int(100)] };
            e.apply("seq", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
        }

        // Branch B: append 200
        let bb;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            bb = e.uuid();
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(1), TypedValue::int(200)]);
            let diff = OwnedDequeDiff { consumed: 0, removed_back: 0, pushed: vec![TypedValue::int(200)] };
            e.apply("seq", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
        }

        // Force persist
        { j.entry_mut(ba).await.unwrap().next().await.unwrap(); }
        { j.entry_mut(bb).await.unwrap().next().await.unwrap(); }

        // 각 브랜치 독립
        {
            let e = j.entry(ba).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(1), Value::int(100)]);
        }
        {
            let e = j.entry(bb).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(1), Value::int(200)]);
        }
        // 부모 불변
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(1)]);
        }
    }

    // =========================================================================
    // Patch(Rec) — field-level history preservation via BlobStore
    // =========================================================================

    #[tokio::test]
    async fn patch_rec_history_preserved_across_persist() {
        // Patch(Rec)로 필드별 변경 후 각 턴 시점으로 정확히 복원 가능한지 검증.
        let interner = Interner::new();
        let (mut j, _root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(), interner.clone(), 1,
        ).await.unwrap();
        let root = _root;
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // Turn 1: {a: 1, b: 2}
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(1)), (b, Value::int(2))]));
            e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(obj, Ty::error())), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // Turn 2: a = 10 (b unchanged)
            e.apply("state", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(10)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }

        let n3;
        {
            let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
            // Turn 3: add c = 3, remove b
            e.apply("state", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(c, PatchDiff::Set(Value::int(3)))]),
                removals: vec![b],
            }, ty: Ty::error() });
        }

        // Force persist
        { j.entry_mut(n3).await.unwrap().next().await.unwrap(); }

        // n3: {a: 10, c: 3}
        {
            let e = j.entry(n3).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!("expected Object") };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(10))));
            assert_eq!(f.get(&b), None);
            assert_eq!(f.get(&c), Some(&Value::Pure(PureValue::Int(3))));
        }
        // n2: {a: 10, b: 2}
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!("expected Object") };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(10))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(2))));
            assert_eq!(f.get(&c), None);
        }
        // n1: {a: 1, b: 2}
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!("expected Object") };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(2))));
        }
    }

    #[tokio::test]
    async fn patch_rec_nested_history_preserved() {
        // 중첩 Rec 패치가 persist 후에도 올바르게 복원되는지 검증.
        let interner = Interner::new();
        let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(), interner.clone(), 1,
        ).await.unwrap();
        let x = interner.intern("x");
        let inner = interner.intern("inner");
        let y = interner.intern("y");

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // {x: 1, inner: {y: 10}}
            let obj = Value::object(FxHashMap::from_iter([
                (x, Value::int(1)),
                (inner, Value::object(FxHashMap::from_iter([(y, Value::int(10))]))),
            ]));
            e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(obj, Ty::error())), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // inner.y = 99 (nested Rec)
            e.apply("state", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(inner, PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(y, PatchDiff::Set(Value::int(99)))]),
                    removals: vec![],
                })]),
                removals: vec![],
            }, ty: Ty::error() });
        }

        // Force persist
        { j.entry_mut(n2).await.unwrap().next().await.unwrap(); }

        // n2: {x: 1, inner: {y: 99}}
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&x), Some(&Value::Pure(PureValue::Int(1))));
            let Value::Lazy(LazyValue::Object(inner_f)) = f.get(&inner).unwrap() else { panic!() };
            assert_eq!(inner_f.get(&y), Some(&Value::Pure(PureValue::Int(99))));
        }
        // n1: {x: 1, inner: {y: 10}}
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&x), Some(&Value::Pure(PureValue::Int(1))));
            let Value::Lazy(LazyValue::Object(inner_f)) = f.get(&inner).unwrap() else { panic!() };
            assert_eq!(inner_f.get(&y), Some(&Value::Pure(PureValue::Int(10))));
        }
    }

    #[tokio::test]
    async fn patch_set_replaces_any_value_type() {
        // PatchDiff::Set이 문자열, 숫자 등 모든 값 타입에서 동작하는지 검증.
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("val", StorageOps::Patch { diff: PatchDiff::set(TypedValue::string("hello")), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // 문자열 → 숫자로 타입 자체를 교체
            e.apply("val", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(42)), ty: Ty::error() });
        }

        // Force persist
        { j.entry_mut(n2).await.unwrap().next().await.unwrap(); }

        // n2: 42
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(e.get("val").unwrap().value(), Value::Pure(PureValue::Int(42))));
        }
        // n1: "hello"
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(e.get("val").unwrap().value(), Value::Pure(PureValue::String(s)) if s == "hello"));
        }
    }

    #[tokio::test]
    async fn patch_rec_fork_independent_branches() {
        // Patch(Rec) fork 후 각 브랜치에서 독립적으로 필드 변경되는지 검증.
        let interner = Interner::new();
        let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(), interner.clone(), 1,
        ).await.unwrap();
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

        // Force persist
        { j.entry_mut(ba).await.unwrap().next().await.unwrap(); }
        { j.entry_mut(bb).await.unwrap().next().await.unwrap(); }

        // ba: a=100, bb: a=200, n1: a=0
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

    #[tokio::test]
    async fn mixed_set_sequence_rec_coexist() {
        // Set(스냅샷), Sequence, Rec(패치)가 같은 journal에서 공존하며
        // 각각 독립적으로 히스토리를 보존하는지 검증.
        let interner = Interner::new();
        let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(), interner.clone(), 1,
        ).await.unwrap();
        let a = interner.intern("a");

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // counter: 숫자 (Set)
            e.apply("counter", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(0)), ty: Ty::error() });
            // log: 시퀀스
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(10)]);
            let diff = OwnedDequeDiff { consumed: 0, removed_back: 0, pushed: vec![TypedValue::int(10)] };
            e.apply("log", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
            // config: 오브젝트 (Rec)
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(100))]));
            e.apply("config", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(obj, Ty::error())), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // counter = 1
            e.apply("counter", StorageOps::Patch { diff: PatchDiff::set(TypedValue::int(1)), ty: Ty::error() });
            // log append 20
            let squashed = TrackedDeque::from_vec(vec![TypedValue::int(10), TypedValue::int(20)]);
            let diff = OwnedDequeDiff { consumed: 0, removed_back: 0, pushed: vec![TypedValue::int(20)] };
            e.apply("log", StorageOps::Sequence { squashed, diff, ty: Ty::error() });
            // config.a = 200
            e.apply("config", StorageOps::Patch { diff: PatchDiff::Rec {
                updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(200)))]),
                removals: vec![],
            }, ty: Ty::error() });
        }

        // Force persist
        { j.entry_mut(n2).await.unwrap().next().await.unwrap(); }

        // n2: counter=1, log=[10,20], config.a=200
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(e.get("counter").unwrap().value(), Value::Pure(PureValue::Int(1))));
            let val = e.get("log").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(10), Value::int(20)]);
            let val = e.get("config").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(200))));
        }

        // n1: counter=0, log=[10], config.a=100
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(e.get("counter").unwrap().value(), Value::Pure(PureValue::Int(0))));
            let val = e.get("log").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(10)]);
            let val = e.get("config").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(100))));
        }
    }

    #[tokio::test]
    async fn patch_compute_roundtrip_via_blobstore() {
        // PatchDiff::compute로 생성한 diff가 BlobStore를 통해 persist/restore 후에도
        // 올바르게 적용되는지 검증.
        let interner = Interner::new();
        let (mut j, root) = BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(), interner.clone(), 1,
        ).await.unwrap();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");

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

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.apply("state", StorageOps::Patch { diff: PatchDiff::set(TypedValue::new(old_val.clone(), Ty::error())), ty: Ty::error() });
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            let diff = PatchDiff::compute(&old_val, &new_val).expect("should have diff");
            e.apply("state", StorageOps::Patch { diff, ty: Ty::error() });
        }

        // Force persist
        { j.entry_mut(n2).await.unwrap().next().await.unwrap(); }

        // n2: {a: 1, b: 99} (c removed)
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(99))));
            assert_eq!(f.get(&c), None);
        }
        // n1: {a: 1, b: 2, c: 3}
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("state").unwrap();
            let Value::Lazy(LazyValue::Object(f)) = val.value() else { panic!() };
            assert_eq!(f.get(&a), Some(&Value::Pure(PureValue::Int(1))));
            assert_eq!(f.get(&b), Some(&Value::Pure(PureValue::Int(2))));
            assert_eq!(f.get(&c), Some(&Value::Pure(PureValue::Int(3))));
        }
    }

    #[tokio::test]
    async fn patch_set_non_object_compute_returns_set() {
        // non-object 값의 compute가 PatchDiff::Set을 반환하는지 검증.
        let old = Value::int(1);
        let new = Value::int(2);
        let diff = PatchDiff::compute(&old, &new);
        assert!(matches!(diff, Some(PatchDiff::Set(Value::Pure(PureValue::Int(2))))));

        // 동일한 값이면 None
        let same = PatchDiff::compute(&old, &old);
        assert!(same.is_none());

        // 문자열 → 숫자 크로스 타입
        let old_str = Value::string("hello".to_string());
        let new_int = Value::int(42);
        let cross = PatchDiff::compute(&old_str, &new_int);
        assert!(matches!(cross, Some(PatchDiff::Set(Value::Pure(PureValue::Int(42))))));
    }
}
