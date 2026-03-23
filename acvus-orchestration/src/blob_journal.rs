use acvus_interpreter::{ConcreteValue, LazyValue, SequenceChain, TypedValue, Value};
use acvus_utils::{Interner, OwnedDequeDiff, TrackedDeque};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use acvus_mir::ty::Ty;

use crate::blob::{BlobHash, BlobStore};
use crate::storage::{EntryMut, EntryRef, Journal, JournalError, PatchDiff, Prune};

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
    /// Patch diff blob hash (non-Sequence keys changed this turn).
    #[serde(rename = "f", skip_serializing_if = "Option::is_none")]
    diff_hash: Option<[u8; 32]>,
    /// Sequence diff blob hash (Sequence keys changed this turn).
    /// Stored separately from patch diffs so that Sequence keys never
    /// go through the ConcreteValue serialization path.
    #[serde(rename = "q", skip_serializing_if = "Option::is_none", default)]
    seq_diff_hash: Option<[u8; 32]>,
}

// ── Internal tree ───────────────────────────────────────────────────

struct NodeEntry {
    uuid: Uuid,
    parent: Option<usize>,
    children: Vec<usize>,
    depth: usize,
    snapshot_hash: Option<BlobHash>,
    diff_hash: Option<BlobHash>,
    seq_diff_hash: Option<BlobHash>,
}

struct TreeMeta {
    nodes: Vec<NodeEntry>,
    uuid_to_idx: FxHashMap<Uuid, usize>,
    /// Pruned UUIDs. Append-only for CRDT merge.
    tombstones: std::collections::HashSet<Uuid>,
}

struct HotNode {
    idx: usize,
    /// Full accumulated state at this node (all keys, all types).
    state: FxHashMap<String, TypedValue>,
    /// Patch keys changed during this node's turn.
    /// Stored as PatchDiff — only the changed fields, not the full post-apply value.
    turn_patches: FxHashMap<String, (PatchDiff, Ty)>,
    /// Sequence keys changed during this node's turn.
    /// Stored as diff only — serialized separately, never via ConcreteValue.
    turn_sequence_diffs: FxHashMap<String, OwnedDequeDiff<Value>>,
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
        serde_json::from_slice(bytes).map_err(|e| JournalError::Deserialization(e.to_string()))?;
    Ok(concrete
        .into_iter()
        .map(|(k, cv, ser_ty)| {
            let ty = ser_ty.to_ty(interner);
            (k, TypedValue::from_concrete(&cv, interner, ty))
        })
        .collect())
}

// ── Sequence diff serialization ─────────────────────────────────────
//
// Sequence diffs are stored separately from patch diffs.
// They NEVER go through ConcreteValue — only the pushed items are
// converted to ConcreteValue for individual element serialization.

#[derive(serde::Serialize, serde::Deserialize)]
struct SerSequenceDiff {
    #[serde(rename = "c")]
    consumed: usize,
    #[serde(rename = "r")]
    removed_back: usize,
    #[serde(rename = "p")]
    pushed: Vec<ConcreteValue>,
    /// Element type — needed to reconstruct TypedValue on load.
    #[serde(rename = "t")]
    elem_ty: acvus_mir::ser_ty::SerTy,
}

fn serialize_seq_diffs(
    diffs: &FxHashMap<String, OwnedDequeDiff<Value>>,
    interner: &Interner,
) -> Result<Vec<u8>, JournalError> {
    let ser: Vec<(String, SerSequenceDiff)> = diffs
        .iter()
        .map(|(k, diff)| {
            let pushed_concrete: Vec<ConcreteValue> = diff
                .pushed
                .iter()
                .map(|v| v.to_concrete(interner))
                .collect();
            (
                k.clone(),
                SerSequenceDiff {
                    consumed: diff.consumed,
                    removed_back: diff.removed_back,
                    pushed: pushed_concrete,
                    // Element type inferred from first pushed item, or Int as fallback.
                    // TODO: carry elem_ty through record_sequence_diff.
                    elem_ty: acvus_mir::ty::Ty::Int.to_ser(interner),
                },
            )
        })
        .collect();
    serde_json::to_vec(&ser).map_err(|e| JournalError::Serialization(e.to_string()))
}

fn deserialize_seq_diffs(
    bytes: &[u8],
    interner: &Interner,
) -> Result<FxHashMap<String, OwnedDequeDiff<Value>>, JournalError> {
    let ser: Vec<(String, SerSequenceDiff)> =
        serde_json::from_slice(bytes).map_err(|e| JournalError::Deserialization(e.to_string()))?;
    Ok(ser
        .into_iter()
        .map(|(k, sd)| {
            let pushed: Vec<Value> = sd
                .pushed
                .iter()
                .map(|cv| Value::from_concrete(cv, interner))
                .collect();
            (
                k,
                OwnedDequeDiff {
                    consumed: sd.consumed,
                    removed_back: sd.removed_back,
                    pushed,
                },
            )
        })
        .collect())
}

// ── Patch diff serialization ────────────────────────────────────────
//
// Patch diffs are stored as PatchDiff structures, not as post-apply values.
// This means PatchDiff::Rec stores only the changed keys, not the full object.
// PatchDiff::Set stores the full replacement value (unavoidable).

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "k")]
enum SerPatchDiff {
    #[serde(rename = "s")]
    Set { v: ConcreteValue },
    #[serde(rename = "r")]
    Rec {
        #[serde(rename = "u")]
        updates: Vec<(String, SerPatchDiff)>,
        #[serde(rename = "d", skip_serializing_if = "Vec::is_empty", default)]
        removals: Vec<String>,
    },
}

impl SerPatchDiff {
    fn from_patch_diff(diff: &PatchDiff, interner: &Interner) -> Self {
        match diff {
            PatchDiff::Set(v) => SerPatchDiff::Set {
                v: v.to_concrete(interner),
            },
            PatchDiff::Rec { updates, removals } => SerPatchDiff::Rec {
                updates: updates
                    .iter()
                    .map(|(k, d)| {
                        (
                            interner.resolve(*k).to_string(),
                            Self::from_patch_diff(d, interner),
                        )
                    })
                    .collect(),
                removals: removals
                    .iter()
                    .map(|k| interner.resolve(*k).to_string())
                    .collect(),
            },
        }
    }

    fn to_patch_diff(&self, interner: &Interner) -> PatchDiff {
        match self {
            SerPatchDiff::Set { v } => PatchDiff::Set(Value::from_concrete(v, interner)),
            SerPatchDiff::Rec { updates, removals } => PatchDiff::Rec {
                updates: updates
                    .iter()
                    .map(|(k, d)| (interner.intern(k), d.to_patch_diff(interner)))
                    .collect(),
                removals: removals.iter().map(|k| interner.intern(k)).collect(),
            },
        }
    }
}

fn serialize_patch_diffs(
    diffs: &FxHashMap<String, (PatchDiff, Ty)>,
    interner: &Interner,
) -> Result<Vec<u8>, JournalError> {
    let ser: Vec<(String, SerPatchDiff, acvus_mir::ser_ty::SerTy)> = diffs
        .iter()
        .map(|(k, (diff, ty))| {
            (
                k.clone(),
                SerPatchDiff::from_patch_diff(diff, interner),
                ty.to_ser(interner),
            )
        })
        .collect();
    serde_json::to_vec(&ser).map_err(|e| JournalError::Serialization(e.to_string()))
}

fn deserialize_patch_diffs(
    bytes: &[u8],
    interner: &Interner,
) -> Result<FxHashMap<String, (PatchDiff, Ty)>, JournalError> {
    let ser: Vec<(String, SerPatchDiff, acvus_mir::ser_ty::SerTy)> =
        serde_json::from_slice(bytes).map_err(|e| JournalError::Deserialization(e.to_string()))?;
    Ok(ser
        .into_iter()
        .map(|(k, sd, ser_ty)| {
            let ty = ser_ty.to_ty(interner);
            (k, (sd.to_patch_diff(interner), ty))
        })
        .collect())
}

// ── StoragePatch application ─────────────────────────────────────────

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
            seq_diff_hash: None,
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
        let ser: SerTreeMeta = serde_json::from_slice(&tree_bytes)
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
                    let remote_hash = actual.ok_or_else(|| {
                        JournalError::MissingBlob("tree ref disappeared during flush".into())
                    })?;
                    let remote_bytes = self
                        .store
                        .get(&remote_hash)
                        .await
                        .ok_or_else(|| JournalError::MissingBlob("remote tree blob".into()))?;
                    let remote_ser: SerTreeMeta = serde_json::from_slice(&remote_bytes)
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
            if let Some(h) = node.seq_diff_hash {
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
            // Patch diffs: apply PatchDiff to existing state.
            if let Some(diff_hash) = self.tree.nodes[node_idx].diff_hash {
                let diff_bytes = self
                    .store
                    .get(&diff_hash)
                    .await
                    .ok_or_else(|| JournalError::MissingBlob("diff".into()))?;
                let patch_diffs = deserialize_patch_diffs(&diff_bytes, &self.interner)?;
                for (k, (patch, ty)) in patch_diffs {
                    let existing = state.get(&k);
                    let new_val = patch.apply(existing, ty);
                    state.insert(k, new_val);
                }
            }
            // Sequence diffs: apply OwnedDequeDiff to existing sequence state.
            if let Some(seq_diff_hash) = self.tree.nodes[node_idx].seq_diff_hash {
                let seq_diff_bytes = self
                    .store
                    .get(&seq_diff_hash)
                    .await
                    .ok_or_else(|| JournalError::MissingBlob("seq_diff".into()))?;
                let seq_diffs = deserialize_seq_diffs(&seq_diff_bytes, &self.interner)?;
                for (k, diff) in seq_diffs {
                    let prev_items = state
                        .get(&k)
                        .map(|tv| {
                            let sc = tv
                                .value()
                                .expect_ref::<SequenceChain>("load_state seq_diff");
                            sc.origin().as_slice().to_vec()
                        })
                        .unwrap_or_default();
                    let new_items = diff.apply(prev_items);
                    let new_deque = TrackedDeque::from_vec(new_items);
                    let sc = SequenceChain::from_stored(new_deque);
                    // Preserve the existing type if available.
                    let ty = state
                        .get(&k)
                        .map(|tv| tv.ty().clone())
                        .unwrap_or_else(|| acvus_mir::ty::Ty::error());
                    state.insert(k, TypedValue::new(Value::sequence(sc), ty));
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
        let Some(ref hot) = self.hot else {
            return Ok(());
        };
        let idx = hot.idx;
        let depth = self.tree.nodes[idx].depth;

        // Store patch diffs if non-empty (PatchDiff, not post-apply values).
        if !hot.turn_patches.is_empty() {
            let diff_bytes = serialize_patch_diffs(&hot.turn_patches, &self.interner)?;
            let diff_hash = self.store.put(diff_bytes).await;
            self.tree.nodes[idx].diff_hash = Some(diff_hash);
        }

        // Store sequence diffs if non-empty (separate blob, separate path).
        if !hot.turn_sequence_diffs.is_empty() {
            let seq_diff_bytes = serialize_seq_diffs(&hot.turn_sequence_diffs, &self.interner)?;
            let seq_diff_hash = self.store.put(seq_diff_bytes).await;
            self.tree.nodes[idx].seq_diff_hash = Some(seq_diff_hash);
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
            turn_patches: FxHashMap::default(),
            turn_sequence_diffs: FxHashMap::default(),
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
                    seq_diff_hash: n.seq_diff_hash.map(|h| *h.as_bytes()),
                })
                .collect(),
            tombstones: self.tree.tombstones.iter().map(|u| u.to_string()).collect(),
        }
    }

    fn deser_tree(ser: SerTreeMeta) -> Result<TreeMeta, JournalError> {
        match ser {
            SerTreeMeta::V1 {
                nodes: ser_nodes,
                tombstones: ser_tombstones,
            } => {
                let tombstones: std::collections::HashSet<Uuid> = ser_tombstones
                    .iter()
                    .map(|s| {
                        Uuid::parse_str(s)
                            .map_err(|e| JournalError::CorruptedData(format!("invalid uuid: {e}")))
                    })
                    .collect::<Result<_, _>>()?;

                let mut nodes = Vec::with_capacity(ser_nodes.len());
                let mut uuid_to_idx = FxHashMap::default();

                let uuids: Vec<Uuid> = ser_nodes
                    .iter()
                    .map(|n| {
                        Uuid::parse_str(&n.uuid)
                            .map_err(|e| JournalError::CorruptedData(format!("invalid uuid: {e}")))
                    })
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
                        seq_diff_hash: sn.seq_diff_hash.map(BlobHash::from_bytes),
                    });
                }

                for (i, sn) in ser_nodes.iter().enumerate() {
                    if let Some(ref parent_str) = sn.parent {
                        let parent_uuid = Uuid::parse_str(parent_str).map_err(|e| {
                            JournalError::CorruptedData(format!("invalid uuid: {e}"))
                        })?;
                        if let Some(&pidx) = uuid_to_idx.get(&parent_uuid) {
                            nodes[i].parent = Some(pidx);
                            if uuid_to_idx.contains_key(&uuids[i]) {
                                nodes[pidx].children.push(i);
                            }
                        }
                    }
                }

                Ok(TreeMeta {
                    nodes,
                    uuid_to_idx,
                    tombstones,
                })
            }
        }
    }

    /// Merge two SerTreeMeta. Both must be the same version.
    fn merge_ser(a: SerTreeMeta, b: SerTreeMeta) -> SerTreeMeta {
        match (a, b) {
            (
                SerTreeMeta::V1 {
                    nodes: a_nodes,
                    tombstones: a_tombstones,
                },
                SerTreeMeta::V1 {
                    nodes: b_nodes,
                    tombstones: b_tombstones,
                },
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
    type Ref<'a>
        = BlobEntryRef<'a, S>
    where
        S: 'a;
    type Mut<'a>
        = BlobEntryMut<'a, S>
    where
        S: 'a;

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
        self.tree.nodes[idx]
            .parent
            .map(|pidx| self.tree.nodes[pidx].uuid)
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
    type Ref<'x>
        = BlobEntryRef<'x, S>
    where
        'a: 'x,
        Self: 'x;

    fn get(&self, key: &str) -> Option<TypedValue> {
        let hot = self.journal.hot.as_ref().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        hot.state.get(key).cloned()
    }

    fn record_patch(&mut self, key: &str, diff: PatchDiff, ty: Ty) {
        let hot = self.journal.hot.as_mut().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        debug_assert!(
            self.journal.tree.nodes[self.idx].children.is_empty(),
            "record_patch on non-leaf"
        );
        // Guard: Sequence keys must use record_sequence_diff.
        if let Some(existing) = hot.state.get(key) {
            assert!(
                !matches!(existing.value(), Value::Lazy(LazyValue::Sequence(_))),
                "record_patch called on Sequence key {key:?} — use record_sequence_diff instead"
            );
        }
        // Apply to state for in-memory reads.
        let existing = hot.state.get(key);
        let new_val = diff.clone().apply(existing, ty.clone());
        let key_owned = key.to_string();
        hot.state.insert(key_owned.clone(), new_val);
        // Store diff itself, not post-apply value.
        hot.turn_patches.insert(key_owned, (diff, ty));
    }

    fn record_sequence_diff(
        &mut self,
        key: &str,
        working: TrackedDeque<Value>,
        ty: Ty,
    ) -> TrackedDeque<Value> {
        let hot = self.journal.hot.as_mut().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        debug_assert!(
            self.journal.tree.nodes[self.idx].children.is_empty(),
            "record_sequence_diff on non-leaf"
        );

        // Guard: non-Sequence keys must use record_patch.
        let existing = hot.state.get(key);
        if let Some(tv) = existing {
            assert!(
                matches!(tv.value(), Value::Lazy(LazyValue::Sequence(_))),
                "record_sequence_diff called on non-Sequence key {key:?} — use record_patch instead"
            );
        }

        // First turn: no existing sequence.
        if existing.is_none() {
            let items = working.into_vec();
            let diff = OwnedDequeDiff {
                consumed: 0,
                removed_back: 0,
                pushed: items.clone(),
            };
            let new_deque = TrackedDeque::from_vec(items);
            let sc = SequenceChain::from_stored(new_deque.clone());
            let stored = TypedValue::new(Value::sequence(sc), ty);
            let key_owned = key.to_string();
            hot.state.insert(key_owned.clone(), stored);
            hot.turn_sequence_diffs.insert(key_owned, diff);
            return new_deque;
        }

        // Subsequent turn: into_diff verifies checksum, extracts diff.
        let existing_tv = existing.unwrap();
        let sc = existing_tv
            .value()
            .expect_ref::<SequenceChain>("record_sequence_diff");
        let origin = sc.origin();
        let (squashed, diff) = working.into_diff(origin);

        let new_sc = SequenceChain::from_stored(squashed.clone());
        let stored = TypedValue::new(Value::sequence(new_sc), ty);
        let key_owned = key.to_string();
        hot.state.insert(key_owned.clone(), stored);
        hot.turn_sequence_diffs.insert(key_owned, diff);
        squashed
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
            seq_diff_hash: None,
        });
        journal.tree.nodes[idx].children.push(new_idx);
        journal.tree.uuid_to_idx.insert(new_uuid, new_idx);

        // New child becomes hot.
        journal.hot = Some(HotNode {
            idx: new_idx,
            state: parent_state,
            turn_patches: FxHashMap::default(),
            turn_sequence_diffs: FxHashMap::default(),
        });

        Ok(BlobEntryMut {
            journal,
            idx: new_idx,
        })
    }

    async fn fork(self) -> Result<Self, JournalError> {
        let journal = self.journal;
        let idx = self.idx;
        let parent_idx = journal.tree.nodes[idx].parent.expect("cannot fork root");

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
            seq_diff_hash: None,
        });
        journal.tree.nodes[parent_idx].children.push(new_idx);
        journal.tree.uuid_to_idx.insert(new_uuid, new_idx);

        // Sibling becomes hot.
        journal.hot = Some(HotNode {
            idx: new_idx,
            state: parent_state,
            turn_patches: FxHashMap::default(),
            turn_sequence_diffs: FxHashMap::default(),
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
                journal.tree.nodes[idx].seq_diff_hash = None;
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
                    journal.tree.nodes[node_idx].seq_diff_hash = None;
                    journal.tree.nodes[node_idx].children.clear();
                }
            }
        }

        // Clear hot if pruned.
        if let Some(ref hot) = journal.hot {
            if !journal
                .tree
                .uuid_to_idx
                .contains_key(&journal.tree.nodes[hot.idx].uuid)
            {
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
    use acvus_interpreter::{LazyValue, PureValue, SequenceChain, TypedValue};
    use acvus_mir::ty::{Effect, Origin, Ty};
    use acvus_utils::{Astr, TrackedDeque};

    use super::*;
    use crate::blob::MemBlobStore;
    use crate::storage::PatchDiff;

    fn seq_ty() -> Ty {
        Ty::Sequence(Box::new(Ty::Int), Origin::Concrete(0), Effect::Pure)
    }

    /// Tests use interval=1 to match TreeJournal behavior (snapshot every turn).
    async fn new_journal() -> (BlobStoreJournal<MemBlobStore>, Uuid) {
        BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), Interner::new(), 1)
            .await
            .unwrap()
    }

    // ── Basic get/apply ──

    #[tokio::test]
    async fn apply_and_get() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.record_patch(
            "x",
            PatchDiff::set(TypedValue::string("hello")),
            Ty::error(),
        );
        assert!(matches!(
            e.get("x").unwrap().value(),
            Value::Pure(PureValue::String(v)) if v == "hello"
        ));
        assert!(e.get("y").is_none());
    }

    #[tokio::test]
    async fn overwrite() {
        let interner = Interner::new();
        let (mut j, root) = BlobStoreJournal::new(MemBlobStore::new(), interner.clone())
            .await
            .unwrap();
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.record_patch(
            "x",
            PatchDiff::set(TypedValue::string("first")),
            Ty::error(),
        );
        e.record_patch(
            "x",
            PatchDiff::set(TypedValue::new(
                Value::object(FxHashMap::from_iter([(
                    interner.intern("v"),
                    Value::int(2),
                )])),
                Ty::error(),
            )),
            Ty::error(),
        );
        assert!(matches!(
            e.get("x").unwrap().value(),
            Value::Lazy(LazyValue::Object(_))
        ));
    }

    #[tokio::test]
    async fn sequence_stores_as_sequence() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let deque = TrackedDeque::from_vec(vec![Value::int(1), Value::int(2)]);
        e.record_sequence_diff("q", deque, seq_ty());
        let val = e.get("q").unwrap();
        let sc = val
            .value()
            .expect_ref::<SequenceChain>("sequence_stores_as_sequence");
        let expected = vec![Value::int(1), Value::int(2)];
        assert_eq!(sc.origin().as_slice(), &expected[..]);
    }

    #[tokio::test]
    async fn object_diff_updates_and_removals() {
        let interner = Interner::new();
        let (mut j, root) = BlobStoreJournal::new(MemBlobStore::new(), interner.clone())
            .await
            .unwrap();
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");
        e.record_patch(
            "obj",
            PatchDiff::set(TypedValue::new(
                Value::object(FxHashMap::from_iter([
                    (a, Value::int(1)),
                    (b, Value::int(2)),
                ])),
                Ty::error(),
            )),
            Ty::error(),
        );
        let diff = PatchDiff::Rec {
            updates: FxHashMap::from_iter([
                (a, PatchDiff::Set(Value::int(100))),
                (c, PatchDiff::Set(Value::int(3))),
            ]),
            removals: vec![b],
        };
        e.record_patch("obj", diff, Ty::error());
        let val = e.get("obj").unwrap();
        let fields = val
            .value()
            .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
        assert_eq!(fields.get(&a), Some(&Value::int(100)));
        assert_eq!(fields.get(&b), None);
        assert_eq!(fields.get(&c), Some(&Value::int(3)));
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
            e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
            e.record_patch("y", PatchDiff::set(TypedValue::int(2)), Ty::error());
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // Child sees parent's values.
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(1))
            ));
            assert!(matches!(
                e.get("y").unwrap().value(),
                Value::Pure(PureValue::Int(2))
            ));
            // Modifying child doesn't affect parent.
            e.record_patch("x", PatchDiff::set(TypedValue::int(99)), Ty::error());
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(99))
            ));
        }

        // Parent unchanged.
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(1))
            ));
        }
        // Child has override.
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(99))
            ));
        }
    }

    #[tokio::test]
    async fn fork_creates_sibling() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.record_patch("x", PatchDiff::set(TypedValue::int(2)), Ty::error());
        }

        // Fork from n2 → sibling of n2 (child of n1).
        {
            let e = j.entry_mut(n2).await.unwrap().fork().await.unwrap();
            assert_eq!(e.depth(), 2);
            // Sees n1's state (x=1), not n2's (x=2).
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(1))
            ));
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
            e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
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
            e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
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
        assert!(matches!(
            j.entry(n2).await.unwrap().get("x").unwrap().value(),
            Value::Pure(PureValue::Int(1))
        ));
        assert!(matches!(
            j.entry(n3).await.unwrap().get("x").unwrap().value(),
            Value::Pure(PureValue::Int(1))
        ));

        // Modifying one child doesn't affect the other.
        {
            let mut e = j.entry_mut(n2).await.unwrap();
            e.record_patch("x", PatchDiff::set(TypedValue::int(99)), Ty::error());
        }
        assert!(matches!(
            j.entry(n2).await.unwrap().get("x").unwrap().value(),
            Value::Pure(PureValue::Int(99))
        ));
        assert!(matches!(
            j.entry(n3).await.unwrap().get("x").unwrap().value(),
            Value::Pure(PureValue::Int(1))
        ));
    }

    #[tokio::test]
    async fn get_prefers_turn_diff() {
        let (mut j, root) = new_journal().await;

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
        }
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            // x=1 from parent.
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(1))
            ));
            // Override.
            e.record_patch("x", PatchDiff::set(TypedValue::int(2)), Ty::error());
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(2))
            ));
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
            e.record_patch("x", PatchDiff::set(TypedValue::int(42)), Ty::error());
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
            e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.record_patch("y", PatchDiff::set(TypedValue::int(2)), Ty::error());
        }

        // Switch back to n1 — triggers hot swap.
        // n2's state should be persisted, n1's state loaded.
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(1))
            ));
            assert!(e.get("y").is_none()); // y was added on n2, not n1
        }

        // n2 still has its data.
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(
                e.get("y").unwrap().value(),
                Value::Pure(PureValue::Int(2))
            ));
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(1))
            )); // inherited
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
        let root_snap = j.tree.nodes[j.tree.uuid_to_idx[&root]]
            .snapshot_hash
            .unwrap();
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
            e.record_patch("x", PatchDiff::set(TypedValue::int(42)), Ty::error());
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
        let root_snap = j.tree.nodes[j.tree.uuid_to_idx[&root]]
            .snapshot_hash
            .unwrap();
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
            let (mut j, root) = BlobStoreJournal::new(MemBlobStore::new(), interner.clone())
                .await
                .unwrap();

            {
                let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
                n1 = e.uuid();
                e.record_patch("x", PatchDiff::set(TypedValue::int(10)), Ty::error());
            }
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                n2 = e.uuid();
                e.record_patch("y", PatchDiff::set(TypedValue::int(20)), Ty::error());
            }

            j.flush_tree().await.unwrap();
            store = j.store;
        }

        // Re-open from the same blob store.
        let j2 = BlobStoreJournal::open(store, interner)
            .await
            .expect("should open")
            .unwrap();

        // Verify data is intact.
        {
            let e = j2.entry(n1).await.unwrap();
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(10))
            ));
        }
        {
            let e = j2.entry(n2).await.unwrap();
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(10))
            ));
            assert!(matches!(
                e.get("y").unwrap().value(),
                Value::Pure(PureValue::Int(20))
            ));
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
            e.record_patch(
                &format!("k{i}"),
                PatchDiff::set(TypedValue::int(i as i64)),
                Ty::error(),
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
        e.record_patch("x", PatchDiff::set(TypedValue::int(7)), Ty::error());
        let r = e.as_ref();
        assert!(matches!(
            r.get("x").unwrap().value(),
            Value::Pure(PureValue::Int(7))
        ));
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
            e.record_patch("a", PatchDiff::set(TypedValue::int(1)), Ty::error());
            e.record_patch("b", PatchDiff::set(TypedValue::int(2)), Ty::error());
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
            e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
        }
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.record_patch("y", PatchDiff::set(TypedValue::int(2)), Ty::error());
        }

        // Switch to n1 to mutate.
        // This persists n2, loads n1.
        // n1 already has children so apply would fail (non-leaf assert).
        // But we can read.
        {
            let e = j.entry(n1).await.unwrap();
            assert!(matches!(
                e.get("x").unwrap().value(),
                Value::Pure(PureValue::Int(1))
            ));
            assert!(e.get("y").is_none());
        }

        // n2 data still intact.
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(
                e.get("y").unwrap().value(),
                Value::Pure(PureValue::Int(2))
            ));
        }
    }

    // ── Snapshot interval ──

    #[tokio::test]
    async fn interval_only_snapshots_at_boundaries() {
        // Interval=4: snapshots at depth 0, 4, 8, ...
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), Interner::new(), 4)
                .await
                .unwrap();

        let mut cursor = root;
        let mut uuids = vec![root];
        for i in 1..=8 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            uuids.push(cursor);
            e.record_patch(
                &format!("k{i}"),
                PatchDiff::set(TypedValue::int(i as i64)),
                Ty::error(),
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
            if idx.is_none() {
                continue;
            }
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
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), Interner::new(), 4)
                .await
                .unwrap();

        let mut cursor = root;
        for i in 1..=7 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            e.record_patch(
                &format!("k{i}"),
                PatchDiff::set(TypedValue::int(i as i64)),
                Ty::error(),
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
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), Interner::new(), 4)
                .await
                .unwrap();

        let mut cursor = root;
        for i in 1..=3 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            e.record_patch(
                &format!("k{i}"),
                PatchDiff::set(TypedValue::int(i as i64)),
                Ty::error(),
            );
        }

        // Fork from depth 3 → sibling at depth 3, parent at depth 2 (no snapshot).
        // Must reconstruct parent from root snapshot + diffs at depth 1, 2.
        let forked = j.entry_mut(cursor).await.unwrap().fork().await.unwrap();
        assert_eq!(forked.depth(), 3);
        // Sees state up to depth 2 (parent), not depth 3 changes.
        assert!(matches!(
            forked.get("k1").unwrap().value(),
            Value::Pure(PureValue::Int(1))
        ));
        assert!(matches!(
            forked.get("k2").unwrap().value(),
            Value::Pure(PureValue::Int(2))
        ));
        assert!(forked.get("k3").is_none());
    }

    #[tokio::test]
    async fn interval_deep_chain_no_data_loss() {
        // Interval=128 (default), 200 turns.
        let (mut j, root) = BlobStoreJournal::new(MemBlobStore::new(), Interner::new())
            .await
            .unwrap();
        let mut cursor = root;
        for i in 1..=200 {
            let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            e.record_patch(
                &format!("k{i}"),
                PatchDiff::set(TypedValue::int(i as i64)),
                Ty::error(),
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
        let snap_count = j
            .tree
            .nodes
            .iter()
            .filter(|n| j.tree.uuid_to_idx.contains_key(&n.uuid))
            .filter(|n| n.snapshot_hash.is_some())
            .count();
        assert_eq!(
            snap_count, 2,
            "only root and depth-128 should have snapshots"
        );
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
            )
            .await
            .unwrap();
            let mut cursor = root;
            for i in 1..=8 {
                let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
                cursor = e.uuid();
                // Each turn adds a distinct key — snapshot grows, diff stays small.
                e.record_patch(
                    &format!("k{i}"),
                    PatchDiff::set(TypedValue::int(i as i64)),
                    Ty::error(),
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

    fn get_seq_items(val: &TypedValue) -> Vec<Value> {
        let sc = val.value().expect_ref::<SequenceChain>("get_seq_items");
        sc.origin().as_slice().to_vec()
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
            e.record_sequence_diff(
                "seq",
                TrackedDeque::from_vec(vec![Value::int(1), Value::int(2)]),
                seq_ty(),
            );
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // Turn 2: seq = [1, 2, 3] (appended 3)
            let mut working = e
                .get("seq")
                .unwrap()
                .value()
                .expect_ref::<SequenceChain>("n2 seq")
                .origin()
                .clone();
            working.checkpoint();
            working.push(Value::int(3));
            e.record_sequence_diff("seq", working, seq_ty());
        }

        let n3;
        {
            let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
            // Turn 3: seq = [2, 3, 4] (consumed 1, appended 4)
            let mut working = e
                .get("seq")
                .unwrap()
                .value()
                .expect_ref::<SequenceChain>("n3 seq")
                .origin()
                .clone();
            working.checkpoint();
            working.consume(1);
            working.push(Value::int(4));
            e.record_sequence_diff("seq", working, seq_ty());
        }

        // Force persist by advancing
        {
            j.entry_mut(n3).await.unwrap().next().await.unwrap();
        }

        // 각 시점의 값이 정확히 복원되는지 검증
        {
            let e = j.entry(n3).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(
                get_seq_items(&val),
                vec![Value::int(2), Value::int(3), Value::int(4)]
            );
        }
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("seq").unwrap();
            assert_eq!(
                get_seq_items(&val),
                vec![Value::int(1), Value::int(2), Value::int(3)]
            );
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
            e.record_sequence_diff("seq", TrackedDeque::from_vec(vec![Value::int(1)]), seq_ty());
        }

        // Branch A: append 100
        let ba;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            ba = e.uuid();
            let mut working = e
                .get("seq")
                .unwrap()
                .value()
                .expect_ref::<SequenceChain>("ba seq")
                .origin()
                .clone();
            working.checkpoint();
            working.push(Value::int(100));
            e.record_sequence_diff("seq", working, seq_ty());
        }

        // Branch B: append 200
        let bb;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            bb = e.uuid();
            let mut working = e
                .get("seq")
                .unwrap()
                .value()
                .expect_ref::<SequenceChain>("bb seq")
                .origin()
                .clone();
            working.checkpoint();
            working.push(Value::int(200));
            e.record_sequence_diff("seq", working, seq_ty());
        }

        // Force persist
        {
            j.entry_mut(ba).await.unwrap().next().await.unwrap();
        }
        {
            j.entry_mut(bb).await.unwrap().next().await.unwrap();
        }

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
        let (mut j, _root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), interner.clone(), 1)
                .await
                .unwrap();
        let root = _root;
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // Turn 1: {a: 1, b: 2}
            let obj = Value::object(FxHashMap::from_iter([
                (a, Value::int(1)),
                (b, Value::int(2)),
            ]));
            e.record_patch(
                "state",
                PatchDiff::set(TypedValue::new(obj, Ty::error())),
                Ty::error(),
            );
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // Turn 2: a = 10 (b unchanged)
            e.record_patch(
                "state",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(10)))]),
                    removals: vec![],
                },
                Ty::error(),
            );
        }

        let n3;
        {
            let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
            n3 = e.uuid();
            // Turn 3: add c = 3, remove b
            e.record_patch(
                "state",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(c, PatchDiff::Set(Value::int(3)))]),
                    removals: vec![b],
                },
                Ty::error(),
            );
        }

        // Force persist
        {
            j.entry_mut(n3).await.unwrap().next().await.unwrap();
        }

        // n3: {a: 10, c: 3}
        {
            let e = j.entry(n3).await.unwrap();
            let val = e.get("state").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(10)));
            assert_eq!(f.get(&b), None);
            assert_eq!(f.get(&c), Some(&Value::int(3)));
        }
        // n2: {a: 10, b: 2}
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(10)));
            assert_eq!(f.get(&b), Some(&Value::int(2)));
            assert_eq!(f.get(&c), None);
        }
        // n1: {a: 1, b: 2}
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("state").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(1)));
            assert_eq!(f.get(&b), Some(&Value::int(2)));
        }
    }

    #[tokio::test]
    async fn patch_rec_nested_history_preserved() {
        // 중첩 Rec 패치가 persist 후에도 올바르게 복원되는지 검증.
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), interner.clone(), 1)
                .await
                .unwrap();
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
                (
                    inner,
                    Value::object(FxHashMap::from_iter([(y, Value::int(10))])),
                ),
            ]));
            e.record_patch(
                "state",
                PatchDiff::set(TypedValue::new(obj, Ty::error())),
                Ty::error(),
            );
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // inner.y = 99 (nested Rec)
            e.record_patch(
                "state",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(
                        inner,
                        PatchDiff::Rec {
                            updates: FxHashMap::from_iter([(y, PatchDiff::Set(Value::int(99)))]),
                            removals: vec![],
                        },
                    )]),
                    removals: vec![],
                },
                Ty::error(),
            );
        }

        // Force persist
        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        // n2: {x: 1, inner: {y: 99}}
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&x), Some(&Value::int(1)));
            let inner_f = f
                .get(&inner)
                .unwrap()
                .expect_ref::<FxHashMap<Astr, Value>>("expected inner Object");
            assert_eq!(inner_f.get(&y), Some(&Value::int(99)));
        }
        // n1: {x: 1, inner: {y: 10}}
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("state").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&x), Some(&Value::int(1)));
            let inner_f = f
                .get(&inner)
                .unwrap()
                .expect_ref::<FxHashMap<Astr, Value>>("expected inner Object");
            assert_eq!(inner_f.get(&y), Some(&Value::int(10)));
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
            e.record_patch(
                "val",
                PatchDiff::set(TypedValue::string("hello")),
                Ty::error(),
            );
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // 문자열 → 숫자로 타입 자체를 교체
            e.record_patch("val", PatchDiff::set(TypedValue::int(42)), Ty::error());
        }

        // Force persist
        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        // n2: 42
        {
            let e = j.entry(n2).await.unwrap();
            assert!(matches!(
                e.get("val").unwrap().value(),
                Value::Pure(PureValue::Int(42))
            ));
        }
        // n1: "hello"
        {
            let e = j.entry(n1).await.unwrap();
            assert!(
                matches!(e.get("val").unwrap().value(), Value::Pure(PureValue::String(s)) if s == "hello")
            );
        }
    }

    #[tokio::test]
    async fn patch_rec_fork_independent_branches() {
        // Patch(Rec) fork 후 각 브랜치에서 독립적으로 필드 변경되는지 검증.
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), interner.clone(), 1)
                .await
                .unwrap();
        let a = interner.intern("a");

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(0))]));
            e.record_patch(
                "s",
                PatchDiff::set(TypedValue::new(obj, Ty::error())),
                Ty::error(),
            );
        }

        let ba;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            ba = e.uuid();
            e.record_patch(
                "s",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(100)))]),
                    removals: vec![],
                },
                Ty::error(),
            );
        }

        let bb;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            bb = e.uuid();
            e.record_patch(
                "s",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(200)))]),
                    removals: vec![],
                },
                Ty::error(),
            );
        }

        // Force persist
        {
            j.entry_mut(ba).await.unwrap().next().await.unwrap();
        }
        {
            j.entry_mut(bb).await.unwrap().next().await.unwrap();
        }

        // ba: a=100, bb: a=200, n1: a=0
        {
            let e = j.entry(ba).await.unwrap();
            let val = e.get("s").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(100)));
        }
        {
            let e = j.entry(bb).await.unwrap();
            let val = e.get("s").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(200)));
        }
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("s").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(0)));
        }
    }

    #[tokio::test]
    async fn mixed_set_sequence_rec_coexist() {
        // Set(스냅샷), Sequence, Rec(패치)가 같은 journal에서 공존하며
        // 각각 독립적으로 히스토리를 보존하는지 검증.
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), interner.clone(), 1)
                .await
                .unwrap();
        let a = interner.intern("a");

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            // counter: 숫자 (Set)
            e.record_patch("counter", PatchDiff::set(TypedValue::int(0)), Ty::error());
            // log: 시퀀스
            let deque = TrackedDeque::from_vec(vec![Value::int(10)]);
            e.record_sequence_diff("log", deque, seq_ty());
            // config: 오브젝트 (Rec)
            let obj = Value::object(FxHashMap::from_iter([(a, Value::int(100))]));
            e.record_patch(
                "config",
                PatchDiff::set(TypedValue::new(obj, Ty::error())),
                Ty::error(),
            );
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            // counter = 1
            e.record_patch("counter", PatchDiff::set(TypedValue::int(1)), Ty::error());
            // log append 20
            let val = e.get("log").unwrap();
            let mut working = val
                .value()
                .expect_ref::<SequenceChain>("n2 log")
                .origin()
                .clone();
            working.checkpoint();
            working.push(Value::int(20));
            e.record_sequence_diff("log", working, seq_ty());
            // config.a = 200
            e.record_patch(
                "config",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(200)))]),
                    removals: vec![],
                },
                Ty::error(),
            );
        }

        // Force persist
        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        // n2: counter=1, log=[10,20], config.a=200
        {
            let e = j.entry(n2).await.unwrap();
            assert_eq!(
                e.get("counter")
                    .unwrap()
                    .value()
                    .expect_ref::<i64>("expected Int"),
                &1
            );
            let val = e.get("log").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(10), Value::int(20)]);
            let val = e.get("config").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(200)));
        }

        // n1: counter=0, log=[10], config.a=100
        {
            let e = j.entry(n1).await.unwrap();
            assert_eq!(
                e.get("counter")
                    .unwrap()
                    .value()
                    .expect_ref::<i64>("expected Int"),
                &0
            );
            let val = e.get("log").unwrap();
            assert_eq!(get_seq_items(&val), vec![Value::int(10)]);
            let val = e.get("config").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(100)));
        }
    }

    #[tokio::test]
    async fn patch_compute_roundtrip_via_blobstore() {
        // PatchDiff::compute로 생성한 diff가 BlobStore를 통해 persist/restore 후에도
        // 올바르게 적용되는지 검증.
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(MemBlobStore::new(), interner.clone(), 1)
                .await
                .unwrap();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");

        let old_val = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)),
            (b, Value::int(2)),
            (c, Value::int(3)),
        ]));
        let new_val = Value::object(FxHashMap::from_iter([
            (a, Value::int(1)), // unchanged
            (b, Value::int(99)), // changed
                                // c removed
        ]));

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.record_patch(
                "state",
                PatchDiff::set(TypedValue::new(old_val.clone(), Ty::error())),
                Ty::error(),
            );
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            let diff = PatchDiff::compute(&old_val, &new_val).expect("should have diff");
            e.record_patch("state", diff, Ty::error());
        }

        // Force persist
        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        // n2: {a: 1, b: 99} (c removed)
        {
            let e = j.entry(n2).await.unwrap();
            let val = e.get("state").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(1)));
            assert_eq!(f.get(&b), Some(&Value::int(99)));
            assert_eq!(f.get(&c), None);
        }
        // n1: {a: 1, b: 2, c: 3}
        {
            let e = j.entry(n1).await.unwrap();
            let val = e.get("state").unwrap();
            let f = val
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(1)));
            assert_eq!(f.get(&b), Some(&Value::int(2)));
            assert_eq!(f.get(&c), Some(&Value::int(3)));
        }
    }

    #[tokio::test]
    async fn patch_set_non_object_compute_returns_set() {
        // non-object 값의 compute가 PatchDiff::Set을 반환하는지 검증.
        let old = Value::int(1);
        let new = Value::int(2);
        let diff = PatchDiff::compute(&old, &new);
        assert!(matches!(
            diff,
            Some(PatchDiff::Set(Value::Pure(PureValue::Int(2))))
        ));

        // 동일한 값이면 None
        let same = PatchDiff::compute(&old, &old);
        assert!(same.is_none());

        // 문자열 → 숫자 크로스 타입
        let old_str = Value::string("hello".to_string());
        let new_int = Value::int(42);
        let cross = PatchDiff::compute(&old_str, &new_int);
        assert!(matches!(
            cross,
            Some(PatchDiff::Set(Value::Pure(PureValue::Int(42))))
        ));
    }

    // =========================================================================
    // SpyBlobStore — tracks put sizes for diff-only storage verification
    // =========================================================================

    /// BlobStore wrapper that records every put's byte length.
    /// Used to verify that Sequence diffs store only the diff, not the full state.
    struct SpyBlobStore {
        inner: MemBlobStore,
        /// (hash, byte_len) for every put call, in order.
        put_log: Vec<(BlobHash, usize)>,
    }

    impl SpyBlobStore {
        fn new() -> Self {
            Self {
                inner: MemBlobStore::new(),
                put_log: Vec::new(),
            }
        }

        /// Returns the byte sizes of all blobs put (excluding tree metadata).
        fn put_sizes(&self) -> &[(BlobHash, usize)] {
            &self.put_log
        }

        /// Find the size of a specific blob by hash.
        fn size_of(&self, hash: &BlobHash) -> Option<usize> {
            self.put_log
                .iter()
                .find(|(h, _)| h == hash)
                .map(|(_, s)| *s)
        }
    }

    impl BlobStore for SpyBlobStore {
        async fn put(&mut self, data: Vec<u8>) -> BlobHash {
            let len = data.len();
            let hash = self.inner.put(data).await;
            self.put_log.push((hash, len));
            hash
        }

        async fn get(&self, hash: &BlobHash) -> Option<Vec<u8>> {
            self.inner.get(hash).await
        }

        async fn remove(&mut self, hash: &BlobHash) {
            self.inner.remove(hash).await;
        }

        async fn ref_get(&self, name: &str) -> Option<BlobHash> {
            self.inner.ref_get(name).await
        }

        async fn ref_cas(
            &mut self,
            name: &str,
            expected: Option<BlobHash>,
            new: BlobHash,
        ) -> Result<(), Option<BlobHash>> {
            self.inner.ref_cas(name, expected, new).await
        }

        async fn ref_remove(&mut self, name: &str) {
            self.inner.ref_remove(name).await;
        }

        async fn batch_put(&mut self, blobs: Vec<Vec<u8>>) -> Vec<BlobHash> {
            let mut hashes = Vec::with_capacity(blobs.len());
            for data in blobs {
                hashes.push(self.put(data).await);
            }
            hashes
        }

        async fn batch_get(&self, hashes: &[BlobHash]) -> Vec<Option<Vec<u8>>> {
            self.inner.batch_get(hashes).await
        }

        async fn batch_remove(&mut self, hashes: Vec<BlobHash>) {
            self.inner.batch_remove(hashes).await;
        }
    }

    // =========================================================================
    // Blackbox tests — Sequence diff-only storage verification
    // =========================================================================
    //
    // These tests are the PRIMARY defense against Sequence storage regressions.
    //
    // Design principle: treat the journal as a black box. We feed operations
    // in, flush to blob store, reopen from scratch, and assert on final values.
    // The SpyBlobStore additionally tracks blob sizes so we can verify that
    // diff blobs contain only the diff — not the full accumulated state.
    //
    // Why these tests matter:
    // - Sequence persistence stores diffs (OwnedDequeDiff), not full state.
    // - If full state leaks into the diff blob, storage grows O(n²) over turns.
    // - If the diff chain is applied incorrectly on restore, values are wrong.
    // - These tests catch BOTH failure modes: size-based AND value-based.
    //
    // Test categories:
    // A. SIZE TESTS — "diff blob must be small when the change is small"
    //    Fail if full state leaks into the diff blob.
    // B. ROUNDTRIP TESTS — "flush → reopen → values must be exact"
    //    Fail if diff chain reconstruction is broken.
    // C. COMBINED — both size and value checks in one scenario.
    //
    // =========================================================================

    // ── A. Size tests ────────────────────────────────────────────────────

    /// A1. Append 1 item to a 1000-item sequence.
    ///
    /// The diff blob should contain only the 1 pushed item + metadata.
    /// If the full 1001-item state leaks in, the diff blob will be ≈ snapshot size.
    ///
    /// Regression: serialize_entries was storing ConcreteValue::Sequence (full items)
    /// in the diff blob instead of just the OwnedDequeDiff.
    #[tokio::test]
    async fn size_append_one_to_large_sequence() {
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), Interner::new(), 1)
                .await
                .unwrap();

        // Turn 1: initial 1000-item sequence.
        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let items: Vec<Value> = (0..1000).map(|i| Value::int(i)).collect();
            let deque = TrackedDeque::from_vec(items);
            e.record_sequence_diff("big", deque, seq_ty());
        }

        // Turn 2: append 1 item.
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            let val = e.get("big").unwrap();
            let mut working = val
                .value()
                .expect_ref::<SequenceChain>("n2 big")
                .origin()
                .clone();
            working.checkpoint();
            working.push(Value::int(9999));
            e.record_sequence_diff("big", working, seq_ty());
        }

        // Persist n2 by advancing.
        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        let n2_idx = j.tree.uuid_to_idx[&n2];
        let n2_seq_diff_hash = j.tree.nodes[n2_idx]
            .seq_diff_hash
            .expect("n2 should have a seq_diff blob");
        // n2 must NOT have a patch diff blob (only sequence changed).
        assert!(
            j.tree.nodes[n2_idx].diff_hash.is_none(),
            "n2 should NOT have a patch diff blob — only sequence was changed"
        );
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap_hash = j.tree.nodes[n1_idx]
            .snapshot_hash
            .expect("n1 should have a snapshot (interval=1)");

        let n2_seq_diff_size = j.store.size_of(&n2_seq_diff_hash).unwrap();
        let n1_snap_size = j.store.size_of(&n1_snap_hash).unwrap();

        assert!(
            n2_seq_diff_size < n1_snap_size / 5,
            "seq diff blob ({n2_seq_diff_size} bytes) should be much smaller than \
             snapshot ({n1_snap_size} bytes) — full state may have leaked into diff"
        );
    }

    /// A2. Consume from front of a large sequence (no push).
    ///
    /// The diff is {consumed: 1, pushed: []} — essentially zero payload.
    /// If full state leaks, diff ≈ snapshot.
    #[tokio::test]
    async fn size_consume_only_from_large_sequence() {
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), Interner::new(), 1)
                .await
                .unwrap();

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let items: Vec<Value> = (0..500).map(|i| Value::int(i)).collect();
            let deque = TrackedDeque::from_vec(items);
            e.record_sequence_diff("q", deque, seq_ty());
        }

        // Consume 1 item, push nothing.
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            let val = e.get("q").unwrap();
            let mut working = val
                .value()
                .expect_ref::<SequenceChain>("n2")
                .origin()
                .clone();
            working.checkpoint();
            working.consume(1);
            e.record_sequence_diff("q", working, seq_ty());
        }

        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        let n2_idx = j.tree.uuid_to_idx[&n2];
        let n2_seq_diff_hash = j.tree.nodes[n2_idx]
            .seq_diff_hash
            .expect("n2 should have a seq_diff blob");
        assert!(
            j.tree.nodes[n2_idx].diff_hash.is_none(),
            "n2 should NOT have a patch diff blob"
        );
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap_hash = j.tree.nodes[n1_idx].snapshot_hash.unwrap();

        let n2_seq_diff_size = j.store.size_of(&n2_seq_diff_hash).unwrap();
        let n1_snap_size = j.store.size_of(&n1_snap_hash).unwrap();

        // A consume-only diff has zero pushed items — must be tiny.
        assert!(
            n2_seq_diff_size < n1_snap_size / 10,
            "consume-only diff ({n2_seq_diff_size} bytes) should be negligible vs \
             snapshot ({n1_snap_size} bytes)"
        );
    }

    /// A3. Multiple turns of small diffs on a large sequence.
    ///
    /// Every turn's diff blob should be small, not growing with accumulated state.
    /// Catches the case where diff blob size is O(turn * n) instead of O(diff).
    #[tokio::test]
    async fn size_diff_stays_small_across_many_turns() {
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), Interner::new(), 1)
                .await
                .unwrap();

        // Turn 1: seed with 500 items.
        let mut cursor;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            let items: Vec<Value> = (0..500).map(|i| Value::int(i)).collect();
            let deque = TrackedDeque::from_vec(items);
            e.record_sequence_diff("q", deque, seq_ty());
        }

        // Turns 2–11: each turn appends 1 item.
        let mut diff_hashes = Vec::new();
        for i in 500..510 {
            let prev = cursor;
            let mut e = j.entry_mut(prev).await.unwrap().next().await.unwrap();
            cursor = e.uuid();
            let val = e.get("q").unwrap();
            let mut working = val
                .value()
                .expect_ref::<SequenceChain>("turn")
                .origin()
                .clone();
            working.checkpoint();
            working.push(Value::int(i));
            e.record_sequence_diff("q", working, seq_ty());
            drop(e);

            // Persist by advancing.
            let e2 = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
            let next = e2.uuid();
            drop(e2);

            let idx = j.tree.uuid_to_idx[&cursor];
            if let Some(dh) = j.tree.nodes[idx].seq_diff_hash {
                diff_hashes.push(dh);
            }
            cursor = next;
        }

        // All diff blobs should be small (each adds just 1 item).
        // The snapshot keeps growing, but diff blobs should NOT.
        // Read initial snapshot size now (after it's been persisted by the first next() call).
        let initial_idx = j
            .tree
            .uuid_to_idx
            .values()
            .find(|&&idx| j.tree.nodes[idx].depth == 1)
            .copied()
            .unwrap();
        let initial_snap_size = j.tree.nodes[initial_idx]
            .snapshot_hash
            .and_then(|h| j.store.size_of(&h))
            .expect("initial node should have been persisted with a snapshot");

        for (i, dh) in diff_hashes.iter().enumerate() {
            let diff_size = j.store.size_of(dh).unwrap();
            assert!(
                diff_size < initial_snap_size / 5,
                "turn {}: diff blob ({diff_size} bytes) should be much smaller than \
                 initial snapshot ({initial_snap_size} bytes) — diff may be accumulating state",
                i + 2,
            );
        }
    }

    /// A4. Consume + push on a large sequence — diff reflects only the delta.
    ///
    /// consume 5, push 2 on a 500-item seq → diff payload = 2 items + metadata.
    #[tokio::test]
    async fn size_consume_and_push_on_large_sequence() {
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), Interner::new(), 1)
                .await
                .unwrap();

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            let items: Vec<Value> = (0..500).map(|i| Value::int(i)).collect();
            let deque = TrackedDeque::from_vec(items);
            e.record_sequence_diff("q", deque, seq_ty());
        }

        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            let val = e.get("q").unwrap();
            let mut working = val
                .value()
                .expect_ref::<SequenceChain>("n2")
                .origin()
                .clone();
            working.checkpoint();
            // Consume 5 from front, push 2 new items.
            working.consume(5);
            working.push(Value::int(8888));
            working.push(Value::int(9999));
            e.record_sequence_diff("q", working, seq_ty());
        }

        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        let n2_idx = j.tree.uuid_to_idx[&n2];
        let n2_seq_diff_hash = j.tree.nodes[n2_idx].seq_diff_hash.unwrap();
        assert!(
            j.tree.nodes[n2_idx].diff_hash.is_none(),
            "n2 should NOT have a patch diff blob"
        );
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap_hash = j.tree.nodes[n1_idx].snapshot_hash.unwrap();

        let n2_seq_diff_size = j.store.size_of(&n2_seq_diff_hash).unwrap();
        let n1_snap_size = j.store.size_of(&n1_snap_hash).unwrap();

        assert!(
            n2_seq_diff_size < n1_snap_size / 5,
            "consume+push diff ({n2_seq_diff_size} bytes) should be much smaller than \
             snapshot ({n1_snap_size} bytes)"
        );
    }

    // ── A'. Size tests — Patch diff blob ────────────────────────────────
    //
    // PatchDiff::Rec patches a subset of fields. The diff blob should
    // contain only the changed fields, not the entire object.
    // If the full post-apply value leaks into the diff blob, size explodes.

    /// A'1. Large object (100 fields) + Rec patch on 1 field.
    ///
    /// diff blob should be much smaller than the full 100-field snapshot.
    /// Currently expected to FAIL: turn_patches stores full post-apply value.
    #[tokio::test]
    async fn size_patch_rec_one_field_of_large_object() {
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 1)
                .await
                .unwrap();

        // Build a 100-field object.
        let fields: FxHashMap<_, _> = (0..100)
            .map(|i| (interner.intern(&format!("f{i}")), Value::int(i)))
            .collect();
        let target_field = interner.intern("f50");

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.record_patch(
                "obj",
                PatchDiff::set(TypedValue::new(Value::object(fields), Ty::error())),
                Ty::error(),
            );
        }

        // Rec patch: change only f50 = 9999.
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.record_patch(
                "obj",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(
                        target_field,
                        PatchDiff::Set(Value::int(9999)),
                    )]),
                    removals: vec![],
                },
                Ty::error(),
            );
        }

        // Persist.
        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        let n2_idx = j.tree.uuid_to_idx[&n2];
        let n2_diff_hash = j.tree.nodes[n2_idx]
            .diff_hash
            .expect("n2 should have a patch diff blob");
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap_hash = j.tree.nodes[n1_idx].snapshot_hash.unwrap();

        let n2_diff_size = j.store.size_of(&n2_diff_hash).unwrap();
        let n1_snap_size = j.store.size_of(&n1_snap_hash).unwrap();

        assert!(
            n2_diff_size < n1_snap_size / 5,
            "Rec patch diff ({n2_diff_size} bytes) should be much smaller than \
             100-field snapshot ({n1_snap_size} bytes) — full object may have leaked"
        );
    }

    /// A'2. Nested object: patch only a deep inner field.
    ///
    /// outer = {a: {b: {c: 1, d: 2, ...(50 fields)}, e: ...}, f: ...(50 fields)}
    /// Patch: outer.a.b.c = 999
    /// diff blob should contain only the nested Rec path, not the entire tree.
    #[tokio::test]
    async fn size_patch_rec_nested_deep_field() {
        let interner = Interner::new();
        let (mut j, root) =
            BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 1)
                .await
                .unwrap();

        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");

        // Build a wide+deep object.
        let inner_fields: FxHashMap<_, _> = (0..50)
            .map(|i| (interner.intern(&format!("i{i}")), Value::int(i)))
            .collect();
        let mut inner = inner_fields;
        inner.insert(c, Value::int(0));

        let outer_fields: FxHashMap<_, _> = (0..50)
            .map(|i| (interner.intern(&format!("o{i}")), Value::int(i)))
            .collect();
        let mut outer = outer_fields;
        outer.insert(
            a,
            Value::object(FxHashMap::from_iter([(b, Value::object(inner))])),
        );

        let n1;
        {
            let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
            n1 = e.uuid();
            e.record_patch(
                "obj",
                PatchDiff::set(TypedValue::new(Value::object(outer), Ty::error())),
                Ty::error(),
            );
        }

        // Nested Rec: only change a.b.c = 999.
        let n2;
        {
            let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
            n2 = e.uuid();
            e.record_patch(
                "obj",
                PatchDiff::Rec {
                    updates: FxHashMap::from_iter([(
                        a,
                        PatchDiff::Rec {
                            updates: FxHashMap::from_iter([(
                                b,
                                PatchDiff::Rec {
                                    updates: FxHashMap::from_iter([(
                                        c,
                                        PatchDiff::Set(Value::int(999)),
                                    )]),
                                    removals: vec![],
                                },
                            )]),
                            removals: vec![],
                        },
                    )]),
                    removals: vec![],
                },
                Ty::error(),
            );
        }

        {
            j.entry_mut(n2).await.unwrap().next().await.unwrap();
        }

        let n2_idx = j.tree.uuid_to_idx[&n2];
        let n2_diff_hash = j.tree.nodes[n2_idx].diff_hash.unwrap();
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap_hash = j.tree.nodes[n1_idx].snapshot_hash.unwrap();

        let n2_diff_size = j.store.size_of(&n2_diff_hash).unwrap();
        let n1_snap_size = j.store.size_of(&n1_snap_hash).unwrap();

        assert!(
            n2_diff_size < n1_snap_size / 3,
            "nested Rec diff ({n2_diff_size} bytes) should be much smaller than \
             full object snapshot ({n1_snap_size} bytes) — full tree may have leaked"
        );
    }

    // ── B. Roundtrip tests — flush → reopen → values exact ──────────────

    /// B1. Multi-turn: push, consume, push → flush → reopen.
    ///
    /// Each turn's historical state must be reconstructed exactly from
    /// the snapshot + diff chain. Tests the core restore path.
    #[tokio::test]
    async fn roundtrip_multi_turn_consume_and_push() {
        let interner = Interner::new();

        let n1;
        let n2;
        let n3;
        let store;
        {
            let (mut j, root) =
                BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 1)
                    .await
                    .unwrap();

            // Turn 1: [1, 2, 3]
            {
                let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
                n1 = e.uuid();
                let deque =
                    TrackedDeque::from_vec(vec![Value::int(1), Value::int(2), Value::int(3)]);
                e.record_sequence_diff("seq", deque, seq_ty());
            }
            // Turn 2: consume 1, push 4 → [2, 3, 4]
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                n2 = e.uuid();
                let val = e.get("seq").unwrap();
                let mut working = val
                    .value()
                    .expect_ref::<SequenceChain>("t2")
                    .origin()
                    .clone();
                working.checkpoint();
                working.consume(1);
                working.push(Value::int(4));
                e.record_sequence_diff("seq", working, seq_ty());
            }
            // Turn 3: push 5, 6 → [2, 3, 4, 5, 6]
            {
                let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
                n3 = e.uuid();
                let val = e.get("seq").unwrap();
                let mut working = val
                    .value()
                    .expect_ref::<SequenceChain>("t3")
                    .origin()
                    .clone();
                working.checkpoint();
                working.push(Value::int(5));
                working.push(Value::int(6));
                e.record_sequence_diff("seq", working, seq_ty());
            }

            j.flush_tree().await.unwrap();
            store = j.store.inner;
        }

        let j2 = BlobStoreJournal::open(store, interner)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(
            get_seq_items(&j2.entry(n1).await.unwrap().get("seq").unwrap()),
            vec![Value::int(1), Value::int(2), Value::int(3)]
        );
        assert_eq!(
            get_seq_items(&j2.entry(n2).await.unwrap().get("seq").unwrap()),
            vec![Value::int(2), Value::int(3), Value::int(4)]
        );
        assert_eq!(
            get_seq_items(&j2.entry(n3).await.unwrap().get("seq").unwrap()),
            vec![
                Value::int(2),
                Value::int(3),
                Value::int(4),
                Value::int(5),
                Value::int(6)
            ]
        );
    }

    /// B2. Fork: two branches diverge from same parent → flush → reopen.
    ///
    /// Each branch must see its own changes, not the other's.
    /// Parent must be unchanged.
    #[tokio::test]
    async fn roundtrip_fork_independent_branches() {
        let interner = Interner::new();

        let n1;
        let branch_a;
        let branch_b;
        let store;
        {
            let (mut j, root) =
                BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 1)
                    .await
                    .unwrap();

            {
                let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
                n1 = e.uuid();
                let deque = TrackedDeque::from_vec(vec![Value::int(1)]);
                e.record_sequence_diff("seq", deque, seq_ty());
            }
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                branch_a = e.uuid();
                let val = e.get("seq").unwrap();
                let mut working = val
                    .value()
                    .expect_ref::<SequenceChain>("ba")
                    .origin()
                    .clone();
                working.checkpoint();
                working.push(Value::int(100));
                e.record_sequence_diff("seq", working, seq_ty());
            }
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                branch_b = e.uuid();
                let val = e.get("seq").unwrap();
                let mut working = val
                    .value()
                    .expect_ref::<SequenceChain>("bb")
                    .origin()
                    .clone();
                working.checkpoint();
                working.push(Value::int(200));
                e.record_sequence_diff("seq", working, seq_ty());
            }

            j.flush_tree().await.unwrap();
            store = j.store.inner;
        }

        let j2 = BlobStoreJournal::open(store, interner)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(
            get_seq_items(&j2.entry(n1).await.unwrap().get("seq").unwrap()),
            vec![Value::int(1)]
        );
        assert_eq!(
            get_seq_items(&j2.entry(branch_a).await.unwrap().get("seq").unwrap()),
            vec![Value::int(1), Value::int(100)]
        );
        assert_eq!(
            get_seq_items(&j2.entry(branch_b).await.unwrap().get("seq").unwrap()),
            vec![Value::int(1), Value::int(200)]
        );
    }

    /// B3. Mixed types: Sequence + Patch + Object coexist → flush → reopen.
    ///
    /// Verifies that Sequence diff storage doesn't corrupt or interfere with
    /// Patch-based storage for other keys in the same node.
    #[tokio::test]
    async fn roundtrip_mixed_types_coexist() {
        let interner = Interner::new();
        let a = interner.intern("a");

        let n1;
        let n2;
        let store;
        {
            let (mut j, root) =
                BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 1)
                    .await
                    .unwrap();

            {
                let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
                n1 = e.uuid();
                e.record_patch("counter", PatchDiff::set(TypedValue::int(0)), Ty::error());
                let deque = TrackedDeque::from_vec(vec![Value::int(10)]);
                e.record_sequence_diff("seq", deque, seq_ty());
                let obj = Value::object(FxHashMap::from_iter([(a, Value::int(1))]));
                e.record_patch(
                    "obj",
                    PatchDiff::set(TypedValue::new(obj, Ty::error())),
                    Ty::error(),
                );
            }
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                n2 = e.uuid();
                e.record_patch("counter", PatchDiff::set(TypedValue::int(1)), Ty::error());
                let val = e.get("seq").unwrap();
                let mut working = val
                    .value()
                    .expect_ref::<SequenceChain>("t2 seq")
                    .origin()
                    .clone();
                working.checkpoint();
                working.push(Value::int(20));
                e.record_sequence_diff("seq", working, seq_ty());
                e.record_patch(
                    "obj",
                    PatchDiff::Rec {
                        updates: FxHashMap::from_iter([(a, PatchDiff::Set(Value::int(99)))]),
                        removals: vec![],
                    },
                    Ty::error(),
                );
            }

            j.flush_tree().await.unwrap();
            store = j.store.inner;
        }

        let j2 = BlobStoreJournal::open(store, interner.clone())
            .await
            .unwrap()
            .unwrap();

        // Turn 2
        {
            let e = j2.entry(n2).await.unwrap();
            assert_eq!(
                e.get("counter")
                    .unwrap()
                    .value()
                    .expect_ref::<i64>("expected Int"),
                &1
            );
            assert_eq!(
                get_seq_items(&e.get("seq").unwrap()),
                vec![Value::int(10), Value::int(20)]
            );
            let obj = e.get("obj").unwrap();
            let f = obj
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(99)));
        }
        // Turn 1 (history preserved)
        {
            let e = j2.entry(n1).await.unwrap();
            assert_eq!(
                e.get("counter")
                    .unwrap()
                    .value()
                    .expect_ref::<i64>("expected Int"),
                &0
            );
            assert_eq!(get_seq_items(&e.get("seq").unwrap()), vec![Value::int(10)]);
            let obj = e.get("obj").unwrap();
            let f = obj
                .value()
                .expect_ref::<FxHashMap<Astr, Value>>("expected Object");
            assert_eq!(f.get(&a), Some(&Value::int(1)));
        }
    }

    /// B4. Snapshot interval > 1: restore requires walking diff chain.
    ///
    /// interval=4, 10 turns of push. Nodes at depth 1,2,3,5,6,7,9,10 have
    /// no snapshot — they MUST be reconstructed from the nearest snapshot
    /// ancestor + diff chain. If diff storage is broken, this fails.
    #[tokio::test]
    async fn roundtrip_across_snapshot_interval() {
        let interner = Interner::new();

        let mut uuids = Vec::new();
        let store;
        {
            let (mut j, root) =
                BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 4)
                    .await
                    .unwrap();

            let mut cursor = root;
            for i in 1..=10 {
                let mut e = j.entry_mut(cursor).await.unwrap().next().await.unwrap();
                cursor = e.uuid();
                uuids.push(cursor);
                if i == 1 {
                    e.record_sequence_diff(
                        "seq",
                        TrackedDeque::from_vec(vec![Value::int(i)]),
                        seq_ty(),
                    );
                } else {
                    let val = e.get("seq").unwrap();
                    let mut working = val
                        .value()
                        .expect_ref::<SequenceChain>("interval")
                        .origin()
                        .clone();
                    working.checkpoint();
                    working.push(Value::int(i));
                    e.record_sequence_diff("seq", working, seq_ty());
                }
            }

            j.flush_tree().await.unwrap();
            store = j.store.inner;
        }

        let j2 = BlobStoreJournal::open_with_snapshot_interval(store, interner, 4)
            .await
            .unwrap()
            .unwrap();

        // Every turn's accumulated state must be exact.
        for (turn, uuid) in uuids.iter().enumerate() {
            let expected: Vec<Value> = (1..=(turn as i64 + 1)).map(|i| Value::int(i)).collect();
            let e = j2.entry(*uuid).await.unwrap();
            assert_eq!(
                get_seq_items(&e.get("seq").unwrap()),
                expected,
                "turn {} (depth {}) has wrong seq items",
                turn + 1,
                turn + 1,
            );
        }
    }

    /// B5. Consume + removed_back together → flush → reopen.
    ///
    /// Tests that both front-consume and back-removal are correctly persisted
    /// and restored through the diff chain.
    #[tokio::test]
    async fn roundtrip_consume_and_remove_back() {
        let interner = Interner::new();

        let n1;
        let n2;
        let store;
        {
            let (mut j, root) =
                BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 1)
                    .await
                    .unwrap();

            // Turn 1: [1, 2, 3, 4, 5]
            {
                let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
                n1 = e.uuid();
                let items: Vec<Value> = (1..=5).map(|i| Value::int(i)).collect();
                e.record_sequence_diff("seq", TrackedDeque::from_vec(items), seq_ty());
            }
            // Turn 2: consume 2 from front, remove 1 from back, push 10
            // [1,2,3,4,5] → consume 2 → [3,4,5] → remove_back 1 → [3,4] → push 10 → [3,4,10]
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                n2 = e.uuid();
                let mut working = e
                    .get("seq")
                    .unwrap()
                    .value()
                    .expect_ref::<SequenceChain>("t2")
                    .origin()
                    .clone();
                working.checkpoint();
                working.consume(2);
                working.pop();
                working.push(Value::int(10));
                e.record_sequence_diff("seq", working, seq_ty());
            }

            j.flush_tree().await.unwrap();
            store = j.store.inner;
        }

        let j2 = BlobStoreJournal::open(store, interner)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(
            get_seq_items(&j2.entry(n1).await.unwrap().get("seq").unwrap()),
            vec![
                Value::int(1),
                Value::int(2),
                Value::int(3),
                Value::int(4),
                Value::int(5)
            ]
        );
        assert_eq!(
            get_seq_items(&j2.entry(n2).await.unwrap().get("seq").unwrap()),
            vec![Value::int(3), Value::int(4), Value::int(10)]
        );
    }

    // ── C. Combined: size + roundtrip in one scenario ───────────────────

    /// C1. Large sequence, multiple diverse diffs, flush+reopen, check both
    /// blob sizes AND restored values.
    ///
    /// This is the most comprehensive single test. If this passes with correct
    /// sizes AND correct values, the diff-only storage pipeline is working.
    #[tokio::test]
    async fn combined_large_sequence_diverse_diffs() {
        let interner = Interner::new();

        let n1; // [0..500]
        let n2; // consume 10, push [900,901] → [10..500, 900, 901]
        let n3; // consume 5, remove_back 2 → [15..500]
        let store;
        let n1_snap_size;
        let n2_diff_size;
        let n3_diff_size;
        {
            let (mut j, root) =
                BlobStoreJournal::with_snapshot_interval(SpyBlobStore::new(), interner.clone(), 1)
                    .await
                    .unwrap();

            // Turn 1: 500 items.
            {
                let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
                n1 = e.uuid();
                let items: Vec<Value> = (0..500).map(|i| Value::int(i)).collect();
                e.record_sequence_diff("q", TrackedDeque::from_vec(items), seq_ty());
            }
            // Turn 2: consume 10, push 2.
            {
                let mut e = j.entry_mut(n1).await.unwrap().next().await.unwrap();
                n2 = e.uuid();
                let mut working = e
                    .get("q")
                    .unwrap()
                    .value()
                    .expect_ref::<SequenceChain>("t2")
                    .origin()
                    .clone();
                working.checkpoint();
                working.consume(10);
                working.push(Value::int(900));
                working.push(Value::int(901));
                e.record_sequence_diff("q", working, seq_ty());
            }
            // Turn 3: consume 5, remove_back 2, no push.
            {
                let mut e = j.entry_mut(n2).await.unwrap().next().await.unwrap();
                n3 = e.uuid();
                let mut working = e
                    .get("q")
                    .unwrap()
                    .value()
                    .expect_ref::<SequenceChain>("t3")
                    .origin()
                    .clone();
                working.checkpoint();
                working.consume(5);
                working.pop();
                working.pop();
                e.record_sequence_diff("q", working, seq_ty());
            }

            // Persist all.
            {
                j.entry_mut(n3).await.unwrap().next().await.unwrap();
            }

            // Capture sizes.
            let n1_idx = j.tree.uuid_to_idx[&n1];
            n1_snap_size = j
                .store
                .size_of(&j.tree.nodes[n1_idx].snapshot_hash.unwrap())
                .unwrap();
            let n2_idx = j.tree.uuid_to_idx[&n2];
            n2_diff_size = j
                .store
                .size_of(&j.tree.nodes[n2_idx].seq_diff_hash.unwrap())
                .unwrap();
            let n3_idx = j.tree.uuid_to_idx[&n3];
            n3_diff_size = j
                .store
                .size_of(&j.tree.nodes[n3_idx].seq_diff_hash.unwrap())
                .unwrap();

            j.flush_tree().await.unwrap();
            store = j.store.inner;
        }

        // Size checks.
        assert!(
            n2_diff_size < n1_snap_size / 5,
            "t2 diff ({n2_diff_size}B) too large vs snapshot ({n1_snap_size}B)"
        );
        assert!(
            n3_diff_size < n1_snap_size / 10,
            "t3 diff ({n3_diff_size}B) too large vs snapshot ({n1_snap_size}B) — no pushed items"
        );

        // Value checks after reopen.
        let j2 = BlobStoreJournal::open(store, interner)
            .await
            .unwrap()
            .unwrap();

        // Turn 1: [0..500]
        {
            let expected: Vec<Value> = (0..500).map(|i| Value::int(i)).collect();
            assert_eq!(
                get_seq_items(&j2.entry(n1).await.unwrap().get("q").unwrap()),
                expected
            );
        }
        // Turn 2: [10..500, 900, 901]
        {
            let mut expected: Vec<Value> = (10..500).map(|i| Value::int(i)).collect();
            expected.push(Value::int(900));
            expected.push(Value::int(901));
            assert_eq!(
                get_seq_items(&j2.entry(n2).await.unwrap().get("q").unwrap()),
                expected
            );
        }
        // Turn 3: [15..500] (consumed 5 more from front, removed 2 from back: 901 and 900)
        {
            let expected: Vec<Value> = (15..500).map(|i| Value::int(i)).collect();
            assert_eq!(
                get_seq_items(&j2.entry(n3).await.unwrap().get("q").unwrap()),
                expected
            );
        }
    }

    #[tokio::test]
    #[should_panic(expected = "checksum mismatch")]
    async fn sequence_checksum_mismatch_panics() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let deque = TrackedDeque::from_vec(vec![Value::int(1)]);
        e.record_sequence_diff("q", deque, seq_ty());
        // Fake deque with different checksum
        let mut fake = TrackedDeque::from_vec(vec![Value::int(1)]);
        fake.checkpoint();
        fake.push(Value::int(99));
        e.record_sequence_diff("q", fake, seq_ty());
    }

    // ── Soundness: cross-type guard ─────────────────────────────────

    /// record_patch on a Sequence key must panic.
    #[tokio::test]
    #[should_panic(expected = "record_patch called on Sequence key")]
    async fn record_patch_on_sequence_key_panics() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        let deque = TrackedDeque::from_vec(vec![Value::int(1)]);
        e.record_sequence_diff("q", deque, seq_ty());
        // This must panic — "q" is a Sequence key.
        e.record_patch("q", PatchDiff::set(TypedValue::int(999)), Ty::error());
    }

    /// record_sequence_diff on a Patch key must panic.
    #[tokio::test]
    #[should_panic(expected = "record_sequence_diff called on non-Sequence key")]
    async fn record_sequence_diff_on_patch_key_panics() {
        let (mut j, root) = new_journal().await;
        let mut e = j.entry_mut(root).await.unwrap().next().await.unwrap();
        e.record_patch("x", PatchDiff::set(TypedValue::int(1)), Ty::error());
        // This must panic — "x" is a Patch key.
        let deque = TrackedDeque::from_vec(vec![Value::int(99)]);
        e.record_sequence_diff("x", deque, seq_ty());
    }
}
