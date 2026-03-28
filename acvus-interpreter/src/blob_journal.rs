//! Blob-backed journal — tree-structured context storage with snapshot + diff.
//!
//! Simplified from the orchestration version:
//! - No PatchDiff (recursive Rec diffs) — fields are replaced wholesale.
//! - No TypedValue / SerTy — stores `Value` directly via `SerValue` mirror.
//! - Field diffs and sequence diffs combined into one blob per node.

use std::collections::HashMap;

use acvus_mir::ty::Effect;
use acvus_utils::{DequeChecksum, Interner, OwnedDequeDiff, TrackedDeque};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::blob::{BlobHash, BlobStore};
use crate::iter::SequenceChain;
use crate::journal::{EntryLifecycle, EntryMut, EntryRef, Journal};
use crate::value::Value;

// ── Error ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum JournalError {
    Serialization(String),
    Deserialization(String),
    MissingBlob(String),
    CorruptedData(String),
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JournalError::Serialization(e) => write!(f, "serialization error: {e}"),
            JournalError::Deserialization(e) => write!(f, "deserialization error: {e}"),
            JournalError::MissingBlob(e) => write!(f, "missing blob: {e}"),
            JournalError::CorruptedData(e) => write!(f, "corrupted data: {e}"),
        }
    }
}

impl std::error::Error for JournalError {}

// ── SerValue — serializable mirror of Value ─────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
enum SerValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Unit,
    Byte(u8),
    String(String),
    List(Vec<SerValue>),
    Object(Vec<(String, SerValue)>),
    Tuple(Vec<SerValue>),
    Deque {
        items: Vec<SerValue>,
        checksum: DequeChecksum,
    },
    Variant {
        tag: String,
        payload: Option<Box<SerValue>>,
    },
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
}

impl SerValue {
    fn from_value(v: &Value, interner: &Interner) -> Self {
        match v {
            Value::Int(n) => SerValue::Int(*n),
            Value::Float(f) => SerValue::Float(*f),
            Value::Bool(b) => SerValue::Bool(*b),
            Value::Unit => SerValue::Unit,
            Value::Byte(b) => SerValue::Byte(*b),
            Value::String(s) => SerValue::String(s.to_string()),
            Value::List(items) => SerValue::List(
                items
                    .iter()
                    .map(|v| SerValue::from_value(v, interner))
                    .collect(),
            ),
            Value::Object(obj) => SerValue::Object(
                obj.iter()
                    .map(|(k, v)| {
                        (
                            interner.resolve(*k).to_string(),
                            SerValue::from_value(v, interner),
                        )
                    })
                    .collect(),
            ),
            Value::Tuple(elems) => SerValue::Tuple(
                elems
                    .iter()
                    .map(|v| SerValue::from_value(v, interner))
                    .collect(),
            ),
            Value::Deque(d) => SerValue::Deque {
                items: d
                    .as_slice()
                    .iter()
                    .map(|v| SerValue::from_value(v, interner))
                    .collect(),
                checksum: d.checksum(),
            },
            Value::Variant { tag, payload } => SerValue::Variant {
                tag: interner.resolve(*tag).to_string(),
                payload: payload
                    .as_ref()
                    .map(|p| Box::new(SerValue::from_value(p, interner))),
            },
            Value::Range(r) => SerValue::Range {
                start: r.start,
                end: r.end,
                inclusive: r.inclusive,
            },
            Value::Sequence(sc) => {
                // Serialize the origin deque items + checksum.
                SerValue::Deque {
                    items: sc
                        .origin()
                        .as_slice()
                        .iter()
                        .map(|v| SerValue::from_value(v, interner))
                        .collect(),
                    checksum: sc.origin().checksum(),
                }
            }
            // Empty, Fn, Iterator, Handle, Opaque — not storable.
            other => panic!("SerValue::from_value: unstorable value {other:?}"),
        }
    }

    fn to_value(self, interner: &Interner) -> Value {
        match self {
            SerValue::Int(n) => Value::int(n),
            SerValue::Float(f) => Value::float(f),
            SerValue::Bool(b) => Value::bool_(b),
            SerValue::Unit => Value::unit(),
            SerValue::Byte(b) => Value::byte(b),
            SerValue::String(s) => Value::string(s),
            SerValue::List(items) => {
                Value::list(items.into_iter().map(|v| v.to_value(interner)).collect())
            }
            SerValue::Object(pairs) => {
                let map: FxHashMap<acvus_utils::Astr, Value> = pairs
                    .into_iter()
                    .map(|(k, v)| (interner.intern(&k), v.to_value(interner)))
                    .collect();
                Value::object(map)
            }
            SerValue::Tuple(elems) => {
                Value::tuple(elems.into_iter().map(|v| v.to_value(interner)).collect())
            }
            SerValue::Deque { items, checksum } => {
                let deque = TrackedDeque::from_vec_with_checksum(
                    items.into_iter().map(|v| v.to_value(interner)).collect(),
                    checksum,
                );
                Value::deque(deque)
            }
            SerValue::Variant { tag, payload } => {
                let astr_tag = interner.intern(&tag);
                let payload_val = payload.map(|p| p.to_value(interner));
                Value::variant(astr_tag, payload_val)
            }
            SerValue::Range {
                start,
                end,
                inclusive,
            } => Value::range(start, end, inclusive),
        }
    }
}

// ── SerSequenceDiff ─────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct SerSequenceDiff {
    #[serde(rename = "c")]
    consumed: usize,
    #[serde(rename = "r")]
    removed_back: usize,
    #[serde(rename = "p")]
    pushed: Vec<SerValue>,
}

// ── SerTurnDiff — combined field + sequence diffs ───────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct SerTurnDiff {
    /// Changed fields: key → new value.
    #[serde(rename = "f", skip_serializing_if = "Vec::is_empty", default)]
    fields: Vec<(String, SerValue)>,
    /// Sequence diffs: key → diff.
    #[serde(rename = "s", skip_serializing_if = "Vec::is_empty", default)]
    sequences: Vec<(String, SerSequenceDiff)>,
}

// ── Serialization types (tree metadata) ─────────────────────────────

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
    /// Combined field + sequence diff blob hash.
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
    state: HashMap<String, Value>,
    /// Fields changed this turn (SSA: each key written at most once).
    changed_fields: HashMap<String, Value>,
    /// Sequence diffs this turn.
    changed_sequences: HashMap<String, OwnedDequeDiff<Value>>,
}

// ── Value serialization helpers ─────────────────────────────────────

fn serialize_entries(
    entries: &HashMap<String, Value>,
    interner: &Interner,
) -> Result<Vec<u8>, JournalError> {
    let ser: Vec<(String, SerValue)> = entries
        .iter()
        .map(|(k, v)| (k.clone(), SerValue::from_value(v, interner)))
        .collect();
    serde_json::to_vec(&ser).map_err(|e| JournalError::Serialization(e.to_string()))
}

fn deserialize_entries(
    bytes: &[u8],
    interner: &Interner,
) -> Result<HashMap<String, Value>, JournalError> {
    let ser: Vec<(String, SerValue)> =
        serde_json::from_slice(bytes).map_err(|e| JournalError::Deserialization(e.to_string()))?;
    Ok(ser
        .into_iter()
        .map(|(k, sv)| (k, sv.to_value(interner)))
        .collect())
}

fn serialize_turn_diff(
    changed_fields: &HashMap<String, Value>,
    changed_sequences: &HashMap<String, OwnedDequeDiff<Value>>,
    interner: &Interner,
) -> Result<Vec<u8>, JournalError> {
    let diff = SerTurnDiff {
        fields: changed_fields
            .iter()
            .map(|(k, v)| (k.clone(), SerValue::from_value(v, interner)))
            .collect(),
        sequences: changed_sequences
            .iter()
            .map(|(k, d)| {
                (
                    k.clone(),
                    SerSequenceDiff {
                        consumed: d.consumed,
                        removed_back: d.removed_back,
                        pushed: d
                            .pushed
                            .iter()
                            .map(|v| SerValue::from_value(v, interner))
                            .collect(),
                    },
                )
            })
            .collect(),
    };
    serde_json::to_vec(&diff).map_err(|e| JournalError::Serialization(e.to_string()))
}

fn deserialize_turn_diff(
    bytes: &[u8],
    interner: &Interner,
) -> Result<
    (
        HashMap<String, Value>,
        HashMap<String, OwnedDequeDiff<Value>>,
    ),
    JournalError,
> {
    let diff: SerTurnDiff =
        serde_json::from_slice(bytes).map_err(|e| JournalError::Deserialization(e.to_string()))?;
    let fields = diff
        .fields
        .into_iter()
        .map(|(k, sv)| (k, sv.to_value(interner)))
        .collect();
    let sequences = diff
        .sequences
        .into_iter()
        .map(|(k, sd)| {
            (
                k,
                OwnedDequeDiff {
                    consumed: sd.consumed,
                    removed_back: sd.removed_back,
                    pushed: sd
                        .pushed
                        .into_iter()
                        .map(|sv| sv.to_value(interner))
                        .collect(),
                },
            )
        })
        .collect();
    Ok((fields, sequences))
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
        let empty = serialize_entries(&HashMap::new(), &interner)?;
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
    pub async fn open(store: S, interner: Interner) -> Result<Option<Self>, JournalError> {
        Self::open_with_snapshot_interval(store, interner, Self::DEFAULT_SNAPSHOT_INTERVAL).await
    }

    /// Load with a custom snapshot interval.
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
                    self.tree = Self::deser_tree(my_ser)?;
                    return Ok(());
                }
                Err(actual) => {
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

    /// Access the underlying blob store.
    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }
}

// ── Internal helpers ────────────────────────────────────────────────

impl<S: BlobStore> BlobStoreJournal<S> {
    /// Load the full accumulated state for a given node index.
    ///
    /// If the node has a snapshot → single blob load (O(1)).
    /// Otherwise → walk up to the nearest snapshot ancestor, apply diffs forward.
    async fn load_state(&self, idx: usize) -> Result<HashMap<String, Value>, JournalError> {
        // Check hot node first.
        if let Some(ref hot) = self.hot
            && hot.idx == idx
        {
            return Ok(hot.state.clone());
        }

        // Walk up to the nearest node with a snapshot.
        let mut path = Vec::new();
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
                let (fields, sequences) = deserialize_turn_diff(&diff_bytes, &self.interner)?;

                // Apply field diffs: replace values.
                for (k, v) in fields {
                    state.insert(k, v);
                }

                // Apply sequence diffs with checksum preservation.
                for (k, diff) in sequences {
                    let (prev_items, prev_checksum) = match state.get(&k) {
                        Some(Value::Sequence(sc)) => {
                            (sc.origin().as_slice().to_vec(), sc.origin().checksum())
                        }
                        Some(Value::Deque(d)) => (d.as_slice().to_vec(), d.checksum()),
                        _ => {
                            // No existing sequence — apply diff to empty, use fresh deque for checksum.
                            let fresh = TrackedDeque::<Value>::new();
                            (Vec::new(), fresh.checksum())
                        }
                    };
                    let (new_items, new_checksum) =
                        diff.apply_with_checksum(prev_items, prev_checksum);
                    let new_deque = TrackedDeque::from_vec_with_checksum(new_items, new_checksum);
                    let sc = SequenceChain::from_stored(new_deque, Effect::pure());
                    state.insert(k, Value::sequence(sc));
                }
            }
        }

        Ok(state)
    }

    /// Persist the current hot node's state to the blob store.
    ///
    /// - Diffs: always stored if non-empty.
    /// - Snapshot: only at depth % snapshot_interval == 0 (root always qualifies).
    async fn persist_hot_node(&mut self) -> Result<(), JournalError> {
        let Some(ref hot) = self.hot else {
            return Ok(());
        };
        let idx = hot.idx;
        let depth = self.tree.nodes[idx].depth;

        // Store combined diffs if non-empty.
        if !hot.changed_fields.is_empty() || !hot.changed_sequences.is_empty() {
            let diff_bytes =
                serialize_turn_diff(&hot.changed_fields, &hot.changed_sequences, &self.interner)?;
            let diff_hash = self.store.put(diff_bytes).await;
            self.tree.nodes[idx].diff_hash = Some(diff_hash);
        }

        // Snapshot only at interval boundaries.
        if depth.is_multiple_of(self.snapshot_interval) {
            let snap_bytes = serialize_entries(&hot.state, &self.interner)?;
            let snap_hash = self.store.put(snap_bytes).await;
            self.tree.nodes[idx].snapshot_hash = Some(snap_hash);
        }

        Ok(())
    }

    /// Ensure the given node is the hot node.
    /// Persists the previous hot node if switching.
    async fn ensure_hot(&mut self, target_idx: usize) -> Result<(), JournalError> {
        if let Some(ref hot) = self.hot
            && hot.idx == target_idx
        {
            return Ok(());
        }
        self.persist_hot_node().await?;
        let state = self.load_state(target_idx).await?;
        self.hot = Some(HotNode {
            idx: target_idx,
            state,
            changed_fields: HashMap::new(),
            changed_sequences: HashMap::new(),
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

pub struct BlobEntryRef<'a, S: BlobStore> {
    _journal: &'a BlobStoreJournal<S>,
    state: HashMap<String, Value>,
}

pub struct BlobEntryMut<'a, S: BlobStore> {
    journal: &'a mut BlobStoreJournal<S>,
    idx: usize,
}

impl<S: BlobStore> EntryRef for BlobEntryRef<'_, S> {
    fn get(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }
}

impl<S: BlobStore> BlobEntryMut<'_, S> {
    /// UUID of the current node.
    pub fn uuid(&self) -> Uuid {
        self.journal.tree.nodes[self.idx].uuid
    }

    /// Depth of the current node.
    pub fn depth(&self) -> usize {
        self.journal.tree.nodes[self.idx].depth
    }
}

impl<S: BlobStore> EntryRef for BlobEntryMut<'_, S> {
    fn get(&self, key: &str) -> Option<&Value> {
        let hot = self.journal.hot.as_ref().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        hot.state.get(key)
    }
}

impl<S: BlobStore> EntryMut for BlobEntryMut<'_, S> {
    fn apply_field(&mut self, key: &str, _path: &[&str], value: Value) {
        let hot = self.journal.hot.as_mut().unwrap();
        debug_assert_eq!(hot.idx, self.idx);
        // For now, ignore path — just replace root.
        let key_owned = key.to_string();
        hot.state.insert(key_owned.clone(), value.clone());
        hot.changed_fields.insert(key_owned, value);
    }

    fn apply_diff(&mut self, key: &str, working: TrackedDeque<Value>) {
        let hot = self.journal.hot.as_mut().unwrap();
        debug_assert_eq!(hot.idx, self.idx);

        let existing = hot.state.get(key);

        // First turn: no existing sequence.
        if existing.is_none() {
            let items = working.into_vec();
            let diff = OwnedDequeDiff {
                consumed: 0,
                removed_back: 0,
                pushed: items.clone(),
            };
            let new_deque = TrackedDeque::from_vec(items);
            let sc = SequenceChain::from_stored(new_deque, Effect::pure());
            let key_owned = key.to_string();
            hot.state.insert(key_owned.clone(), Value::sequence(sc));
            hot.changed_sequences.insert(key_owned, diff);
            return;
        }

        // Subsequent turn: compute diff against origin.
        let existing_val = existing.unwrap();
        let origin = match existing_val {
            Value::Sequence(sc) => sc.origin().clone(),
            Value::Deque(d) => {
                let mut td =
                    TrackedDeque::from_vec_with_checksum(d.as_slice().to_vec(), d.checksum());
                td.checkpoint();
                td
            }
            _ => panic!("apply_diff called on non-sequence key {key:?}"),
        };
        let (squashed, diff) = working.into_diff(&origin);

        let new_sc = SequenceChain::from_stored(squashed, Effect::pure());
        let key_owned = key.to_string();
        hot.state.insert(key_owned.clone(), Value::sequence(new_sc));
        hot.changed_sequences.insert(key_owned, diff);
    }
}

impl<S: BlobStore> EntryLifecycle for BlobEntryMut<'_, S> {
    fn next(self) -> Self {
        let journal = self.journal;
        let idx = self.idx;

        // Persist current node synchronously — the blob store is sync in-memory for now.
        // For async blob stores, this would need to be async.
        // TODO: make this properly async when needed.
        futures::executor::block_on(async {
            journal
                .persist_hot_node()
                .await
                .expect("persist_hot_node failed in next()");
        });

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
            changed_fields: HashMap::new(),
            changed_sequences: HashMap::new(),
        });

        BlobEntryMut {
            journal,
            idx: new_idx,
        }
    }

    fn fork(self) -> Self {
        let journal = self.journal;
        let idx = self.idx;
        let parent_idx = journal.tree.nodes[idx].parent.expect("cannot fork root");

        // Persist current hot node before switching.
        futures::executor::block_on(async {
            journal
                .persist_hot_node()
                .await
                .expect("persist_hot_node failed in fork()");
        });

        // Load parent's state.
        let parent_state = futures::executor::block_on(async {
            journal
                .load_state(parent_idx)
                .await
                .expect("load_state failed in fork()")
        });

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
            changed_fields: HashMap::new(),
            changed_sequences: HashMap::new(),
        });

        BlobEntryMut {
            journal,
            idx: new_idx,
        }
    }
}

impl<S: BlobStore> Journal for BlobStoreJournal<S> {
    type Ref<'a>
        = BlobEntryRef<'a, S>
    where
        S: 'a;
    type Mut<'a>
        = BlobEntryMut<'a, S>
    where
        S: 'a;

    fn entry_ref(&self, id: Uuid) -> Self::Ref<'_> {
        let idx = self.tree.uuid_to_idx[&id];
        let state = futures::executor::block_on(async {
            self.load_state(idx)
                .await
                .expect("load_state failed in entry_ref()")
        });
        BlobEntryRef {
            _journal: self,
            state,
        }
    }

    fn entry_mut(&mut self, id: Uuid) -> Self::Mut<'_> {
        let idx = self.tree.uuid_to_idx[&id];
        futures::executor::block_on(async {
            self.ensure_hot(idx)
                .await
                .expect("ensure_hot failed in entry_mut()")
        });
        BlobEntryMut { journal: self, idx }
    }

    fn contains(&self, id: Uuid) -> bool {
        self.tree.uuid_to_idx.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemBlobStore;

    fn new_journal() -> (BlobStoreJournal<MemBlobStore>, Uuid) {
        futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            Interner::new(),
            1,
        ))
        .unwrap()
    }

    fn new_journal_with_interner(interner: Interner) -> (BlobStoreJournal<MemBlobStore>, Uuid) {
        futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            interner,
            1,
        ))
        .unwrap()
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Cluster 1: Basic CRUD
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn apply_and_get() {
        let (mut j, root) = new_journal();
        let mut e = j.entry_mut(root).next();
        e.apply_field("x", &[], Value::string("hello"));
        assert_eq!(*e.get("x").unwrap(), Value::string("hello"));
        assert!(e.get("y").is_none());
    }

    #[test]
    fn overwrite() {
        let interner = Interner::new();
        let (mut j, root) = new_journal_with_interner(interner.clone());
        let mut e = j.entry_mut(root).next();
        e.apply_field("x", &[], Value::string("first"));
        e.apply_field(
            "x",
            &[],
            Value::object(FxHashMap::from_iter([(
                interner.intern("v"),
                Value::Int(2),
            )])),
        );
        match e.get("x").unwrap() {
            Value::Object(_) => {} // OK — replaced with object
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn sequence_stores_as_sequence() {
        let (mut j, root) = new_journal();
        let mut e = j.entry_mut(root).next();
        let deque = TrackedDeque::from_vec(vec![Value::Int(1), Value::Int(2)]);
        e.apply_diff("q", deque);
        match e.get("q").unwrap() {
            Value::Sequence(sc) => {
                assert_eq!(sc.origin().as_slice(), &[Value::Int(1), Value::Int(2)]);
            }
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn get_prefers_current_turn() {
        let (mut j, root) = new_journal();
        let mut e = j.entry_mut(root).next();
        e.apply_field("x", &[], Value::Int(1));
        e.apply_field("x", &[], Value::Int(2));
        assert_eq!(*e.get("x").unwrap(), Value::Int(2));
    }

    #[test]
    fn multiple_keys_independent() {
        let (mut j, root) = new_journal();
        let mut e = j.entry_mut(root).next();
        e.apply_field("a", &[], Value::Int(1));
        e.apply_field("b", &[], Value::string("hello"));
        e.apply_field("c", &[], Value::Bool(true));
        assert_eq!(*e.get("a").unwrap(), Value::Int(1));
        assert_eq!(*e.get("b").unwrap(), Value::string("hello"));
        assert_eq!(*e.get("c").unwrap(), Value::Bool(true));
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Cluster 2: Tree structure (next/fork)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn next_inherits_parent_state() {
        let (mut j, root) = new_journal();
        {
            let mut e = j.entry_mut(root).next();
            e.apply_field("x", &[], Value::Int(1));
            e.apply_field("y", &[], Value::Int(2));
        }
        // Read back from root's child — need to find its uuid.
        // Use entry_ref on root to verify root is unchanged (depth 0, no data).
        let root_ref = j.entry_ref(root);
        assert!(root_ref.get("x").is_none()); // root has no data
    }

    #[test]
    fn next_child_sees_parent_values() {
        let (mut j, root) = new_journal();

        // Turn 1: write to child of root.
        let mut e1 = j.entry_mut(root).next();
        e1.apply_field("x", &[], Value::Int(1));
        e1.apply_field("y", &[], Value::Int(2));
        let n1 = e1.uuid();
        drop(e1);

        // Turn 2: child of n1.
        let mut e2 = j.entry_mut(n1).next();
        // Child sees parent's values.
        assert_eq!(*e2.get("x").unwrap(), Value::Int(1));
        assert_eq!(*e2.get("y").unwrap(), Value::Int(2));
        // Modify child.
        e2.apply_field("x", &[], Value::Int(99));
        assert_eq!(*e2.get("x").unwrap(), Value::Int(99));
        let n2 = e2.uuid();
        drop(e2);

        // Parent unchanged.
        let e1_ref = j.entry_ref(n1);
        assert_eq!(*e1_ref.get("x").unwrap(), Value::Int(1));

        // Child has override.
        let e2_ref = j.entry_ref(n2);
        assert_eq!(*e2_ref.get("x").unwrap(), Value::Int(99));
    }

    #[test]
    fn fork_sees_parent_state() {
        let (mut j, root) = new_journal();

        // Turn 1.
        let mut e1 = j.entry_mut(root).next();
        e1.apply_field("x", &[], Value::Int(1));
        let n1 = e1.uuid();
        drop(e1);

        // Turn 2.
        let mut e2 = j.entry_mut(n1).next();
        e2.apply_field("x", &[], Value::Int(2));
        let n2 = e2.uuid();
        drop(e2);

        // Fork from n2 → sibling of n2 (child of n1).
        let e3 = j.entry_mut(n2).fork();
        // Sees n1's state (x=1), not n2's (x=2).
        assert_eq!(*e3.get("x").unwrap(), Value::Int(1));
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Cluster 3: Persistence (snapshot/flush/open)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn flush_and_open_round_trip() {
        let interner = Interner::new();
        let (mut j, root) = new_journal_with_interner(interner.clone());

        // Write some data.
        let mut e = j.entry_mut(root).next();
        e.apply_field("x", &[], Value::Int(42));
        e.apply_field("msg", &[], Value::string("hello"));
        let n1 = e.uuid();
        drop(e);

        // Flush.
        futures::executor::block_on(j.flush_tree()).unwrap();

        // Open from same store.
        let store = std::mem::replace(j.store_mut(), MemBlobStore::new());
        let j2 = futures::executor::block_on(BlobStoreJournal::open_with_snapshot_interval(
            store, interner, 1,
        ))
        .unwrap()
        .expect("journal should exist");

        // Verify data survived.
        let e = j2.entry_ref(n1);
        assert_eq!(*e.get("x").unwrap(), Value::Int(42));
        assert_eq!(*e.get("msg").unwrap(), Value::string("hello"));
    }

    #[test]
    fn sequence_survives_persist_as_deque() {
        let interner = Interner::new();
        let (mut j, root) = new_journal_with_interner(interner.clone());

        // Write a sequence.
        let mut e = j.entry_mut(root).next();
        let deque = TrackedDeque::from_vec(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
        e.apply_diff("seq", deque);
        let n1 = e.uuid();
        drop(e);

        // Flush + reopen.
        futures::executor::block_on(j.flush_tree()).unwrap();
        let store = std::mem::replace(j.store_mut(), MemBlobStore::new());
        let j2 = futures::executor::block_on(BlobStoreJournal::open_with_snapshot_interval(
            store, interner, 1,
        ))
        .unwrap()
        .expect("journal should exist");

        // Sequence (self-modifying) is stored as Deque (Purity::Pure).
        // Only Purity::Pure values can be persisted to context.
        let e = j2.entry_ref(n1);
        match e.get("seq").unwrap() {
            Value::Deque(d) => {
                assert_eq!(
                    d.as_slice(),
                    &[Value::Int(10), Value::Int(20), Value::Int(30)]
                );
            }
            other => panic!("expected Deque (persisted from Sequence), got {other:?}"),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Cluster 4: Snapshot interval
    // ═══════════════════════════════════════════════════════════════════

    fn new_journal_interval(interval: usize) -> (BlobStoreJournal<MemBlobStore>, Uuid) {
        futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            Interner::new(),
            interval,
        ))
        .unwrap()
    }

    #[test]
    fn interval_only_snapshots_at_boundaries() {
        // Interval=4: snapshots at depth 0, 4, 8, ...
        let (mut j, root) = new_journal_interval(4);

        let mut cursor = root;
        let mut uuids = vec![root];
        for i in 1..=8 {
            let mut e = j.entry_mut(cursor).next();
            let uuid = e.uuid();
            uuids.push(uuid);
            e.apply_field(&format!("k{i}"), &[], Value::Int(i as i64));
            cursor = uuid;
            drop(e);
        }
        // Force persist of last node.
        {
            let e = j.entry_mut(cursor).next();
            uuids.push(e.uuid());
        }

        // depth 0 (root): snapshot. depth 1,2,3: no. depth 4: snapshot. etc.
        for (i, &uuid) in uuids.iter().enumerate() {
            let Some(&idx) = j.tree.uuid_to_idx.get(&uuid) else {
                continue;
            };
            let has_snap = j.tree.nodes[idx].snapshot_hash.is_some();
            let expected = i % 4 == 0;
            assert_eq!(
                has_snap, expected,
                "depth {i}: expected snapshot={expected}, got {has_snap}"
            );
        }
    }

    #[test]
    fn interval_reconstruction_correct() {
        // Interval=4: intermediate nodes have diffs only.
        let (mut j, root) = new_journal_interval(4);

        let mut cursor = root;
        for i in 1..=7 {
            let mut e = j.entry_mut(cursor).next();
            cursor = e.uuid();
            e.apply_field(&format!("k{i}"), &[], Value::Int(i as i64));
            drop(e);
        }

        // depth 7: no snapshot, must reconstruct from depth 4 + diffs at 5,6,7.
        let e = j.entry_ref(cursor);
        for i in 1..=7 {
            assert_eq!(
                *e.get(&format!("k{i}")).unwrap(),
                Value::Int(i as i64),
                "missing k{i} at depth 7"
            );
        }
    }

    #[test]
    fn interval_fork_reconstructs_parent() {
        // Interval=4: fork from depth 3 → parent at depth 2 has no snapshot.
        let (mut j, root) = new_journal_interval(4);

        let mut cursor = root;
        for i in 1..=3 {
            let mut e = j.entry_mut(cursor).next();
            cursor = e.uuid();
            e.apply_field(&format!("k{i}"), &[], Value::Int(i as i64));
            drop(e);
        }

        // Fork from depth 3 → sibling at depth 3, parent at depth 2 (no snapshot).
        let forked = j.entry_mut(cursor).fork();
        assert_eq!(forked.depth(), 3);
        assert_eq!(*forked.get("k1").unwrap(), Value::Int(1));
        assert_eq!(*forked.get("k2").unwrap(), Value::Int(2));
        assert!(forked.get("k3").is_none());
    }

    #[test]
    fn interval_deep_chain_no_data_loss() {
        // Interval=128 (default), 200 turns.
        let (mut j, root) = futures::executor::block_on(BlobStoreJournal::new(
            MemBlobStore::new(),
            Interner::new(),
        ))
        .unwrap();

        let mut cursor = root;
        for i in 1..=200 {
            let mut e = j.entry_mut(cursor).next();
            cursor = e.uuid();
            e.apply_field(&format!("k{i}"), &[], Value::Int(i as i64));
            drop(e);
        }

        // Leaf at depth 200 sees all values.
        let e = j.entry_ref(cursor);
        for i in 1..=200 {
            assert_eq!(
                *e.get(&format!("k{i}")).unwrap(),
                Value::Int(i as i64),
                "missing k{i} at depth 200"
            );
        }

        // Count snapshots: should be at depths 0, 128 = 2 snapshots only.
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

    // ═══════════════════════════════════════════════════════════════════
    //  SpyBlobStore — tracks blob sizes for size verification tests
    // ═══════════════════════════════════════════════════════════════════

    struct SpyBlobStore {
        inner: MemBlobStore,
        put_log: Vec<(BlobHash, usize)>,
    }

    impl SpyBlobStore {
        fn new() -> Self {
            Self {
                inner: MemBlobStore::new(),
                put_log: Vec::new(),
            }
        }

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

    // ═══════════════════════════════════════════════════════════════════
    //  Cluster 7: Size verification — diff must be O(change), not O(state)
    // ═══════════════════════════════════════════════════════════════════

    /// Append 1 item to a 1000-item sequence.
    /// Diff blob must be much smaller than snapshot.
    #[test]
    fn size_append_one_to_large_sequence() {
        let (mut j, root) = futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            SpyBlobStore::new(),
            Interner::new(),
            1,
        ))
        .unwrap();

        // Turn 1: initial 1000-item sequence.
        let mut e = j.entry_mut(root).next();
        let n1 = e.uuid();
        let items: Vec<Value> = (0..1000).map(|i| Value::Int(i)).collect();
        let deque = TrackedDeque::from_vec(items);
        e.apply_diff("big", deque);
        drop(e);

        // Turn 2: append 1 item.
        let mut e = j.entry_mut(n1).next();
        let n2 = e.uuid();
        let existing = match e.get("big").unwrap() {
            Value::Sequence(sc) => sc.origin().clone(),
            other => panic!("expected Sequence, got {other:?}"),
        };
        let mut working = existing;
        working.checkpoint();
        working.push(Value::Int(9999));
        e.apply_diff("big", working);
        drop(e);

        // Persist n2 by advancing.
        let _ = j.entry_mut(n2).next();

        let n2_idx = j.tree.uuid_to_idx[&n2];
        let n2_diff_hash = j.tree.nodes[n2_idx]
            .diff_hash
            .expect("n2 should have a diff blob");
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap_hash = j.tree.nodes[n1_idx]
            .snapshot_hash
            .expect("n1 should have a snapshot (interval=1)");

        let n2_diff_size = j.store.size_of(&n2_diff_hash).unwrap();
        let n1_snap_size = j.store.size_of(&n1_snap_hash).unwrap();

        assert!(
            n2_diff_size < n1_snap_size / 5,
            "diff blob ({n2_diff_size} bytes) should be much smaller than \
             snapshot ({n1_snap_size} bytes) — full state may have leaked into diff"
        );
    }

    /// Consume from front of a large sequence (no push).
    /// Diff is {consumed: 1, pushed: []} — near-zero payload.
    #[test]
    fn size_consume_only_from_large_sequence() {
        let (mut j, root) = futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            SpyBlobStore::new(),
            Interner::new(),
            1,
        ))
        .unwrap();

        // Turn 1: initial 500-item sequence.
        let mut e = j.entry_mut(root).next();
        let n1 = e.uuid();
        let items: Vec<Value> = (0..500).map(|i| Value::Int(i)).collect();
        let deque = TrackedDeque::from_vec(items);
        e.apply_diff("q", deque);
        drop(e);

        // Turn 2: consume 1, push nothing.
        let mut e = j.entry_mut(n1).next();
        let n2 = e.uuid();
        let existing = match e.get("q").unwrap() {
            Value::Sequence(sc) => sc.origin().clone(),
            other => panic!("expected Sequence, got {other:?}"),
        };
        let mut working = existing;
        working.checkpoint();
        working.consume(1);
        e.apply_diff("q", working);
        drop(e);

        // Persist.
        let _ = j.entry_mut(n2).next();

        let n2_idx = j.tree.uuid_to_idx[&n2];
        let n2_diff_hash = j.tree.nodes[n2_idx]
            .diff_hash
            .expect("n2 should have a diff blob");
        let n1_idx = j.tree.uuid_to_idx[&n1];
        let n1_snap_hash = j.tree.nodes[n1_idx]
            .snapshot_hash
            .expect("n1 should have a snapshot");

        let n2_diff_size = j.store.size_of(&n2_diff_hash).unwrap();
        let n1_snap_size = j.store.size_of(&n1_snap_hash).unwrap();

        assert!(
            n2_diff_size < n1_snap_size / 5,
            "consume-only diff ({n2_diff_size} bytes) should be tiny vs \
             snapshot ({n1_snap_size} bytes)"
        );
    }

    /// Many turns of small changes on a large sequence.
    /// Each diff blob must stay small — no O(n²) accumulation.
    #[test]
    fn size_diff_stays_small_across_many_turns() {
        let (mut j, root) = futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            SpyBlobStore::new(),
            Interner::new(),
            1,
        ))
        .unwrap();

        // Turn 1: initial 500-item sequence.
        let mut e = j.entry_mut(root).next();
        let mut cursor = e.uuid();
        let items: Vec<Value> = (0..500).map(|i| Value::Int(i)).collect();
        let deque = TrackedDeque::from_vec(items);
        e.apply_diff("q", deque);
        drop(e);

        let n1_idx = j.tree.uuid_to_idx[&cursor];
        // Force persist to get snapshot size reference.
        let mut e = j.entry_mut(cursor).next();
        cursor = e.uuid();
        // Just read, no change this turn.
        drop(e);

        let snap_size = j
            .store
            .size_of(&j.tree.nodes[n1_idx].snapshot_hash.unwrap())
            .unwrap();

        // Turns 3..12: append 1 item each.
        for i in 0..10 {
            let mut e = j.entry_mut(cursor).next();
            let new_cursor = e.uuid();
            let existing = match e.get("q").unwrap() {
                Value::Sequence(sc) => sc.origin().clone(),
                other => panic!("expected Sequence, got {other:?}"),
            };
            let mut working = existing;
            working.checkpoint();
            working.push(Value::Int(5000 + i));
            e.apply_diff("q", working);
            cursor = new_cursor;
            drop(e);

            // Persist.
            let next_e = j.entry_mut(cursor).next();
            let next_cursor = next_e.uuid();
            drop(next_e);

            // Check diff size.
            let idx = j.tree.uuid_to_idx[&cursor];
            if let Some(diff_hash) = j.tree.nodes[idx].diff_hash {
                let diff_size = j.store.size_of(&diff_hash).unwrap();
                assert!(
                    diff_size < snap_size / 3,
                    "turn {i}: diff ({diff_size} bytes) should be much smaller than \
                     snapshot ({snap_size} bytes)"
                );
            }
            cursor = next_cursor;
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Cluster 8: Roundtrip — flush → reopen → values must be exact
    // ═══════════════════════════════════════════════════════════════════

    fn get_deque_items(val: &Value) -> Vec<Value> {
        match val {
            Value::Sequence(sc) => sc.origin().as_slice().to_vec(),
            Value::Deque(d) => d.as_slice().to_vec(),
            other => panic!("expected Sequence or Deque, got {other:?}"),
        }
    }

    fn seq_working(e: &BlobEntryMut<'_, impl BlobStore>, key: &str) -> TrackedDeque<Value> {
        let existing = match e.get(key).unwrap() {
            Value::Sequence(sc) => sc.origin().clone(),
            Value::Deque(d) => {
                // After persist+reload, Sequence becomes Deque. Preserve checksum.
                TrackedDeque::from_vec_with_checksum(d.as_slice().to_vec(), d.checksum())
            }
            other => panic!("expected Sequence or Deque for '{key}', got {other:?}"),
        };
        let mut working = existing;
        working.checkpoint();
        working
    }

    #[test]
    fn roundtrip_multi_turn_consume_and_push() {
        let interner = Interner::new();
        let (mut j, root) = futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            interner.clone(),
            1,
        ))
        .unwrap();

        // Turn 1: [1, 2, 3]
        let mut e = j.entry_mut(root).next();
        let n1 = e.uuid();
        let deque = TrackedDeque::from_vec(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        e.apply_diff("seq", deque);
        drop(e);

        // Turn 2: consume 1, push 4 → [2, 3, 4]
        let mut e = j.entry_mut(n1).next();
        let n2 = e.uuid();
        let mut working = seq_working(&e, "seq");
        working.consume(1);
        working.push(Value::Int(4));
        e.apply_diff("seq", working);
        drop(e);

        // Turn 3: push 5, 6 → [2, 3, 4, 5, 6]
        let mut e = j.entry_mut(n2).next();
        let n3 = e.uuid();
        let mut working = seq_working(&e, "seq");
        working.push(Value::Int(5));
        working.push(Value::Int(6));
        e.apply_diff("seq", working);
        drop(e);

        // Flush + reopen.
        futures::executor::block_on(j.flush_tree()).unwrap();
        let store = std::mem::replace(j.store_mut(), MemBlobStore::new());
        let j2 = futures::executor::block_on(BlobStoreJournal::open_with_snapshot_interval(
            store, interner, 1,
        ))
        .unwrap()
        .expect("journal should exist");

        assert_eq!(
            get_deque_items(j2.entry_ref(n1).get("seq").unwrap()),
            vec![Value::Int(1), Value::Int(2), Value::Int(3)]
        );
        assert_eq!(
            get_deque_items(j2.entry_ref(n2).get("seq").unwrap()),
            vec![Value::Int(2), Value::Int(3), Value::Int(4)]
        );
        assert_eq!(
            get_deque_items(j2.entry_ref(n3).get("seq").unwrap()),
            vec![
                Value::Int(2),
                Value::Int(3),
                Value::Int(4),
                Value::Int(5),
                Value::Int(6)
            ]
        );
    }

    #[test]
    fn roundtrip_fork_independent_branches() {
        let interner = Interner::new();
        let (mut j, root) = futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            interner.clone(),
            1,
        ))
        .unwrap();

        // Turn 1: [1]
        let mut e = j.entry_mut(root).next();
        let n1 = e.uuid();
        let deque = TrackedDeque::from_vec(vec![Value::Int(1)]);
        e.apply_diff("seq", deque);
        drop(e);

        // Branch A: push 100 → [1, 100]
        let mut ea = j.entry_mut(n1).next();
        let branch_a = ea.uuid();
        let mut working = seq_working(&ea, "seq");
        working.push(Value::Int(100));
        ea.apply_diff("seq", working);
        drop(ea);

        // Branch B: push 200 → [1, 200] (from n1, not branch_a)
        let mut eb = j.entry_mut(n1).next();
        let branch_b = eb.uuid();
        let mut working = seq_working(&eb, "seq");
        working.push(Value::Int(200));
        eb.apply_diff("seq", working);
        drop(eb);

        // Flush + reopen.
        futures::executor::block_on(j.flush_tree()).unwrap();
        let store = std::mem::replace(j.store_mut(), MemBlobStore::new());
        let j2 = futures::executor::block_on(BlobStoreJournal::open_with_snapshot_interval(
            store, interner, 1,
        ))
        .unwrap()
        .expect("journal should exist");

        assert_eq!(
            get_deque_items(j2.entry_ref(n1).get("seq").unwrap()),
            vec![Value::Int(1)]
        );
        assert_eq!(
            get_deque_items(j2.entry_ref(branch_a).get("seq").unwrap()),
            vec![Value::Int(1), Value::Int(100)]
        );
        assert_eq!(
            get_deque_items(j2.entry_ref(branch_b).get("seq").unwrap()),
            vec![Value::Int(1), Value::Int(200)]
        );
    }

    #[test]
    fn roundtrip_mixed_types_coexist() {
        let interner = Interner::new();
        let a = interner.intern("a");
        let (mut j, root) = futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
            MemBlobStore::new(),
            interner.clone(),
            1,
        ))
        .unwrap();

        // Turn 1: counter=0, seq=[10], obj={a:1}
        let mut e = j.entry_mut(root).next();
        let n1 = e.uuid();
        e.apply_field("counter", &[], Value::Int(0));
        let deque = TrackedDeque::from_vec(vec![Value::Int(10)]);
        e.apply_diff("seq", deque);
        e.apply_field(
            "obj",
            &[],
            Value::object(FxHashMap::from_iter([(a, Value::Int(1))])),
        );
        drop(e);

        // Turn 2: counter=1, seq=[10,20], obj={a:2}
        let mut e = j.entry_mut(n1).next();
        let n2 = e.uuid();
        e.apply_field("counter", &[], Value::Int(1));
        let mut working = seq_working(&e, "seq");
        working.push(Value::Int(20));
        e.apply_diff("seq", working);
        e.apply_field(
            "obj",
            &[],
            Value::object(FxHashMap::from_iter([(a, Value::Int(2))])),
        );
        drop(e);

        // Flush + reopen.
        futures::executor::block_on(j.flush_tree()).unwrap();
        let store = std::mem::replace(j.store_mut(), MemBlobStore::new());
        let j2 = futures::executor::block_on(BlobStoreJournal::open_with_snapshot_interval(
            store,
            interner.clone(),
            1,
        ))
        .unwrap()
        .expect("journal should exist");

        // Turn 1.
        let e1 = j2.entry_ref(n1);
        assert_eq!(*e1.get("counter").unwrap(), Value::Int(0));
        assert_eq!(
            get_deque_items(e1.get("seq").unwrap()),
            vec![Value::Int(10)]
        );
        match e1.get("obj").unwrap() {
            Value::Object(o) => assert_eq!(*o.get(&a).unwrap(), Value::Int(1)),
            other => panic!("expected Object, got {other:?}"),
        }

        // Turn 2.
        let e2 = j2.entry_ref(n2);
        assert_eq!(*e2.get("counter").unwrap(), Value::Int(1));
        assert_eq!(
            get_deque_items(e2.get("seq").unwrap()),
            vec![Value::Int(10), Value::Int(20)]
        );
        match e2.get("obj").unwrap() {
            Value::Object(o) => assert_eq!(*o.get(&a).unwrap(), Value::Int(2)),
            other => panic!("expected Object, got {other:?}"),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Cluster 4 continued: Snapshot interval
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn interval_blob_count_much_lower() {
        // Compare blob count: interval=1 vs interval=4 over 8 turns.
        let count_with_interval = |interval: usize| {
            let (mut j, root) =
                futures::executor::block_on(BlobStoreJournal::with_snapshot_interval(
                    MemBlobStore::new(),
                    Interner::new(),
                    interval,
                ))
                .unwrap();
            let mut cursor = root;
            for i in 1..=8 {
                let mut e = j.entry_mut(cursor).next();
                cursor = e.uuid();
                e.apply_field(&format!("k{i}"), &[], Value::Int(i as i64));
                drop(e);
            }
            futures::executor::block_on(j.flush_tree()).unwrap();
            j.store().blob_count()
        };

        let count_1 = count_with_interval(1);
        let count_4 = count_with_interval(4);

        assert!(
            count_4 < count_1,
            "interval=4 ({count_4}) should use fewer blobs than interval=1 ({count_1})"
        );
    }
}
