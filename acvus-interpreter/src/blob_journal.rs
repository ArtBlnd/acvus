//! Blob-backed journal — tree-structured context storage with snapshot + diff.
//!
//! Simplified from the orchestration version:
//! - No PatchDiff (recursive Rec diffs) — fields are replaced wholesale.
//! - No TypedValue / SerTy — stores `Value` directly via `SerValue` mirror.
//! - Field diffs and sequence diffs combined into one blob per node.

use std::collections::HashMap;

use acvus_mir::ty::Effect;
use acvus_utils::{Interner, OwnedDequeDiff, TrackedDeque};
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
    Deque(Vec<SerValue>),
    Variant { tag: String, payload: Option<Box<SerValue>> },
    Range { start: i64, end: i64, inclusive: bool },
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
            Value::List(items) => {
                SerValue::List(items.iter().map(|v| SerValue::from_value(v, interner)).collect())
            }
            Value::Object(obj) => {
                SerValue::Object(
                    obj.iter()
                        .map(|(k, v)| {
                            (interner.resolve(*k).to_string(), SerValue::from_value(v, interner))
                        })
                        .collect(),
                )
            }
            Value::Tuple(elems) => {
                SerValue::Tuple(elems.iter().map(|v| SerValue::from_value(v, interner)).collect())
            }
            Value::Deque(d) => {
                SerValue::Deque(
                    d.as_slice().iter().map(|v| SerValue::from_value(v, interner)).collect(),
                )
            }
            Value::Variant { tag, payload } => SerValue::Variant {
                tag: interner.resolve(*tag).to_string(),
                payload: payload.as_ref().map(|p| Box::new(SerValue::from_value(p, interner))),
            },
            Value::Range(r) => SerValue::Range {
                start: r.start,
                end: r.end,
                inclusive: r.inclusive,
            },
            Value::Sequence(sc) => {
                // Serialize the origin deque items.
                SerValue::Deque(
                    sc.origin()
                        .as_slice()
                        .iter()
                        .map(|v| SerValue::from_value(v, interner))
                        .collect(),
                )
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
            SerValue::Deque(items) => {
                let deque =
                    TrackedDeque::from_vec(items.into_iter().map(|v| v.to_value(interner)).collect());
                Value::deque(deque)
            }
            SerValue::Variant { tag, payload } => {
                let astr_tag = interner.intern(&tag);
                let payload_val = payload.map(|p| p.to_value(interner));
                Value::variant(astr_tag, payload_val)
            }
            SerValue::Range { start, end, inclusive } => Value::range(start, end, inclusive),
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
        if let Some(ref hot) = self.hot {
            if hot.idx == idx {
                return Ok(hot.state.clone());
            }
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
                let (fields, sequences) =
                    deserialize_turn_diff(&diff_bytes, &self.interner)?;

                // Apply field diffs: replace values.
                for (k, v) in fields {
                    state.insert(k, v);
                }

                // Apply sequence diffs.
                for (k, diff) in sequences {
                    let prev_items = state
                        .get(&k)
                        .map(|v| match v {
                            Value::Sequence(sc) => sc.origin().as_slice().to_vec(),
                            Value::Deque(d) => d.as_slice().to_vec(),
                            _ => Vec::new(),
                        })
                        .unwrap_or_default();
                    let new_items = diff.apply(prev_items);
                    let new_deque = TrackedDeque::from_vec(new_items);
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
            let diff_bytes = serialize_turn_diff(
                &hot.changed_fields,
                &hot.changed_sequences,
                &self.interner,
            )?;
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
                let mut td = TrackedDeque::from_vec(d.as_slice().to_vec());
                td.checkpoint();
                td
            }
            _ => panic!("apply_diff called on non-sequence key {key:?}"),
        };
        let (squashed, diff) = working.into_diff(&origin);

        let new_sc = SequenceChain::from_stored(squashed, Effect::pure());
        let key_owned = key.to_string();
        hot.state
            .insert(key_owned.clone(), Value::sequence(new_sc));
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
            journal.persist_hot_node().await.expect("persist_hot_node failed in next()");
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
            journal.persist_hot_node().await.expect("persist_hot_node failed in fork()");
        });

        // Load parent's state.
        let parent_state = futures::executor::block_on(async {
            journal.load_state(parent_idx).await.expect("load_state failed in fork()")
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
            self.load_state(idx).await.expect("load_state failed in entry_ref()")
        });
        BlobEntryRef {
            _journal: self,
            state,
        }
    }

    fn entry_mut(&mut self, id: Uuid) -> Self::Mut<'_> {
        let idx = self.tree.uuid_to_idx[&id];
        futures::executor::block_on(async {
            self.ensure_hot(idx).await.expect("ensure_hot failed in entry_mut()")
        });
        BlobEntryMut { journal: self, idx }
    }

    fn contains(&self, id: Uuid) -> bool {
        self.tree.uuid_to_idx.contains_key(&id)
    }
}
