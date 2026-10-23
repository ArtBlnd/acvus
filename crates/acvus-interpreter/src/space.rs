//! A space (RFC-0033): content-addressed, append-only nodes, and a head per
//! identity moved by compare-and-exchange. An extension value's log is a
//! chain of op nodes from a checkpoint; a value of a language shape is a
//! state node. A nested extension value has its own chain, named from its
//! parent's node by head, so a log is structural.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use acvus_extern::{Holding, NodeHash, Owned, SpaceError, SpaceResult};
use acvus_mir::ser_ty::SerTy;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

use crate::host::{Codec, Storage, StorageError};
use crate::port::Held;
use crate::layout::{self, Nested, ZeroWidth};
use crate::runtime::AcvusRuntime;
use crate::value::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NodeKind {
    /// The whole value as canonical bytes.
    State = 0,
    /// One op on the parent's value.
    Op = 1,
}

/// One node of a log: its kind, its parent (`None` at the first state),
/// and its bytes. Its address is the hash of exactly this.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Node {
    kind: NodeKind,
    parent: Option<NodeHash>,
    bytes: Vec<u8>,
}

impl Node {
    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    /// The node this one follows; `None` at a log's first state.
    pub fn parent(&self) -> Option<NodeHash> {
        self.parent
    }

    /// The payload: a state's canonical bytes, or one op's.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The node's bytes: kind, a parent flag and the parent, then the
    /// payload. The address is the BLAKE3 hash of exactly these.
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + NodeHash::LEN + self.bytes.len());
        out.push(self.kind as u8);
        match self.parent {
            Some(p) => {
                out.push(1);
                out.extend_from_slice(&p.0);
            }
            None => out.push(0),
        }
        out.extend_from_slice(&self.bytes);
        out
    }

    fn from_bytes(bytes: &[u8]) -> SpaceResult<Self> {
        let (&kind, rest) = bytes
            .split_first()
            .ok_or_else(|| SpaceError::new("empty node"))?;
        let kind = match kind {
            0 => NodeKind::State,
            1 => NodeKind::Op,
            other => return Err(SpaceError::new(format!("node kind {other}"))),
        };
        let (&flag, rest) = rest
            .split_first()
            .ok_or_else(|| SpaceError::new("truncated node"))?;
        let (parent, rest) = match flag {
            0 => (None, rest),
            1 => {
                let (hash, rest) = rest
                    .split_first_chunk::<{ NodeHash::LEN }>()
                    .ok_or_else(|| SpaceError::new("truncated node parent"))?;
                (Some(NodeHash(*hash)), rest)
            }
            other => return Err(SpaceError::new(format!("node parent flag {other}"))),
        };
        Ok(Node {
            kind,
            parent,
            bytes: rest.to_vec(),
        })
    }

    fn address(&self) -> NodeHash {
        NodeHash(*blake3::hash(&self.to_bytes()).as_bytes())
    }
}

/// How a space keeps an extension value that changes in place: at a
/// commit that changed it, what follows the head it was loaded at.
///
/// A commit whose nested value's head moved always ends in a state node,
/// whatever the mode answers: the parent's state is what names its
/// children's heads, and an op does not.
pub trait Mode: Send + Sync {
    fn record(&self, commit: &Commit) -> Record;
}

/// What a commit brings to its mode.
pub struct Commit {
    /// The ops recorded since the value was loaded.
    pub ops: usize,
    /// The op nodes between the head and the last state node before it.
    pub ops_since_state: usize,
}

/// What a commit writes after the head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Record {
    /// One state node; the ops are not kept.
    State,
    /// One op node per op, then a state node where `then_state`.
    Ops { then_state: bool },
}

/// Every commit is a state node; the ops are not kept.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plain;

impl Mode for Plain {
    fn record(&self, _: &Commit) -> Record {
        Record::State
    }
}

/// A commit appends its ops, and ends in a state node once
/// `checkpoint_every` ops have accrued since the last state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Log {
    pub checkpoint_every: usize,
}

impl Mode for Log {
    fn record(&self, commit: &Commit) -> Record {
        Record::Ops {
            then_state: commit.ops_since_state + commit.ops >= self.checkpoint_every,
        }
    }
}

/// The head of one identity: the node and the type its value has, written
/// apart from any interner, so a space opened by one program is read by
/// another.
#[derive(Clone, PartialEq, Debug)]
pub struct Head {
    pub hash: NodeHash,
    pub ty: SerTy,
}

/// What a backing store implements: nodes by address, heads by identity,
/// and the compare-and-exchange that moves a head.
pub trait Store: Send + Sync {
    fn put(&self, hash: NodeHash, bytes: &[u8]) -> SpaceResult<()>;
    fn get(&self, hash: NodeHash) -> SpaceResult<Option<Vec<u8>>>;
    fn head(&self, id: &str) -> SpaceResult<Option<Head>>;
    fn identities(&self) -> SpaceResult<Vec<String>>;
    fn node_count(&self) -> SpaceResult<usize>;
    /// Move `id`'s head from `expected` to `new`; on a head that is not
    /// `expected`, the current head comes back and nothing moves.
    fn cmpxchg(
        &self,
        id: &str,
        expected: Option<NodeHash>,
        new: Head,
    ) -> SpaceResult<Result<(), Option<NodeHash>>>;
}

#[derive(Default)]
pub struct MemoryStore {
    nodes: Mutex<HashMap<NodeHash, Vec<u8>>>,
    heads: Mutex<HashMap<String, Head>>,
}

impl Store for MemoryStore {
    fn put(&self, hash: NodeHash, bytes: &[u8]) -> SpaceResult<()> {
        self.nodes
            .lock()
            .expect("nodes")
            .entry(hash)
            .or_insert_with(|| bytes.to_vec());
        Ok(())
    }

    fn get(&self, hash: NodeHash) -> SpaceResult<Option<Vec<u8>>> {
        Ok(self.nodes.lock().expect("nodes").get(&hash).cloned())
    }

    fn head(&self, id: &str) -> SpaceResult<Option<Head>> {
        Ok(self.heads.lock().expect("heads").get(id).cloned())
    }

    fn identities(&self) -> SpaceResult<Vec<String>> {
        let mut ids: Vec<String> = self.heads.lock().expect("heads").keys().cloned().collect();
        ids.sort();
        Ok(ids)
    }

    fn node_count(&self) -> SpaceResult<usize> {
        Ok(self.nodes.lock().expect("nodes").len())
    }

    fn cmpxchg(
        &self,
        id: &str,
        expected: Option<NodeHash>,
        new: Head,
    ) -> SpaceResult<Result<(), Option<NodeHash>>> {
        let mut heads = self.heads.lock().expect("heads");
        let current = heads.get(id).map(|h| h.hash);
        if current != expected {
            return Ok(Err(current));
        }
        heads.insert(id.to_string(), new);
        Ok(Ok(()))
    }
}

/// A store in a directory: `nodes/<hex>` holds a node's bytes,
/// `heads/<id>.json` holds `{ "head": <hex>, "ty": <SerTy> }`. Heads move
/// under one process-wide lock per store; two processes on one directory
/// are not coordinated.
pub struct DirStore {
    root: std::path::PathBuf,
    lock: Mutex<()>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HeadFile {
    head: String,
    ty: acvus_mir::ser_ty::SerTy,
}

impl DirStore {
    pub fn open(root: impl Into<std::path::PathBuf>) -> SpaceResult<Self> {
        let root = root.into();
        for sub in ["nodes", "heads"] {
            std::fs::create_dir_all(root.join(sub))
                .map_err(|e| SpaceError::new(format!("{}: {e}", root.display())))?;
        }
        Ok(Self {
            root,
            lock: Mutex::new(()),
        })
    }

    fn io(&self, e: std::io::Error) -> SpaceError {
        SpaceError::new(format!("{}: {e}", self.root.display()))
    }

    fn head_path(&self, id: &str) -> std::path::PathBuf {
        self.root.join("heads").join(format!("{id}.json"))
    }

    fn read_head(&self, id: &str) -> SpaceResult<Option<Head>> {
        let text = match std::fs::read_to_string(self.head_path(id)) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(self.io(e)),
        };
        let file: HeadFile =
            serde_json::from_str(&text).map_err(|e| SpaceError::new(format!("{id}: {e}")))?;
        Ok(Some(Head {
            hash: unhex(&file.head)?,
            ty: file.ty,
        }))
    }
}

impl Store for DirStore {
    fn put(&self, hash: NodeHash, bytes: &[u8]) -> SpaceResult<()> {
        let path = self.root.join("nodes").join(hex(&hash));
        if path.exists() {
            return Ok(());
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes).map_err(|e| self.io(e))?;
        std::fs::rename(&tmp, &path).map_err(|e| self.io(e))
    }

    fn get(&self, hash: NodeHash) -> SpaceResult<Option<Vec<u8>>> {
        match std::fs::read(self.root.join("nodes").join(hex(&hash))) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(self.io(e)),
        }
    }

    fn head(&self, id: &str) -> SpaceResult<Option<Head>> {
        let _guard = self.lock.lock().expect("store lock");
        self.read_head(id)
    }

    fn identities(&self) -> SpaceResult<Vec<String>> {
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(self.root.join("heads")).map_err(|e| self.io(e))? {
            let entry = entry.map_err(|e| self.io(e))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_suffix(".json") {
                ids.push(id.to_string());
            }
        }
        ids.sort();
        Ok(ids)
    }

    fn node_count(&self) -> SpaceResult<usize> {
        Ok(std::fs::read_dir(self.root.join("nodes"))
            .map_err(|e| self.io(e))?
            .count())
    }

    fn cmpxchg(
        &self,
        id: &str,
        expected: Option<NodeHash>,
        new: Head,
    ) -> SpaceResult<Result<(), Option<NodeHash>>> {
        let _guard = self.lock.lock().expect("store lock");
        let current = self.read_head(id)?.map(|h| h.hash);
        if current != expected {
            return Ok(Err(current));
        }
        let file = HeadFile {
            head: hex(&new.hash),
            ty: new.ty,
        };
        let text =
            serde_json::to_string(&file).map_err(|e| SpaceError::new(format!("{id}: {e}")))?;
        let path = self.head_path(id);
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| self.io(e))?;
        std::fs::rename(&tmp, &path).map_err(|e| self.io(e))?;
        Ok(Ok(()))
    }
}

pub struct Space {
    mode: Box<dyn Mode>,
    store: Box<dyn Store>,
}

impl Space {
    /// The zero-width elements one load may read in all, its nested values'
    /// included (`ZeroWidth`, RFC-0033 rule 2). A unit decodes to one
    /// 16-byte word, so the bound's units hold 16 MiB; an element that is a
    /// tuple or object of units holds its box besides. A commit is not held
    /// to it: a value holding more commits, and its load is refused.
    pub const ZERO_WIDTH_ELEMENTS: usize = 1 << 20;

    pub fn new<M>(mode: M) -> Self
    where
        M: Mode + 'static,
    {
        Self::over(mode, Box::new(MemoryStore::default()))
    }

    pub fn over<M>(mode: M, store: Box<dyn Store>) -> Self
    where
        M: Mode + 'static,
    {
        Self {
            mode: Box::new(mode),
            store,
        }
    }

    pub fn mode(&self) -> &dyn Mode {
        self.mode.as_ref()
    }

    pub fn head(&self, id: &str) -> Option<NodeHash> {
        self.store.head(id).ok().flatten().map(|h| h.hash)
    }

    pub fn identities(&self, interner: &Interner) -> SpaceResult<Vec<Identity>> {
        let mut out = Vec::new();
        for id in self.store.identities()? {
            if let Some(head) = self.store.head(&id)? {
                out.push(Identity {
                    id,
                    ty: head.ty.to_ty(interner),
                });
            }
        }
        Ok(out)
    }

    pub fn node_count(&self) -> usize {
        self.store.node_count().expect("node count")
    }

    fn put(&self, node: Node) -> SpaceResult<NodeHash> {
        let hash = node.address();
        self.store.put(hash, &node.to_bytes())?;
        Ok(hash)
    }

    /// The node at `hash`: a head, or any node a head's chain reaches
    /// through its parents. The store's bytes are refused unless they hash
    /// to `hash`, so short of a BLAKE3 collision no node names itself or a
    /// descendant as its parent or a nested head, and a chain and a nesting
    /// end.
    pub fn get(&self, hash: NodeHash) -> SpaceResult<Node> {
        let bytes = self
            .store
            .get(hash)?
            .ok_or_else(|| SpaceError::new(format!("no node {}", hex(&hash))))?;
        if blake3::hash(&bytes).as_bytes() != &hash.0 {
            return Err(SpaceError::new(format!(
                "the node stored at {} hashes elsewhere",
                hex(&hash)
            )));
        }
        Node::from_bytes(&bytes)
    }

    pub fn cmpxchg(
        &self,
        id: &str,
        expected: Option<NodeHash>,
        new: Head,
    ) -> SpaceResult<Result<(), Option<NodeHash>>> {
        self.store.cmpxchg(id, expected, new)
    }
}

/// One identity a space holds, at the type its head records.
#[derive(Clone, PartialEq, Debug)]
pub struct Identity {
    pub id: String,
    pub ty: Ty,
}

/// A value word loaded from or committed to the space: the runtime's and its
/// tooling's (RFC-0090 rule 6). A host reads and writes a context through a
/// page over a `SpaceStorage`.
macro_rules! space_values {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl Space {
            /// The value of `id` as its head names it.
            $v fn load(&self, rt: &AcvusRuntime, id: &str, ty: &Ty) -> SpaceResult<Option<Value>> {
                match self.head(id) {
                    Some(head) => self
                        .load_at(rt, ty, head, &ZeroWidth::allowing(Self::ZERO_WIDTH_ELEMENTS))
                        .map(Some),
                    None => Ok(None),
                }
            }

            /// Commit `value` as the new head of `id`. An extension value carries
            /// the head it was loaded at, and the commit is refused when the head
            /// has moved since; a value that carries none, or one of a language
            /// shape, replaces whatever the head is.
            $v fn commit(
                &self,
                rt: &AcvusRuntime,
                id: &str,
                ty: &Ty,
                value: &mut Value,
            ) -> SpaceResult<NodeHash> {
                let loaded_at = match layout::extension(rt, ty) {
                    Ok((hooks, _)) => (hooks.head)(rt, value),
                    Err(_) => None,
                };
                let expected = loaded_at.or_else(|| self.head(id));
                let new = self.commit_value(rt, ty, value)?;
                let head = Head {
                    hash: new,
                    ty: ty.to_ser(&rt.shared.interner),
                };
                self.cmpxchg(id, expected, head)?.map_err(|current| {
                    SpaceError::new(format!(
                        "@{id}: head moved to {} since it was loaded",
                        current.map_or("nothing".to_string(), |h| hex(&h))
                    ))
                })?;
                Ok(new)
            }
        }
    };
}
tooling_vis!(space_values);

impl Space {
    fn load_stored(&self, rt: &AcvusRuntime, id: &str) -> SpaceResult<Option<Stored>> {
        let Some(head) = self.store.head(id)? else {
            return Ok(None);
        };
        let ty = head.ty.to_ty(&rt.shared.interner);
        let zero_width = ZeroWidth::allowing(Self::ZERO_WIDTH_ELEMENTS);
        let value = self.load_at(rt, &ty, head.hash, &zero_width)?;
        Ok(Some(Stored { ty, value }))
    }

    /// The value `head` names. Every node it reads is read whole: a
    /// language shape's state by `layout::decode_all`, and an extension's
    /// state and each op refused when its hooks leave a byte unread. What
    /// the hooks read is the type's own promise (RFC-0080 rule 4); the
    /// space bounds the zero-width parts they read through it.
    fn load_at(
        &self,
        rt: &AcvusRuntime,
        ty: &Ty,
        head: NodeHash,
        zero_width: &ZeroWidth,
    ) -> SpaceResult<Value> {
        if !matches!(ty, Ty::UserDefined { .. }) {
            let node = self.get(head)?;
            if node.kind != NodeKind::State {
                return Err(SpaceError::new(
                    "a value of a language shape is a state node",
                ));
            }
            return layout::decode_all(rt, self, ty, &node.bytes, zero_width);
        }
        let (hooks, args) = layout::extension(rt, ty)?;
        let mut ops: Vec<Node> = Vec::new();
        let mut at = head;
        let state = loop {
            let node = self.get(at)?;
            match node.kind {
                NodeKind::State => break node,
                NodeKind::Op => {
                    at = node
                        .parent
                        .ok_or_else(|| SpaceError::new("an op node has a parent"))?;
                    ops.push(node);
                }
            }
        };
        let decode =
            |t: &Ty, input: &mut &[u8]| layout::decode_part(rt, self, t, input, zero_width);
        let shown = ty.display(&rt.shared.interner);
        let mut input = state.bytes.as_slice();
        let value = (hooks.decode_state)(rt, &args, &decode, &mut input)?;
        // In a holder, so a value refused after its state is read is released.
        // SAFETY: `decode_state` made the word, and no other holder owns it.
        let mut value: Owned<AcvusRuntime> = unsafe { Owned::from_value(Holding::new(), value) };
        layout::all_read(input, &format_args!("the state of {shown}"))?;
        for op in ops.iter().rev() {
            let mut input = op.bytes.as_slice();
            // SAFETY: `apply_op` edits the payload in place through the
            // type's hooks and writes no word into the holder.
            (hooks.apply_op)(rt, unsafe { value.value_mut(Holding::new()) }, &args, &decode, &mut input)?;
            layout::all_read(input, &format_args!("an op of {shown}"))?;
        }
        // SAFETY: `set_head` writes the head into the payload, not a word
        // into the holder.
        (hooks.set_head)(rt, unsafe { value.value_mut(Holding::new()) }, head);
        // SAFETY: the word moves to the caller, which owns it from then on.
        Ok(value.into_value(unsafe { Holding::new() }))
    }

    /// The new head for `value`, chained onto the head it carries.
    fn commit_value(&self, rt: &AcvusRuntime, ty: &Ty, value: &mut Value) -> SpaceResult<NodeHash> {
        if !matches!(ty, Ty::UserDefined { .. }) {
            let mut bytes = Vec::new();
            layout::encode(rt, self, ty, value, &mut bytes)?;
            return self.put(Node {
                kind: NodeKind::State,
                parent: None,
                bytes,
            });
        }
        let (hooks, args) = layout::extension(rt, ty)?;
        // Children first: their heads are what the parent's bytes name,
        // and a child whose head moved is a change of the parent.
        let mut children_moved = false;
        if args.iter().any(layout::holds_extension) {
            (hooks.children)(rt, value, &args, &mut |child_ty, child| {
                // SAFETY: `commit_nested` writes no word into the child.
                self.commit_nested(rt, child_ty, unsafe { child.value_mut(acvus_extern::Holding::new()) }, &mut children_moved)
            })?;
        }
        let encode = |t: &Ty, v: &Value, out: &mut Vec<u8>| layout::encode(rt, self, t, v, out);
        let ops = (hooks.take_ops)(rt, value, &args, &encode)?;
        let state = |parent: Option<NodeHash>| -> SpaceResult<Node> {
            let mut bytes = Vec::new();
            (hooks.encode_state)(rt, value, &args, &encode, &mut bytes)?;
            Ok(Node {
                kind: NodeKind::State,
                parent,
                bytes,
            })
        };
        let Some(mut head) = (hooks.head)(rt, value) else {
            // Never held: the state as it stands is the first node.
            let head = self.put(state(None)?)?;
            (hooks.set_head)(rt, value, head);
            return Ok(head);
        };
        if ops.is_empty() && !children_moved {
            return Ok(head);
        }
        let commit = Commit {
            ops: ops.len(),
            ops_since_state: self.ops_since_state(head)?,
        };
        match self.mode.record(&commit) {
            Record::State => {
                head = self.put(state(Some(head))?)?;
            }
            Record::Ops { then_state } => {
                for op in ops {
                    head = self.put(Node {
                        kind: NodeKind::Op,
                        parent: Some(head),
                        bytes: op,
                    })?;
                }
                if children_moved || then_state {
                    head = self.put(state(Some(head))?)?;
                }
            }
        }
        (hooks.set_head)(rt, value, head);
        Ok(head)
    }

    /// Commit every extension value inside `value`, walking the language
    /// shapes by type down to each; `moved` records whether any head moved.
    /// It edits an extension's payload in place through the type's hooks and
    /// writes no word into `value` or any part of it, which is what each
    /// `value_mut` below relies on.
    fn commit_nested(
        &self,
        rt: &AcvusRuntime,
        ty: &Ty,
        value: &mut Value,
        moved: &mut bool,
    ) -> SpaceResult<()> {
        match ty {
            Ty::UserDefined { .. } => {
                let hooks = layout::extension(rt, ty)?.0;
                let before = (hooks.head)(rt, value);
                let head = self.commit_value(rt, ty, value)?;
                *moved |= before != Some(head);
                Ok(())
            }
            Ty::Array(elem, _) => {
                // SAFETY (each composite): the type is the runtime's witness
                // of the value's shape.
                for v in unsafe { value.as_array_mut() }.0.iter_mut() {
                    // SAFETY: `commit_nested` writes no word into the part.
                    self.commit_nested(rt, elem, unsafe { v.value_mut(acvus_extern::Holding::new()) }, moved)?;
                }
                Ok(())
            }
            Ty::Tuple(elems) => {
                for (v, t) in unsafe { value.as_tuple_mut() }.0.iter_mut().zip(elems) {
                    // SAFETY: as the array's.
                    self.commit_nested(rt, t, unsafe { v.value_mut(acvus_extern::Holding::new()) }, moved)?;
                }
                Ok(())
            }
            Ty::Object(fields) => {
                let laid = layout::sorted_fields(&rt.shared.interner, fields);
                let types: Vec<Ty> = laid.iter().map(|(_, t)| (*t).clone()).collect();
                let values = unsafe { value.as_object_mut() };
                for (t, v) in types.iter().zip(values.iter_mut()) {
                    // SAFETY: as the array's.
                    self.commit_nested(rt, t, unsafe { v.value_mut(acvus_extern::Holding::new()) }, moved)?;
                }
                Ok(())
            }
            Ty::Option(inner) => {
                if let Some(v) = value.option_payload_mut() {
                    self.commit_nested(rt, inner, v, moved)?;
                }
                Ok(())
            }
            Ty::Result(ok, err) => {
                let variant = unsafe { value.as_variant_mut() };
                // SAFETY: the same witness — a variant's first register is its tag.
                let tag = unsafe { variant.tag().as_tag() };
                let (_, held) = layout::result_side(rt, tag, ok, err)?;
                // SAFETY: as the array's.
                self.commit_nested(rt, held, unsafe { variant.payload_mut().value_mut(acvus_extern::Holding::new()) }, moved)
            }
            Ty::Enum { variants, .. } => {
                let variant = unsafe { value.as_variant_mut() };
                // SAFETY: the same witness — a variant's first register is its tag.
                let tag = unsafe { variant.tag().as_tag() };
                if let Some(Some(t)) = variants.get(&tag) {
                    let t = t.as_ref().clone();
                    // SAFETY: as the array's.
                    self.commit_nested(rt, &t, unsafe { variant.payload_mut().value_mut(acvus_extern::Holding::new()) }, moved)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn ops_since_state(&self, mut at: NodeHash) -> SpaceResult<usize> {
        let mut n = 0;
        loop {
            let node = self.get(at)?;
            match (node.kind, node.parent) {
                (NodeKind::State, _) => return Ok(n),
                (NodeKind::Op, Some(parent)) => {
                    n += 1;
                    at = parent;
                }
                (NodeKind::Op, None) => return Err(SpaceError::new("an op node has a parent")),
            }
        }
    }
}

impl Nested for Space {
    fn commit(&self, rt: &AcvusRuntime, ty: &Ty, value: &Value) -> SpaceResult<NodeHash> {
        // A nested value was committed by `children` before its parent
        // was encoded; its head is what the parent names.
        let (hooks, _) = layout::extension(rt, ty)?;
        (hooks.head)(rt, value)
            .ok_or_else(|| SpaceError::new("a nested value is committed before its parent"))
    }

    fn load(
        &self,
        rt: &AcvusRuntime,
        ty: &Ty,
        head: NodeHash,
        zero_width: &ZeroWidth,
    ) -> SpaceResult<Value> {
        self.load_at(rt, ty, head, zero_width)
    }
}

pub fn hex(hash: &NodeHash) -> String {
    hash.0.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> SpaceResult<NodeHash> {
    let bad = || SpaceError::new(format!("not a node hash: {text}"));
    if text.len() != NodeHash::LEN * 2 {
        return Err(bad());
    }
    let mut out = [0u8; NodeHash::LEN];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
    }
    Ok(NodeHash(out))
}

// -- The storage over a space ------------------------------------------

/// A value as a head names it, at the type the head records.
struct Stored {
    ty: Ty,
    value: Value,
}

/// A page's storage over a space (RFC-0033): a context loads at the type its
/// head records, a store waits in the storage until `commit` moves the head
/// of every context stored since the last commit, and a restore of a holder
/// the space gave writes nothing.
pub struct SpaceStorage<'s> {
    space: &'s Space,
    stored: BTreeMap<String, Held>,
    /// The keys whose stored holder a load gave out, which a restore puts
    /// back among the stored.
    lent: BTreeSet<String>,
    committed: Vec<Committed>,
}

/// A head a commit moved.
#[derive(Clone, PartialEq, Debug)]
pub struct Committed {
    pub id: String,
    pub head: NodeHash,
}

impl<'s> SpaceStorage<'s> {
    pub fn new(space: &'s Space) -> Self {
        SpaceStorage {
            space,
            stored: BTreeMap::new(),
            lent: BTreeSet::new(),
            committed: Vec::new(),
        }
    }

    pub fn space(&self) -> &'s Space {
        self.space
    }

    /// The heads the last commit moved, in identity order.
    pub fn committed(&self) -> &[Committed] {
        &self.committed
    }
}

impl Storage for SpaceStorage<'_> {
    fn load(&mut self, key: &str, codec: &Codec<'_>) -> Result<Option<Held>, StorageError> {
        if let Some(held) = self.stored.remove(key) {
            self.lent.insert(key.to_owned());
            return Ok(Some(held));
        }
        let Some(Stored { ty, value }) = self.space.load_stored(codec.rt(), key)? else {
            return Ok(None);
        };
        // SAFETY: `load_stored` decodes a fresh word, which no other holder
        // owns.
        let value = unsafe { Owned::from_value(Holding::new(), value) };
        Ok(Some(Held::new(value, Arc::new(ty), codec.rt().shared.compilation)))
    }

    fn store(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
        self.lent.remove(key);
        self.stored.insert(key.to_owned(), held);
        Ok(())
    }

    fn restore(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
        if self.lent.remove(key) {
            self.stored.entry(key.to_owned()).or_insert(held);
        }
        Ok(())
    }

    fn commit(&mut self, codec: &Codec<'_>) -> Result<(), StorageError> {
        self.committed.clear();
        while let Some((id, mut held)) = self.stored.pop_first() {
            let ty = held.ty().clone();
            // SAFETY: `Space::commit` edits an extension's payload in place
            // through its hooks, as `commit_nested` does, and writes no word
            // into the value.
            let value = unsafe { held.value_mut() };
            let head = match self.space.commit(codec.rt(), &id, &ty, value) {
                Ok(head) => head,
                Err(error) => {
                    self.stored.insert(id, held);
                    return Err(error.into());
                }
            };
            self.committed.push(Committed { id, head });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use acvus_extern::{SpaceHooks, SpaceResult};
    use acvus_mir::graph::QualifiedRef;
    use acvus_mir::ty::LenTerm;
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::executor::SequentialExecutor;
    use crate::interpreter::InterpreterContext;

    fn refuse<T>() -> SpaceResult<T> {
        Err(SpaceError::new("not in this test"))
    }

    /// An extension type's hooks as a type would write them: the state is a
    /// count and that many units, read one at a time through the space, and
    /// an op is one byte. The value itself is a unit; only what the space
    /// does around the hooks is under test.
    fn units_hooks() -> SpaceHooks<AcvusRuntime> {
        SpaceHooks {
            encode_state: Box::new(|_, _, _, _, _| refuse()),
            decode_state: Box::new(|_, _, elem, input| {
                let (count, rest) = input
                    .split_first_chunk::<8>()
                    .ok_or_else(|| SpaceError::new("Units: truncated count"))?;
                *input = rest;
                for _ in 0..u64::from_le_bytes(*count) {
                    drop(elem(&Ty::Unit, input)?);
                }
                Ok(Value::unit())
            }),
            take_ops: Box::new(|_, _, _, _| refuse()),
            apply_op: Box::new(|_, _, _, _, op| {
                let (_, rest) = op
                    .split_first()
                    .ok_or_else(|| SpaceError::new("Units: empty op"))?;
                *op = rest;
                Ok(())
            }),
            children: Box::new(|_, _, _, _| Ok(())),
            head: Box::new(|_, _| None),
            set_head: Box::new(|_, _, _| {}),
        }
    }

    fn units_runtime(i: &Interner) -> (AcvusRuntime, Ty) {
        let id = QualifiedRef::root(i.intern("Units"));
        let hooks = [(id, units_hooks())].into_iter().collect();
        let rt = InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor))
            .with_space(hooks)
            .runtime_over_an_empty_page();
        let ty = Ty::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        (rt, ty)
    }

    /// A store and the nodes written into it, each at its own address.
    fn put(store: &MemoryStore, kind: NodeKind, parent: Option<NodeHash>, bytes: Vec<u8>) -> NodeHash {
        let node = Node { kind, parent, bytes };
        let hash = node.address();
        store.put(hash, &node.to_bytes()).unwrap();
        hash
    }

    fn held_at(i: &Interner, store: MemoryStore, id: &str, ty: &Ty, hash: NodeHash) -> Space {
        let head = Head {
            hash,
            ty: ty.to_ser(i),
        };
        store.cmpxchg(id, None, head).unwrap().unwrap();
        Space::over(Plain, Box::new(store))
    }

    fn count(n: u64) -> Vec<u8> {
        n.to_le_bytes().to_vec()
    }

    #[test]
    fn an_extension_node_is_read_whole() {
        let i = Interner::new();
        let (rt, ty) = units_runtime(&i);
        // the state and an op, each read whole
        let store = MemoryStore::default();
        let state = put(&store, NodeKind::State, None, count(2));
        let op = put(&store, NodeKind::Op, Some(state), vec![1]);
        let space = held_at(&i, store, "u", &ty, op);
        assert!(space.load(&rt, "u", &ty).unwrap().is_some());
        // a byte after the state
        let store = MemoryStore::default();
        let state = put(&store, NodeKind::State, None, [count(2), vec![9]].concat());
        let space = held_at(&i, store, "u", &ty, state);
        let refusal = space.load(&rt, "u", &ty).expect_err("a byte after the state");
        assert!(refusal.0.contains("left unread after the state of Units"), "{}", refusal.0);
        // a byte after an op
        let store = MemoryStore::default();
        let state = put(&store, NodeKind::State, None, count(0));
        let op = put(&store, NodeKind::Op, Some(state), vec![1, 9]);
        let space = held_at(&i, store, "u", &ty, op);
        let refusal = space.load(&rt, "u", &ty).expect_err("a byte after the op");
        assert!(refusal.0.contains("left unread after an op of Units"), "{}", refusal.0);
    }

    /// A count an extension's hooks read themselves, of zero-width parts
    /// read through the space, takes from the load's allowance.
    #[test]
    fn a_zero_width_count_a_hook_reads_is_bounded() {
        let i = Interner::new();
        let (rt, ty) = units_runtime(&i);
        let store = MemoryStore::default();
        let state = put(&store, NodeKind::State, None, count(u64::MAX));
        let space = held_at(&i, store, "u", &ty, state);
        let started = Instant::now();
        let refusal = space.load(&rt, "u", &ty).expect_err("u64::MAX units");
        assert!(refusal.0.contains("zero-width elements"), "{}", refusal.0);
        assert!(started.elapsed() < Duration::from_secs(5), "refused after {:?}", started.elapsed());
    }

    /// Nested values share their parent's allowance: two nested values of
    /// just over half the bound each are refused together.
    #[test]
    fn nested_loads_share_the_allowance() {
        let i = Interner::new();
        let (rt, units) = units_runtime(&i);
        let half = (Space::ZERO_WIDTH_ELEMENTS / 2) as u64;
        let pair = Ty::Array(Box::new(units.clone()), LenTerm::Known(2));
        let loaded = |each: u64| {
            let store = MemoryStore::default();
            // one node, named twice: each naming is a load of its own
            let child = put(&store, NodeKind::State, None, count(each));
            let parent = put(&store, NodeKind::State, None, [count(2), child.0.to_vec(), child.0.to_vec()].concat());
            let space = held_at(&i, store, "p", &pair, parent);
            let value = space.load(&rt, "p", &pair)?.expect("held");
            // SAFETY: the load made the word, and no other holder owns it.
            drop(unsafe { Owned::<AcvusRuntime>::from_value(Holding::new(), value) });
            SpaceResult::Ok(())
        };
        assert!(loaded(half).is_ok());
        assert!(loaded(half + 1).is_err_and(|e| e.0.contains("zero-width elements")));
    }

    /// The store's bytes are the node only when they hash to its address:
    /// an op stored as its own parent is refused, not followed forever.
    #[test]
    fn a_node_that_names_itself_is_refused() {
        let i = Interner::new();
        let (rt, ty) = units_runtime(&i);
        let store = MemoryStore::default();
        let at = NodeHash([3; NodeHash::LEN]);
        let node = Node {
            kind: NodeKind::Op,
            parent: Some(at),
            bytes: vec![1],
        };
        store.put(at, &node.to_bytes()).unwrap();
        let space = held_at(&i, store, "u", &ty, at);
        let refusal = space.load(&rt, "u", &ty).expect_err("a cycle");
        assert!(refusal.0.contains("hashes elsewhere"), "{}", refusal.0);
    }
}
