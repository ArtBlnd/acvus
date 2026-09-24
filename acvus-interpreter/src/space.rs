//! A space (RFC-0033): content-addressed, append-only nodes, and a head per
//! identity moved by compare-and-exchange. An extension value's log is a
//! chain of op nodes from a checkpoint; a value of a language shape is a
//! state node. A nested extension value has its own chain, named from its
//! parent's node by head, so a log is structural.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex};

use acvus_extern::{
    ArgAt, Borrowed, Crossing, Declared, Holding, NodeHash, OneValue, Owned, Project, SpaceError,
    SpaceResult,
};
use acvus_mir::ty::Ty;

use crate::host::{Contexts, Page, PageError, held_as};
use crate::layout::{self, Nested};
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

/// The head of one identity: the node and the type its value has.
#[derive(Clone, PartialEq, Debug)]
pub struct Head {
    pub hash: NodeHash,
    pub ty: Ty,
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
    interner: acvus_utils::Interner,
    lock: Mutex<()>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HeadFile {
    head: String,
    ty: acvus_mir::ser_ty::SerTy,
}

impl DirStore {
    pub fn open(
        root: impl Into<std::path::PathBuf>,
        interner: &acvus_utils::Interner,
    ) -> SpaceResult<Self> {
        let root = root.into();
        for sub in ["nodes", "heads"] {
            std::fs::create_dir_all(root.join(sub))
                .map_err(|e| SpaceError::new(format!("{}: {e}", root.display())))?;
        }
        Ok(Self {
            root,
            interner: interner.clone(),
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
            ty: file.ty.to_ty(&self.interner),
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
            ty: new.ty.to_ser(&self.interner),
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

    /// Every identity the space holds, with its type.
    pub fn identities(&self) -> SpaceResult<Vec<(String, Ty)>> {
        let mut out = Vec::new();
        for id in self.store.identities()? {
            if let Some(head) = self.store.head(&id)? {
                out.push((id, head.ty));
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
    /// through its parents.
    pub fn get(&self, hash: NodeHash) -> SpaceResult<Node> {
        let bytes = self
            .store
            .get(hash)?
            .ok_or_else(|| SpaceError::new(format!("no node {}", hex(&hash))))?;
        Node::from_bytes(&bytes)
    }

    /// Move `id`'s head from `expected` to `new` under `ty`.
    pub fn cmpxchg(
        &self,
        id: &str,
        expected: Option<NodeHash>,
        new: NodeHash,
        ty: &Ty,
    ) -> SpaceResult<Result<(), Option<NodeHash>>> {
        self.store.cmpxchg(
            id,
            expected,
            Head {
                hash: new,
                ty: ty.clone(),
            },
        )
    }

    /// The value of `id` as its head names it.
    pub fn load(&self, rt: &AcvusRuntime, id: &str, ty: &Ty) -> SpaceResult<Option<Value>> {
        match self.head(id) {
            Some(head) => self.load_at(rt, ty, head).map(Some),
            None => Ok(None),
        }
    }

    /// Commit `value` as the new head of `id`. An extension value carries
    /// the head it was loaded at, and the commit is refused when the head
    /// has moved since; a value that carries none, or one of a language
    /// shape, replaces whatever the head is.
    pub fn commit(
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
        self.cmpxchg(id, expected, new, ty)?.map_err(|current| {
            SpaceError::new(format!(
                "@{id}: head moved to {} since it was loaded",
                current.map_or("nothing".to_string(), |h| hex(&h))
            ))
        })?;
        Ok(new)
    }

    fn load_at(&self, rt: &AcvusRuntime, ty: &Ty, head: NodeHash) -> SpaceResult<Value> {
        if !matches!(ty, Ty::UserDefined { .. }) {
            let node = self.get(head)?;
            if node.kind != NodeKind::State {
                return Err(SpaceError::new(
                    "a value of a language shape is a state node",
                ));
            }
            return layout::decode(rt, self, ty, &mut node.bytes.as_slice());
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
        let decode = |t: &Ty, input: &mut &[u8]| layout::decode_owned(rt, self, t, input);
        let mut value = (hooks.decode_state)(rt, &args, &decode, &mut state.bytes.as_slice())?;
        for op in ops.iter().rev() {
            (hooks.apply_op)(rt, &mut value, &args, &decode, &mut op.bytes.as_slice())?;
        }
        (hooks.set_head)(rt, &mut value, head);
        Ok(value)
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

    fn load(&self, rt: &AcvusRuntime, ty: &Ty, head: NodeHash) -> SpaceResult<Value> {
        self.load_at(rt, ty, head)
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

// -- The page over a space ---------------------------------------------

/// A run's page over a space (RFC-0033): a context is loaded from the
/// space when the run first fetches it, and every context the run holds
/// is committed when the host asks.
pub struct SpacePage {
    space: Arc<Space>,
    types: HashMap<String, Ty>,
    held: Mutex<HashMap<String, Owned<AcvusRuntime>>>,
    declared: Option<Contexts>,
}

impl SpacePage {
    /// A page over `space` for the identities it holds, plus `seed`:
    /// values for identities the space does not hold yet, or replaces.
    pub fn new(
        space: Arc<Space>,
        seed: HashMap<String, (Ty, Owned<AcvusRuntime>)>,
    ) -> SpaceResult<Self> {
        let mut types: HashMap<String, Ty> = space.identities()?.into_iter().collect();
        let mut held = HashMap::new();
        for (id, (ty, value)) in seed {
            types.insert(id.clone(), ty);
            held.insert(id, value);
        }
        Ok(Self {
            space,
            types,
            held: Mutex::new(held),
            declared: None,
        })
    }

    pub fn of(space: Arc<Space>, contexts: &Contexts) -> SpaceResult<Self> {
        let interner = &contexts.rt().shared.interner;
        let mut types: HashMap<String, Ty> = space.identities()?.into_iter().collect();
        for (key, declared) in contexts.settled() {
            match types.get(key) {
                Some(stored) if !stored.same_erased(declared) => {
                    return Err(SpaceError::new(format!(
                        "@{key} is stored as {}, and declared as {}",
                        stored.display(interner),
                        declared.display(interner)
                    )));
                }
                Some(_) => {}
                None => {
                    types.insert(key.clone(), declared.clone());
                }
            }
        }
        Ok(Self {
            space,
            types,
            held: Mutex::new(HashMap::new()),
            declared: Some(contexts.clone()),
        })
    }

    pub fn read<T>(&mut self, key: &str) -> Result<<T as Borrowed>::Ref<'_>, PageError>
    where
        T: Declared + Project<AcvusRuntime>,
    {
        let (contexts, ty, value) = self.loaded_as::<T>(key)?;
        let table = T::table(ArgAt {
            interner: &contexts.rt().shared.interner,
            ty,
        });
        // SAFETY: the page holds `key` at `ty`, which is `T`'s, and a value
        // enters the page only at the type it holds the key at.
        Ok(unsafe { T::project(contexts.rt(), value, &table) })
    }

    pub fn update<T, F, U>(&mut self, key: &str, f: F) -> Result<U, PageError>
    where
        T: Declared + Project<AcvusRuntime>,
        F: FnOnce(<T as Borrowed>::Mut<'_>) -> U,
    {
        let (contexts, ty, value) = self.loaded_as::<T>(key)?;
        let table = T::table(ArgAt {
            interner: &contexts.rt().shared.interner,
            ty,
        });
        // SAFETY: `&mut self` names the value exclusively, and a projection
        // writes inside the storage the word names, never the word itself.
        let word = unsafe { value.value_mut(Holding::new()) };
        // SAFETY: as `read`, exclusively.
        Ok(f(unsafe { T::project_mut(contexts.rt(), word, &table) }))
    }

    pub fn insert<T>(&mut self, key: &str, value: T) -> Result<(), PageError>
    where
        T: Declared + OneValue<AcvusRuntime>,
    {
        let Some(contexts) = &self.declared else {
            return Err(PageError::Undeclared {
                key: key.to_owned(),
            });
        };
        held_as::<T>(&contexts.rt().shared.interner, key, self.types.get(key))?;
        // SAFETY: `value` crosses at `T`, the type the page holds `key` at.
        let held = Owned::erased(unsafe { Crossing::new(contexts.rt()) }, value);
        self.held
            .get_mut()
            .expect("page")
            .insert(key.to_owned(), held);
        Ok(())
    }

    fn loaded_as<T>(
        &mut self,
        key: &str,
    ) -> Result<(&Contexts, &Ty, &mut Owned<AcvusRuntime>), PageError>
    where
        T: Declared,
    {
        let Some(contexts) = &self.declared else {
            return Err(PageError::Undeclared {
                key: key.to_owned(),
            });
        };
        let ty = held_as::<T>(&contexts.rt().shared.interner, key, self.types.get(key))?;
        let value = match self.held.get_mut().expect("page").entry(key.to_owned()) {
            Entry::Occupied(held) => held.into_mut(),
            Entry::Vacant(vacant) => {
                let Some(loaded) = self
                    .space
                    .load(contexts.rt(), key, ty)
                    .map_err(PageError::Space)?
                else {
                    return Err(PageError::Absent {
                        key: key.to_owned(),
                    });
                };
                // SAFETY: `load` decodes a fresh word, which no other holder
                // owns.
                vacant.insert(unsafe { Owned::from_value(Holding::new(), loaded) })
            }
        };
        Ok((contexts, ty, value))
    }

    pub fn types(&self) -> &HashMap<String, Ty> {
        &self.types
    }

    /// Commit every context the page holds; the new head of each, in
    /// identity order. A context the run never fetched is not touched.
    pub fn commit(&self, rt: &AcvusRuntime) -> SpaceResult<Vec<(String, NodeHash)>> {
        let mut held = std::mem::take(&mut *self.held.lock().expect("page"));
        let mut ids: Vec<String> = held.keys().cloned().collect();
        ids.sort();
        let mut out = Vec::new();
        for id in ids {
            let ty = self
                .types
                .get(&id)
                .ok_or_else(|| SpaceError::new(format!("@{id}: no type")))?;
            let mut value = held.remove(&id).expect("listed");
            // SAFETY: `commit` edits the value in place as `commit_nested`
            // does and writes no word into it.
            let head = self.space.commit(rt, &id, ty, unsafe { value.value_mut(acvus_extern::Holding::new()) })?;
            out.push((id, head));
        }
        Ok(out)
    }
}

impl Page for SpacePage {
    fn held_type(&self, key: &str) -> Option<&Ty> {
        self.types.get(key)
    }
}

impl crate::journal::RuntimeContext for SpacePage {
    fn take(&self, rt: &AcvusRuntime, key: &str) -> Option<Owned<AcvusRuntime>> {
        if let Some(v) = self.held.lock().expect("page").remove(key) {
            return Some(v);
        }
        let ty = self.types.get(key)?;
        self.space
            .load(rt, key, ty)
            .unwrap_or_else(|e| panic!("context fetch: @{key}: {e}"))
            // SAFETY: `load` decodes a fresh word, which no other holder
            // owns.
            .map(|value| unsafe { Owned::from_value(acvus_extern::Holding::new(), value) })
    }

    unsafe fn set(&self, key: &str, value: Owned<AcvusRuntime>) {
        self.held
            .lock()
            .expect("page")
            .insert(key.to_string(), value);
    }

    fn take_writes(&self) -> Vec<crate::journal::ContextWrite> {
        Vec::new()
    }
}
