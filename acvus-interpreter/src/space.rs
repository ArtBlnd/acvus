//! A space (RFC-0033): content-addressed, append-only nodes, and a head per
//! identity moved by compare-and-exchange. An extension value's log is a
//! chain of op nodes from a checkpoint; a value of a language shape is a
//! state node. A nested extension value has its own chain, named from its
//! parent's node by head, so a log is structural.

use std::collections::HashMap;
use std::sync::Mutex;

use acvus_extern::{NodeHash, SpaceError, SpaceResult};
use acvus_mir::ty::Ty;

use crate::layout::{self, Nested};
use crate::runtime::AcvusRuntime;
use crate::value::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Kind {
    /// The whole value as canonical bytes.
    State = 0,
    /// One op on the parent's value.
    Op = 1,
}

/// One node of a log: its kind, its parent (`None` at the first state),
/// and its bytes. Its address is the hash of exactly this.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Node {
    kind: Kind,
    parent: Option<NodeHash>,
    bytes: Vec<u8>,
}

impl Node {
    fn address(&self) -> NodeHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[self.kind as u8]);
        match self.parent {
            Some(p) => {
                hasher.update(&[1]);
                hasher.update(&p.0);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(&self.bytes);
        NodeHash(*hasher.finalize().as_bytes())
    }
}

/// How a space keeps a value that changes in place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Every commit is a state node; history is not kept.
    Plain,
    /// A commit appends the ops recorded since the last one, and ends in
    /// a state node once `checkpoint_every` ops have accrued since the last
    /// state, or when a nested value's head moved.
    Log { checkpoint_every: usize },
}

pub struct Space {
    mode: Mode,
    nodes: Mutex<HashMap<NodeHash, Node>>,
    heads: Mutex<HashMap<String, NodeHash>>,
}

impl Space {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            nodes: Mutex::new(HashMap::new()),
            heads: Mutex::new(HashMap::new()),
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn head(&self, id: &str) -> Option<NodeHash> {
        self.heads.lock().expect("heads").get(id).copied()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.lock().expect("nodes").len()
    }

    fn put(&self, node: Node) -> NodeHash {
        let hash = node.address();
        self.nodes
            .lock()
            .expect("nodes")
            .entry(hash)
            .or_insert(node);
        hash
    }

    fn get(&self, hash: NodeHash) -> SpaceResult<Node> {
        self.nodes
            .lock()
            .expect("nodes")
            .get(&hash)
            .cloned()
            .ok_or_else(|| SpaceError::new(format!("no node {}", hex(&hash))))
    }

    /// Move `id`'s head from `expected` to `new`; on a head that is not
    /// `expected`, the current head comes back and nothing moves.
    pub fn cmpxchg(
        &self,
        id: &str,
        expected: Option<NodeHash>,
        new: NodeHash,
    ) -> Result<(), Option<NodeHash>> {
        let mut heads = self.heads.lock().expect("heads");
        let current = heads.get(id).copied();
        if current != expected {
            return Err(current);
        }
        heads.insert(id.to_string(), new);
        Ok(())
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
        self.cmpxchg(id, expected, new).map_err(|current| {
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
            if node.kind != Kind::State {
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
                Kind::State => break node,
                Kind::Op => {
                    at = node
                        .parent
                        .ok_or_else(|| SpaceError::new("an op node has a parent"))?;
                    ops.push(node);
                }
            }
        };
        let decode = |t: &Ty, input: &mut &[u8]| layout::decode(rt, self, t, input);
        let mut value = (hooks.decode_state)(rt, args, &decode, &mut state.bytes.as_slice())?;
        for op in ops.iter().rev() {
            (hooks.apply_op)(rt, &mut value, args, &decode, &mut op.bytes.as_slice())?;
        }
        (hooks.set_head)(rt, &mut value, head);
        Ok(value)
    }

    /// The new head for `value`, chained onto the head it carries.
    fn commit_value(&self, rt: &AcvusRuntime, ty: &Ty, value: &mut Value) -> SpaceResult<NodeHash> {
        if !matches!(ty, Ty::UserDefined { .. }) {
            let mut bytes = Vec::new();
            layout::encode(rt, self, ty, value, &mut bytes)?;
            return Ok(self.put(Node {
                kind: Kind::State,
                parent: None,
                bytes,
            }));
        }
        let (hooks, args) = layout::extension(rt, ty)?;
        // Children first: their heads are what the parent's bytes name,
        // and a child whose head moved is a change of the parent.
        let mut children_moved = false;
        (hooks.children)(rt, value, args, &mut |child_ty, child| {
            let child_hooks = layout::extension(rt, child_ty)?.0;
            let before = (child_hooks.head)(rt, child);
            let head = self.commit_value(rt, child_ty, child)?;
            children_moved |= before != Some(head);
            Ok(())
        })?;
        let encode = |t: &Ty, v: &Value, out: &mut Vec<u8>| layout::encode(rt, self, t, v, out);
        let ops = (hooks.take_ops)(rt, value, args, &encode)?;
        let state = |parent: Option<NodeHash>| -> SpaceResult<Node> {
            let mut bytes = Vec::new();
            (hooks.encode_state)(rt, value, args, &encode, &mut bytes)?;
            Ok(Node {
                kind: Kind::State,
                parent,
                bytes,
            })
        };
        let Some(mut head) = (hooks.head)(rt, value) else {
            // Never held: the state as it stands is the first node.
            let head = self.put(state(None)?);
            (hooks.set_head)(rt, value, head);
            return Ok(head);
        };
        if ops.is_empty() && !children_moved {
            return Ok(head);
        }
        match self.mode {
            Mode::Plain => {
                head = self.put(state(Some(head))?);
            }
            Mode::Log { checkpoint_every } => {
                let mut since_state = self.ops_since_state(head)?;
                for op in ops {
                    head = self.put(Node {
                        kind: Kind::Op,
                        parent: Some(head),
                        bytes: op,
                    });
                    since_state += 1;
                }
                if children_moved || since_state >= checkpoint_every {
                    head = self.put(state(Some(head))?);
                }
            }
        }
        (hooks.set_head)(rt, value, head);
        Ok(head)
    }

    fn ops_since_state(&self, mut at: NodeHash) -> SpaceResult<usize> {
        let mut n = 0;
        loop {
            let node = self.get(at)?;
            match (node.kind, node.parent) {
                (Kind::State, _) => return Ok(n),
                (Kind::Op, Some(parent)) => {
                    n += 1;
                    at = parent;
                }
                (Kind::Op, None) => return Err(SpaceError::new("an op node has a parent")),
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

fn hex(hash: &NodeHash) -> String {
    hash.0.iter().map(|b| format!("{b:02x}")).collect()
}
