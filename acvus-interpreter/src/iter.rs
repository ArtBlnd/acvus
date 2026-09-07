//! Lazy iterator and sequence handles.

use std::collections::VecDeque;

use acvus_utils::TrackedDeque;
use sync_wrapper::SyncWrapper;

use crate::value::{FnValue, Value};

// ── IterOp ───────────────────────────────────────────────────────────

pub enum IterOp {
    Map(FnValue),
    Filter(FnValue),
    Take { remaining: usize },
    Skip { remaining: usize },
    Flatten,
    FlatMap(FnValue),
}

// ── IterSource ───────────────────────────────────────────────────────

pub enum IterSource {
    Leaf(LeafSource),
    Chain(VecDeque<IterHandle>),
}

pub enum LeafSource {
    Values {
        items: Vec<Value>,
        offset: usize,
    },
    Generator {
        next_fn: SyncWrapper<Box<dyn FnMut() -> Option<Value> + Send>>,
    },
    Done,
}

impl LeafSource {
    pub(crate) fn pull(&mut self) -> Option<Value> {
        match self {
            Self::Values { items, offset } => {
                let val = items.get(*offset).cloned();
                if val.is_some() {
                    *offset += 1;
                }
                val
            }
            Self::Generator { next_fn } => next_fn.get_mut()(),
            Self::Done => None,
        }
    }
}

// ── IterHandle ───────────────────────────────────────────────────────

pub(crate) struct ExpansionFrame {
    pub(crate) next_op: usize,
    pub(crate) items: VecDeque<Value>,
}

pub struct IterHandle {
    pub(crate) source: IterSource,
    pub(crate) ops: Vec<IterOp>,
    pub(crate) expansions: Vec<ExpansionFrame>,
    pub(crate) exhausted: bool,
}

impl IterHandle {
    fn from_source(source: IterSource) -> Self {
        Self {
            source,
            ops: Vec::new(),
            expansions: Vec::new(),
            exhausted: false,
        }
    }

    pub fn from_list(items: Vec<Value>) -> Self {
        Self::from_source(IterSource::Leaf(LeafSource::Values { items, offset: 0 }))
    }

    pub fn from_fn(f: impl FnMut() -> Option<Value> + Send + 'static) -> Self {
        Self::from_source(IterSource::Leaf(LeafSource::Generator {
            next_fn: SyncWrapper::new(Box::new(f)),
        }))
    }

    pub fn done() -> Self {
        Self::from_source(IterSource::Leaf(LeafSource::Done))
    }

    pub fn map(self, f: FnValue) -> Self {
        self.push_op(IterOp::Map(f))
    }
    pub fn filter(self, f: FnValue) -> Self {
        self.push_op(IterOp::Filter(f))
    }
    pub fn take(self, n: usize) -> Self {
        self.push_op(IterOp::Take { remaining: n })
    }
    pub fn skip(self, n: usize) -> Self {
        self.push_op(IterOp::Skip { remaining: n })
    }
    pub fn flatten(self) -> Self {
        self.push_op(IterOp::Flatten)
    }
    pub fn flat_map(self, f: FnValue) -> Self {
        self.push_op(IterOp::FlatMap(f))
    }

    pub fn chain(self, other: IterHandle) -> Self {
        match self {
            IterHandle {
                source: IterSource::Chain(mut parts),
                ops,
                expansions,
                exhausted,
            } if ops.is_empty() => {
                parts.push_back(other);
                IterHandle {
                    source: IterSource::Chain(parts),
                    ops,
                    expansions,
                    exhausted,
                }
            }
            first => Self::from_source(IterSource::Chain(VecDeque::from([first, other]))),
        }
    }

    fn push_op(mut self, op: IterOp) -> Self {
        self.ops.push(op);
        self
    }
}

impl std::fmt::Debug for IterHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let src = match &self.source {
            IterSource::Leaf(LeafSource::Values { items, offset }) => {
                format!("values(len={}, off={offset})", items.len())
            }
            IterSource::Leaf(LeafSource::Generator { .. }) => "generator".to_string(),
            IterSource::Leaf(LeafSource::Done) => "done".to_string(),
            IterSource::Chain(parts) => format!("chain({})", parts.len()),
        };
        write!(
            f,
            "Iter({src}, ops={}, expansions={}, exhausted={})",
            self.ops.len(),
            self.expansions.len(),
            self.exhausted
        )
    }
}

impl PartialEq for IterHandle {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

// ── SequenceChain ────────────────────────────────────────────────────

pub struct SequenceChain {
    origin: TrackedDeque<Value>,
    ops: Vec<SequenceOp>,
}

pub enum SequenceOp {
    Take(usize),
    Skip(usize),
    Chain(IterHandle),
}

impl SequenceChain {
    pub fn new(origin: TrackedDeque<Value>) -> Self {
        Self {
            origin,
            ops: Vec::new(),
        }
    }

    pub fn from_stored(stored: TrackedDeque<Value>) -> Self {
        let mut working = stored;
        working.checkpoint();
        Self::new(working)
    }

    pub fn empty() -> Self {
        let mut deque = TrackedDeque::new();
        deque.checkpoint();
        Self::new(deque)
    }

    pub fn take(mut self, n: usize) -> Self {
        self.ops.push(SequenceOp::Take(n));
        self
    }
    pub fn skip(mut self, n: usize) -> Self {
        self.ops.push(SequenceOp::Skip(n));
        self
    }
    pub fn chain(mut self, iter: IterHandle) -> Self {
        self.ops.push(SequenceOp::Chain(iter));
        self
    }

    pub fn origin(&self) -> &TrackedDeque<Value> {
        &self.origin
    }
    pub fn ops(&self) -> &[SequenceOp] {
        &self.ops
    }
    pub fn has_ops(&self) -> bool {
        !self.ops.is_empty()
    }
    pub fn origin_checksum(&self) -> acvus_utils::DequeChecksum {
        self.origin.checksum()
    }

    pub fn into_origin(self) -> TrackedDeque<Value> {
        assert!(
            self.ops.is_empty(),
            "into_origin with {} pending ops",
            self.ops.len()
        );
        self.origin
    }

    pub fn into_iter_handle(self) -> IterHandle {
        let mut handle = IterHandle::from_list(self.origin.into_vec());
        for op in self.ops {
            handle = match op {
                SequenceOp::Take(n) => handle.take(n),
                SequenceOp::Skip(n) => handle.skip(n),
                SequenceOp::Chain(ih) => handle.chain(ih),
            };
        }
        handle
    }
}

impl std::fmt::Debug for SequenceChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Seq(len={}, ops={}, cksum={:#x})",
            self.origin.len(),
            self.ops.len(),
            self.origin.checksum()
        )
    }
}

impl PartialEq for SequenceChain {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
