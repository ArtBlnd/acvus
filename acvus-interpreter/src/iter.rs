use std::sync::{Arc, Mutex};

use acvus_mir::ty::Effect;
use acvus_utils::TrackedDeque;

use crate::value::{FnValue, Value};

// ── IterOp — lazy pipeline operation ─────────────────────────────────

/// A single lazy operation in an iterator pipeline.
#[derive(Clone)]
pub enum IterOp {
    Map(FnValue),
    Filter(FnValue),
    Take(usize),
    Skip(usize),
    Chain(Vec<Value>),
    Flatten,
    FlatMap(FnValue),
}

// ── PureInit — source + ops before collection ────────────────────────

/// Data needed to collect a pure iterator. Consumed on first exec_next.
#[derive(Clone)]
pub struct PureInit {
    pub source: Vec<Value>,
    pub ops: Vec<IterOp>,
}

// ── IterHandle ───────────────────────────────────────────────────────

/// A lazy iterator handle. Two fundamentally different execution paths
/// based on effect.
///
/// ## Pure (OnceLock pattern)
///
/// First `exec_next` collects all elements through the ops pipeline,
/// stores them in a shared `Arc<[Value]>`. Subsequent calls just index.
/// No mutex after initialization. Clone = Arc::clone + fresh index.
///
/// ## Effectful (lazy pull)
///
/// Each `exec_next` pulls one element, applying ops lazily.
/// Clone = deep copy (independent execution).
pub enum IterHandle {
    /// Pure iterator — collect once, index after.
    Pure {
        /// Shared collected items. `None` = not yet collected.
        items: Arc<Mutex<Option<Arc<[Value]>>>>,
        /// Source data + ops for collection (consumed on first access).
        init: Arc<Mutex<Option<PureInit>>>,
        /// Current read position (per-clone).
        index: usize,
    },
    /// Effectful iterator — lazy pull, no memo, no sync.
    /// Move-only: sole owner guaranteed by SSA.
    Effectful {
        state: EffectfulState,
        effect: Effect,
    },
}

/// State for effectful (lazy) iteration.
pub enum EffectfulState {
    Suspended {
        source: Vec<Value>,
        /// Element-level ops only (Map, Filter, Flatten, FlatMap).
        elem_ops: Vec<IterOp>,
        offset: usize,
        /// Remaining elements to take (None = unlimited).
        take_remaining: Option<usize>,
    },
    Done,
}

impl IterHandle {
    /// Create from source + ops.
    pub fn new(source: Vec<Value>, ops: Vec<IterOp>, effect: Effect) -> Self {
        if effect.is_pure() {
            Self::Pure {
                items: Arc::new(Mutex::new(None)),
                init: Arc::new(Mutex::new(Some(PureInit { source, ops }))),
                index: 0,
            }
        } else {
            // Split ops into iterator-level (applied to source) and element-level.
            let mut src = source;
            let mut elem_ops = Vec::new();
            let mut take_remaining: Option<usize> = None;
            let mut skip = 0usize;

            for op in ops {
                match op {
                    IterOp::Skip(n) => skip += n,
                    IterOp::Take(n) => {
                        take_remaining = Some(match take_remaining {
                            Some(prev) => prev.min(n),
                            None => n,
                        });
                    }
                    IterOp::Chain(extra) => src.extend(extra),
                    other => elem_ops.push(other),
                }
            }

            Self::Effectful {
                state: EffectfulState::Suspended {
                    source: src,
                    elem_ops,
                    offset: skip,
                    take_remaining,
                },
                effect,
            }
        }
    }

    /// Create from a plain list (no ops).
    pub fn from_list(items: Vec<Value>, effect: Effect) -> Self {
        Self::new(items, Vec::new(), effect)
    }

    /// Create a done (empty) iterator.
    pub fn done(effect: Effect) -> Self {
        if effect.is_pure() {
            Self::Pure {
                items: Arc::new(Mutex::new(Some(Arc::from(Vec::<Value>::new())))),
                init: Arc::new(Mutex::new(None)),
                index: 0,
            }
        } else {
            Self::Effectful {
                state: EffectfulState::Done,
                effect,
            }
        }
    }

    pub fn effect(&self) -> &Effect {
        match self {
            Self::Pure { .. } => {
                // Pure iterator's effect is always pure — use a static ref.
                static PURE: std::sync::OnceLock<Effect> = std::sync::OnceLock::new();
                PURE.get_or_init(Effect::pure)
            }
            Self::Effectful { effect, .. } => effect,
        }
    }

    // ── Pure: access collected items ─────────────────────────────

    /// For pure iterators: get or initialize the collected items.
    /// Returns the shared Arc. If not yet collected, returns None
    /// (caller must collect via interpreter and call `set_collected`).
    pub fn get_collected(&self) -> Option<Arc<[Value]>> {
        match self {
            Self::Pure { items, .. } => {
                items.lock().unwrap().clone()
            }
            _ => panic!("get_collected on effectful iterator"),
        }
    }

    /// Take the init data for collection (consumed once).
    pub fn take_init(&self) -> Option<PureInit> {
        match self {
            Self::Pure { init, .. } => init.lock().unwrap().take(),
            _ => panic!("take_init on effectful iterator"),
        }
    }

    /// Store collected items after first exec_next.
    pub fn set_collected(&self, collected: Arc<[Value]>) {
        match self {
            Self::Pure { items, .. } => {
                *items.lock().unwrap() = Some(collected);
            }
            _ => panic!("set_collected on effectful iterator"),
        }
    }

    /// Current index (pure only).
    pub fn index(&self) -> usize {
        match self {
            Self::Pure { index, .. } => *index,
            _ => panic!("index on effectful iterator"),
        }
    }

    /// Create a clone with advanced index (pure only).
    pub fn with_next_index(&self) -> Self {
        match self {
            Self::Pure { items, init, index } => Self::Pure {
                items: Arc::clone(items),
                init: Arc::clone(init),
                index: index + 1,
            },
            _ => panic!("with_next_index on effectful iterator"),
        }
    }

    // ── Lazy builders (push ops) ─────────────────────────────────

    pub fn map(self, f: FnValue) -> Self { self.push_op(IterOp::Map(f)) }
    pub fn filter(self, f: FnValue) -> Self { self.push_op(IterOp::Filter(f)) }
    pub fn take(self, n: usize) -> Self { self.push_op(IterOp::Take(n)) }
    pub fn skip(self, n: usize) -> Self { self.push_op(IterOp::Skip(n)) }
    pub fn flatten(self) -> Self { self.push_op(IterOp::Flatten) }
    pub fn flat_map(self, f: FnValue) -> Self { self.push_op(IterOp::FlatMap(f)) }

    pub fn chain(self, other: IterHandle) -> Self {
        // For chain, we need to collect the other iter's source.
        // For now, just push a Chain op with the other's source.
        match other {
            IterHandle::Pure { init, items, .. } => {
                // If already collected, use those items.
                if let Some(collected) = items.lock().unwrap().as_ref() {
                    self.push_op(IterOp::Chain(collected.to_vec()))
                } else if let Some(pure_init) = init.lock().unwrap().as_ref() {
                    // Not yet collected — take source (no ops support for now).
                    self.push_op(IterOp::Chain(pure_init.source.clone()))
                } else {
                    self.push_op(IterOp::Chain(Vec::new()))
                }
            }
            IterHandle::Effectful { .. } => {
                panic!("chain: cannot chain effectful iterator as source")
            }
        }
    }

    fn push_op(self, op: IterOp) -> Self {
        match self {
            Self::Pure { init, items, index } => {
                let mut guard = init.lock().unwrap();
                if let Some(pinit) = guard.as_mut() {
                    pinit.ops.push(op);
                    drop(guard);
                    Self::Pure { init, items, index }
                } else {
                    // Already collected — need to re-wrap.
                    // Create new init from collected items.
                    let collected = items.lock().unwrap().clone().unwrap_or_default();
                    drop(guard);
                    Self::Pure {
                        items: Arc::new(Mutex::new(None)),
                        init: Arc::new(Mutex::new(Some(PureInit {
                            source: collected.to_vec(),
                            ops: vec![op],
                        }))),
                        index,
                    }
                }
            }
            Self::Effectful { mut state, effect } => {
                if let EffectfulState::Suspended { source, elem_ops, offset, take_remaining } = &mut state {
                    match op {
                        IterOp::Skip(n) => *offset += n,
                        IterOp::Take(n) => {
                            *take_remaining = Some(take_remaining.map_or(n, |prev| prev.min(n)));
                        }
                        IterOp::Chain(extra) => source.extend(extra),
                        other => elem_ops.push(other),
                    }
                }
                Self::Effectful { state, effect }
            }
        }
    }
}

impl Clone for IterHandle {
    fn clone(&self) -> Self {
        match self {
            Self::Pure { items, init, index } => Self::Pure {
                items: Arc::clone(items),
                init: Arc::clone(init),
                index: *index,
            },
            Self::Effectful { .. } => {
                panic!("clone: effectful Iterator is move-only")
            }
        }
    }
}

impl std::fmt::Debug for IterHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pure { items, index, .. } => {
                let collected = items.lock().unwrap().is_some();
                write!(f, "Iter::Pure(idx={index}, collected={collected})")
            }
            Self::Effectful { state, .. } => {
                match state {
                    EffectfulState::Suspended { source, elem_ops, offset, .. } => {
                        write!(f, "Iter::Effectful(len={}, ops={}, off={})", source.len(), elem_ops.len(), offset)
                    }
                    EffectfulState::Done => write!(f, "Iter::Effectful(done)"),
                }
            }
        }
    }
}

impl PartialEq for IterHandle {
    fn eq(&self, _other: &Self) -> bool { false }
}

// ── SequenceChain ────────────────────────────────────────────────────

/// Lazy sequence over a [`TrackedDeque`] with origin tracking.
///
/// Only structural operations (take, skip, chain) are allowed — element
/// transforms must go through Iterator coercion.
#[derive(Clone)]
pub struct SequenceChain {
    origin: TrackedDeque<Value>,
    ops: Vec<SequenceOp>,
    effect: Effect,
}

#[derive(Clone)]
pub enum SequenceOp {
    Take(usize),
    Skip(usize),
    Chain(IterHandle),
}

impl SequenceChain {
    pub fn new(origin: TrackedDeque<Value>, effect: Effect) -> Self {
        debug_assert!(origin.is_dirty() || true, "origin must have checkpoint");
        Self { origin, ops: Vec::new(), effect }
    }

    pub fn from_stored(stored: TrackedDeque<Value>, effect: Effect) -> Self {
        let mut working = stored;
        working.checkpoint();
        Self::new(working, effect)
    }

    pub fn empty(effect: Effect) -> Self {
        let mut deque = TrackedDeque::new();
        deque.checkpoint();
        Self::new(deque, effect)
    }

    pub fn take(mut self, n: usize) -> Self { self.ops.push(SequenceOp::Take(n)); self }
    pub fn skip(mut self, n: usize) -> Self { self.ops.push(SequenceOp::Skip(n)); self }
    pub fn chain(mut self, iter: IterHandle) -> Self { self.ops.push(SequenceOp::Chain(iter)); self }

    pub fn origin(&self) -> &TrackedDeque<Value> { &self.origin }
    pub fn ops(&self) -> &[SequenceOp] { &self.ops }
    pub fn effect(&self) -> &Effect { &self.effect }
    pub fn has_ops(&self) -> bool { !self.ops.is_empty() }
    pub fn origin_checksum(&self) -> u64 { self.origin.checksum() }

    pub fn into_origin(self) -> TrackedDeque<Value> {
        assert!(self.ops.is_empty(), "into_origin with {} pending ops", self.ops.len());
        self.origin
    }

    /// Convert to Iterator (breaks origin relationship).
    pub fn into_iter_handle(self) -> IterHandle {
        let effect = self.effect.clone();
        let source = self.origin.into_vec();
        let ops: Vec<IterOp> = self.ops.into_iter().map(|op| match op {
            SequenceOp::Take(n) => IterOp::Take(n),
            SequenceOp::Skip(n) => IterOp::Skip(n),
            SequenceOp::Chain(ih) => {
                // Collect the chained iterator's items.
                if let Some(collected) = ih.get_collected() {
                    IterOp::Chain(collected.to_vec())
                } else {
                    IterOp::Chain(Vec::new())
                }
            }
        }).collect();
        IterHandle::new(source, ops, effect)
    }
}

impl std::fmt::Debug for SequenceChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Seq(len={}, ops={}, cksum={:#x})",
            self.origin.len(), self.ops.len(), self.origin.checksum())
    }
}

impl PartialEq for SequenceChain {
    fn eq(&self, _other: &Self) -> bool { false }
}
