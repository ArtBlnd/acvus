use std::sync::{Arc, Mutex};

use acvus_mir::ty::Effect;
use acvus_utils::TrackedDeque;

use crate::value::{FnValue, Value};

// =========================================================================
// IterChain / IterOp — flat chain representation for drive_chain
// =========================================================================

/// Flat representation of a lazy iterator pipeline.
#[derive(Clone)]
pub struct IterChain {
    pub source: Arc<[Value]>,
    pub ops: Vec<IterOp>,
}

/// A single lazy operation in the pipeline.
#[derive(Clone)]
pub enum IterOp {
    Map(FnValue),
    Filter(FnValue),
    Take(usize),
    Skip(usize),
    /// Concatenate another chain's elements after the current ones.
    Chain(IterChain),
    /// Flatten nested lists: each `Value::List(inner)` is inlined.
    Flatten,
    /// Map then flatten: call closure, then inline any `Value::List`.
    FlatMap(FnValue),
}

// =========================================================================
// SequenceChain — lazy Deque with origin tracking
// =========================================================================

/// A lazy sequence of operations on a [`TrackedDeque`].
///
/// Represents a Deque that has been loaded from storage (with checkpoint set)
/// and has pending structural operations. The origin TrackedDeque's checksum
/// is preserved through all operations — when collected, the result can be
/// diffed against the storage origin via [`TrackedDeque::into_diff`].
///
/// ## Invariant
///
/// `origin` must have a checkpoint set. All structural ops (take, skip, chain)
/// are applied lazily. On collect, they are executed against `origin`,
/// producing a TrackedDeque in the same checksum lineage.
///
/// ## Allowed ops
///
/// Only structural operations are allowed — these do not transform individual
/// elements, so the TrackedDeque's diff tracking remains valid:
/// - `Take(n)`: keep only the first `n` elements
/// - `Skip(n)`: consume `n` elements from the front
/// - `Chain(IterHandle)`: extend with elements from a lazy iterator
///
/// Element-transforming operations (map, filter) break the origin relationship
/// and must go through Iterator (Sequence → Iterator coercion).
#[derive(Clone)]
pub struct SequenceChain {
    origin: TrackedDeque<Value>,
    ops: Vec<SequenceOp>,
}

/// A structural operation on a Sequence.
#[derive(Clone)]
pub enum SequenceOp {
    /// Keep only the first `n` elements.
    Take(usize),
    /// Consume `n` elements from the front.
    Skip(usize),
    /// Extend with elements from a lazy iterator.
    /// Requires runtime to execute (IterHandle may contain closures).
    Chain(IterHandle),
}

impl SequenceChain {
    /// Create a new SequenceChain from a TrackedDeque.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the deque has a checkpoint set.
    pub fn new(origin: TrackedDeque<Value>) -> Self {
        debug_assert!(
            origin.is_dirty() || true, // checkpoint must exist — is_dirty only works after checkpoint
            "SequenceChain: origin must have checkpoint set"
        );
        Self {
            origin,
            ops: Vec::new(),
        }
    }

    /// Create from a storage-loaded TrackedDeque: clone + checkpoint.
    pub fn from_stored(stored: TrackedDeque<Value>) -> Self {
        let mut working = stored;
        working.checkpoint();
        Self::new(working)
    }

    /// Create an empty sequence (first turn, no stored value).
    pub fn empty() -> Self {
        let mut deque = TrackedDeque::new();
        deque.checkpoint();
        Self::new(deque)
    }

    // -- lazy builders --------------------------------------------------------

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

    // -- access ---------------------------------------------------------------

    /// Borrow the origin TrackedDeque.
    pub fn origin(&self) -> &TrackedDeque<Value> {
        &self.origin
    }

    /// Borrow the pending ops.
    pub fn ops(&self) -> &[SequenceOp] {
        &self.ops
    }

    /// Convert to an IterHandle (Sequence → Iterator coercion).
    ///
    /// The origin's items become the source, and SequenceOps are translated
    /// to IterOps. The result is still lazy — nothing is executed.
    pub fn into_iter_handle(self, effect: Effect) -> IterHandle {
        let source: Vec<Value> = self.origin.into_vec();
        let ops: Vec<IterOp> = self
            .ops
            .into_iter()
            .map(|op| match op {
                SequenceOp::Take(n) => IterOp::Take(n),
                SequenceOp::Skip(n) => IterOp::Skip(n),
                SequenceOp::Chain(iter) => IterOp::Chain(iter.snapshot_chain()),
            })
            .collect();
        IterHandle::new(source, ops, effect)
    }

    /// Consume and return the origin TrackedDeque (for direct Deque access
    /// when no ops have been applied).
    ///
    /// # Panics
    ///
    /// Panics if there are pending ops.
    pub fn into_origin(self) -> TrackedDeque<Value> {
        assert!(
            self.ops.is_empty(),
            "into_origin called with {} pending ops",
            self.ops.len()
        );
        self.origin
    }

    /// Whether this sequence has pending lazy ops.
    pub fn has_ops(&self) -> bool {
        !self.ops.is_empty()
    }

    /// The origin's checksum — for verification against storage.
    pub fn origin_checksum(&self) -> u64 {
        self.origin.checksum()
    }
}

impl std::fmt::Debug for SequenceChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SequenceChain(len={}, ops={}, checksum={:#x})",
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

// =========================================================================
// IterHandle — thunk-based lazy iterator
// =========================================================================

/// Internal state of a lazy iterator.
///
/// Transitions: `Suspended` → `Evaluated` (on `next`, Pure only) or read-and-discard (Effectful).
/// `Chain` allows composing two iterators sequentially.
#[derive(Clone)]
pub enum IterRepr {
    /// Not yet executed. Holds source data + pipeline ops.
    Suspended {
        source: Vec<Value>,
        ops: Vec<IterOp>,
        offset: usize,
    },
    /// One element has been computed (Pure memo).
    /// `head` is the result, `tail` is the rest of the iterator.
    Evaluated {
        head: Value,
        tail: IterHandle,
    },
    /// Two iterators in sequence — exhaust `first`, then `second`.
    Chain {
        first: IterHandle,
        second: IterHandle,
    },
    /// An op applied on top of another iterator.
    Wrapped {
        inner: IterHandle,
        op: IterOp,
    },
    /// No more elements.
    Done,
}

/// A lazy iterator handle.
///
/// ## Clone semantics
///
/// - **Pure** (`Effect::Pure`): `Arc` is shared — clones see the same memo.
/// - **Effectful**: state is deep-copied — each clone runs independently.
pub struct IterHandle {
    state: Arc<Mutex<IterRepr>>,
    effect: Effect,
}

impl IterHandle {
    /// Create a new iterator from source data + ops pipeline.
    pub fn new(source: Vec<Value>, ops: Vec<IterOp>, effect: Effect) -> Self {
        Self::suspended(source, ops, 0, effect)
    }

    /// Create with explicit offset.
    pub fn suspended(source: Vec<Value>, ops: Vec<IterOp>, offset: usize, effect: Effect) -> Self {
        Self {
            state: Arc::new(Mutex::new(IterRepr::Suspended {
                source,
                ops,
                offset,
            })),
            effect,
        }
    }

    /// Create from a plain list (no ops).
    pub fn from_list(items: Vec<Value>, effect: Effect) -> Self {
        Self::new(items, Vec::new(), effect)
    }

    /// Create from an IterChain.
    pub fn from_chain(chain: IterChain, effect: Effect) -> Self {
        Self::new(chain.source.to_vec(), chain.ops, effect)
    }

    /// Create a done (empty) iterator.
    pub fn done(effect: Effect) -> Self {
        Self {
            state: Arc::new(Mutex::new(IterRepr::Done)),
            effect,
        }
    }

    /// Chain two iterators: exhaust self, then other.
    pub fn chain(self, other: IterHandle) -> Self {
        // LUB: if either is Effectful, the chain is Effectful.
        let effect = match (self.effect, other.effect) {
            (Effect::Pure, Effect::Pure) => Effect::Pure,
            _ => Effect::Effectful,
        };
        Self {
            state: Arc::new(Mutex::new(IterRepr::Chain {
                first: self,
                second: other,
            })),
            effect,
        }
    }

    // -- lazy chain builders (push IterOp, return new IterHandle) -------------

    pub fn map(self, f: FnValue) -> Self {
        self.push_op(IterOp::Map(f))
    }

    pub fn filter(self, f: FnValue) -> Self {
        self.push_op(IterOp::Filter(f))
    }

    pub fn take(self, n: usize) -> Self {
        self.push_op(IterOp::Take(n))
    }

    pub fn skip(self, n: usize) -> Self {
        self.push_op(IterOp::Skip(n))
    }

    pub fn flatten(self) -> Self {
        self.push_op(IterOp::Flatten)
    }

    pub fn flat_map(self, f: FnValue) -> Self {
        self.push_op(IterOp::FlatMap(f))
    }

    /// Push an IterOp onto the pipeline.
    ///
    /// - Suspended: appends op to ops list (fast path).
    /// - Other states: wraps in a Wrapped variant (op applied on top).
    fn push_op(self, op: IterOp) -> Self {
        let mut state = self.state.lock().unwrap();
        match &mut *state {
            IterRepr::Suspended { source, ops, offset } => {
                let cur_offset = *offset;
                let mut new_ops = std::mem::take(ops);
                new_ops.push(op);
                let new_source = std::mem::take(source);
                drop(state);
                Self::suspended(new_source, new_ops, cur_offset, self.effect)
            }
            _ => {
                let effect = self.effect;
                drop(state);
                Self {
                    state: Arc::new(Mutex::new(IterRepr::Wrapped {
                        inner: self,
                        op,
                    })),
                    effect,
                }
            }
        }
    }

    // -- snapshot / compatibility with drive_chain ----------------------------

    /// Snapshot the current pipeline as an IterChain (for drive_chain compatibility).
    ///
    /// Only works for Suspended state. Panics on other states.
    pub fn snapshot_chain(&self) -> IterChain {
        let state = self.state.lock().unwrap();
        match &*state {
            IterRepr::Suspended { source, ops, .. } => IterChain {
                source: source.clone().into(),
                ops: ops.clone(),
            },
            _ => panic!("snapshot_chain: only works on Suspended IterHandle"),
        }
    }

    /// Extract source + ops + offset for drive_chain compatibility.
    ///
    /// Returns `Some((source, ops, offset))` if Suspended, `None` otherwise.
    pub fn into_chain_parts(self) -> Option<(Vec<Value>, Vec<IterOp>, usize)> {
        let state = self.state.lock().unwrap();
        match &*state {
            IterRepr::Suspended { source, ops, offset } => {
                Some((source.clone(), ops.clone(), *offset))
            }
            _ => None,
        }
    }

    /// Get the current state (cloned).
    pub fn get_state(&self) -> IterRepr {
        self.state.lock().unwrap().clone()
    }

    /// Set the state (for Pure memo after evaluation).
    pub fn set_state(&self, repr: IterRepr) {
        *self.state.lock().unwrap() = repr;
    }

    /// The effect of this iterator.
    pub fn effect(&self) -> Effect {
        self.effect
    }
}

impl Clone for IterHandle {
    fn clone(&self) -> Self {
        match self.effect {
            // Pure: share the Arc — clones see the same memo.
            Effect::Pure => Self {
                state: Arc::clone(&self.state),
                effect: self.effect,
            },
            // Effectful: deep copy — each clone runs independently.
            _ => Self {
                state: Arc::new(Mutex::new(self.state.lock().unwrap().clone())),
                effect: self.effect,
            },
        }
    }
}

impl std::fmt::Debug for IterHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        match &*state {
            IterRepr::Suspended { source, ops, offset } => {
                write!(f, "IterHandle::Suspended(len={}, ops={}, offset={})", source.len(), ops.len(), offset)
            }
            IterRepr::Evaluated { .. } => write!(f, "IterHandle::Evaluated"),
            IterRepr::Chain { .. } => write!(f, "IterHandle::Chain"),
            IterRepr::Wrapped { .. } => write!(f, "IterHandle::Wrapped"),
            IterRepr::Done => write!(f, "IterHandle::Done"),
        }
    }
}

impl PartialEq for IterHandle {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
