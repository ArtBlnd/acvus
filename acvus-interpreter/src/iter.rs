use std::sync::{Arc, Mutex};

use crate::value::{FnValue, Value};

/// A shared, lazy iterator over `Value`s.
///
/// The chain of operations is stored as a flat list (`IterChain`).
/// Actual computation is deferred until `collect` (or a consuming HOF).
/// Cloning a `SharedIter` is cheap (Arc refcount bump).
#[derive(Clone)]
pub struct SharedIter {
    inner: Arc<Mutex<SharedIterInner>>,
}

struct SharedIterInner {
    chain: IterChain,
    /// Cached result after the first `collect`.
    buffer: Option<Vec<Value>>,
}

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

impl SharedIter {
    pub fn from_list(items: Vec<Value>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SharedIterInner {
                chain: IterChain {
                    source: items.into(),
                    ops: Vec::new(),
                },
                buffer: None,
            })),
        }
    }

    pub fn from_list_rev(mut items: Vec<Value>) -> Self {
        items.reverse();
        Self::from_list(items)
    }

    // -- lazy chain builders (return a *new* SharedIter) ----------------------

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

    pub fn chain(self, other: SharedIter) -> Self {
        let other_chain = other.snapshot_chain();
        self.push_op(IterOp::Chain(other_chain))
    }

    pub fn flatten(self) -> Self {
        self.push_op(IterOp::Flatten)
    }

    pub fn flat_map(self, f: FnValue) -> Self {
        self.push_op(IterOp::FlatMap(f))
    }

    // -- snapshot / internal --------------------------------------------------

    /// Snapshot the current chain.
    /// If the chain has already been collected, use the cached buffer as the new source.
    pub fn snapshot_chain(&self) -> IterChain {
        let inner = self.inner.lock().unwrap();
        if let Some(buf) = &inner.buffer {
            IterChain {
                source: buf.clone().into(),
                ops: Vec::new(),
            }
        } else {
            inner.chain.clone()
        }
    }

    /// Get the cached result, if available.
    pub fn cached(&self) -> Option<Vec<Value>> {
        self.inner.lock().unwrap().buffer.clone()
    }

    /// Store the computed result in the cache.
    pub fn set_cache(&self, items: Vec<Value>) {
        let mut inner = self.inner.lock().unwrap();
        if inner.buffer.is_none() {
            inner.buffer = Some(items);
        }
    }

    fn push_op(self, op: IterOp) -> Self {
        let chain = self.snapshot_chain();
        let mut ops = chain.ops;
        ops.push(op);
        Self {
            inner: Arc::new(Mutex::new(SharedIterInner {
                chain: IterChain {
                    source: chain.source,
                    ops,
                },
                buffer: None,
            })),
        }
    }
}

impl std::fmt::Debug for SharedIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedIter(...)")
    }
}

impl PartialEq for SharedIter {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
