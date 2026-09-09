//! Lazy iterator pipeline.

use std::collections::VecDeque;

use acvus_interpreter::{FnValue, ExternValue, RuntimeError, Value};
use acvus_mir::graph::QualifiedRef;
use acvus_utils::Interner;
use futures::future::BoxFuture;
use sync_wrapper::SyncWrapper;

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
    fn pull(&mut self) -> Option<Value> {
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

struct ExpansionFrame {
    next_op: usize,
    items: VecDeque<Value>,
}

pub struct IterHandle {
    source: IterSource,
    ops: Vec<IterOp>,
    expansions: Vec<ExpansionFrame>,
    exhausted: bool,
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

// ── Value bridge ─────────────────────────────────────────────────────

pub fn iterator_qref(interner: &Interner) -> QualifiedRef {
    QualifiedRef::root(interner.intern("Iterator"))
}

pub fn iter_value(interner: &Interner, handle: IterHandle) -> Value {
    Value::extern_value(ExternValue::new(iterator_qref(interner), handle))
}

pub fn into_iter_handle(value: Value) -> IterHandle {
    match value {
        Value::Extern(o) => match o.into_owned::<IterHandle>() {
            Ok(handle) => handle,
            Err(o) => panic!("expected a uniquely owned Iterator, got {o:?}"),
        },
        other => panic!("expected Iterator, got {other:?}"),
    }
}

// ── Pulling ──────────────────────────────────────────────────────────

pub fn exec_next(iter: &mut IterHandle) -> BoxFuture<'_, Result<Option<Value>, RuntimeError>> {
    Box::pin(async move {
        loop {
            let Some((start_op, val)) = next_input(iter).await? else {
                return Ok(None);
            };
            if let Some(out) = run_ops(iter, start_op, val).await? {
                return Ok(Some(out));
            }
        }
    })
}

async fn next_input(iter: &mut IterHandle) -> Result<Option<(usize, Value)>, RuntimeError> {
    while let Some(frame) = iter.expansions.last_mut() {
        if let Some(val) = frame.items.pop_front() {
            return Ok(Some((frame.next_op, val)));
        }
        iter.expansions.pop();
    }
    if iter.exhausted {
        return Ok(None);
    }

    let raw = match &mut iter.source {
        IterSource::Leaf(leaf) => leaf.pull(),
        IterSource::Chain(parts) => loop {
            let Some(front) = parts.front_mut() else {
                break None;
            };
            match exec_next(front).await? {
                Some(val) => break Some(val),
                None => {
                    parts.pop_front();
                }
            }
        },
    };
    Ok(raw.map(|val| (0, val)))
}

async fn run_ops(
    iter: &mut IterHandle,
    start_op: usize,
    mut val: Value,
) -> Result<Option<Value>, RuntimeError> {
    let mut i = start_op;
    while i < iter.ops.len() {
        match &mut iter.ops[i] {
            IterOp::Map(f) => val = f.call(val).await?,
            IterOp::Filter(f) => {
                if !f.call(val.clone()).await?.as_bool() {
                    return Ok(None);
                }
            }
            IterOp::Skip { remaining } => {
                if *remaining > 0 {
                    *remaining -= 1;
                    return Ok(None);
                }
            }
            IterOp::Take { remaining } => {
                if *remaining == 0 {
                    return Ok(None);
                }
                *remaining -= 1;
                if *remaining == 0 {
                    iter.exhausted = true;
                }
            }
            IterOp::Flatten => match val {
                Value::List(l) => {
                    let Some(first) = expand(iter, i, l.iter().cloned().collect()) else {
                        return Ok(None);
                    };
                    val = first;
                }
                other => val = other,
            },
            IterOp::FlatMap(f) => match f.call(val).await? {
                Value::List(l) => {
                    let Some(first) = expand(iter, i, l.iter().cloned().collect()) else {
                        return Ok(None);
                    };
                    val = first;
                }
                other => val = other,
            },
        }
        i += 1;
    }
    Ok(Some(val))
}

fn expand(iter: &mut IterHandle, op_index: usize, mut items: VecDeque<Value>) -> Option<Value> {
    let first = items.pop_front()?;
    iter.expansions.push(ExpansionFrame {
        next_op: op_index + 1,
        items,
    });
    Some(first)
}
