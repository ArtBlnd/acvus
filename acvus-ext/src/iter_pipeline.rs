//! The `Iterator` extension type: a lazy, move-only pipeline.
//!
//! The pipeline itself is erased: it moves the runtime's values and calls
//! the runtime's closures. `Iter<T, E, I, Rt>` is the typed view on it. A
//! typed source enters through `from_items` or `generate`, and a typed
//! consumer pulls through `next`; those two edges and the closure calls are
//! the only places a value crosses the runtime boundary, through the
//! runtime's `erase`/`materialize`.

use std::collections::VecDeque;
use std::marker::PhantomData;

use acvus_extern::{
    BoxFuture, ClosureFn, EffectVar, ExternType, Fn1, IdentityVar, Ref, Runtime, TyVar,
};
use sync_wrapper::SyncWrapper;

#[derive(ExternType)]
#[extern_type(name = "Iterator")]
pub struct Iter<T, E, I, Rt>(Pipeline<Rt>, PhantomData<(T, E, I)>)
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime;

/// How one item becomes many, for Flatten and FlatMap: the typed layer
/// supplies it, since only the typed layer knows the item's shape.
type Expand<Rt> =
    Box<dyn Fn(&Rt, <Rt as Runtime>::Value) -> Vec<<Rt as Runtime>::Value> + Send + Sync>;

/// A closure as the pipeline keeps it: its element types are erased with
/// the pipeline's, so one op shape serves every `T`.
type Lambda<Rt> = Fn1<<Rt as Runtime>::Value, <Rt as Runtime>::Value, (), Rt>;

enum Op<Rt: Runtime> {
    Map(Lambda<Rt>),
    Filter(Lambda<Rt>),
    Take { remaining: usize },
    Skip { remaining: usize },
    Flatten(Expand<Rt>),
    FlatMap(Lambda<Rt>, Expand<Rt>),
}

type Generator<Rt> = SyncWrapper<Box<dyn FnMut(&Rt) -> Option<<Rt as Runtime>::Value> + Send>>;

enum Source<Rt: Runtime> {
    Generator(Generator<Rt>),
    Chain(VecDeque<Pipeline<Rt>>),
    Done,
}

/// Items produced by one Flatten or FlatMap expansion, still to be fed to
/// the ops after it.
struct ExpansionFrame<Rt: Runtime> {
    next_op: usize,
    items: VecDeque<Rt::Value>,
}

pub struct Pipeline<Rt: Runtime> {
    source: Source<Rt>,
    ops: Vec<Op<Rt>>,
    expansions: Vec<ExpansionFrame<Rt>>,
    exhausted: bool,
}

impl<Rt: Runtime> Pipeline<Rt> {
    fn from_source(source: Source<Rt>) -> Self {
        Self {
            source,
            ops: Vec::new(),
            expansions: Vec::new(),
            exhausted: false,
        }
    }

    fn push_op(mut self, op: Op<Rt>) -> Self {
        self.ops.push(op);
        self
    }

    fn chain(self, other: Pipeline<Rt>) -> Self {
        match self {
            Pipeline {
                source: Source::Chain(mut parts),
                ops,
                expansions,
                exhausted,
            } if ops.is_empty() => {
                parts.push_back(other);
                Pipeline {
                    source: Source::Chain(parts),
                    ops,
                    expansions,
                    exhausted,
                }
            }
            first => Self::from_source(Source::Chain(VecDeque::from([first, other]))),
        }
    }
}

impl<T, E, I, Rt> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    fn erased(pipeline: Pipeline<Rt>) -> Self {
        Self(pipeline, PhantomData)
    }

    fn retype<U: TyVar>(self) -> Iter<U, E, I, Rt> {
        Iter(self.0, PhantomData)
    }

    pub fn empty() -> Self {
        Self::erased(Pipeline::from_source(Source::Done))
    }

    pub fn from_items(items: Vec<T>) -> Self {
        let mut items = items.into_iter();
        Self::generate(move || items.next())
    }

    pub fn from_fn(f: impl FnMut(&Rt) -> Option<Rt::Value> + Send + 'static) -> Self {
        Self::erased(Pipeline::from_source(Source::Generator(SyncWrapper::new(
            Box::new(f),
        ))))
    }

    /// A typed generator: each item crosses into the runtime as it is pulled.
    pub fn generate(mut f: impl FnMut() -> Option<T> + Send + 'static) -> Self {
        Self::from_fn(move |rt| f().map(|item| unsafe { rt.erase::<T>(item) }))
    }

    pub fn map<U: TyVar>(self, f: Fn1<T, U, E, Rt>) -> Iter<U, E, I, Rt> {
        Self::erased(self.0.push_op(Op::Map(f.erased()))).retype()
    }

    pub fn filter(self, f: Fn1<Ref<T>, bool, E, Rt>) -> Self {
        Self::erased(self.0.push_op(Op::Filter(f.erased())))
    }

    pub fn take(self, n: usize) -> Self {
        Self::erased(self.0.push_op(Op::Take { remaining: n }))
    }

    pub fn skip(self, n: usize) -> Self {
        Self::erased(self.0.push_op(Op::Skip { remaining: n }))
    }

    /// The two sources joined; the result is a source of its own.
    pub fn chain<J, K>(self, other: Iter<T, E, J, Rt>) -> Iter<T, E, K, Rt>
    where
        J: IdentityVar,
        K: IdentityVar,
    {
        Iter(self.0.chain(other.0), PhantomData)
    }

    /// The same pipeline under another identity: what a constructor that
    /// makes a new source from an existing one returns.
    pub fn reidentify<K: IdentityVar>(self) -> Iter<T, E, K, Rt> {
        Iter(self.0, PhantomData)
    }

    /// Flatten items that are themselves sequences of `U`.
    pub fn flatten<U>(self) -> Iter<U, E, I, Rt>
    where
        T: IntoIterator<Item = U>,
        U: TyVar,
    {
        Self::erased(self.0.push_op(Op::Flatten(expand_as::<T, U, Rt>()))).retype()
    }

    /// Map each item to a sequence of `U` and flatten.
    pub fn flat_map<S, U>(self, f: Fn1<T, S, E, Rt>) -> Iter<U, E, I, Rt>
    where
        S: TyVar + IntoIterator<Item = U>,
        U: TyVar,
    {
        Self::erased(self.0.push_op(Op::FlatMap(f.erased(), expand_as::<S, U, Rt>()))).retype()
    }

    pub async fn next(&mut self, rt: &Rt) -> Result<Option<T>, Rt::Error>
    where
    {
        match pull(&mut self.0, rt).await? {
            Some(value) => Ok(Some(unsafe { rt.materialize::<T>(value) })),
            None => Ok(None),
        }
    }
}

fn expand_as<S, U, Rt>() -> Expand<Rt>
where
    S: TyVar + IntoIterator<Item = U>,
    U: TyVar,
    Rt: Runtime,
{
    Box::new(|rt, value| {
        let items = unsafe { rt.materialize::<S>(value) };
        items
            .into_iter()
            .map(|u| unsafe { rt.erase::<U>(u) })
            .collect()
    })
}

fn pull<'a, Rt>(
    pipeline: &'a mut Pipeline<Rt>,
    rt: &'a Rt,
) -> BoxFuture<'a, Result<Option<Rt::Value>, Rt::Error>>
where
    Rt: Runtime,
{
    Box::pin(async move {
        loop {
            let Some((start_op, val)) = next_input(pipeline, rt).await? else {
                return Ok(None);
            };
            if let Some(out) = run_ops(pipeline, start_op, val, rt).await? {
                return Ok(Some(out));
            }
        }
    })
}

async fn next_input<Rt>(
    pipeline: &mut Pipeline<Rt>,
    rt: &Rt,
) -> Result<Option<(usize, Rt::Value)>, Rt::Error>
where
    Rt: Runtime,
{
    while let Some(frame) = pipeline.expansions.last_mut() {
        if let Some(val) = frame.items.pop_front() {
            return Ok(Some((frame.next_op, val)));
        }
        pipeline.expansions.pop();
    }
    if pipeline.exhausted {
        return Ok(None);
    }

    let raw = match &mut pipeline.source {
        Source::Generator(next_fn) => next_fn.get_mut()(rt),
        Source::Done => None,
        Source::Chain(parts) => loop {
            let Some(front) = parts.front_mut() else {
                break None;
            };
            match pull(front, rt).await? {
                Some(val) => break Some(val),
                None => {
                    parts.pop_front();
                }
            }
        },
    };
    Ok(raw.map(|val| (0, val)))
}

async fn run_ops<Rt: Runtime>(
    pipeline: &mut Pipeline<Rt>,
    start_op: usize,
    mut val: Rt::Value,
    rt: &Rt,
) -> Result<Option<Rt::Value>, Rt::Error> {
    let mut i = start_op;
    while i < pipeline.ops.len() {
        match &mut pipeline.ops[i] {
            Op::Map(f) => val = f.call(rt, (val,)).await?,
            Op::Filter(f) => {
                let keep = f.call(rt, (unsafe { rt.reference(&val) },)).await?;
                if !unsafe { rt.materialize::<bool>(keep) } {
                    return Ok(None);
                }
            }
            Op::Skip { remaining } => {
                if *remaining > 0 {
                    *remaining -= 1;
                    return Ok(None);
                }
            }
            Op::Take { remaining } => {
                if *remaining == 0 {
                    return Ok(None);
                }
                *remaining -= 1;
                if *remaining == 0 {
                    pipeline.exhausted = true;
                }
            }
            Op::Flatten(expand) => {
                let items = expand(rt, val);
                let Some(first) = push_expansion(pipeline, i, items.into()) else {
                    return Ok(None);
                };
                val = first;
            }
            Op::FlatMap(f, expand) => {
                let mapped = f.call(rt, (val,)).await?;
                let items = expand(rt, mapped);
                let Some(first) = push_expansion(pipeline, i, items.into()) else {
                    return Ok(None);
                };
                val = first;
            }
        }
        i += 1;
    }
    Ok(Some(val))
}

fn push_expansion<Rt: Runtime>(
    pipeline: &mut Pipeline<Rt>,
    op_index: usize,
    mut items: VecDeque<Rt::Value>,
) -> Option<Rt::Value> {
    let first = items.pop_front()?;
    pipeline.expansions.push(ExpansionFrame {
        next_op: op_index + 1,
        items,
    });
    Some(first)
}
