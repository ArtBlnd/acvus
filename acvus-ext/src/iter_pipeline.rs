//! The `Iterator` extension type: a lazy, move-only pipeline.
//!
//! The pipeline itself is erased: it moves the runtime's values and calls
//! the runtime's closures. `Iter<T, E, Rt>` is the typed view on it. A
//! typed source enters through `from_items` or `from_fn`, and a typed
//! consumer pulls through `next`; those two edges and the closure calls are
//! the only places a value changes representation.

use std::collections::VecDeque;
use std::marker::PhantomData;

use acvus_extern::{
    BoxFuture, EffectVar, ExternType, FromValue, IdentityVar, Interner, IntoValue, Runtime, TyVar,
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
type Expand<Rt> = Box<
    dyn Fn(
            <Rt as Runtime>::Value,
            &Interner,
        ) -> Result<Vec<<Rt as Runtime>::Value>, <Rt as Runtime>::Error>
        + Send
        + Sync,
>;

enum Op<Rt: Runtime> {
    Map(Rt::Closure),
    Filter(Rt::Closure),
    Take { remaining: usize },
    Skip { remaining: usize },
    Flatten(Expand<Rt>),
    FlatMap(Rt::Closure, Expand<Rt>),
}

type Generator<Rt> =
    SyncWrapper<Box<dyn FnMut(&Interner) -> Option<<Rt as Runtime>::Value> + Send>>;

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

    pub fn from_items(items: Vec<T>) -> Self
    where
        T: IntoValue<Rt>,
    {
        let mut items = items.into_iter();
        Self::from_fn(move |i| items.next().map(|item| item.into_value(i)))
    }

    pub fn from_fn(f: impl FnMut(&Interner) -> Option<Rt::Value> + Send + 'static) -> Self {
        Self::erased(Pipeline::from_source(Source::Generator(SyncWrapper::new(
            Box::new(f),
        ))))
    }

    /// A typed generator: each item crosses into the runtime as it is pulled.
    pub fn generate(mut f: impl FnMut() -> Option<T> + Send + 'static) -> Self
    where
        T: IntoValue<Rt>,
    {
        Self::from_fn(move |i| f().map(|item| item.into_value(i)))
    }

    pub fn map<U: TyVar>(self, f: Rt::Closure) -> Iter<U, E, I, Rt> {
        Self::erased(self.0.push_op(Op::Map(f))).retype()
    }

    pub fn filter(self, f: Rt::Closure) -> Self {
        Self::erased(self.0.push_op(Op::Filter(f)))
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
        T: FromValue<Rt> + IntoIterator<Item = U>,
        U: TyVar + IntoValue<Rt>,
    {
        Self::erased(self.0.push_op(Op::Flatten(expand_as::<T, U, Rt>()))).retype()
    }

    /// Map each item to a sequence of `U` and flatten.
    pub fn flat_map<S, U>(self, f: Rt::Closure) -> Iter<U, E, I, Rt>
    where
        S: FromValue<Rt> + IntoIterator<Item = U>,
        U: TyVar + IntoValue<Rt>,
    {
        Self::erased(self.0.push_op(Op::FlatMap(f, expand_as::<S, U, Rt>()))).retype()
    }

    pub async fn next(&mut self, interner: &Interner) -> Result<Option<T>, Rt::Error>
    where
        T: FromValue<Rt>,
    {
        match pull(&mut self.0, interner).await? {
            Some(value) => Ok(Some(T::from_value(value, interner)?)),
            None => Ok(None),
        }
    }
}

fn expand_as<S, U, Rt>() -> Expand<Rt>
where
    S: FromValue<Rt> + IntoIterator<Item = U>,
    U: IntoValue<Rt>,
    Rt: Runtime,
{
    Box::new(|value, i| {
        let items = S::from_value(value, i)?;
        Ok(items.into_iter().map(|u| u.into_value(i)).collect())
    })
}

fn pull<'a, Rt: Runtime>(
    pipeline: &'a mut Pipeline<Rt>,
    interner: &'a Interner,
) -> BoxFuture<'a, Result<Option<Rt::Value>, Rt::Error>> {
    Box::pin(async move {
        loop {
            let Some((start_op, val)) = next_input(pipeline, interner).await? else {
                return Ok(None);
            };
            if let Some(out) = run_ops(pipeline, start_op, val, interner).await? {
                return Ok(Some(out));
            }
        }
    })
}

async fn next_input<Rt: Runtime>(
    pipeline: &mut Pipeline<Rt>,
    interner: &Interner,
) -> Result<Option<(usize, Rt::Value)>, Rt::Error> {
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
        Source::Generator(next_fn) => next_fn.get_mut()(interner),
        Source::Done => None,
        Source::Chain(parts) => loop {
            let Some(front) = parts.front_mut() else {
                break None;
            };
            match pull(front, interner).await? {
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
    interner: &Interner,
) -> Result<Option<Rt::Value>, Rt::Error> {
    let mut i = start_op;
    while i < pipeline.ops.len() {
        match &mut pipeline.ops[i] {
            Op::Map(f) => val = Rt::call(f, vec![val]).await?,
            Op::Filter(f) => {
                let keep = Rt::call(f, vec![val.clone()]).await?;
                if !Rt::into_bool(keep)? {
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
                let items = expand(val, interner)?;
                let Some(first) = push_expansion(pipeline, i, items.into()) else {
                    return Ok(None);
                };
                val = first;
            }
            Op::FlatMap(f, expand) => {
                let mapped = Rt::call(f, vec![val]).await?;
                let items = expand(mapped, interner)?;
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
