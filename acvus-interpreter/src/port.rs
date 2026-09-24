//! A run's reach into the storage behind its page (RFC-0090 rule 3,
//! RFC-0025 rule 2): a `Fetch` loads, a `Commit` that wrote stores, one
//! that did not restores, and nothing else reaches the storage.

use std::collections::{BTreeSet, HashMap};
use std::ptr::NonNull;
use std::sync::Arc;

use acvus_extern::{Borrows, Holding, Lendable, Owned, Shared};
use acvus_mir::ty::{PolyTy, Ty};
use acvus_utils::Interner;
use futures::channel::{mpsc, oneshot};
use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::host::{Codec, HostError, Part, Storage, StorageError};
use crate::interpreter::Compilation;
use crate::runtime::AcvusRuntime;
use crate::value::Value;

// -- Held --------------------------------------------------------------

/// One context's value as a storage keeps it: the runtime's word and the
/// type the word was crossed at. Neither is readable outside the runtime, so
/// a storage moves a holder whole and never reads inside it (RFC-0090
/// rule 6).
pub struct Held {
    value: Owned<AcvusRuntime>,
    ty: Arc<Ty>,
    made_by: Compilation,
}

impl Held {
    pub(crate) fn new(value: Owned<AcvusRuntime>, ty: Arc<Ty>, made_by: Compilation) -> Self {
        Held { value, ty, made_by }
    }

    pub(crate) fn ty(&self) -> &Ty {
        &self.ty
    }

    pub(crate) fn made_by(&self) -> Compilation {
        self.made_by
    }

    pub(crate) fn lend<Q, O, F>(&self, rt: &AcvusRuntime, interner: &Interner, f: F) -> Result<O, PolyTy>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        // SAFETY: the word was crossed at the type the holder carries, the
        // holder is borrowed for the call, and a shared parameter writes
        // nothing.
        unsafe { acvus_extern::lend(rt, interner, &self.value, &self.ty, f) }
    }

    pub(crate) fn lend_mut<Q, O, F>(&mut self, rt: &AcvusRuntime, interner: &Interner, f: F) -> Result<O, PolyTy>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        // SAFETY: `&mut self` names the holder exclusively for the call, and
        // a lent parameter writes inside the storage the word names or the
        // word in place, never another holder's.
        let word = unsafe { self.value.value_mut(Holding::new()) };
        // SAFETY: the word was crossed at the type the holder carries, and
        // `word` is its only live name for the call.
        unsafe { acvus_extern::lend(rt, interner, word, &self.ty, f) }
    }

    /// # Safety
    /// The caller writes no word into the value; it may edit inside the
    /// storage the word names.
    pub(crate) unsafe fn value_mut(&mut self) -> &mut Value {
        // SAFETY: the caller's contract, and `&mut self` names the holder
        // exclusively.
        unsafe { self.value.value_mut(Holding::new()) }
    }

    /// The word, moved into a register the checker typed at the holder's
    /// type, which owns it from then on.
    pub(crate) fn into_word(self) -> Value {
        // SAFETY: the caller defines the register the word goes to, which
        // owns it from then on.
        self.value.into_value(unsafe { Holding::new() })
    }
}

macro_rules! held_values {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl Held {
            $v fn into_value(self) -> Owned<AcvusRuntime> {
                self.value
            }
        }
    };
}
tooling_vis!(held_values);

/// The final value of a context a run stored.
#[cfg(feature = "tooling")]
#[derive(Debug)]
pub struct ContextWrite {
    pub key: String,
    pub value: Owned<AcvusRuntime>,
}

// -- Ending a run ------------------------------------------------------

pub(crate) struct Ended(pub(crate) HostError);

/// A synchronous frame, or a closure an extern handler called, has no way
/// out but the unwinder, so a run ends the way RFC-0048 rule 8 ends a
/// run-time failure: it releases nothing. `resume_unwind` runs no panic
/// hook, so nothing is printed.
pub(crate) fn end_run(error: HostError) -> ! {
    std::panic::resume_unwind(Box::new(Ended(error)))
}

pub(crate) fn ended(payload: Box<dyn std::any::Any + Send>) -> HostError {
    match payload.downcast::<Ended>() {
        Ok(ended) => ended.0,
        Err(trap) => HostError::Trapped {
            message: trap_message(trap),
        },
    }
}

const TRAP_WITHOUT_MESSAGE: &str = "the run trapped with a payload that is not a message";

fn trap_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_owned();
    }
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(_) => TRAP_WITHOUT_MESSAGE.to_owned(),
    }
}

// -- The gate ----------------------------------------------------------

/// A storage the page borrows, reached by the run's frames and tasks under
/// synchronous access. `close` takes the lock each access holds, so it waits
/// for an access in progress and none follows it.
pub(crate) struct Gate {
    reach: Mutex<Option<Reach>>,
}

struct Reach(NonNull<dyn Storage>);

// SAFETY: `Storage: Send`, and a `Reach` is dereferenced only under the
// gate's lock, one access at a time.
unsafe impl Send for Reach {}

impl Gate {
    /// # Safety
    /// `storage` is not touched otherwise, and outlives every access, until
    /// `close` returns.
    pub(crate) unsafe fn open(storage: &mut dyn Storage) -> Gate {
        let reach: NonNull<dyn Storage + '_> = NonNull::from(storage);
        // SAFETY: the caller's contract bounds every dereference by `close`,
        // which the erased lifetime does not.
        let reach: NonNull<dyn Storage + 'static> = unsafe { std::mem::transmute(reach) };
        Gate {
            reach: Mutex::new(Some(Reach(reach))),
        }
    }

    pub(crate) fn close(&self) {
        *self.reach.lock() = None;
    }

    fn with<T, F>(&self, f: F) -> Result<T, StorageError>
    where
        F: FnOnce(&mut dyn Storage) -> Result<T, StorageError>,
    {
        let mut reach = self.reach.lock();
        let Some(Reach(storage)) = reach.as_mut() else {
            return Err(StorageError::new(
                "the run's page was closed before this access".to_owned(),
            ));
        };
        // SAFETY: `open`'s contract: until `close`, which this lock excludes,
        // the storage outlives the access and nothing else touches it.
        f(unsafe { storage.as_mut() })
    }
}

// -- Requests ----------------------------------------------------------

/// Under waited access the run's own future serves each request with the
/// storage it borrows, so no access outlives the borrow.
pub(crate) enum Request {
    Load {
        key: String,
        reply: oneshot::Sender<Result<Option<Held>, StorageError>>,
    },
    Store {
        key: String,
        held: Held,
        reply: oneshot::Sender<Result<(), StorageError>>,
    },
    Restore {
        key: String,
        held: Held,
        reply: oneshot::Sender<Result<(), StorageError>>,
    },
}

pub(crate) type Requests = mpsc::UnboundedReceiver<Request>;

// -- Seeded (tooling) --------------------------------------------------

#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
pub(crate) struct Seeded {
    holders: HashMap<String, Held>,
    stored: BTreeSet<String>,
}

// -- Port --------------------------------------------------------------

enum Reaches {
    /// A runtime a program lends through, which runs nothing that touches
    /// a context; an access ends the run.
    Nothing,
    Gate(Gate),
    Queue(mpsc::UnboundedSender<Request>),
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    Seeded(Mutex<Seeded>),
}

/// A reach an operation that cannot wait can use.
enum Now<'p> {
    Gate(&'p Gate),
    Seeded(parking_lot::MutexGuard<'p, Seeded>),
}

pub(crate) struct Port {
    reaches: Reaches,
    /// The keys whose init a fetch ran, in the order it ran them.
    filled: Mutex<Vec<String>>,
}

pub(crate) type Loaded = Result<Option<Held>, HostError>;

impl Port {
    fn of(reaches: Reaches) -> Arc<Port> {
        Arc::new(Port {
            reaches,
            filled: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn nothing() -> Arc<Port> {
        Port::of(Reaches::Nothing)
    }

    pub(crate) fn gate(gate: Gate) -> Arc<Port> {
        Port::of(Reaches::Gate(gate))
    }

    pub(crate) fn queue() -> (Arc<Port>, Requests) {
        let (sender, requests) = mpsc::unbounded();
        (Port::of(Reaches::Queue(sender)), requests)
    }

    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    pub(crate) fn seeded(holders: HashMap<String, Held>) -> Arc<Port> {
        Port::of(Reaches::Seeded(Mutex::new(Seeded {
            holders,
            stored: BTreeSet::new(),
        })))
    }

    pub(crate) fn close(&self) {
        if let Reaches::Gate(gate) = &self.reaches {
            gate.close();
        }
    }

    pub(crate) fn note_filled(&self, key: &str) {
        self.filled.lock().push(key.to_owned());
    }

    pub(crate) fn take_filled(&self) -> Vec<String> {
        std::mem::take(&mut *self.filled.lock())
    }

    fn no_storage() -> StorageError {
        StorageError::new("this run has no page to reach a context through".to_owned())
    }

    fn now(&self) -> Result<Now<'_>, HostError> {
        match &self.reaches {
            Reaches::Gate(gate) => Ok(Now::Gate(gate)),
            Reaches::Seeded(seeded) => Ok(Now::Seeded(seeded.lock())),
            Reaches::Nothing => Err(HostError::Storage(Port::no_storage())),
            Reaches::Queue(_) => Err(HostError::Storage(StorageError::new(
                "a program compiled for synchronous access runs over a storage whose access waits"
                    .to_owned(),
            ))),
        }
    }

    pub(crate) fn load(&self, rt: &AcvusRuntime, key: &str) -> Loaded {
        match self.now()? {
            Now::Gate(gate) => gate
                .with(|storage| storage.load(key, &Codec::of(rt)))
                .map_err(HostError::Storage),
            Now::Seeded(mut seeded) => Ok(seeded.holders.remove(key)),
        }
    }

    pub(crate) fn store(&self, key: &str, held: Held) -> Result<(), HostError> {
        match self.now()? {
            Now::Gate(gate) => gate
                .with(|storage| storage.store(key, held))
                .map_err(HostError::Storage),
            Now::Seeded(mut seeded) => {
                seeded.stored.insert(key.to_owned());
                seeded.holders.insert(key.to_owned(), held);
                Ok(())
            }
        }
    }

    pub(crate) fn restore(&self, key: &str, held: Held) -> Result<(), HostError> {
        match self.now()? {
            Now::Gate(gate) => gate
                .with(|storage| storage.restore(key, held))
                .map_err(HostError::Storage),
            Now::Seeded(mut seeded) => {
                seeded.holders.entry(key.to_owned()).or_insert(held);
                Ok(())
            }
        }
    }

    pub(crate) fn load_waited(self: &Arc<Self>, rt: &AcvusRuntime, key: &str) -> BoxFuture<'static, Loaded> {
        let Reaches::Queue(queue) = &self.reaches else {
            return Box::pin(std::future::ready(self.load(rt, key)));
        };
        let (reply, answer) = oneshot::channel();
        let sent = queue.unbounded_send(Request::Load {
            key: key.to_owned(),
            reply,
        });
        Box::pin(async move {
            sent.map_err(|_| HostError::Storage(Port::gone()))?;
            answer
                .await
                .map_err(|_| HostError::Storage(Port::gone()))?
                .map_err(HostError::Storage)
        })
    }

    pub(crate) fn store_waited(self: &Arc<Self>, key: &str, held: Held, wrote: bool) -> BoxFuture<'static, Result<(), HostError>> {
        let Reaches::Queue(queue) = &self.reaches else {
            let done = match wrote {
                true => self.store(key, held),
                false => self.restore(key, held),
            };
            return Box::pin(std::future::ready(done));
        };
        let (reply, answer) = oneshot::channel();
        let key = key.to_owned();
        let request = match wrote {
            true => Request::Store { key, held, reply },
            false => Request::Restore { key, held, reply },
        };
        let sent = queue.unbounded_send(request);
        Box::pin(async move {
            sent.map_err(|_| HostError::Storage(Port::gone()))?;
            answer
                .await
                .map_err(|_| HostError::Storage(Port::gone()))?
                .map_err(HostError::Storage)
        })
    }

    fn gone() -> StorageError {
        StorageError::new("the run's page is gone, and this access reached no storage".to_owned())
    }

    #[cfg(feature = "tooling")]
    pub(crate) fn take_writes(&self) -> Vec<ContextWrite> {
        let Reaches::Seeded(seeded) = &self.reaches else {
            return Vec::new();
        };
        let mut seeded = seeded.lock();
        let stored = std::mem::take(&mut seeded.stored);
        stored
            .into_iter()
            .map(|key| {
                let held = seeded.holders.remove(&key).unwrap_or_else(|| {
                    panic!("context '{key}' was stored but holds no value at the end of the run")
                });
                ContextWrite {
                    key,
                    value: held.into_value(),
                }
            })
            .collect()
    }
}

pub(crate) async fn serve<S, F>(storage: &mut S, codec: &Codec<'_>, mut requests: Requests, run: F) -> F::Output
where
    S: crate::host::AsyncStorage,
    F: std::future::Future,
{
    use futures::StreamExt;
    use futures::future::{Either, select};
    let mut run = std::pin::pin!(run);
    loop {
        let request = match select(run.as_mut(), requests.next()).await {
            Either::Left((done, _)) => return done,
            Either::Right((request, _)) => request,
        };
        let Some(request) = request else {
            return run.await;
        };
        let (key, unread) = match request {
            Request::Load { key, reply } => {
                let answer = storage.load(&key, codec).await;
                (key, reply.send(answer).err().map(Unread::Load))
            }
            Request::Store { key, held, reply } => {
                let answer = storage.store(&key, held).await;
                (key, reply.send(answer).err().map(Unread::Stored))
            }
            Request::Restore { key, held, reply } => {
                let answer = storage.restore(&key, held).await;
                (key, reply.send(answer).err().map(Unread::Stored))
            }
        };
        if let Some(unread) = unread {
            unread.log(&key);
        }
    }
}

/// An answer whose asker was dropped with its run before it arrived.
enum Unread {
    Load(Result<Option<Held>, StorageError>),
    Stored(Result<(), StorageError>),
}

impl Unread {
    fn log(self, key: &str) {
        match self {
            Unread::Load(Ok(Some(_))) => tracing::warn!(
                key,
                "a loaded holder reached no run and was released; the storage no longer holds it"
            ),
            Unread::Load(Err(error)) | Unread::Stored(Err(error)) => {
                tracing::warn!(key, %error, "a storage error reached no run")
            }
            Unread::Load(Ok(None)) | Unread::Stored(Ok(())) => {}
        }
    }
}

// -- Checking a loaded holder ------------------------------------------

/// A holder another compilation made, or one at another type than
/// `settled`, is refused before its word is read (RFC-0090 rules 4, 6).
pub(crate) fn refusal_of(rt: &AcvusRuntime, key: &str, held: &Held, settled: &Ty) -> Option<HostError> {
    if held.made_by() != rt.shared.compilation {
        let message = format!("the storage gave `@{key}` a holder another compilation made");
        return Some(HostError::Storage(StorageError::new(message)));
    }
    if held.ty().same_erased(settled) {
        return None;
    }
    let interner = &rt.shared.interner;
    Some(HostError::Mismatched {
        what: Part::Context(key.to_owned()),
        held: held.ty().display(interner).to_string(),
        asked: settled.display(interner).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held(v: Value) -> Held {
        let ty = match v.is_string() {
            true => Ty::String,
            false => Ty::Int(acvus_mir::ty::IntTy::I64),
        };
        let compilation = crate::interpreter::InterpreterContext::new(
            &acvus_utils::Interner::new(),
            rustc_hash::FxHashMap::default(),
            Arc::new(crate::executor::SequentialExecutor),
        )
        .compilation;
        // SAFETY: each value is moved in by the caller and held nowhere else.
        let value = unsafe { Owned::from_value(Holding::new(), v) };
        Held::new(value, Arc::new(ty), compilation)
    }

    #[derive(Default)]
    struct Shelf(HashMap<String, Held>);

    impl Storage for Shelf {
        fn load(&mut self, key: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
            Ok(self.0.remove(key))
        }

        fn store(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
            self.0.insert(key.to_owned(), held);
            Ok(())
        }

        fn restore(&mut self, key: &str, held: Held) -> Result<(), StorageError> {
            self.0.entry(key.to_owned()).or_insert(held);
            Ok(())
        }

        fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
            Ok(())
        }
    }

    #[test]
    fn a_closed_gate_reaches_no_storage() {
        let mut shelf = Shelf::default();
        // SAFETY: the gate is closed before `shelf` is touched again.
        let port = Port::gate(unsafe { Gate::open(&mut shelf) });
        port.store("x", held(Value::int(1))).expect("the gate is open");
        port.close();
        assert!(matches!(
            port.store("x", held(Value::int(2))),
            Err(HostError::Storage(_))
        ));
        assert_eq!(shelf.0.len(), 1, "the access after `close` stored nothing");
    }
}
