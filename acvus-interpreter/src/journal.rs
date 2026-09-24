//! Context - runtime context storage for the interpreter.
//!
//! `Context` is a single snapshot of context state. Read/write via `&self`
//! (interior mutability via RwLock). A context place is a storage like a
//! local (RFC-0018). A host reads what a run wrote through the page's typed
//! methods; the runtime's own tooling, built with the `tooling` feature,
//! also drains the raw values a run assigned (RFC-0090 rule 6).

#[cfg(feature = "tooling")]
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "tooling")]
use std::sync::Mutex;
use std::sync::RwLock;

use acvus_extern::{Borrows, Crossing, Declared, Holding, Lendable, OneValue, Owned, Shared};
use acvus_mir::ty::Ty;

use crate::host::{Contexts, Page, PageError, held_as};
use crate::runtime::AcvusRuntime;

// -- Held --------------------------------------------------------------

/// One context's value as a page keeps it: the runtime's word and the type
/// the word was crossed at. Neither is readable outside the runtime, so a
/// storage moves a holder whole and never reads inside it (RFC-0090 rule 6).
/// A run reads a holder only at the type it carries: a storage that gives
/// back another key's holder, or another compilation's, is refused at the
/// fetch before the word is read.
pub struct Held {
    value: Owned<AcvusRuntime>,
    ty: Arc<Ty>,
}

impl Held {
    pub(crate) fn new(value: Owned<AcvusRuntime>, ty: Arc<Ty>) -> Self {
        Held { value, ty }
    }

    pub(crate) fn ty(&self) -> &Ty {
        &self.ty
    }

    /// Lend the value to `f` at the type the holder carries, as the page's
    /// `key`.
    pub(crate) fn lend<Q, O, F>(&mut self, contexts: &Contexts, key: &str, f: F) -> Result<O, PageError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let interner = &contexts.rt().shared.interner;
        // SAFETY: `&mut self` names the holder exclusively for the call, and
        // a lent parameter writes inside the storage the word names or the
        // word in place, never another holder's.
        let word = unsafe { self.value.value_mut(Holding::new()) };
        // SAFETY: the word was crossed at the type the holder carries, and
        // `word` is its only live name for the call.
        unsafe { acvus_extern::lend(contexts.rt(), interner, word, &self.ty, f) }.map_err(|asked| {
            PageError::Mismatched {
                key: key.to_owned(),
                held: self.ty.display(interner).to_string(),
                asked: asked.display(interner).to_string(),
            }
        })
    }
}

macro_rules! held_values {
    ($v:vis) => {
        impl Held {
            /// The holder's value, moved out: the runtime's and its tooling's.
            $v fn into_value(self) -> Owned<AcvusRuntime> {
                self.value
            }
        }
    };
}
tooling_vis!(held_values);

// -- ContextWrite -----------------------------------------------------

/// The final value of a context a run assigned.
#[cfg(feature = "tooling")]
#[derive(Debug)]
pub struct ContextWrite {
    pub key: String,
    pub value: Owned<AcvusRuntime>,
}

// -- Context trait ---------------------------------------------------

/// A storage a run's contexts are fetched from and committed to. It moves
/// whole holders, which it cannot open (RFC-0090 rule 6): a wrong
/// implementation loses or misplaces a holder, which the run refuses when it
/// fetches, and never makes the run read a value at another type.
pub trait RuntimeContext: Send + Sync {
    /// Move the holder out; the key is unset until `set`. `rt` is the run
    /// the page belongs to, which a page that loads a value decodes it with.
    fn take(&self, rt: &AcvusRuntime, key: &str) -> Option<Held>;
    fn set(&self, key: &str, held: Held);
    /// Whether `take` of `key` would give a holder.
    fn holds(&self, key: &str) -> bool;
    /// The final value of every key `set` since the last drain, moved out.
    #[cfg(feature = "tooling")]
    fn take_writes(&self) -> Vec<ContextWrite>;
}

// -- InMemoryContext -------------------------------------------------

/// In-memory Context backed by RwLock<HashMap>. No persistence.
/// Suitable for tests and the sequential executor.
pub struct InMemoryContext {
    data: RwLock<HashMap<String, Held>>,
    #[cfg(feature = "tooling")]
    assigned: Mutex<BTreeSet<String>>,
    solved: Option<Contexts>,
}

impl InMemoryContext {
    pub fn empty() -> Self {
        Self::holding(HashMap::new(), None)
    }

    pub fn of(contexts: &Contexts) -> Self {
        Self::holding(HashMap::new(), Some(contexts.clone()))
    }

    fn holding(data: HashMap<String, Held>, solved: Option<Contexts>) -> Self {
        Self {
            data: RwLock::new(data),
            #[cfg(feature = "tooling")]
            assigned: Mutex::new(BTreeSet::new()),
            solved,
        }
    }

    /// Lend `key`'s value to `f`, whose parameter crosses as a handler's
    /// shared one does (RFC-0090 rule 3).
    pub fn with<Q, F, O>(&mut self, key: &str, f: F) -> Result<O, PageError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let contexts = in_graph(&self.solved, key)?;
        let Some(held) = self.data.get_mut().expect("page").get_mut(key) else {
            return Err(PageError::Absent {
                key: key.to_owned(),
            });
        };
        held.lend(contexts, key, f)
    }

    /// Lend `key`'s value to `f` exclusively, whose parameter crosses as a
    /// handler's does, shared or exclusive.
    pub fn with_mut<Q, F, O>(&mut self, key: &str, f: F) -> Result<O, PageError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let contexts = in_graph(&self.solved, key)?;
        let Some(held) = self.data.get_mut().expect("page").get_mut(key) else {
            return Err(PageError::Absent {
                key: key.to_owned(),
            });
        };
        held.lend(contexts, key, f)
    }

    pub fn insert<T>(&mut self, key: &str, value: T) -> Result<(), PageError>
    where
        T: Declared + OneValue<AcvusRuntime>,
    {
        let contexts = in_graph(&self.solved, key)?;
        let ty = held_as::<T>(&contexts.rt().shared.interner, key, contexts.solved().get(key))?;
        // SAFETY: `value` crosses at `T`, the type the page holds `key` at.
        let value = Owned::erased(unsafe { Crossing::new(contexts.rt()) }, value);
        self.data
            .get_mut()
            .expect("page")
            .insert(key.to_owned(), Held::new(value, Arc::new(ty.clone())));
        Ok(())
    }
}

/// A page seeded with raw holders, each at the type the tooling states for
/// it, in `SpacePage::new`'s seed shape: the runtime's and its tooling's
/// (RFC-0090 rule 6). A host opens a page with `of` and fills it with
/// `insert`.
macro_rules! in_memory_values {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl InMemoryContext {
            $v fn new(initial: HashMap<String, (Ty, Owned<AcvusRuntime>)>) -> Self {
                let data = initial
                    .into_iter()
                    .map(|(key, (ty, value))| (key, Held::new(value, Arc::new(ty))))
                    .collect();
                Self::holding(data, None)
            }
        }
    };
}
tooling_vis!(in_memory_values);

/// The compilation the page was opened for, where it names `key`.
pub(crate) fn in_graph<'c>(solved: &'c Option<Contexts>, key: &str) -> Result<&'c Contexts, PageError> {
    match solved {
        Some(contexts) if contexts.solved().contains_key(key) => Ok(contexts),
        _ => Err(PageError::NotInGraph {
            key: key.to_owned(),
        }),
    }
}

impl Page for InMemoryContext {
    fn held_type(&self, key: &str) -> Option<&Ty> {
        self.solved.as_ref()?.solved().get(key)
    }
}

impl RuntimeContext for InMemoryContext {
    fn take(&self, _: &AcvusRuntime, key: &str) -> Option<Held> {
        self.data.write().unwrap().remove(key)
    }

    fn holds(&self, key: &str) -> bool {
        self.data.read().unwrap().contains_key(key)
    }

    fn set(&self, key: &str, held: Held) {
        #[cfg(feature = "tooling")]
        self.assigned.lock().unwrap().insert(key.to_string());
        self.data.write().unwrap().insert(key.to_string(), held);
    }

    #[cfg(feature = "tooling")]
    fn take_writes(&self) -> Vec<ContextWrite> {
        let assigned = std::mem::take(&mut *self.assigned.lock().unwrap());
        let mut data = self.data.write().unwrap();
        assigned
            .into_iter()
            .map(|key| {
                let held = data.remove(&key).unwrap_or_else(|| {
                    panic!("context '{key}' was assigned but holds no value at the end of the run")
                });
                ContextWrite {
                    key,
                    value: held.into_value(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    fn owned(v: Value) -> Owned<AcvusRuntime> {
        // SAFETY: each value is moved in by the caller and held nowhere else.
        unsafe { Owned::from_value(acvus_extern::Holding::new(), v) }
    }

    /// The settled type of the one kind of value each test stores.
    fn ty_of(v: &Value) -> Ty {
        match v.is_string() {
            true => Ty::String,
            false => Ty::Int(acvus_mir::ty::IntTy::I64),
        }
    }

    fn held(v: Value) -> Held {
        let ty = ty_of(&v);
        Held::new(owned(v), Arc::new(ty))
    }

    fn make_ctx(pairs: Vec<(&str, Value)>) -> InMemoryContext {
        let data: HashMap<String, (Ty, Owned<AcvusRuntime>)> = pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), (ty_of(&v), owned(v))))
            .collect();
        InMemoryContext::new(data)
    }

    /// An `InMemoryContext` reads nothing off the run it is handed, so these
    /// tests hand it one that holds nothing.
    fn run() -> AcvusRuntime {
        crate::interpreter::InterpreterContext::new(
            &acvus_utils::Interner::new(),
            rustc_hash::FxHashMap::default(),
            std::sync::Arc::new(crate::executor::SequentialExecutor),
        )
        .runtime_over_an_empty_page()
    }

    fn take(ctx: &InMemoryContext, key: &str) -> Option<Owned<AcvusRuntime>> {
        ctx.take(&run(), key).map(Held::into_value)
    }

    fn is_int(v: Option<Owned<AcvusRuntime>>, n: i64) -> bool {
        matches!(v, Some(value) if value.kind().is_inline() && value.bits() == n as u64)
    }

    fn is_str(v: Option<Owned<AcvusRuntime>>, s: &str) -> bool {
        let Some(v) = v else {
            return false;
        };
        // SAFETY: is_string checked the vtable.
        v.is_string() && unsafe { v.as_str() } == s
    }

    #[test]
    fn take_moves_the_value_out() {
        let ctx = make_ctx(vec![("x", Value::string("hi"))]);
        assert!(is_str(take(&ctx, "x"), "hi"));
        assert!(take(&ctx, "x").is_none());
    }

    #[test]
    fn take_missing_returns_none() {
        let ctx = make_ctx(vec![]);
        assert!(take(&ctx, "x").is_none());
    }

    #[test]
    fn set_after_take_restores_the_key() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        assert!(is_int(take(&ctx, "x"), 1));
        ctx.set("x", held(Value::int(2)));
        assert!(is_int(take(&ctx, "x"), 2));
    }

    #[cfg(feature = "tooling")]
    #[test]
    fn take_writes_hands_out_final_values() {
        let ctx = make_ctx(vec![("x", Value::int(1)), ("y", Value::int(9))]);
        ctx.set("x", held(Value::int(2)));
        ctx.set("x", held(Value::int(3)));
        let writes = ctx.take_writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].key, "x");
        assert!(is_int(Some(writes.into_iter().next().unwrap().value), 3));
        assert!(
            take(&ctx, "x").is_none(),
            "the drained value left the page"
        );
        assert!(is_int(take(&ctx, "y"), 9), "an unassigned key stays");
    }

    #[test]
    fn concurrent_set_and_take() {
        use std::sync::Arc;
        let ctx = Arc::new(make_ctx(vec![("counter", Value::int(0))]));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let ctx_ref = Arc::clone(&ctx);
                std::thread::spawn(move || {
                                ctx_ref.set("counter", held(Value::int(i)));
                    let _ = take(&ctx_ref, "counter");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
