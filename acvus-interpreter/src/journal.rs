//! The memory of a page: the holders a run fetches its contexts from and
//! commits them to (RFC-0014). A run reaches it through its runtime, which
//! several frames and spawned tasks share. A `Page` owns one and loads it from
//! a `Storage` when it opens (RFC-0090 rule 3); the runtime's own tooling
//! seeds one directly and drains the raw values a run assigned (RFC-0090
//! rule 6).

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use acvus_extern::{Borrows, Holding, Lendable, Owned, Shared};
use acvus_mir::ty::{PolyTy, Ty};
use acvus_utils::Interner;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::interpreter::Compilation;
use crate::runtime::AcvusRuntime;
use crate::value::Value;

// -- Held --------------------------------------------------------------

/// One context's value as a page keeps it: the runtime's word and the type
/// the word was crossed at. Neither is readable outside the runtime, so a
/// storage moves a holder whole and never reads inside it (RFC-0090 rule 6).
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
}

macro_rules! held_values {
    ($v:vis) => {
        impl Held {
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

// -- RuntimeContext ---------------------------------------------------

pub(crate) struct RuntimeContext {
    holders: RwLock<HashMap<String, Held>>,
    changed: Mutex<BTreeSet<String>>,
}

impl RuntimeContext {
    pub(crate) fn new(holders: HashMap<String, Held>) -> Self {
        RuntimeContext {
            holders: RwLock::new(holders),
            changed: Mutex::new(BTreeSet::new()),
        }
    }

    pub(crate) fn empty() -> Self {
        Self::new(HashMap::new())
    }

    pub(crate) fn take(&self, key: &str) -> Option<Held> {
        self.holders.write().remove(key)
    }

    pub(crate) fn set_changed(&self, key: &str, held: Held) {
        self.changed.lock().insert(key.to_owned());
        self.holders.write().insert(key.to_owned(), held);
    }

    pub(crate) fn set_unchanged(&self, key: &str, held: Held) {
        self.holders.write().insert(key.to_owned(), held);
    }

    pub(crate) fn holds(&self, key: &str) -> bool {
        self.holders.read().contains_key(key)
    }

    pub(crate) fn read(&self) -> RwLockReadGuard<'_, HashMap<String, Held>> {
        self.holders.read()
    }

    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, HashMap<String, Held>> {
        self.holders.write()
    }

    pub(crate) fn mark_changed(&self, key: &str) {
        self.changed.lock().insert(key.to_owned());
    }

    pub(crate) fn take_changed(&self) -> BTreeSet<String> {
        std::mem::take(&mut *self.changed.lock())
    }

    #[cfg(feature = "tooling")]
    pub(crate) fn take_writes(&self) -> Vec<ContextWrite> {
        let changed = self.take_changed();
        let mut holders = self.holders.write();
        changed
            .into_iter()
            .map(|key| {
                let held = holders.remove(&key).unwrap_or_else(|| {
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

    fn owned(v: Value) -> Owned<AcvusRuntime> {
        // SAFETY: each value is moved in by the caller and held nowhere else.
        unsafe { Owned::from_value(acvus_extern::Holding::new(), v) }
    }

    fn settled_ty_of(v: &Value) -> Ty {
        match v.is_string() {
            true => Ty::String,
            false => Ty::Int(acvus_mir::ty::IntTy::I64),
        }
    }

    fn held(v: Value) -> Held {
        let ty = settled_ty_of(&v);
        let compilation = crate::interpreter::InterpreterContext::new(
            &acvus_utils::Interner::new(),
            rustc_hash::FxHashMap::default(),
            Arc::new(crate::executor::SequentialExecutor),
        )
        .compilation;
        Held::new(owned(v), Arc::new(ty), compilation)
    }

    fn make_ctx(pairs: Vec<(&str, Value)>) -> RuntimeContext {
        let holders: HashMap<String, Held> = pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), held(v)))
            .collect();
        RuntimeContext::new(holders)
    }

    fn take(ctx: &RuntimeContext, key: &str) -> Option<Owned<AcvusRuntime>> {
        ctx.take(key).map(Held::into_value)
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
        ctx.set_changed("x", held(Value::int(2)));
        assert!(is_int(take(&ctx, "x"), 2));
    }

    #[cfg(feature = "tooling")]
    #[test]
    fn take_writes_hands_out_final_values() {
        let ctx = make_ctx(vec![("x", Value::int(1)), ("y", Value::int(9))]);
        ctx.set_changed("x", held(Value::int(2)));
        ctx.set_changed("x", held(Value::int(3)));
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
        let ctx = Arc::new(make_ctx(vec![("counter", Value::int(0))]));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let ctx_ref = Arc::clone(&ctx);
                std::thread::spawn(move || {
                    ctx_ref.set_changed("counter", held(Value::int(i)));
                    let _ = take(&ctx_ref, "counter");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
