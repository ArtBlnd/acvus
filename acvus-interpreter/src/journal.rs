//! Context - runtime context storage for the interpreter.
//!
//! `Context` is a single snapshot of context state. Read/write via `&self`
//! (interior mutability via RwLock). A context place is a storage like a
//! local (RFC-0018). The page remembers which keys a run assigned, and
//! `take_writes` hands their final values out.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, RwLock};

use acvus_extern::{ArgAt, Borrowed, Crossing, Declared, Holding, OneValue, Owned, Project};
use acvus_mir::ty::Ty;

use crate::host::{Contexts, Page, PageError, held_as};
use crate::runtime::AcvusRuntime;

// -- ContextWrite -----------------------------------------------------

/// The final value of a context a run assigned.
#[derive(Debug)]
pub struct ContextWrite {
    pub key: String,
    pub value: Owned<AcvusRuntime>,
}

// -- Context trait ---------------------------------------------------

/// Single snapshot of context state. Read/write via `&self`.
pub trait RuntimeContext: Send + Sync {
    /// Move the whole value out; the key is unset until `set`. `rt` is the
    /// run the page belongs to, which a page that loads a value reads it with.
    fn take(&self, rt: &AcvusRuntime, key: &str) -> Option<Owned<AcvusRuntime>>;
    /// # Safety
    /// `value` was crossed at the type the checker settled for `key`, which
    /// a typed page reads it back at (`Page::held_type`).
    unsafe fn set(&self, key: &str, value: Owned<AcvusRuntime>);
    /// The final value of every key `set` since the last drain, moved out.
    fn take_writes(&self) -> Vec<ContextWrite>;
}

// -- InMemoryContext -------------------------------------------------

/// In-memory Context backed by RwLock<HashMap>. No persistence.
/// Suitable for tests and the sequential executor.
pub struct InMemoryContext {
    data: RwLock<HashMap<String, Owned<AcvusRuntime>>>,
    assigned: Mutex<BTreeSet<String>>,
    declared: Option<Contexts>,
}

impl InMemoryContext {
    pub fn new(initial: HashMap<String, Owned<AcvusRuntime>>) -> Self {
        Self {
            data: RwLock::new(initial),
            assigned: Mutex::new(BTreeSet::new()),
            declared: None,
        }
    }

    pub fn empty() -> Self {
        Self::new(HashMap::new())
    }

    pub fn of(contexts: &Contexts) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            assigned: Mutex::new(BTreeSet::new()),
            declared: Some(contexts.clone()),
        }
    }

    pub fn read<T>(&mut self, key: &str) -> Result<<T as Borrowed>::Ref<'_>, PageError>
    where
        T: Declared + Project<AcvusRuntime>,
    {
        let contexts = declared(&self.declared, key)?;
        let interner = &contexts.rt().shared.interner;
        let ty = held_as::<T>(interner, key, contexts.settled().get(key))?;
        let data = self.data.get_mut().expect("page");
        let Some(value) = data.get(key) else {
            return Err(PageError::Absent {
                key: key.to_owned(),
            });
        };
        let table = T::table(ArgAt { interner, ty });
        // SAFETY: the page holds `key` at `ty`, which is `T`'s, and a value
        // enters the page only at the type it holds the key at.
        Ok(unsafe { T::project(contexts.rt(), value, &table) })
    }

    pub fn update<T, F, U>(&mut self, key: &str, f: F) -> Result<U, PageError>
    where
        T: Declared + Project<AcvusRuntime>,
        F: FnOnce(<T as Borrowed>::Mut<'_>) -> U,
    {
        let contexts = declared(&self.declared, key)?;
        let interner = &contexts.rt().shared.interner;
        let ty = held_as::<T>(interner, key, contexts.settled().get(key))?;
        let data = self.data.get_mut().expect("page");
        let Some(value) = data.get_mut(key) else {
            return Err(PageError::Absent {
                key: key.to_owned(),
            });
        };
        let table = T::table(ArgAt { interner, ty });
        // SAFETY: `&mut self` names the value exclusively, and a projection
        // writes inside the storage the word names, never the word itself.
        let word = unsafe { value.value_mut(Holding::new()) };
        // SAFETY: as `read`, exclusively.
        Ok(f(unsafe { T::project_mut(contexts.rt(), word, &table) }))
    }

    pub fn insert<T>(&mut self, key: &str, value: T) -> Result<(), PageError>
    where
        T: Declared + OneValue<AcvusRuntime>,
    {
        let contexts = declared(&self.declared, key)?;
        let interner = &contexts.rt().shared.interner;
        held_as::<T>(interner, key, contexts.settled().get(key))?;
        // SAFETY: `value` crosses at `T`, the type the page holds `key` at.
        let held = Owned::erased(unsafe { Crossing::new(contexts.rt()) }, value);
        self.data
            .get_mut()
            .expect("page")
            .insert(key.to_owned(), held);
        Ok(())
    }
}

fn declared<'c>(declared: &'c Option<Contexts>, key: &str) -> Result<&'c Contexts, PageError> {
    let Some(contexts) = declared else {
        return Err(PageError::Undeclared {
            key: key.to_owned(),
        });
    };
    Ok(contexts)
}

impl Page for InMemoryContext {
    fn held_type(&self, key: &str) -> Option<&Ty> {
        self.declared.as_ref()?.settled().get(key)
    }
}

impl RuntimeContext for InMemoryContext {
    fn take(&self, _: &AcvusRuntime, key: &str) -> Option<Owned<AcvusRuntime>> {
        self.data.write().unwrap().remove(key)
    }

    unsafe fn set(&self, key: &str, value: Owned<AcvusRuntime>) {
        self.assigned.lock().unwrap().insert(key.to_string());
        self.data.write().unwrap().insert(key.to_string(), value);
    }

    fn take_writes(&self) -> Vec<ContextWrite> {
        let assigned = std::mem::take(&mut *self.assigned.lock().unwrap());
        let mut data = self.data.write().unwrap();
        assigned
            .into_iter()
            .map(|key| {
                let value = data.remove(&key).unwrap_or_else(|| {
                    panic!("context '{key}' was assigned but holds no value at the end of the run")
                });
                ContextWrite { key, value }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    fn make_ctx(pairs: Vec<(&str, Value)>) -> InMemoryContext {
        let data: HashMap<String, Owned<AcvusRuntime>> = pairs
            .into_iter()
            // SAFETY: each value is moved in by the caller and held nowhere else.
            .map(|(k, v)| (k.to_string(), unsafe { Owned::from_value(acvus_extern::Holding::new(), v) }))
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
        assert!(is_str(ctx.take(&run(), "x"), "hi"));
        assert!(ctx.take(&run(), "x").is_none());
    }

    #[test]
    fn take_missing_returns_none() {
        let ctx = make_ctx(vec![]);
        assert!(ctx.take(&run(), "x").is_none());
    }

    #[test]
    fn set_after_take_restores_the_key() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        assert!(is_int(ctx.take(&run(), "x"), 1));
        // SAFETY: an integer word owns nothing, and this page declares no type.
        unsafe { ctx.set("x", Owned::from_value(acvus_extern::Holding::new(), Value::int(2))) };
        assert!(is_int(ctx.take(&run(), "x"), 2));
    }

    #[test]
    fn take_writes_hands_out_final_values() {
        let ctx = make_ctx(vec![("x", Value::int(1)), ("y", Value::int(9))]);
        // SAFETY: an integer word owns nothing, and this page declares no type.
        unsafe { ctx.set("x", Owned::from_value(acvus_extern::Holding::new(), Value::int(2))) };
        // SAFETY: as above.
        unsafe { ctx.set("x", Owned::from_value(acvus_extern::Holding::new(), Value::int(3))) };
        let writes = ctx.take_writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].key, "x");
        assert!(is_int(Some(writes.into_iter().next().unwrap().value), 3));
        assert!(
            ctx.take(&run(), "x").is_none(),
            "the drained value left the page"
        );
        assert!(is_int(ctx.take(&run(), "y"), 9), "an unassigned key stays");
    }

    #[test]
    fn concurrent_set_and_take() {
        use std::sync::Arc;
        let ctx = Arc::new(make_ctx(vec![("counter", Value::int(0))]));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let ctx_ref = Arc::clone(&ctx);
                std::thread::spawn(move || {
                    // SAFETY: an integer word owns nothing, and this page declares no type.
                    unsafe { ctx_ref.set("counter", Owned::from_value(acvus_extern::Holding::new(), Value::int(i))) };
                    let _ = ctx_ref.take(&run(), "counter");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
