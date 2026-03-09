use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use parking_lot::Mutex;

use crate::Astr;

// ---------------------------------------------------------------------------
// Signal — unified coroutine communication
// ---------------------------------------------------------------------------

enum Signal<V> {
    Empty,
    Yield(V),
    NeedContext {
        name: Astr,
        bindings: HashMap<Astr, V>,
    },
    ContextReady(Arc<V>),
}

// ---------------------------------------------------------------------------
// YieldHandle — producer side
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct YieldHandle<V> {
    shared: Arc<Mutex<Signal<V>>>,
}

impl<V> YieldHandle<V> {
    pub fn yield_val(&self, value: V) -> YieldFuture<'_, V> {
        YieldFuture {
            shared: &self.shared,
            value: Some(value),
        }
    }

    pub fn request_context(&self, name: Astr) -> ContextFuture<'_, V> {
        ContextFuture {
            shared: &self.shared,
            name: Some(name),
            bindings: HashMap::new(),
        }
    }

    pub fn request_context_with(
        &self,
        name: Astr,
        bindings: HashMap<Astr, V>,
    ) -> ContextFuture<'_, V> {
        ContextFuture {
            shared: &self.shared,
            name: Some(name),
            bindings,
        }
    }
}

// ---------------------------------------------------------------------------
// YieldFuture
// ---------------------------------------------------------------------------

pub struct YieldFuture<'a, V> {
    shared: &'a Arc<Mutex<Signal<V>>>,
    value: Option<V>,
}

impl<V> Future for YieldFuture<'_, V>
where
    V: Unpin,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if let Some(value) = this.value.take() {
            *this.shared.lock() = Signal::Yield(value);
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

// ---------------------------------------------------------------------------
// ContextFuture
// ---------------------------------------------------------------------------

pub struct ContextFuture<'a, V> {
    shared: &'a Arc<Mutex<Signal<V>>>,
    name: Option<Astr>,
    bindings: HashMap<Astr, V>,
}

impl<V> Future for ContextFuture<'_, V>
where
    V: Unpin,
{
    type Output = Arc<V>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Arc<V>> {
        let this = self.get_mut();
        let mut signal = this.shared.lock();

        if let Some(name) = this.name.take() {
            *signal = Signal::NeedContext {
                name,
                bindings: std::mem::take(&mut this.bindings),
            };
            Poll::Pending
        } else if let Signal::ContextReady(_) = &*signal {
            let Signal::ContextReady(value) = std::mem::replace(&mut *signal, Signal::Empty) else {
                unreachable!()
            };
            Poll::Ready(value)
        } else {
            Poll::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// Resume API — public types
// ---------------------------------------------------------------------------

pub struct ResumeKey<V>(ResumeKeyInner<V>);

enum ResumeKeyInner<V> {
    Start,
    Context(Arc<V>),
}

pub enum Stepped<V, E> {
    Emit(EmitStepped<V>),
    NeedContext(NeedContextStepped<V>),
    Done,
    Error(E),
}

pub struct EmitStepped<V> {
    value: V,
    key: ResumeKey<V>,
}

impl<V> EmitStepped<V> {
    pub fn into_parts(self) -> (V, ResumeKey<V>) {
        (self.value, self.key)
    }
}

pub struct NeedContextStepped<V> {
    name: Astr,
    bindings: HashMap<Astr, V>,
}

impl<V> NeedContextStepped<V> {
    pub fn name(&self) -> Astr {
        self.name
    }

    pub fn bindings(&self) -> &HashMap<Astr, V> {
        &self.bindings
    }

    pub fn into_parts(self) -> (Astr, HashMap<Astr, V>) {
        (self.name, self.bindings)
    }

    pub fn into_key(self, value: Arc<V>) -> ResumeKey<V> {
        ResumeKey(ResumeKeyInner::Context(value))
    }
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

pub struct Coroutine<V, E> {
    shared: Arc<Mutex<Signal<V>>>,
    fut: Option<Pin<Box<dyn Future<Output = Result<(), E>> + Send>>>,
}

impl<V, E> Coroutine<V, E> {
    pub fn resume(&mut self, key: ResumeKey<V>) -> ResumeFuture<'_, V, E> {
        if let ResumeKeyInner::Context(arc) = key.0 {
            *self.shared.lock() = Signal::ContextReady(arc);
        }
        ResumeFuture { coroutine: self }
    }
}

// ---------------------------------------------------------------------------
// ResumeFuture — async resume
// ---------------------------------------------------------------------------

pub struct ResumeFuture<'a, V, E> {
    coroutine: &'a mut Coroutine<V, E>,
}

impl<V, E> Future for ResumeFuture<'_, V, E> {
    type Output = Stepped<V, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Stepped<V, E>> {
        let this = self.get_mut();
        let fut = match &mut this.coroutine.fut {
            Some(f) => f.as_mut(),
            None => return Poll::Ready(Stepped::Done),
        };

        let poll_result = fut.poll(cx);
        let mut signal = this.coroutine.shared.lock();

        match std::mem::replace(&mut *signal, Signal::Empty) {
            Signal::Yield(value) => {
                if poll_result.is_ready() {
                    this.coroutine.fut = None;
                }
                Poll::Ready(Stepped::Emit(EmitStepped {
                    value,
                    key: ResumeKey(ResumeKeyInner::Start),
                }))
            }
            Signal::NeedContext { name, bindings } => {
                Poll::Ready(Stepped::NeedContext(NeedContextStepped { name, bindings }))
            }
            other => {
                // put back unconsumed signal (e.g. ContextReady mid-processing)
                *signal = other;
                match poll_result {
                    Poll::Ready(Ok(())) => {
                        this.coroutine.fut = None;
                        Poll::Ready(Stepped::Done)
                    }
                    Poll::Ready(Err(e)) => {
                        this.coroutine.fut = None;
                        Poll::Ready(Stepped::Error(e))
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

pub fn coroutine<V, E, F, Fut>(f: F) -> (Coroutine<V, E>, ResumeKey<V>)
where
    F: FnOnce(YieldHandle<V>) -> Fut,
    Fut: Future<Output = Result<(), E>> + Send + 'static,
{
    let shared = Arc::new(Mutex::new(Signal::Empty));
    let handle = YieldHandle {
        shared: Arc::clone(&shared),
    };
    let fut = f(handle);
    (
        Coroutine {
            shared,
            fut: Some(Box::pin(fut)),
        },
        ResumeKey(ResumeKeyInner::Start),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Interner;

    async fn step<V, E>(
        co: &mut Coroutine<V, E>,
        key: ResumeKey<V>,
    ) -> Stepped<V, E> {
        co.resume(key).await
    }

    #[tokio::test]
    async fn empty_coroutine() {
        let (mut co, key) = coroutine::<i32, (), _, _>(|_handle| async move { Ok(()) });
        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn single_yield() {
        let (mut co, key) = coroutine::<_, (), _, _>(|handle| async move {
            handle.yield_val(42).await;
            Ok(())
        });

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, 42);

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn multiple_yields() {
        let (mut co, key) = coroutine::<_, (), _, _>(|handle| async move {
            handle.yield_val(1).await;
            handle.yield_val(2).await;
            handle.yield_val(3).await;
            Ok(())
        });

        let mut key = key;
        for expected in [1, 2, 3] {
            let Stepped::Emit(emit) = step(&mut co, key).await else {
                panic!("expected Emit({expected})");
            };
            let (value, next_key) = emit.into_parts();
            assert_eq!(value, expected);
            key = next_key;
        }

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_request() {
        let interner = Interner::new();
        let user = interner.intern("user");
        let (mut co, key) = coroutine::<_, (), _, _>(|handle| async move {
            let ctx = handle.request_context(user).await;
            handle.yield_val(format!("got: {ctx}")).await;
            Ok(())
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), user);
        assert!(need.bindings().is_empty());

        let key = need.into_key(Arc::new("alice".to_string()));
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "got: alice");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_request_with_bindings() {
        let interner = Interner::new();
        let user = interner.intern("user");
        let role = interner.intern("role");
        let (mut co, key) = coroutine::<_, (), _, _>(|handle| async move {
            let mut bindings = HashMap::new();
            bindings.insert(role, "admin".into());
            let ctx = handle
                .request_context_with(user, bindings)
                .await;
            handle.yield_val(format!("got: {ctx}")).await;
            Ok(())
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), user);
        assert_eq!(need.bindings().get(&role), Some(&"admin".to_string()));

        let key = need.into_key(Arc::new("bob".to_string()));
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "got: bob");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn interleaved_yield_and_context() {
        let interner = Interner::new();
        let name_key = interner.intern("name");
        let age_key = interner.intern("age");
        let (mut co, key) = coroutine::<_, (), _, _>(|handle| async move {
            handle.yield_val("start".to_string()).await;
            let name = handle.request_context(name_key).await;
            handle.yield_val(format!("hello {name}")).await;
            let age = handle.request_context(age_key).await;
            handle.yield_val(format!("{name} is {age}")).await;
            Ok(())
        });

        // yield "start"
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "start");

        // need context "name"
        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), name_key);
        let key = need.into_key(Arc::new("eve".to_string()));

        // yield "hello eve"
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "hello eve");

        // need context "age"
        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), age_key);
        let key = need.into_key(Arc::new("30".to_string()));

        // yield "eve is 30"
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "eve is 30");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn multiple_context_requests_in_sequence() {
        let interner = Interner::new();
        let a_key = interner.intern("a");
        let b_key = interner.intern("b");
        let (mut co, key) = coroutine::<_, (), _, _>(|handle| async move {
            let a = handle.request_context(a_key).await;
            let b = handle.request_context(b_key).await;
            handle.yield_val(format!("{a}+{b}")).await;
            Ok(())
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext a");
        };
        assert_eq!(need.name(), a_key);
        let key = need.into_key(Arc::new("1".to_string()));

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext b");
        };
        assert_eq!(need.name(), b_key);
        let key = need.into_key(Arc::new("2".to_string()));

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "1+2");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn done_after_done_is_idempotent() {
        let (mut co, key) = coroutine::<i32, (), _, _>(|_handle| async move { Ok(()) });

        assert!(matches!(step(&mut co, key).await, Stepped::Done));

        let key = ResumeKey(ResumeKeyInner::Start);
        assert!(matches!(step(&mut co, key).await, Stepped::Done));

        let key = ResumeKey(ResumeKeyInner::Start);
        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn yield_handle_clone() {
        let (mut co, key) = coroutine::<_, (), _, _>(|handle| async move {
            let h2 = handle.clone();
            handle.yield_val(1).await;
            h2.yield_val(2).await;
            Ok(())
        });

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit(1)");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, 1);

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit(2)");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, 2);

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_without_yield() {
        let interner = Interner::new();
        let ignored = interner.intern("ignored");
        let (mut co, key) = coroutine::<String, (), _, _>(|handle| async move {
            let _ctx = handle.request_context(ignored).await;
            Ok(())
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        let key = need.into_key(Arc::new("value".to_string()));
        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn error_propagation() {
        let (mut co, key) = coroutine::<i32, String, _, _>(|_handle| async move {
            Err("something went wrong".to_string())
        });

        let Stepped::Error(e) = step(&mut co, key).await else {
            panic!("expected Error");
        };
        assert_eq!(e, "something went wrong");

        // After error, further resumes return Done
        let key = ResumeKey(ResumeKeyInner::Start);
        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn error_after_yield() {
        let (mut co, key) = coroutine::<i32, String, _, _>(|handle| async move {
            handle.yield_val(1).await;
            Err("failed after yield".to_string())
        });

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, 1);

        let Stepped::Error(e) = step(&mut co, key).await else {
            panic!("expected Error");
        };
        assert_eq!(e, "failed after yield");
    }
}
