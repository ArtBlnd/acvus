use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// CoroutineShared — internal shared state
// ---------------------------------------------------------------------------

struct CoroutineShared<V, K> {
    new_requests: VecDeque<ContextRequest<V, K>>,
    new_extern_requests: VecDeque<ExternCallRequest<V, K>>,
    store_requests: VecDeque<(K, V)>,
    yield_slot: Option<V>,
}

// ---------------------------------------------------------------------------
// ContextSlot — shared between ContextFuture and ContextRequest
// ---------------------------------------------------------------------------

/// None = pending, Some(Some(v)) = resolved, Some(None) = not found.
struct ContextSlot<V> {
    value: Option<Option<V>>,
    waker: Option<Waker>,
}

// ---------------------------------------------------------------------------
// YieldHandle — producer side
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct YieldHandle<V, K> {
    shared: Arc<Mutex<CoroutineShared<V, K>>>,
}

impl<V, K: Send + 'static> YieldHandle<V, K> {
    pub fn yield_val(&self, value: V) -> YieldFuture<'_, V, K> {
        YieldFuture {
            shared: &self.shared,
            value: Some(value),
        }
    }

    /// Request a context value. Panics if not found.
    pub async fn request_context(&self, key: K) -> V
    where K: Copy + Send + Unpin, V: Unpin,
    {
        self.try_request_context(key).await
            .unwrap_or_else(|| panic!("context not found for key"))
    }

    /// Store a value into a context slot. Fire-and-forget within the coroutine.
    pub fn store_context(&self, key: K, value: V) {
        self.shared.lock().store_requests.push_back((key, value));
    }

    /// Request a context value. Returns None if not found.
    pub fn try_request_context(&self, key: K) -> ContextFuture<'_, V, K> {
        ContextFuture {
            shared: &self.shared,
            slot: Arc::new(Mutex::new(ContextSlot {
                value: None,
                waker: None,
            })),
            request_data: Some(key),
        }
    }

    pub fn request_extern_call(&self, key: K, args: Vec<V>) -> ExternCallFuture<'_, V, K> {
        ExternCallFuture {
            shared: &self.shared,
            slot: Arc::new(Mutex::new(ContextSlot {
                value: None,
                waker: None,
            })),
            request_data: Some((key, args)),
        }
    }
}

// ---------------------------------------------------------------------------
// YieldFuture
// ---------------------------------------------------------------------------

pub struct YieldFuture<'a, V, K> {
    shared: &'a Arc<Mutex<CoroutineShared<V, K>>>,
    value: Option<V>,
}

impl<V, K> Future for YieldFuture<'_, V, K>
where
    V: Unpin,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if let Some(value) = this.value.take() {
            this.shared.lock().yield_slot = Some(value);
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

// ---------------------------------------------------------------------------
// ContextFuture
// ---------------------------------------------------------------------------

pub struct ContextFuture<'a, V, K> {
    shared: &'a Arc<Mutex<CoroutineShared<V, K>>>,
    slot: Arc<Mutex<ContextSlot<V>>>,
    request_data: Option<K>,
}

impl<V, K: Send + Unpin + 'static> Future for ContextFuture<'_, V, K>
where
    V: Unpin,
{
    type Output = Option<V>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<V>> {
        let this = self.get_mut();

        if let Some(key) = this.request_data.take() {
            // First poll: register the request
            this.shared.lock().new_requests.push_back(ContextRequest {
                key,
                slot: Arc::clone(&this.slot),
            });
            this.slot.lock().waker = Some(cx.waker().clone());
            Poll::Pending
        } else {
            // Subsequent polls: check if resolved
            let mut slot = this.slot.lock();
            match slot.value.take() {
                Some(resolved) => Poll::Ready(resolved),
                None => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }
}

impl<V, K> Drop for ContextFuture<'_, V, K> {
    fn drop(&mut self) {
        if self.request_data.is_none() {
            // Was registered; remove from new_requests if still pending
            let mut shared = self.shared.lock();
            shared
                .new_requests
                .retain(|r| !Arc::ptr_eq(&r.slot, &self.slot));
        }
    }
}

// ---------------------------------------------------------------------------
// ContextRequest — public, returned to executor via Stepped::NeedContext
// ---------------------------------------------------------------------------

pub struct ContextRequest<V, K> {
    key: K,
    slot: Arc<Mutex<ContextSlot<V>>>,
}

impl<V, K: Copy> ContextRequest<V, K> {
    pub fn key(&self) -> K {
        self.key
    }

    /// Provide the resolved value. Wakes the coroutine if it is waiting.
    pub fn resolve(self, value: V) {
        let mut slot = self.slot.lock();
        slot.value = Some(Some(value));
        if let Some(waker) = slot.waker.take() {
            waker.wake();
        }
    }

    /// Signal that no value was found. Wakes the coroutine.
    pub fn resolve_not_found(self) {
        let mut slot = self.slot.lock();
        slot.value = Some(None);
        if let Some(waker) = slot.waker.take() {
            waker.wake();
        }
    }
}

// ---------------------------------------------------------------------------
// ExternCallRequest — public, returned to executor via Stepped::NeedExternCall
// ---------------------------------------------------------------------------

pub struct ExternCallRequest<V, K> {
    key: K,
    args: Vec<V>,
    slot: Arc<Mutex<ContextSlot<V>>>,
}

impl<V, K: Copy> ExternCallRequest<V, K> {
    pub fn key(&self) -> K {
        self.key
    }

    pub fn args(&self) -> &[V] {
        &self.args
    }

    /// Provide the resolved value. Wakes the coroutine if it is waiting.
    pub fn resolve(self, value: V) {
        let mut slot = self.slot.lock();
        slot.value = Some(Some(value));
        if let Some(waker) = slot.waker.take() {
            waker.wake();
        }
    }
}

// ---------------------------------------------------------------------------
// ExternCallFuture
// ---------------------------------------------------------------------------

pub struct ExternCallFuture<'a, V, K> {
    shared: &'a Arc<Mutex<CoroutineShared<V, K>>>,
    slot: Arc<Mutex<ContextSlot<V>>>,
    request_data: Option<(K, Vec<V>)>,
}

impl<V, K: Send + Unpin + 'static> Future for ExternCallFuture<'_, V, K>
where
    V: Unpin,
{
    type Output = V;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<V> {
        let this = self.get_mut();

        if let Some((key, args)) = this.request_data.take() {
            // First poll: register the request
            this.shared
                .lock()
                .new_extern_requests
                .push_back(ExternCallRequest {
                    key,
                    args,
                    slot: Arc::clone(&this.slot),
                });
            this.slot.lock().waker = Some(cx.waker().clone());
            Poll::Pending
        } else {
            // Subsequent polls: check if resolved
            let mut slot = this.slot.lock();
            match slot.value.take() {
                Some(Some(value)) => Poll::Ready(value),
                Some(None) => panic!("extern call resolved as not found"),
                None => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }
}

impl<V, K> Drop for ExternCallFuture<'_, V, K> {
    fn drop(&mut self) {
        if self.request_data.is_none() {
            // Was registered; remove from new_extern_requests if still pending
            let mut shared = self.shared.lock();
            shared
                .new_extern_requests
                .retain(|r| !Arc::ptr_eq(&r.slot, &self.slot));
        }
    }
}

// ---------------------------------------------------------------------------
// Stepped — resume result
// ---------------------------------------------------------------------------

pub enum Stepped<V, E, K> {
    Emit(V),
    NeedContext(ContextRequest<V, K>),
    NeedExternCall(ExternCallRequest<V, K>),
    StoreContext(K, V),
    Done,
    Error(E),
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

pub struct Coroutine<V, E, K> {
    shared: Arc<Mutex<CoroutineShared<V, K>>>,
    fut: Option<Pin<Box<dyn Future<Output = Result<(), E>> + Send>>>,
}

impl<V, E, K> Coroutine<V, E, K> {
    pub fn resume(&mut self) -> ResumeFuture<'_, V, E, K> {
        ResumeFuture { coroutine: self }
    }

    /// Ownership-passing step. Takes self, returns self back with the stepped result.
    /// Enables use in FuturesUnordered without borrow issues.
    pub async fn step(mut self) -> (Self, Stepped<V, E, K>) {
        let stepped = self.resume().await;
        (self, stepped)
    }
}

// ---------------------------------------------------------------------------
// ResumeFuture — async resume
// ---------------------------------------------------------------------------

pub struct ResumeFuture<'a, V, E, K> {
    coroutine: &'a mut Coroutine<V, E, K>,
}

impl<V, E, K> Future for ResumeFuture<'_, V, E, K> {
    type Output = Stepped<V, E, K>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Stepped<V, E, K>> {
        let this = self.get_mut();
        let fut = match &mut this.coroutine.fut {
            Some(f) => f.as_mut(),
            None => return Poll::Ready(Stepped::Done),
        };

        let poll_result = fut.poll(cx);
        let mut shared = this.coroutine.shared.lock();

        // Check yield slot
        if let Some(value) = shared.yield_slot.take() {
            if poll_result.is_ready() {
                drop(shared);
                this.coroutine.fut = None;
            }
            return Poll::Ready(Stepped::Emit(value));
        }

        // Check for new context requests
        if let Some(request) = shared.new_requests.pop_front() {
            return Poll::Ready(Stepped::NeedContext(request));
        }

        // Check for context store requests
        if let Some((key, value)) = shared.store_requests.pop_front() {
            return Poll::Ready(Stepped::StoreContext(key, value));
        }

        // Check for new extern call requests
        if let Some(request) = shared.new_extern_requests.pop_front() {
            return Poll::Ready(Stepped::NeedExternCall(request));
        }

        // No signals
        match poll_result {
            Poll::Ready(Ok(())) => {
                drop(shared);
                this.coroutine.fut = None;
                Poll::Ready(Stepped::Done)
            }
            Poll::Ready(Err(e)) => {
                drop(shared);
                this.coroutine.fut = None;
                Poll::Ready(Stepped::Error(e))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

pub fn coroutine<V, E, K, F, Fut>(f: F) -> Coroutine<V, E, K>
where
    K: Send + 'static,
    F: FnOnce(YieldHandle<V, K>) -> Fut,
    Fut: Future<Output = Result<(), E>> + Send + 'static,
{
    let shared = Arc::new(Mutex::new(CoroutineShared {
        new_requests: VecDeque::new(),
        new_extern_requests: VecDeque::new(),
        store_requests: VecDeque::new(),
        yield_slot: None,
    }));
    let handle = YieldHandle {
        shared: Arc::clone(&shared),
    };
    let fut = f(handle);
    Coroutine {
        shared,
        fut: Some(Box::pin(fut)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Astr, Interner};

    async fn step<V, E, K>(co: &mut Coroutine<V, E, K>) -> Stepped<V, E, K> {
        co.resume().await
    }

    #[tokio::test]
    async fn empty_coroutine() {
        let mut co = coroutine::<i32, (), Astr, _, _>(|_handle| async move { Ok(()) });
        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn single_yield() {
        let mut co = coroutine::<_, (), Astr, _, _>(|handle| async move {
            handle.yield_val(42).await;
            Ok(())
        });

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, 42);

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn multiple_yields() {
        let mut co = coroutine::<_, (), Astr, _, _>(|handle| async move {
            handle.yield_val(1).await;
            handle.yield_val(2).await;
            handle.yield_val(3).await;
            Ok(())
        });

        for expected in [1, 2, 3] {
            let Stepped::Emit(value) = step(&mut co).await else {
                panic!("expected Emit");
            };
            assert_eq!(value, expected);
        }

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_request() {
        let interner = Interner::new();
        let user = interner.intern("user");
        let mut co = coroutine::<_, (), Astr, _, _>(|handle| async move {
            let ctx = handle.request_context(user).await;
            handle.yield_val(format!("got: {ctx}")).await;
            Ok(())
        });

        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(request.key(), user);
        request.resolve("alice".to_string());

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, "got: alice");

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn extern_call_request() {
        let interner = Interner::new();
        let add = interner.intern("add");
        let mut co = coroutine::<String, (), Astr, _, _>(|handle| async move {
            let result = handle
                .request_extern_call(add, vec!["1".to_string(), "2".to_string()])
                .await;
            handle.yield_val(format!("result: {result}")).await;
            Ok(())
        });

        let Stepped::NeedExternCall(request) = step(&mut co).await else {
            panic!("expected NeedExternCall");
        };
        assert_eq!(request.key(), add);
        assert_eq!(request.args(), &["1".to_string(), "2".to_string()]);
        request.resolve("3".to_string());

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, "result: 3");

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn interleaved_yield_and_context() {
        let interner = Interner::new();
        let name_key = interner.intern("name");
        let age_key = interner.intern("age");
        let mut co = coroutine::<_, (), Astr, _, _>(|handle| async move {
            handle.yield_val("start".to_string()).await;
            let name = handle.request_context(name_key).await;
            handle.yield_val(format!("hello {name}")).await;
            let age = handle.request_context(age_key).await;
            handle.yield_val(format!("{name} is {age}")).await;
            Ok(())
        });

        // yield "start"
        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, "start");

        // need context "name"
        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(request.key(), name_key);
        request.resolve("eve".to_string());

        // yield "hello eve"
        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, "hello eve");

        // need context "age"
        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(request.key(), age_key);
        request.resolve("30".to_string());

        // yield "eve is 30"
        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, "eve is 30");

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn multiple_context_requests_in_sequence() {
        let interner = Interner::new();
        let a_key = interner.intern("a");
        let b_key = interner.intern("b");
        let mut co = coroutine::<_, (), Astr, _, _>(|handle| async move {
            let a = handle.request_context(a_key).await;
            let b = handle.request_context(b_key).await;
            handle.yield_val(format!("{a}+{b}")).await;
            Ok(())
        });

        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext a");
        };
        assert_eq!(request.key(), a_key);
        request.resolve("1".to_string());

        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext b");
        };
        assert_eq!(request.key(), b_key);
        request.resolve("2".to_string());

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, "1+2");

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn done_after_done_is_idempotent() {
        let mut co = coroutine::<i32, (), Astr, _, _>(|_handle| async move { Ok(()) });

        assert!(matches!(step(&mut co).await, Stepped::Done));
        assert!(matches!(step(&mut co).await, Stepped::Done));
        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn yield_handle_clone() {
        let mut co = coroutine::<_, (), Astr, _, _>(|handle| async move {
            let h2 = handle.clone();
            handle.yield_val(1).await;
            h2.yield_val(2).await;
            Ok(())
        });

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit(1)");
        };
        assert_eq!(value, 1);

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit(2)");
        };
        assert_eq!(value, 2);

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_without_yield() {
        let interner = Interner::new();
        let ignored = interner.intern("ignored");
        let mut co = coroutine::<String, (), Astr, _, _>(|handle| async move {
            let _ctx = handle.request_context(ignored).await;
            Ok(())
        });

        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext");
        };
        request.resolve("value".to_string());
        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn error_propagation() {
        let mut co = coroutine::<i32, String, Astr, _, _>(|_handle| async move {
            Err("something went wrong".to_string())
        });

        let Stepped::Error(e) = step(&mut co).await else {
            panic!("expected Error");
        };
        assert_eq!(e, "something went wrong");

        // After error, further resumes return Done
        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn error_after_yield() {
        let mut co = coroutine::<i32, String, Astr, _, _>(|handle| async move {
            handle.yield_val(1).await;
            Err("failed after yield".to_string())
        });

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, 1);

        let Stepped::Error(e) = step(&mut co).await else {
            panic!("expected Error");
        };
        assert_eq!(e, "failed after yield");
    }
}
