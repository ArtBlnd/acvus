use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

use crate::Astr;

// ---------------------------------------------------------------------------
// CoroutineShared — internal shared state
// ---------------------------------------------------------------------------

struct CoroutineShared<V> {
    new_requests: VecDeque<ContextRequest<V>>,
    new_extern_requests: VecDeque<ExternCallRequest<V>>,
    yield_slot: Option<V>,
}

// ---------------------------------------------------------------------------
// ContextSlot — shared between ContextFuture and ContextRequest
// ---------------------------------------------------------------------------

struct ContextSlot<V> {
    value: Option<V>,
    waker: Option<Waker>,
}

// ---------------------------------------------------------------------------
// YieldHandle — producer side
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct YieldHandle<V> {
    shared: Arc<Mutex<CoroutineShared<V>>>,
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
            slot: Arc::new(Mutex::new(ContextSlot {
                value: None,
                waker: None,
            })),
            request_data: Some(name),
        }
    }

    pub fn request_extern_call(&self, name: Astr, args: Vec<V>) -> ExternCallFuture<'_, V> {
        ExternCallFuture {
            shared: &self.shared,
            slot: Arc::new(Mutex::new(ContextSlot {
                value: None,
                waker: None,
            })),
            request_data: Some((name, args)),
        }
    }
}

// ---------------------------------------------------------------------------
// YieldFuture
// ---------------------------------------------------------------------------

pub struct YieldFuture<'a, V> {
    shared: &'a Arc<Mutex<CoroutineShared<V>>>,
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

pub struct ContextFuture<'a, V> {
    shared: &'a Arc<Mutex<CoroutineShared<V>>>,
    slot: Arc<Mutex<ContextSlot<V>>>,
    request_data: Option<Astr>,
}

impl<V> Future for ContextFuture<'_, V>
where
    V: Unpin,
{
    type Output = V;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<V> {
        let this = self.get_mut();

        if let Some(name) = this.request_data.take() {
            // First poll: register the request
            this.shared.lock().new_requests.push_back(ContextRequest {
                name,
                slot: Arc::clone(&this.slot),
            });
            this.slot.lock().waker = Some(cx.waker().clone());
            Poll::Pending
        } else {
            // Subsequent polls: check if resolved
            let mut slot = this.slot.lock();
            if let Some(value) = slot.value.take() {
                Poll::Ready(value)
            } else {
                slot.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl<V> Drop for ContextFuture<'_, V> {
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

pub struct ContextRequest<V> {
    name: Astr,
    slot: Arc<Mutex<ContextSlot<V>>>,
}

impl<V> ContextRequest<V> {
    pub fn name(&self) -> Astr {
        self.name
    }

    /// Provide the resolved value. Wakes the coroutine if it is waiting.
    pub fn resolve(self, value: V) {
        let mut slot = self.slot.lock();
        slot.value = Some(value);
        if let Some(waker) = slot.waker.take() {
            waker.wake();
        }
    }
}

// ---------------------------------------------------------------------------
// ExternCallRequest — public, returned to executor via Stepped::NeedExternCall
// ---------------------------------------------------------------------------

pub struct ExternCallRequest<V> {
    name: Astr,
    args: Vec<V>,
    slot: Arc<Mutex<ContextSlot<V>>>,
}

impl<V> ExternCallRequest<V> {
    pub fn name(&self) -> Astr {
        self.name
    }

    pub fn args(&self) -> &[V] {
        &self.args
    }

    /// Provide the resolved value. Wakes the coroutine if it is waiting.
    pub fn resolve(self, value: V) {
        let mut slot = self.slot.lock();
        slot.value = Some(value);
        if let Some(waker) = slot.waker.take() {
            waker.wake();
        }
    }
}

// ---------------------------------------------------------------------------
// ExternCallFuture
// ---------------------------------------------------------------------------

pub struct ExternCallFuture<'a, V> {
    shared: &'a Arc<Mutex<CoroutineShared<V>>>,
    slot: Arc<Mutex<ContextSlot<V>>>,
    request_data: Option<(Astr, Vec<V>)>,
}

impl<V> Future for ExternCallFuture<'_, V>
where
    V: Unpin,
{
    type Output = V;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<V> {
        let this = self.get_mut();

        if let Some((name, args)) = this.request_data.take() {
            // First poll: register the request
            this.shared
                .lock()
                .new_extern_requests
                .push_back(ExternCallRequest {
                    name,
                    args,
                    slot: Arc::clone(&this.slot),
                });
            this.slot.lock().waker = Some(cx.waker().clone());
            Poll::Pending
        } else {
            // Subsequent polls: check if resolved
            let mut slot = this.slot.lock();
            if let Some(value) = slot.value.take() {
                Poll::Ready(value)
            } else {
                slot.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl<V> Drop for ExternCallFuture<'_, V> {
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

pub enum Stepped<V, E> {
    Emit(V),
    NeedContext(ContextRequest<V>),
    NeedExternCall(ExternCallRequest<V>),
    Done,
    Error(E),
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

pub struct Coroutine<V, E> {
    shared: Arc<Mutex<CoroutineShared<V>>>,
    fut: Option<Pin<Box<dyn Future<Output = Result<(), E>> + Send>>>,
}

impl<V, E> Coroutine<V, E> {
    pub fn resume(&mut self) -> ResumeFuture<'_, V, E> {
        ResumeFuture { coroutine: self }
    }

    /// Ownership-passing step. Takes self, returns self back with the stepped result.
    /// Enables use in FuturesUnordered without borrow issues.
    pub async fn step(mut self) -> (Self, Stepped<V, E>) {
        let stepped = self.resume().await;
        (self, stepped)
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

pub fn coroutine<V, E, F, Fut>(f: F) -> Coroutine<V, E>
where
    F: FnOnce(YieldHandle<V>) -> Fut,
    Fut: Future<Output = Result<(), E>> + Send + 'static,
{
    let shared = Arc::new(Mutex::new(CoroutineShared {
        new_requests: VecDeque::new(),
        new_extern_requests: VecDeque::new(),
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
    use crate::Interner;

    async fn step<V, E>(co: &mut Coroutine<V, E>) -> Stepped<V, E> {
        co.resume().await
    }

    #[tokio::test]
    async fn empty_coroutine() {
        let mut co = coroutine::<i32, (), _, _>(|_handle| async move { Ok(()) });
        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn single_yield() {
        let mut co = coroutine::<_, (), _, _>(|handle| async move {
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
        let mut co = coroutine::<_, (), _, _>(|handle| async move {
            handle.yield_val(1).await;
            handle.yield_val(2).await;
            handle.yield_val(3).await;
            Ok(())
        });

        for expected in [1, 2, 3] {
            let Stepped::Emit(value) = step(&mut co).await else {
                panic!("expected Emit({expected})");
            };
            assert_eq!(value, expected);
        }

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_request() {
        let interner = Interner::new();
        let user = interner.intern("user");
        let mut co = coroutine::<_, (), _, _>(|handle| async move {
            let ctx = handle.request_context(user).await;
            handle.yield_val(format!("got: {ctx}")).await;
            Ok(())
        });

        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(request.name(), user);
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
        let mut co = coroutine::<String, (), _, _>(|handle| async move {
            let result = handle
                .request_extern_call(add, vec!["1".to_string(), "2".to_string()])
                .await;
            handle.yield_val(format!("result: {result}")).await;
            Ok(())
        });

        let Stepped::NeedExternCall(request) = step(&mut co).await else {
            panic!("expected NeedExternCall");
        };
        assert_eq!(request.name(), add);
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
        let mut co = coroutine::<_, (), _, _>(|handle| async move {
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
        assert_eq!(request.name(), name_key);
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
        assert_eq!(request.name(), age_key);
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
        let mut co = coroutine::<_, (), _, _>(|handle| async move {
            let a = handle.request_context(a_key).await;
            let b = handle.request_context(b_key).await;
            handle.yield_val(format!("{a}+{b}")).await;
            Ok(())
        });

        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext a");
        };
        assert_eq!(request.name(), a_key);
        request.resolve("1".to_string());

        let Stepped::NeedContext(request) = step(&mut co).await else {
            panic!("expected NeedContext b");
        };
        assert_eq!(request.name(), b_key);
        request.resolve("2".to_string());

        let Stepped::Emit(value) = step(&mut co).await else {
            panic!("expected Emit");
        };
        assert_eq!(value, "1+2");

        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn done_after_done_is_idempotent() {
        let mut co = coroutine::<i32, (), _, _>(|_handle| async move { Ok(()) });

        assert!(matches!(step(&mut co).await, Stepped::Done));
        assert!(matches!(step(&mut co).await, Stepped::Done));
        assert!(matches!(step(&mut co).await, Stepped::Done));
    }

    #[tokio::test]
    async fn yield_handle_clone() {
        let mut co = coroutine::<_, (), _, _>(|handle| async move {
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
        let mut co = coroutine::<String, (), _, _>(|handle| async move {
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
        let mut co = coroutine::<i32, String, _, _>(|_handle| async move {
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
        let mut co = coroutine::<i32, String, _, _>(|handle| async move {
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
