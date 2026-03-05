use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct Shared<V> {
    value: Option<V>,
    yielded: bool,
    producer_waker: Option<Waker>,
    context_request: Option<String>,
    context_bindings: HashMap<String, V>,
    context_response: Option<Arc<V>>,
    context_requested: bool,
}

// ---------------------------------------------------------------------------
// YieldHandle — producer side
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct YieldHandle<V> {
    shared: Arc<Mutex<Shared<V>>>,
}

impl<V> YieldHandle<V> {
    pub fn yield_val(&self, value: V) -> YieldFuture<V> {
        YieldFuture {
            shared: Arc::clone(&self.shared),
            value: Some(value),
        }
    }

    pub fn request_context(&self, name: String) -> ContextFuture<V> {
        ContextFuture {
            shared: Arc::clone(&self.shared),
            name: Some(name),
            bindings: HashMap::new(),
        }
    }

    pub fn request_context_with(
        &self,
        name: String,
        bindings: HashMap<String, V>,
    ) -> ContextFuture<V> {
        ContextFuture {
            shared: Arc::clone(&self.shared),
            name: Some(name),
            bindings,
        }
    }
}

// ---------------------------------------------------------------------------
// YieldFuture
// ---------------------------------------------------------------------------

pub struct YieldFuture<V> {
    shared: Arc<Mutex<Shared<V>>>,
    value: Option<V>,
}

impl<V> Future for YieldFuture<V>
where
    V: Unpin,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut shared = this.shared.lock();

        if let Some(value) = this.value.take() {
            shared.value = Some(value);
            shared.yielded = true;
            shared.producer_waker = Some(cx.waker().clone());
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

// ---------------------------------------------------------------------------
// ContextFuture
// ---------------------------------------------------------------------------

pub struct ContextFuture<V> {
    shared: Arc<Mutex<Shared<V>>>,
    name: Option<String>,
    bindings: HashMap<String, V>,
}

impl<V> Future for ContextFuture<V>
where
    V: Unpin,
{
    type Output = Arc<V>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Arc<V>> {
        let this = self.get_mut();
        let mut shared = this.shared.lock();

        if let Some(name) = this.name.take() {
            shared.context_request = Some(name);
            shared.context_bindings = std::mem::take(&mut this.bindings);
            shared.context_requested = true;
            shared.producer_waker = Some(cx.waker().clone());
            Poll::Pending
        } else if let Some(value) = shared.context_response.take() {
            Poll::Ready(value)
        } else {
            shared.producer_waker = Some(cx.waker().clone());
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

pub enum Stepped<V> {
    Emit(EmitStepped<V>),
    NeedContext(NeedContextStepped<V>),
    Done,
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
    name: String,
    bindings: HashMap<String, V>,
}

impl<V> NeedContextStepped<V> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn bindings(&self) -> &HashMap<String, V> {
        &self.bindings
    }

    pub fn into_parts(self) -> (String, HashMap<String, V>) {
        (self.name, self.bindings)
    }

    pub fn into_key(self, value: Arc<V>) -> ResumeKey<V> {
        ResumeKey(ResumeKeyInner::Context(value))
    }
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

pub struct Coroutine<V> {
    shared: Arc<Mutex<Shared<V>>>,
    fut: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl<V> Coroutine<V> {
    pub fn resume(&mut self, key: ResumeKey<V>) -> ResumeFuture<'_, V> {
        if let ResumeKeyInner::Context(arc) = key.0 {
            let mut shared = self.shared.lock();
            shared.context_response = Some(arc);
        }
        ResumeFuture { coroutine: self }
    }
}

// ---------------------------------------------------------------------------
// ResumeFuture — async resume
// ---------------------------------------------------------------------------

pub struct ResumeFuture<'a, V> {
    coroutine: &'a mut Coroutine<V>,
}

impl<V> Future for ResumeFuture<'_, V> {
    type Output = Stepped<V>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Stepped<V>> {
        let this = self.get_mut();
        let fut = match &mut this.coroutine.fut {
            Some(f) => f.as_mut(),
            None => return Poll::Ready(Stepped::Done),
        };

        match fut.poll(cx) {
            Poll::Ready(()) => {
                this.coroutine.fut = None;
                let mut shared = this.coroutine.shared.lock();
                if shared.yielded {
                    shared.yielded = false;
                    if let Some(value) = shared.value.take() {
                        return Poll::Ready(Stepped::Emit(EmitStepped {
                            value,
                            key: ResumeKey(ResumeKeyInner::Start),
                        }));
                    }
                }
                Poll::Ready(Stepped::Done)
            }
            Poll::Pending => {
                let mut shared = this.coroutine.shared.lock();
                if shared.context_requested {
                    shared.context_requested = false;
                    let name = shared.context_request.take().unwrap();
                    let bindings = std::mem::take(&mut shared.context_bindings);
                    Poll::Ready(Stepped::NeedContext(NeedContextStepped { name, bindings }))
                } else if shared.yielded {
                    shared.yielded = false;
                    let value = shared.value.take().unwrap();
                    Poll::Ready(Stepped::Emit(EmitStepped {
                        value,
                        key: ResumeKey(ResumeKeyInner::Start),
                    }))
                } else {
                    // Real async I/O pending — propagate
                    Poll::Pending
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

pub fn coroutine<V, F, Fut>(f: F) -> (Coroutine<V>, ResumeKey<V>)
where
    F: FnOnce(YieldHandle<V>) -> Fut,
    Fut: Future<Output = ()> + Send + 'static,
{
    let shared = Arc::new(Mutex::new(Shared {
        value: None,
        yielded: false,
        producer_waker: None,
        context_request: None,
        context_bindings: HashMap::new(),
        context_response: None,
        context_requested: false,
    }));
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
