use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

use crate::value::Value;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct Shared {
    value: Option<Value>,
    yielded: bool,
    producer_waker: Option<Waker>,
    context_request: Option<String>,
    context_bindings: HashMap<String, Value>,
    context_response: Option<Arc<Value>>,
    context_requested: bool,
}

// ---------------------------------------------------------------------------
// YieldHandle — producer side
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct YieldHandle {
    shared: Arc<Mutex<Shared>>,
}

impl YieldHandle {
    pub fn yield_val(&self, value: Value) -> YieldFuture {
        YieldFuture {
            shared: Arc::clone(&self.shared),
            value: Some(value),
        }
    }

    pub fn request_context(&self, name: String) -> ContextFuture {
        ContextFuture {
            shared: Arc::clone(&self.shared),
            name: Some(name),
            bindings: HashMap::new(),
        }
    }

    pub fn request_context_with(
        &self,
        name: String,
        bindings: HashMap<String, Value>,
    ) -> ContextFuture {
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

pub struct YieldFuture {
    shared: Arc<Mutex<Shared>>,
    value: Option<Value>,
}

impl Future for YieldFuture {
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

pub struct ContextFuture {
    shared: Arc<Mutex<Shared>>,
    name: Option<String>,
    bindings: HashMap<String, Value>,
}

impl Future for ContextFuture {
    type Output = Arc<Value>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Arc<Value>> {
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

pub struct ResumeKey(ResumeKeyInner);

enum ResumeKeyInner {
    Start,
    Context(Arc<Value>),
}

pub enum Stepped {
    Emit(EmitStepped),
    NeedContext(NeedContextStepped),
    Done,
}

pub struct EmitStepped {
    value: Value,
    key: ResumeKey,
}

impl EmitStepped {
    pub fn into_parts(self) -> (Value, ResumeKey) {
        (self.value, self.key)
    }
}

pub struct NeedContextStepped {
    name: String,
    bindings: HashMap<String, Value>,
}

impl NeedContextStepped {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn bindings(&self) -> &HashMap<String, Value> {
        &self.bindings
    }

    pub fn into_parts(self) -> (String, HashMap<String, Value>) {
        (self.name, self.bindings)
    }

    pub fn into_key(self, value: Arc<Value>) -> ResumeKey {
        ResumeKey(ResumeKeyInner::Context(value))
    }
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

pub struct Coroutine {
    shared: Arc<Mutex<Shared>>,
    fut: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl Coroutine {
    pub fn resume(&mut self, key: ResumeKey) -> Stepped {
        // 1. Inject context response if the key carries one.
        if let ResumeKeyInner::Context(arc) = key.0 {
            let mut shared = self.shared.lock();
            shared.context_response = Some(arc);
        }

        let fut = match &mut self.fut {
            Some(f) => f.as_mut(),
            None => return Stepped::Done,
        };

        // 2. Poll with noop waker.
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);

        match fut.poll(&mut cx) {
            Poll::Ready(()) => {
                self.fut = None;
                let mut shared = self.shared.lock();
                if shared.yielded {
                    shared.yielded = false;
                    if let Some(value) = shared.value.take() {
                        return Stepped::Emit(EmitStepped {
                            value,
                            key: ResumeKey(ResumeKeyInner::Start),
                        });
                    }
                }
                Stepped::Done
            }
            Poll::Pending => {
                let mut shared = self.shared.lock();
                if shared.context_requested {
                    shared.context_requested = false;
                    let name = shared.context_request.take().unwrap();
                    let bindings = std::mem::take(&mut shared.context_bindings);
                    Stepped::NeedContext(NeedContextStepped { name, bindings })
                } else if shared.yielded {
                    shared.yielded = false;
                    let value = shared.value.take().unwrap();
                    Stepped::Emit(EmitStepped {
                        value,
                        key: ResumeKey(ResumeKeyInner::Start),
                    })
                } else {
                    Stepped::Done
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

pub fn coroutine<F, Fut>(f: F) -> (Coroutine, ResumeKey)
where
    F: FnOnce(YieldHandle) -> Fut,
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
