use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use futures::Stream;
use parking_lot::Mutex;

use crate::value::Value;

struct Shared {
    value: Option<Value>,
    yielded: bool,
    producer_waker: Option<Waker>,
}

/// Handle passed to the producer async fn. Call `.yield_val(v).await` to emit values.
#[derive(Clone)]
pub struct YieldHandle {
    shared: Arc<Mutex<Shared>>,
}

// YieldHandle is Send + Sync via Arc<Mutex>.

impl YieldHandle {
    pub fn yield_val(&self, value: Value) -> YieldFuture {
        YieldFuture {
            shared: Arc::clone(&self.shared),
            value: Some(value),
        }
    }
}

/// Future returned by `YieldHandle::yield_val`. Completes once the stream
/// consumer has taken the value.
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
            // First poll: deposit value and suspend.
            shared.value = Some(value);
            shared.yielded = true;
            shared.producer_waker = Some(cx.waker().clone());
            Poll::Pending
        } else {
            // Resumed by stream consumer: value has been taken.
            Poll::Ready(())
        }
    }
}

/// Stream that drives a producer future and yields values deposited via `YieldHandle`.
pub struct YieldStream {
    shared: Arc<Mutex<Shared>>,
    fut: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl Stream for YieldStream {
    type Item = Value;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Value>> {
        let this = self.get_mut();

        let fut = match &mut this.fut {
            Some(f) => f.as_mut(),
            None => return Poll::Ready(None),
        };

        // Drive the producer future.
        match fut.poll(cx) {
            Poll::Ready(()) => {
                this.fut = None;
                // Producer finished. Check if there's a final yielded value.
                let mut shared = this.shared.lock();
                if shared.yielded {
                    shared.yielded = false;
                    Poll::Ready(shared.value.take())
                } else {
                    Poll::Ready(None)
                }
            }
            Poll::Pending => {
                let mut shared = this.shared.lock();
                if shared.yielded {
                    shared.yielded = false;
                    let value = shared.value.take();
                    // Wake the producer so it continues on next poll.
                    let waker = shared.producer_waker.take();
                    drop(shared);
                    if let Some(waker) = waker {
                        waker.wake();
                    }
                    Poll::Ready(value)
                } else {
                    // Producer is waiting on something else (e.g. async call).
                    Poll::Pending
                }
            }
        }
    }
}

/// Create a `YieldStream` from an async closure that receives a `YieldHandle`.
pub fn yielder<F, Fut>(f: F) -> YieldStream
where
    F: FnOnce(YieldHandle) -> Fut,
    Fut: Future<Output = ()> + Send + 'static,
{
    let shared = Arc::new(Mutex::new(Shared {
        value: None,
        yielded: false,
        producer_waker: None,
    }));
    let handle = YieldHandle { shared: Arc::clone(&shared) };
    let fut = f(handle);
    YieldStream {
        shared,
        fut: Some(Box::pin(fut)),
    }
}
