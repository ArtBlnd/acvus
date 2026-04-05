//! HostError + ExternFnFuture: Errors and async support at the host boundary.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

#[derive(Debug, Clone)]
pub enum HostError {
    TypeMismatch { expected: &'static str, got: &'static str },
    CallFailed(String),
    SlotEmpty,
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch { expected, got } => write!(f, "type mismatch: expected {expected}, got {got}"),
            Self::CallFailed(msg) => write!(f, "call failed: {msg}"),
            Self::SlotEmpty => write!(f, "slot is empty (double take?)"),
        }
    }
}

impl std::error::Error for HostError {}

// ── ExternFnFuture ─────────────────────────────────────────────────

/// All ExternFn calls return this. Sync handlers resolve immediately (Ready),
/// async handlers return a boxed future (Async).
///
/// The interpreter polls this: if Ready, no allocation. If Async, drive the future.
pub enum ExternFnFuture<'a> {
    /// Sync path: already complete. Zero allocation.
    Ready(Option<Result<(), HostError>>),
    /// Async path: boxed future.
    Async(Pin<Box<dyn Future<Output = Result<(), HostError>> + 'a>>),
}

impl<'a> ExternFnFuture<'a> {
    /// For sync handlers: extract the result immediately.
    /// Panics if this is an Async future.
    pub fn unwrap_ready(self) -> Result<(), HostError> {
        match self {
            ExternFnFuture::Ready(Some(result)) => result,
            ExternFnFuture::Ready(None) => panic!("ExternFnFuture::Ready already consumed"),
            ExternFnFuture::Async(_) => panic!("unwrap_ready called on Async future"),
        }
    }
}

impl<'a> Future for ExternFnFuture<'a> {
    type Output = Result<(), HostError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            ExternFnFuture::Ready(result) => {
                Poll::Ready(result.take().expect("ExternFnFuture::Ready polled twice"))
            }
            ExternFnFuture::Async(fut) => fut.as_mut().poll(cx),
        }
    }
}
