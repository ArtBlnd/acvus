//! Executor trait - controls how spawned computations are executed.
//!
//! Three spawn paths:
//! - `spawn_interpreter`: deferred MIR execution (fork + run)
//! - `spawn_blocking`: sync ExternFn (closure, no interpreter access)
//! - `spawn_async`: async ExternFn (future, no interpreter access)
//!
//! All paths return `HandleValue`, eval'd to the value the work produced.
//! `sleep` is the timer `Runtime::sleep` waits on (RFC-0075 rule 4).

use std::panic::resume_unwind;
use std::pin::Pin;
use std::time::Duration;

use futures::future::BoxFuture;
use sync_wrapper::SyncWrapper;

use crate::interpreter::Interpreter;
use crate::value::{HandleValue, Value};

// -- Trait -------------------------------------------------------------

/// Executor controls spawn/eval execution strategy.
///
/// Implementations decide whether to run sequentially, in parallel
/// (tokio::spawn), or with any other scheduling strategy.
pub trait Executor: Send + Sync {
    /// Spawn a deferred MIR interpreter execution.
    fn spawn_interpreter(&self, interpreter: Interpreter) -> HandleValue;

    /// Spawn a sync blocking closure (ExternFn, no interpreter access).
    fn spawn_blocking(&self, f: Box<dyn FnOnce() -> Value + Send + Sync>) -> HandleValue;

    /// Spawn an async future (ExternFn, no interpreter access).
    fn spawn_async(&self, f: Pin<Box<dyn Future<Output = Value> + Send>>) -> HandleValue;

    /// Force a handle to completion and return its value. A panic in the
    /// spawned work is resumed here, on the awaiting run's thread.
    fn eval(&self, handle: HandleValue) -> BoxFuture<'_, Value>;

    /// A future that waits `d`.
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()>;
}

// -- SequentialExecutor -----------------------------------------------

/// Tag types for HandleValue dispatch in SequentialExecutor.
struct DeferredInterpreter(Interpreter);
struct DeferredBlocking(Box<dyn FnOnce() -> Value + Send + Sync>);
struct DeferredAsync(SyncWrapper<Pin<Box<dyn Future<Output = Value> + Send>>>);

/// Simplest executor - spawn stores the computation, eval runs it immediately.
/// No parallelism. Good for testing and deterministic execution.
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn spawn_interpreter(&self, interpreter: Interpreter) -> HandleValue {
        HandleValue::new(DeferredInterpreter(interpreter))
    }

    fn spawn_blocking(&self, f: Box<dyn FnOnce() -> Value + Send + Sync>) -> HandleValue {
        HandleValue::new(DeferredBlocking(f))
    }

    fn spawn_async(&self, f: Pin<Box<dyn Future<Output = Value> + Send>>) -> HandleValue {
        HandleValue::new(DeferredAsync(SyncWrapper::new(f)))
    }

    fn eval(&self, handle: HandleValue) -> BoxFuture<'_, Value> {
        Box::pin(async move {
            // Try each deferred type via try_downcast chain.
            let handle = match handle.try_downcast::<DeferredInterpreter>() {
                Ok(mut d) => return d.0.execute().await,
                Err(h) => h,
            };
            let handle = match handle.try_downcast::<DeferredBlocking>() {
                Ok(d) => return d.0(),
                Err(h) => h,
            };
            match handle.try_downcast::<DeferredAsync>() {
                Ok(d) => d.0.into_inner().await,
                Err(_) => panic!("eval: unknown handle type"),
            }
        })
    }

    /// Sleeps the thread: this executor runs nothing else while a run
    /// waits, so there is no other work to yield to. The thread is blocked
    /// for `d`, and so is any other task a host runs on it.
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(async move { std::thread::sleep(d) })
    }
}

// -- TokioExecutor ----------------------------------------------------

/// Work starts at spawn as a tokio task and eval joins it. Two spawns
/// that stand before their evals run concurrently; the schedule the
/// compiler emitted (RFC-0007) is what decides that.
pub struct TokioExecutor;

type Joined = tokio::task::JoinHandle<Value>;

impl Executor for TokioExecutor {
    fn spawn_interpreter(&self, mut interpreter: Interpreter) -> HandleValue {
        HandleValue::new(tokio::spawn(async move { interpreter.execute().await }))
    }

    fn spawn_blocking(&self, f: Box<dyn FnOnce() -> Value + Send + Sync>) -> HandleValue {
        HandleValue::new(tokio::task::spawn_blocking(f))
    }

    fn spawn_async(&self, f: Pin<Box<dyn Future<Output = Value> + Send>>) -> HandleValue {
        HandleValue::new(tokio::spawn(f))
    }

    fn eval(&self, handle: HandleValue) -> BoxFuture<'_, Value> {
        Box::pin(async move {
            let joined = handle
                .try_downcast::<Joined>()
                .unwrap_or_else(|_| panic!("eval: handle was not spawned by TokioExecutor"));
            match joined.await {
                Ok(value) => value,
                Err(joined) if joined.is_panic() => resume_unwind(joined.into_panic()),
                Err(joined) => panic!("spawned task failed: {joined}"),
            }
        })
    }

    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep(d))
    }
}
