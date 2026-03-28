//! Executor trait — controls how spawned computations are executed.
//!
//! Three spawn paths:
//! - `spawn_interpreter`: deferred MIR execution (fork + run)
//! - `spawn_blocking`: sync ExternFn (closure, no interpreter access)
//! - `spawn_async`: async ExternFn (future, no interpreter access)
//!
//! All paths return `HandleValue`, eval'd to a uniform `ExecResult`.

use std::pin::Pin;

use futures::future::BoxFuture;
use sync_wrapper::SyncWrapper;

use crate::error::RuntimeError;
use crate::interpreter::{ExecResult, Interpreter};
use crate::value::HandleValue;

// ── Trait ─────────────────────────────────────────────────────────────

/// Executor controls spawn/eval execution strategy.
///
/// Implementations decide whether to run sequentially, in parallel
/// (tokio::spawn), or with any other scheduling strategy.
pub trait Executor: Send + Sync {
    /// Spawn a deferred MIR interpreter execution.
    fn spawn_interpreter(&self, interpreter: Interpreter) -> HandleValue;

    /// Spawn a sync blocking closure (ExternFn, no interpreter access).
    fn spawn_blocking(
        &self,
        f: Box<dyn FnOnce() -> Result<ExecResult, RuntimeError> + Send + Sync>,
    ) -> HandleValue;

    /// Spawn an async future (ExternFn, no interpreter access).
    fn spawn_async(
        &self,
        f: Pin<Box<dyn Future<Output = Result<ExecResult, RuntimeError>> + Send>>,
    ) -> HandleValue;

    /// Force a handle to completion and return its result.
    fn eval(&self, handle: HandleValue) -> BoxFuture<'_, Result<ExecResult, RuntimeError>>;
}

// ── SequentialExecutor ───────────────────────────────────────────────

/// Tag types for HandleValue dispatch in SequentialExecutor.
struct DeferredInterpreter(Interpreter);
struct DeferredBlocking(Box<dyn FnOnce() -> Result<ExecResult, RuntimeError> + Send + Sync>);
struct DeferredAsync(
    SyncWrapper<Pin<Box<dyn Future<Output = Result<ExecResult, RuntimeError>> + Send>>>,
);

/// Simplest executor — spawn stores the computation, eval runs it immediately.
/// No parallelism. Good for testing and deterministic execution.
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn spawn_interpreter(&self, interpreter: Interpreter) -> HandleValue {
        HandleValue::new(DeferredInterpreter(interpreter))
    }

    fn spawn_blocking(
        &self,
        f: Box<dyn FnOnce() -> Result<ExecResult, RuntimeError> + Send + Sync>,
    ) -> HandleValue {
        HandleValue::new(DeferredBlocking(f))
    }

    fn spawn_async(
        &self,
        f: Pin<Box<dyn Future<Output = Result<ExecResult, RuntimeError>> + Send>>,
    ) -> HandleValue {
        HandleValue::new(DeferredAsync(SyncWrapper::new(f)))
    }

    fn eval(&self, handle: HandleValue) -> BoxFuture<'_, Result<ExecResult, RuntimeError>> {
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
}
