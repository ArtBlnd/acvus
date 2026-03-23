//! Executor trait — controls how spawned interpreters are executed.
//!
//! The interpreter calls `executor.spawn(child_interpreter)` on Spawn,
//! and `executor.eval(handle).await` on Eval.

use crate::error::RuntimeError;
use crate::interpreter::{ExecResult, Interpreter};
use crate::value::HandleValue;

// ── Trait ─────────────────────────────────────────────────────────────

/// Executor controls spawn/eval execution strategy.
///
/// Implementations decide whether to run sequentially, in parallel
/// (tokio::spawn), or with any other scheduling strategy.
pub trait Executor: Send + Sync {
    /// Create a handle for deferred execution. May or may not start running.
    fn spawn(&self, interpreter: Interpreter) -> HandleValue;

    /// Force a handle to completion and return its result.
    fn eval(
        &self,
        handle: HandleValue,
    ) -> futures::future::BoxFuture<'_, Result<ExecResult, RuntimeError>>;
}

// ── SequentialExecutor ───────────────────────────────────────────────

/// Simplest executor — spawn stores the interpreter, eval runs it immediately.
/// No parallelism. Good for testing and deterministic execution.
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn spawn(&self, interpreter: Interpreter) -> HandleValue {
        HandleValue::new(interpreter)
    }

    fn eval(
        &self,
        handle: HandleValue,
    ) -> futures::future::BoxFuture<'_, Result<ExecResult, RuntimeError>> {
        Box::pin(async move {
            let mut interp = handle.downcast::<Interpreter>();
            interp.execute().await
        })
    }
}
