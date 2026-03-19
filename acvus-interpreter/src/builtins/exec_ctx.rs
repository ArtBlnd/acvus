//! Execution context trait for async builtins.
//!
//! Defines the two primitive operations that async builtins need from the
//! interpreter: pulling one element from an iterator and calling a closure.
//! The interpreter implements this trait; builtins remain decoupled.

use std::sync::Arc;

use futures::future::BoxFuture;

use acvus_utils::YieldHandle;

use crate::error::RuntimeError;
use crate::iter::IterHandle;
use crate::value::{FnValue, TypedValue, Value};

/// Runtime capabilities required by async builtins.
///
/// Implemented by the interpreter. Async builtin functions are generic over
/// `Ctx: ExecCtx`, so they never depend on the concrete `Interpreter` type.
///
/// ## Ownership
///
/// All methods take `self` by value and return it back, matching the
/// interpreter's ownership-passing pattern for `Send + 'static` futures.
pub trait ExecCtx: Sized {
    /// Pull one element from an iterator.
    ///
    /// Returns `None` if exhausted, `Some((item, rest))` otherwise.
    /// For pure iterators the result is memoized in the handle's state.
    fn exec_next<'a>(
        self,
        ih: IterHandle,
        handle: &'a YieldHandle<TypedValue>,
    ) -> BoxFuture<'a, Result<(Self, Option<(Value, IterHandle)>), RuntimeError>>;

    /// Invoke a closure with the given arguments.
    fn call_closure<'a>(
        self,
        f: FnValue,
        args: Vec<Arc<Value>>,
        handle: &'a YieldHandle<TypedValue>,
    ) -> BoxFuture<'a, Result<(Self, Value), RuntimeError>>;
}

// ── Derived helpers ─────────────────────────────────────────────────

/// Consume an entire iterator into a `Vec<Value>`.
///
/// Built on top of [`ExecCtx::exec_next`] — loops until exhausted.
pub async fn collect_vec<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    ih: IterHandle,
    handle: &'a YieldHandle<TypedValue>,
) -> Result<(Ctx, Vec<Value>), RuntimeError> {
    let mut items = Vec::new();
    let mut current = ih;
    loop {
        let result;
        (ctx, result) = ctx.exec_next(current, handle).await?;
        match result {
            Some((item, rest)) => {
                items.push(item);
                current = rest;
            }
            None => break,
        }
    }
    Ok((ctx, items))
}
