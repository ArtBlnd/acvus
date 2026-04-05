//! Scope: The handler's sole interface to interpreter state.
//!
//! NOT object-safe by design — generic methods force monomorphization.
//! `S: Scope` bound only. No `dyn Scope`. Everything inlines.

use crate::Hosted;
use crate::error::HostError;
use crate::ity::{Callable, EffectParam, ITy};

/// Managed interface through which ExternFn handlers interact with interpreter state.
///
/// All slots (Repr) are pre-allocated by SSA. Scope writes into existing slots.
///
/// **Not object-safe**: generic methods ensure all operations are monomorphized
/// and inlined into the interpreter's codegen. This is intentional — `dyn Scope`
/// would erase type information and destroy all optimization guarantees.
#[allow(async_fn_in_trait)] // intentionally not object-safe
pub trait Scope {
    /// Value location descriptor. Host-specific. Pre-allocated by SSA.
    type Repr: Copy;
    /// Runtime opaque value type. All generic T: Hosted resolve to this at runtime.
    type Owned: Hosted;

    /// Write a value into a pre-allocated slot.
    fn store<T: ITy>(&mut self, repr: Self::Repr, val: T);
    /// Clone a value from a slot.
    fn clone<T: ITy + Clone>(&self, repr: &Self::Repr) -> T;
    /// Move a value out of a slot. The slot is released.
    fn take<T: ITy>(&mut self, repr: Self::Repr) -> T;
    /// Allocate a temporary slot.
    fn alloc_tmp(&mut self) -> Self::Repr;

    /// Call a function with N arguments via CallArgs.
    /// This is the fundamental dispatch — call, call_1, call_2, call_3 delegate here by default.
    /// Interpreters can override call_1/call_2/call_3 for fusion (skip store/take).
    ///
    /// Async: the called function may itself be async.
    async fn call_n<F, Args, R, E>(&mut self, f: &F, args: Args) -> Result<R, HostError>
    where F: Callable<Args, R, E>, Args: CallArgs, R: ITy, E: EffectParam;

    /// Call a function with 0 arguments.
    async fn call<F, R, E>(&mut self, f: &F) -> Result<R, HostError>
    where F: Callable<(), R, E>, R: ITy, E: EffectParam {
        self.call_n(f, ()).await
    }

    /// Call a function with 1 argument.
    async fn call_1<F, A, R, E>(&mut self, f: &F, a: A) -> Result<R, HostError>
    where F: Callable<(A,), R, E>, A: ITy, R: ITy, E: EffectParam {
        self.call_n(f, (a,)).await
    }

    /// Call a function with 2 arguments.
    async fn call_2<F, A, B, R, E>(&mut self, f: &F, a: A, b: B) -> Result<R, HostError>
    where F: Callable<(A, B), R, E>, A: ITy, B: ITy, R: ITy, E: EffectParam {
        self.call_n(f, (a, b)).await
    }

    /// Call a function with 3 arguments.
    async fn call_3<F, A, B, C, R, E>(&mut self, f: &F, a: A, b: B, c: C) -> Result<R, HostError>
    where F: Callable<(A, B, C), R, E>, A: ITy, B: ITy, C: ITy, R: ITy, E: EffectParam {
        self.call_n(f, (a, b, c)).await
    }
}

// ── CallArgs: tuple → argument storage ─────────────────────────────

/// Trait for storing tuple arguments into Scope slots.
///
/// Each element is stored via `Scope::store` with its concrete type.
/// Interpreters override call_1/call_2/call_3 to fuse this away.
pub trait CallArgs: 'static {
    /// Store each argument into pre-allocated slots.
    fn store_all<S: Scope>(self, scope: &mut S, slots: &[S::Repr]);
    /// Number of arguments.
    fn arity() -> usize;
}

impl CallArgs for () {
    fn store_all<S: Scope>(self, _scope: &mut S, _slots: &[S::Repr]) {}
    fn arity() -> usize { 0 }
}

impl<A: ITy> CallArgs for (A,) {
    fn store_all<S: Scope>(self, scope: &mut S, slots: &[S::Repr]) {
        scope.store(slots[0], self.0);
    }
    fn arity() -> usize { 1 }
}

impl<A: ITy, B: ITy> CallArgs for (A, B) {
    fn store_all<S: Scope>(self, scope: &mut S, slots: &[S::Repr]) {
        scope.store(slots[0], self.0);
        scope.store(slots[1], self.1);
    }
    fn arity() -> usize { 2 }
}

impl<A: ITy, B: ITy, C: ITy> CallArgs for (A, B, C) {
    fn store_all<S: Scope>(self, scope: &mut S, slots: &[S::Repr]) {
        scope.store(slots[0], self.0);
        scope.store(slots[1], self.1);
        scope.store(slots[2], self.2);
    }
    fn arity() -> usize { 3 }
}

impl<A: ITy, B: ITy, C: ITy, D: ITy> CallArgs for (A, B, C, D) {
    fn store_all<S: Scope>(self, scope: &mut S, slots: &[S::Repr]) {
        scope.store(slots[0], self.0);
        scope.store(slots[1], self.1);
        scope.store(slots[2], self.2);
        scope.store(slots[3], self.3);
    }
    fn arity() -> usize { 4 }
}
