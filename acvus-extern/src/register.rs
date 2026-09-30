//! Registering an extern fn with a runtime, one function at a time and at
//! once (registry). A registered function is its dispatch: the glue
//! `#[extern_fn]` writes, which takes the call's argument run and opens each
//! argument by its own mode, monomorphised at the runtime and at every
//! parameter's type, so the runtime reaches the body by the one indirect call
//! it makes to the function and nothing else is dispatched or tested.

use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::entry::SiteShape;
use crate::boundary::seal::ExternFns;
use crate::done::Done;
use crate::runtime::Runtime;

/// A function a runtime can register: its path, and its dispatch at the
/// runtime `R`. Only `#[extern_fn]` implements it (the seal); a dispatch
/// written by hand could open an argument at another type than the
/// declaration's.
pub trait ExternFn<R: Runtime>: ExternFns + Send + Sync + 'static {
    /// The declaration's path, the name a compiled call names it by.
    fn path(&self, names: &Interner) -> QualifiedRef;

    /// Runs one call: the entry over `args` and `site`, each argument opened
    /// by its mode, the body, and its result.
    fn call(&self, runtime: &R, args: &[R::Value], site: &SiteShape<R>) -> Done<R>;
}

/// An awaited function a runtime can register: its path, and its dispatch at
/// the runtime `R`, whose call is a future the runtime awaits. The future's
/// type is the function's own; the runtime registering it is generic over the
/// function, so how it holds the function and its future is the runtime's
/// own. Only `#[extern_fn]` implements it (the seal), for an `async fn`.
pub trait AwaitedFn<R: Runtime>: ExternFns + Send + Sync + 'static {
    /// The declaration's path, the name a compiled call names it by.
    fn path(&self, names: &Interner) -> QualifiedRef;

    /// Runs one call: the entry over `args` and `site`, each argument opened
    /// by its mode before the future is made, the body awaited, and its
    /// result.
    fn call<'c>(
        &'c self,
        runtime: &'c R,
        args: &'c [R::Value],
        site: &'c SiteShape<R>,
    ) -> impl Future<Output = Done<R>> + Send + 'c;
}

/// Why a registration is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// A function of this path, returning or awaited, is registered
    /// already.
    PathTaken(QualifiedRef),
}

/// A runtime that registers functions, each at once, and resolves a compiled
/// call's path to the registered function once, when it prepares the call.
pub trait Registers: Runtime {
    fn register<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self>;

    fn register_awaited<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: AwaitedFn<Self>;

    /// The returning function registered at `path`.
    fn registered(&self, path: QualifiedRef) -> Option<&dyn ExternFn<Self>>;

    /// Calls the awaited function registered at `path`, none being
    /// registered there as awaited. How the runtime holds the function and
    /// the future its call makes is its own.
    fn call_awaited<'c>(
        &'c self,
        path: QualifiedRef,
        args: &'c [Self::Value],
        site: &'c SiteShape<Self>,
    ) -> Option<impl Future<Output = Done<Self>> + Send + 'c>;
}
