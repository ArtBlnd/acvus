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

/// Why a registration is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// A function of this path is registered already.
    PathTaken(QualifiedRef),
}

/// A runtime that registers functions, each at once, and resolves a compiled
/// call's path to the registered function once, when it prepares the call.
pub trait Registers: Runtime {
    fn register<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self>;

    /// The function registered at `path`.
    fn registered(&self, path: QualifiedRef) -> Option<&dyn ExternFn<Self>>;
}
