//! Registering an extern fn with a runtime, one function at a time and at
//! once (registry). A registered function is its dispatch: the glue
//! `#[extern_fn]` writes, which takes the call's argument run and opens each
//! argument by its own mode, monomorphised at the runtime and at every
//! parameter's type, so the runtime reaches the body by the one indirect call
//! it makes to the function and nothing else is dispatched or tested.

use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::entry::SiteShape;
use crate::boundary::seal::ExternFns;
use crate::decl::Declaration;
use crate::done::Done;
use crate::runtime::Runtime;

/// A function's dispatch at the runtime `R`: a registered function's, or a
/// function value's. Only `#[extern_fn]` implements it (the seal); a dispatch
/// written by hand could open an argument at another type than the
/// declaration's.
pub trait ExternFn<R: Runtime>: ExternFns + Send + Sync + 'static {
    /// Runs one call: the entry over `args` and `site`, each argument opened
    /// by its mode, the body, and its result.
    fn call(&self, runtime: &R, args: &[R::Value], site: &SiteShape<R>) -> Done<R>;
}

/// A function a registry declares: its path and what the compiler knows of
/// it. A function value is a dispatch only, and has neither. It is stated at
/// the runtime `R` only because a parameter's Rust type may name it, as a
/// type variable's `Uniform<R, T>` does; no part of it depends on `R`.
///
/// `#[extern_fn]` writes it from a signature, and refuses what a declaration
/// cannot state. A key the attribute does not know:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(pure)]
/// fn f(a: i64) -> i64 { a }
/// ```
///
/// A generic parameter bounded by anything but one `Var<kind::..>`, beside
/// the runtime:
///
/// ```compile_fail
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::extern_fn]
/// fn f<R: Runtime, T: Clone>(a: acvus_extern::Uniform<R, T>) {}
/// ```
///
/// A capture not lent as `&T`:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f(#[capture] tag: String, a: i64) -> i64 { a }
/// ```
///
/// A result that borrows from more than one lent parameter:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f<'a, 'b>(a: &'a str, b: &'b str) -> (&'a str, &'b str) { (a, b) }
/// ```
///
/// An effect that is neither a level nor one of the function's effect
/// variables:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(effect = F)]
/// fn f(a: i64) -> i64 { a }
/// ```
///
/// An awaited body whose result borrows:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// async fn f(text: &str) -> &str { text }
/// ```
pub trait Declared<R: Runtime> {
    /// The declaration's path, the name a compiled call names it by.
    fn path(&self, names: &Interner) -> QualifiedRef;

    /// What the compiler knows of the function.
    fn declaration(&self, names: &Interner) -> Declaration;
}

/// An awaited function's dispatch at the runtime `R`, whose call is a future
/// the runtime awaits. The future's
/// type is the function's own; the runtime registering it is generic over the
/// function, so how it holds the function and its future is the runtime's
/// own. Only `#[extern_fn]` implements it (the seal), for an `async fn`.
pub trait AwaitedFn<R: Runtime>: ExternFns + Send + Sync + 'static {
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
        F: ExternFn<Self> + Declared<Self>;

    fn register_awaited<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: AwaitedFn<Self> + Declared<Self>;


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
