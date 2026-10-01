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
///
/// `N` is its arity and `M` its requirements: a call's argument run is
/// `[R::Value; N]` and its site's impls `[R::Value; M]`, of static size, so a
/// runtime that builds a call's operation at `F`, `N` and `M` passes them
/// without a length and a compare.
pub trait ExternFn<R: Runtime, const N: usize, const M: usize>: ExternFns + Send + Sync + 'static {
    /// Runs one call: the entry over `args` and `site`, each argument opened
    /// by its mode, the body, and its result.
    fn call(&self, runtime: &R, args: [R::Value; N], site: &SiteShape<R, M>) -> Done<R>;
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
/// An effect that is neither a level nor one of the function's effect
/// variables:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(effect = F)]
/// fn f(a: i64) -> i64 { a }
/// ```
///
/// `'static`, which names no loan a call is tracked by:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f(text: &'static str) -> usize { text.len() }
/// ```
///
/// A result whose lifetime is the context's alone:
///
/// ```compile_fail
/// use acvus_extern::context::Returning;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::extern_fn]
/// fn f<'c, R: Runtime>(#[context] cx: &'c Returning<'c, R>, n: i64) -> &'c str { "" }
/// ```
///
/// A relation speaking of what is no argument, the context or a capture:
///
/// ```compile_fail
/// use acvus_extern::context::Returning;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::extern_fn(relation(result <= cx))]
/// fn f<R: Runtime>(#[context] cx: &Returning<'_, R>, n: u64) -> u64 { n }
/// ```
///
/// A relation that is no `=`, `<=` or `<`, or a term no affine one:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(relation(result >= n))]
/// fn f(n: u64) -> u64 { n }
/// ```
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(relation(result = n * n))]
/// fn f(n: u64) -> u64 { n * n }
/// ```
///
/// An ending that is neither `returns` nor `total`:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(ends = forever)]
/// fn f(n: u64) -> u64 { n }
/// ```
///
/// A fact of a place on what is no argument, or naming what is none:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f(#[capture] #[touch(read)] tag: &String, n: u64) -> u64 { n }
/// ```
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f(#[touch(store(nothing))] text: &mut String, n: u64) -> u64 { n }
/// ```
///
/// A touch that is no mode the compiler knows:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f(#[touch(write)] text: &mut String) {}
/// ```
///
/// An awaited body whose result borrows:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// async fn f(text: &str) -> &str { text }
/// ```
pub trait Declared<R: Runtime> {
    /// The declaration's path, the name a compiled call names it by: the
    /// function's own, whatever it captures, so another declaration names it
    /// without a value of it.
    fn path(names: &Interner) -> QualifiedRef;

    /// What the compiler knows of the function.
    fn declaration(&self, names: &Interner) -> Declaration;
}

/// An awaited function's dispatch at the runtime `R`, whose call is a future
/// the runtime awaits. The future's
/// type is the function's own; the runtime registering it is generic over the
/// function, so how it holds the function and its future is the runtime's
/// own. Only `#[extern_fn]` implements it (the seal), for an `async fn`.
pub trait AwaitedFn<R: Runtime, const N: usize, const M: usize>: ExternFns + Send + Sync + 'static {
    /// Runs one call: the entry over `args` and `site`, each argument opened
    /// by its mode before the future is made, the body awaited, and its
    /// result.
    fn call<'c>(
        &'c self,
        runtime: &'c R,
        args: [R::Value; N],
        site: &'c SiteShape<R, M>,
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
    fn register<F, const N: usize, const M: usize>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self, N, M> + Declared<Self>;

    fn register_awaited<F, const N: usize, const M: usize>(&mut self, f: F) -> Result<(), Refused>
    where
        F: AwaitedFn<Self, N, M> + Declared<Self>;
}
