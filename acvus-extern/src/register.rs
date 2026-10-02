//! Registering an extern fn with a runtime, one function at a time and at
//! once (registry). A registered function is its dispatch: the glue
//! `#[extern_fn]` writes, which takes the call's argument run and opens each
//! argument at the Rust type its parameter writes, monomorphised at the runtime and at every
//! parameter's type, so the runtime reaches the body by the one indirect call
//! it makes to the function and nothing else is dispatched or tested.

use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::entry::SiteShape;
use crate::boundary::seal::{ExternFns, Families};
use crate::decl::{Declaration, GenericDeclaration};
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
    /// at its parameter's Rust type, the body, and its result.
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
/// A generic parameter bounded by anything but one `TypeVar<R>`, `EffectVar`
/// or `IdentityVar`, beside the runtime:
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
/// A lifetime elided in a path, which a signature names:
///
/// ```compile_fail
/// use acvus_extern::Loaned;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::extern_fn]
/// fn f<R: Runtime>(text: Loaned<String, R>) {}
/// ```
///
/// A result whose lifetime is the executor's alone:
///
/// ```compile_fail
/// use acvus_extern::executor::Executor;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::extern_fn]
/// fn f<'c, R: Runtime>(#[executor] cx: &'c Executor<'c, R>, n: i64) -> &'c str { "" }
/// ```
///
/// A relation speaking of what is no argument, the executor or a capture:
///
/// ```compile_fail
/// use acvus_extern::executor::Executor;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::extern_fn(relation(result <= cx))]
/// fn f<R: Runtime>(#[executor] cx: &Executor<'_, R>, n: u64) -> u64 { n }
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
/// A bound written between lifetimes, which no signature writes:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f<'a, 'b: 'a>(x: &'a str, y: &'b str) -> &'a str { y }
/// ```
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn f<'a, 'b>(x: &'a str, y: &'b str) -> &'a str where 'b: 'a { y }
/// ```
///
/// A returning body named for a function that is one itself:
///
/// ```compile_fail
/// fn g(n: u64) -> u64 { n }
///
/// #[acvus_extern::extern_fn(sync = g)]
/// fn f(n: u64) -> u64 { n }
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
/// A combine's law the compiler does not know, and an update naming no
/// combine:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(combine(idempotent))]
/// fn f(a: i64, b: i64) -> i64 { a.max(b) }
/// ```
///
/// ```compile_fail
/// #[acvus_extern::extern_fn]
/// fn g(#[update(f)] counter: &mut i64, by: i64) {}
/// ```
pub trait Declared<R: Runtime> {
    /// The declaration's path, the name a compiled call names it by: the
    /// function's own, whatever it captures, so another declaration names it
    /// without a value of it.
    fn path(names: &Interner) -> QualifiedRef;

    /// What the compiler knows of the function.
    fn declaration(&self, names: &Interner) -> Declaration;
}

/// A generic function's declaration (extern fn §6): a name with a signature
/// and no body, implemented by `#[generic_fn]` on the generic function's
/// family marker. It has no dispatch: a call of it is a call of the impl it
/// reaches.
///
/// Only `#[generic_fn]` makes a family (the seal `Families`): a family
/// written by hand, which could state a run its impls do not take, does not
/// compile.
///
/// ```compile_fail,E0277
/// #[allow(non_camel_case_types)]
/// struct fake {}
///
/// impl acvus_extern::instance::GenericFunction for fake {
///     type Takes = acvus_extern::instance::receiver::Shared;
/// }
/// ```
///
/// A generic function states nothing a body states, so the attribute refuses
/// an effect, a fact, an executor, a capture and an awaited signature:
///
/// ```compile_fail
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn(effect = pure)]
/// fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;
/// ```
///
/// ```compile_fail
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// async fn same<R: Runtime, T: TypeVar<R>>(a: T, b: T) -> bool;
/// ```
///
/// ```compile_fail
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::executor::Executor;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: Runtime, T: TypeVar<R>>(#[executor] cx: &Executor<'_, R>, a: &T, b: &T) -> bool;
/// ```
///
/// An impl's signature is its generic function's at the choice it writes;
/// another signature does not compile:
///
/// ```compile_fail
/// use acvus_extern::Uniform;
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;
///
/// #[acvus_extern::extern_fn(impl_of = same<Uniform<R, i64>>)]
/// fn same_i64<R: Runtime>(a: &Uniform<R, i64>) -> bool { true }
/// ```
///
/// A requirement is written by its role, and its Rust type is that role's:
/// a handle at a type for a generic function taking its receiver by `&`,
/// the receiver with its impl for one taking it by value or by `&mut`:
///
/// ```compile_fail
/// use acvus_extern::callback::CallsBack;
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::executor::Executor;
/// use acvus_extern::instance::Instance;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn bump<R: Runtime, T: TypeVar<R>>(counter: &mut T) -> u64;
///
/// #[acvus_extern::extern_fn]
/// fn f<R: CallsBack, T: TypeVar<R>>(#[executor] cx: &Executor<'_, R>, #[requires] c: Instance<R, bump, (T,)>) -> u64 { 0 }
/// ```
///
/// ```compile_fail
/// use acvus_extern::callback::CallsBack;
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::executor::Executor;
/// use acvus_extern::instance::InstanceOf;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn bump<R: Runtime, T: TypeVar<R>>(counter: &mut T) -> u64;
///
/// #[acvus_extern::extern_fn]
/// fn f<R: CallsBack, T: TypeVar<R>>(#[executor] cx: &Executor<'_, R>, #[requires] c: &InstanceOf<R, bump, (T,)>) -> u64 { 0 }
/// ```
///
/// ```compile_fail
/// use acvus_extern::callback::CallsBack;
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::executor::Executor;
/// use acvus_extern::instance::Instance;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;
///
/// #[acvus_extern::extern_fn]
/// fn f<R: CallsBack, T: TypeVar<R>>(#[executor] cx: &Executor<'_, R>, #[instance] c: Instance<R, same, (T,)>) -> u64 { 0 }
/// ```
///
/// A type variable's place takes a held value, `Uniform<R, X>`: a Rust
/// value is its type's own representation, which never stands for a type
/// variable (runtime contract §1), whether an impl chooses it or a
/// requirement does:
///
/// ```compile_fail
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;
///
/// #[acvus_extern::extern_fn(impl_of = same<i64>)]
/// fn same_i64(a: &i64, b: &i64) -> bool { a == b }
/// ```
///
/// ```compile_fail
/// use acvus_extern::callback::CallsBack;
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::executor::Executor;
/// use acvus_extern::instance::InstanceOf;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;
///
/// #[acvus_extern::extern_fn]
/// fn f<R: CallsBack>(#[executor] cx: &Executor<'_, R>, #[requires] eq: &InstanceOf<R, same, (i64,)>) -> bool { true }
/// ```
///
/// An impl takes no requirement as a parameter: the impls it requires are
/// its value's captures (impls §2):
///
/// ```compile_fail
/// use acvus_extern::Uniform;
/// use acvus_extern::callback::CallsBack;
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::instance::InstanceOf;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;
///
/// #[acvus_extern::extern_fn(impl_of = same<Uniform<R, i64>>)]
/// fn same_i64<R: CallsBack>(#[requires] eq: &InstanceOf<R, same, (Uniform<R, i64>,)>, a: &Uniform<R, i64>, b: &Uniform<R, i64>) -> bool { true }
/// ```
///
/// An impl states no name: it is reached only through its generic function:
///
/// ```compile_fail
/// use acvus_extern::Uniform;
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::runtime::Runtime;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: Runtime, T: TypeVar<R>>(a: &T, b: &T) -> bool;
///
/// #[acvus_extern::extern_fn(impl_of = same<Uniform<R, i64>>, name = "same_i64")]
/// fn same_i64<R: Runtime>(a: &Uniform<R, i64>, b: &Uniform<R, i64>) -> bool { true }
/// ```
///
/// `generic` marks an impl, so it is stated with `impl_of`:
///
/// ```compile_fail
/// #[acvus_extern::extern_fn(generic)]
/// fn f(n: u64) -> u64 { n }
/// ```
pub trait GenericDeclared<R: Runtime>: Families {
    /// The generic function's path: its Rust module path and its name.
    fn path(names: &Interner) -> QualifiedRef;

    /// What the compiler knows of the generic function.
    fn declaration(names: &Interner) -> GenericDeclaration;
}

/// An awaited function's dispatch at the runtime `R`, whose call is a future
/// the runtime awaits. The future's
/// type is the function's own; the runtime registering it is generic over the
/// function, so how it holds the function and its future is the runtime's
/// own. Only `#[extern_fn]` implements it (the seal), for an `async fn`.
pub trait AwaitedFn<R: Runtime, const N: usize, const M: usize>: ExternFns + Send + Sync + 'static {
    /// Runs one call: the entry over `args` and `site`, each argument opened
    /// at its parameter's Rust type before the future is made, the body
    /// awaited, and its result.
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

    /// Registers a function with both bodies (macros §1): which one a call
    /// runs is the runtime's choice, the returning one wherever the call's
    /// effect does not suspend.
    fn register_pair<F, const N: usize, const M: usize>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self, N, M> + AwaitedFn<Self, N, M> + Declared<Self>;
}
