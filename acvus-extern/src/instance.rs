//! A requirement is a bound on a handler's own type variable, and the
//! shape of the call it admits is the signature's (RFC-0067 Decision 1).
//!
//! A handler declares the requirement by taking the instance as a
//! parameter; `#[extern_fn]` reads it and records it on
//! `FnDecl::requires`, which is what `Externs::combine` meets with the
//! signature's instances.

use futures::future::BoxFuture;

use crate::runtime::Runtime;

/// A resolved instance's handler as a plain function: the values ABI of
/// `Handler::call` without the `&self` and with the window lent rather than
/// moved (RFC-0067 Decision 3).
///
/// The window is lent because an entry's caller is another handler, which
/// was handed the window by value and keeps it for its own further calls.
/// There is no `Runtime::reborrow` for it to make a second handle with, and
/// that absence is a decision: one method on every host buys one word.
///
/// Letting `Handler`'s own closure take the window this way too was built
/// and withdrawn. `Handler::call` then has to give the moved window a stack
/// slot to lend, the address escapes into the closure, and
/// `benches/asm_probe.rs` counted sixteen `Op::run` bodies that ended in the
/// cleanup landing pad instead of the tail jump — `flatten`, `vec_deque`,
/// `skip`, `filter` and `map`.
pub type EntryFn<Rt> = for<'a, 'w> unsafe fn(
    &'a Rt,
    &'a mut <Rt as Runtime>::Frame<'w>,
    &'a [<Rt as Runtime>::Value],
    &'a mut [<Rt as Runtime>::Value],
);

/// A resolved instance whose Rust body is an `async fn`, as a plain
/// function: the run and the caller's window as `EntryFn` takes them, and
/// the future the body is.
///
/// `AsyncCall::call`'s own form — `fn(Rt, &[Value]) -> BoxFuture<'static,
/// Value>` — is not what an entry takes, for two reasons. `Runtime` is
/// not `Clone`, and an entry is called from a handler holding `&Rt`. And
/// `'static` is a claim an entry cannot keep: the receiver in its run is a
/// reference into the calling handler's own storage, so the future borrows
/// the call. The shape here is `AsyncGlue`'s own inner closure type, with
/// the values ABI's run in place of the taken arguments.
pub type AsyncEntryFn<Rt> = for<'a, 'w, 'r> unsafe fn(
    &'a Rt,
    &'a mut <Rt as Runtime>::Frame<'w>,
    &'r [<Rt as Runtime>::Value],
) -> BoxFuture<'a, <Rt as Runtime>::Value>;

/// The task an instance's body runs at, as the function that runs it
/// (RFC-0046).
///
/// Two node kinds, one Rust type each, were the other candidate and are not
/// built. Which form a node has is the registry's answer at the ground
/// type, and what reaches a requiring handler is one untyped word in a
/// `Bounds` slice, so the two kinds would be chosen by casting that word to
/// one of two pointer types — and the wrong cast compiles.
pub enum EntryRun<Rt>
where
    Rt: Runtime,
{
    Sync(EntryFn<Rt>),
    Await(AsyncEntryFn<Rt>),
}

impl<Rt> Clone for EntryRun<Rt>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Rt> Copy for EntryRun<Rt> where Rt: Runtime {}

/// A shared signature as a Rust caller of one of its instances sees it:
/// the shape of a call and nothing about a receiver beyond how the first
/// parameter takes it. `extern_signature!` writes the impl, so that a
/// handler which requires a signature restates none of its modes and none
/// of its widths.
pub trait Signature<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    /// What the signature's first parameter stands at: the variable a bound
    /// names, which RFC-0019 makes the one an instance is matched by.
    type This;
    /// The first parameter's mode: `&'a This`, `&'a mut This`, or `This`.
    /// The mode reaches a requiring handler through this projection alone,
    /// so the handler's own `I::call` is where a wrong mode is refused.
    type Recv<'a>;
    /// The arguments after the first.
    type Rest<'a>;
    type Ret;
}
