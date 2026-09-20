//! A requirement is a bound, and the instance it names is a tree the call
//! site resolved (RFC-0067 Decisions 1 and 4).
//!
//! A handler requires a shared signature of one of its type variables by
//! writing `T: InstanceOf<sig::S<T, Rt>>`, and calls it as `T::call(&x, rt,
//! frame, rest)`. Nothing is looked up while the call runs: the site
//! resolved the instance, and the instances that instance's own bounds
//! require, before the glue was given its site.

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::Mutex;

use futures::future::BoxFuture;

use crate::handler::ArgAt;
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{Var, kind};

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

/// One instance as a site resolved it: the function it runs through, and
/// the entry of the instance each of its own bounds required, in the
/// declaration's `requires` order.
pub struct EntryNode<Rt>
where
    Rt: Runtime,
{
    run: EntryRun<Rt>,
    children: Box<[Entry<Rt>]>,
}

/// A resolved instance, as the one word it is: the address of a node in the
/// arena of the `InstanceEntries` the site table was filled from.
///
/// The word is what `Kind::Entry` carries, so a run holds one wherever the
/// callee's own bounds have to reach an instance the callee's site does not
/// know.
pub struct Entry<Rt>(*const EntryNode<Rt>)
where
    Rt: Runtime;

impl<Rt> Clone for Entry<Rt>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Rt> Copy for Entry<Rt> where Rt: Runtime {}

// SAFETY: the node behind the pointer is immutable once the arena has
// handed its address out, and `EntryFn` is a plain function pointer.
unsafe impl<Rt> Send for Entry<Rt> where Rt: Runtime {}
// SAFETY: as `Send`'s.
unsafe impl<Rt> Sync for Entry<Rt> where Rt: Runtime {}

impl<Rt> std::fmt::Debug for Entry<Rt>
where
    Rt: Runtime,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entry({:p})", self.0)
    }
}

impl<Rt> Entry<Rt>
where
    Rt: Runtime,
{
    /// The one word this entry is, as a host's value word.
    pub fn addr(self) -> usize {
        self.0 as usize
    }

    /// # Safety
    /// `addr` is what `Entry::addr` read off an entry whose arena is alive.
    pub unsafe fn from_addr(addr: usize) -> Self {
        Entry(addr as *const EntryNode<Rt>)
    }

    /// # Safety
    /// The arena that made this entry is alive: the site table that resolved
    /// it holds the arena, and a run's entry word is the caller's own.
    pub unsafe fn run(self) -> EntryRun<Rt> {
        // SAFETY: the caller's contract.
        unsafe { (*self.0).run }
    }

    /// What this instance's own bounds resolved to, in `requires` order.
    ///
    /// # Safety
    /// As `run`'s.
    pub unsafe fn bounds(self) -> Bounds<Rt> {
        // SAFETY: the caller's contract.
        Bounds(unsafe { &raw const *(*self.0).children })
    }
}

/// The entries one declaration's bounds resolved to, as the site table
/// holds them: a slice in the arena, read by position in `requires` order.
pub struct Bounds<Rt>(*const [Entry<Rt>])
where
    Rt: Runtime;

impl<Rt> Clone for Bounds<Rt>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Rt> Copy for Bounds<Rt> where Rt: Runtime {}

// SAFETY: as `Entry`'s.
unsafe impl<Rt> Send for Bounds<Rt> where Rt: Runtime {}
// SAFETY: as `Entry`'s.
unsafe impl<Rt> Sync for Bounds<Rt> where Rt: Runtime {}

impl<Rt> Bounds<Rt>
where
    Rt: Runtime,
{
    /// The bounds of a declaration that has none.
    pub fn none() -> Self {
        Bounds(&raw const *(&[] as &[Entry<Rt>]))
    }

    /// # Safety
    /// The arena that made these entries is alive, as `Entry::run`'s.
    pub unsafe fn at(self, index: usize) -> Entry<Rt> {
        // SAFETY: the caller's contract.
        unsafe { (*self.0)[index] }
    }
}

/// Where the nodes a site resolved live: an append-only arena owned by the
/// registry's `InstanceTable` and kept alive by every site table that read
/// one, so that a node outlives every call the glue holding its address can
/// make.
pub struct NodeArena<Rt>
where
    Rt: Runtime,
{
    nodes: Mutex<Vec<Box<EntryNode<Rt>>>>,
    lists: Mutex<Vec<Box<[Entry<Rt>]>>>,
}

impl<Rt> Default for NodeArena<Rt>
where
    Rt: Runtime,
{
    fn default() -> Self {
        NodeArena {
            nodes: Mutex::new(Vec::new()),
            lists: Mutex::new(Vec::new()),
        }
    }
}

impl<Rt> NodeArena<Rt>
where
    Rt: Runtime,
{
    pub fn node(&self, run: EntryRun<Rt>, children: Vec<Entry<Rt>>) -> Entry<Rt> {
        let node = Box::new(EntryNode {
            run,
            children: children.into_boxed_slice(),
        });
        let at: *const EntryNode<Rt> = &raw const *node;
        self.nodes.lock().expect("the arena's lock").push(node);
        Entry(at)
    }

    /// The entries a declaration's own bounds resolved to, where no node
    /// stands above them: the requiring handler is a handler and not an
    /// instance, so nothing was resolved for it to be a child of.
    pub fn bounds(&self, entries: Vec<Entry<Rt>>) -> Bounds<Rt> {
        let list = entries.into_boxed_slice();
        let at: *const [Entry<Rt>] = &raw const *list;
        self.lists.lock().expect("the arena's lock").push(list);
        Bounds(at)
    }
}

/// A bounded type variable's carrier: the value one argument of one call
/// site passed, with the entry of each instance the declaration required
/// beside it. `#[extern_fn]` writes one per bounded variable.
///
/// A carrier is built at the call and never stored: what crosses into an
/// extension type's payload is `Held<Rt>`, the value alone.
pub trait Carrier<Rt>: Var<kind::Type> + Sized
where
    Rt: Runtime,
{
    /// One entry per `InstanceOf` bound, in the order the declaration's
    /// `where` clause wrote them; `#[extern_fn]` writes both halves, the
    /// reads here and the `requires` list it puts on the `FnDecl`.
    fn entries(at: ArgAt<'_, Rt>) -> Bounds<Rt>;

    fn of(value: Rt::Value, bounds: Bounds<Rt>) -> Self;

    /// The value where it lies: `call_entry` takes this reference's address
    /// as the storage the instance's `&T` parameter names.
    fn value(&self) -> &Rt::Value;

    fn into_value(self) -> Rt::Value;
}

/// What a site resolved for one bounded variable, as the body of a pattern
/// instance receives it: the entries a carrier of that variable is built
/// with.
pub struct Bound<C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
    bounds: Bounds<Rt>,
    at: PhantomData<fn() -> C>,
}

impl<C, Rt> Clone for Bound<C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, Rt> Copy for Bound<C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
}

impl<C, Rt> Bound<C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
    pub fn new(bounds: Bounds<Rt>) -> Self {
        Bound {
            bounds,
            at: PhantomData,
        }
    }
}

/// A bounded variable's value where a value keeps it: one of the runtime's
/// values and nothing else, so that an extension type holding one is a
/// payload naming no type parameter and both the handler that built it and
/// the instance that reads it back name one Rust type (RFC-0067, step 3
/// second half).
#[repr(transparent)]
pub struct Held<Rt>(Owned<Rt>)
where
    Rt: Runtime;

impl<Rt> Held<Rt>
where
    Rt: Runtime,
{
    /// The value a carrier stands at, kept without its entries.
    pub fn of<C>(carrier: C) -> Self
    where
        C: Carrier<Rt>,
    {
        Held(Owned::from_value(carrier.into_value()))
    }

    /// The same field where the declaration's variable carries no bound, so
    /// that what the call handed over is the runtime's value itself.
    pub fn of_value(value: Rt::Value) -> Self {
        Held(Owned::from_value(value))
    }

    /// The carrier this value stands at, for the length of one call: the
    /// entries are the site's, and what the call wrote into the value is
    /// written back here when the guard drops.
    pub fn at<'a, C>(&'a mut self, bound: &Bound<C, Rt>) -> HeldMut<'a, C, Rt>
    where
        C: Carrier<Rt>,
    {
        let value = std::mem::take(&mut self.0).into_value();
        HeldMut {
            carrier: ManuallyDrop::new(C::of(value, bound.bounds)),
            slot: self,
        }
    }
}

/// A carrier built over a `Held` value for one call, writing the value back
/// where it came from: an instance whose receiver is `&mut I` may write its
/// own value word, and the field that lent it is where that word belongs.
pub struct HeldMut<'a, C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
    carrier: ManuallyDrop<C>,
    slot: &'a mut Held<Rt>,
}

impl<C, Rt> std::ops::Deref for HeldMut<'_, C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
    type Target = C;

    fn deref(&self) -> &C {
        &self.carrier
    }
}

impl<C, Rt> std::ops::DerefMut for HeldMut<'_, C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
    fn deref_mut(&mut self) -> &mut C {
        &mut self.carrier
    }
}

impl<C, Rt> Drop for HeldMut<'_, C, Rt>
where
    C: Carrier<Rt>,
    Rt: Runtime,
{
    fn drop(&mut self) {
        // SAFETY: `ManuallyDrop::take` is called once, in the guard's own
        // `Drop`, and nothing reads the carrier afterwards.
        let carrier = unsafe { ManuallyDrop::take(&mut self.carrier) };
        self.slot.0 = Owned::from_value(carrier.into_value());
    }
}

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
    type This: Carrier<Rt>;
    /// The first parameter's mode: `&'a This`, `&'a mut This`, or `This`.
    /// The mode reaches a requiring handler through this projection alone,
    /// so the handler's own `I::call` is where a wrong mode is refused;
    /// `acvus-extern-macro/tests/compile_fail/instance_wrong_mode.rs` is
    /// that refusal, executed.
    type Recv<'a>;
    /// The arguments after the first.
    type Rest<'a>;
    type Ret;

    /// The carrier a receiver in any of the three modes stands at, which is
    /// where the entry of this signature lies.
    fn as_this<'a>(recv: &'a Self::Recv<'_>) -> &'a Self::This;

    /// # Safety
    /// `entry` is the entry of this signature's instance at the acvus type
    /// `this` holds, which is what the site `this` came from resolved, and
    /// the arena that entry belongs to is alive.
    unsafe fn call_entry(
        entry: Entry<Rt>,
        rt: &Rt,
        frame: &mut Rt::Frame<'_>,
        this: Self::Recv<'_>,
        rest: Self::Rest<'_>,
    ) -> Self::Ret;

    /// The future borrows the call. That lifetime is not consumed here: it
    /// is what holds the calling handler's storage alive for the reference
    /// an async instance's run carries into it, and the entry `#[extern_fn]`
    /// writes is where that reference is read.
    ///
    /// # Safety
    /// As `call_entry`'s.
    unsafe fn call_entry_async<'a>(
        entry: Entry<Rt>,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
        this: Self::Recv<'a>,
        rest: Self::Rest<'a>,
    ) -> BoxFuture<'a, Self::Ret>
    where
        Self::Ret: Send;
}

/// One of the runtime's values holding what `value` crosses as. The run an
/// entry is called with is built out of these, one per parameter after the
/// first, and ends in the entry's own word.
pub fn one_value<Rt, T>(rt: &Rt, value: T) -> Rt::Value
where
    Rt: Runtime,
    T: crate::obj::Cross<Rt, Form = crate::obj::One>,
{
    let mut out = [<Rt::Value as Default>::default()];
    value.into_run(rt, &mut out);
    out[0]
}

/// A type with an instance of the shared signature `S` (RFC-0067 Decision
/// 1). `call` is the handle the marker bound `HasInstance<S>` never had,
/// and for want of which that bound was deleted.
///
/// What a call takes and gives is `S`'s and not this trait's, so a handler
/// that requires a signature writes the requirement and nothing more: the
/// modes, the widths and the result are read off the signature's own
/// declaration wherever the bound is used.
pub trait InstanceOf<S, Rt>: Sized
where
    Rt: Runtime,
    S: Signature<Rt, This = Self>,
{
    fn call(
        this: <S as Signature<Rt>>::Recv<'_>,
        rt: &Rt,
        frame: &mut Rt::Frame<'_>,
        rest: <S as Signature<Rt>>::Rest<'_>,
    ) -> <S as Signature<Rt>>::Ret;
}

/// The same predicate at the async task: what a requiring handler's `async`
/// body bounds its variable by, where the `_now` twin bounds it by
/// `InstanceOf`. RFC-0046's site effect picks which body runs, so a
/// pipeline runs sync until its first async stage and async from there.
///
/// The direction is one way, and the node is where it is kept: a sync
/// instance's node runs under either trait, and an async instance's node
/// runs only under this one. There is no impl the other way — an async
/// instance suspends, and a sync caller has nowhere to suspend to.
pub trait InstanceOfAsync<S, Rt>: Sized
where
    Rt: Runtime,
    S: Signature<Rt, This = Self>,
{
    fn call<'a, 'w>(
        this: <S as Signature<Rt>>::Recv<'a>,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'w>,
        rest: <S as Signature<Rt>>::Rest<'a>,
    ) -> BoxFuture<'a, <S as Signature<Rt>>::Ret>
    where
        <S as Signature<Rt>>::Ret: Send;
}
