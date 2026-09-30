//! The checking adapter (runtime contract §3, K15). Only a trusting runtime
//! implements the contract; for every such runtime X, `Adapter<X>` implements
//! it again with X's own values and X's own operations, and checks what X
//! trusts. The host chooses whether to run X or `Adapter<X>`, and an extern
//! cannot tell which.
//!
//! The detection is partial and grows. Lent storage is tracked by address in a
//! model after Tree Borrows (`Borrows`), and a typed read is checked by
//! `Validate`, which X provides outside the contract. No tracked model is
//! known to cover every breach, so a breach the tracking does not cover passes
//! as it would under X, and a use the tracking cannot yet tell apart from a
//! breach may be refused.

use std::ops::{Deref, DerefMut, Range};
use std::sync::Mutex;
use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::callback::{CallArgs, CallsBack};
use crate::impls::task::TaskKind;
use crate::impls::{Family, InstanceEntry, Instances, Task};
use crate::runtime::{Crosses, Mutability, RefTarget, Runtime};

/// What a runtime answers, from its own representation, about a value the
/// crossing is about to read at a type. It is not part of the contract: only
/// the adapter asks it, and a runtime run without the adapter is never asked.
pub trait Validate: Runtime {
    /// Whether `v` is a value of `T`.
    fn validate<T: Crosses>(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
    /// Whether `v` is a program's array whose elements are values of `T`.
    fn validate_run<T: Crosses>(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
    /// Whether `v` is a `String`.
    fn validate_text(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
    /// Whether `v` is an impl value of the generic function `F` at task `K`.
    fn validate_instance<F: Family, K: TaskKind>(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
    /// Whether `v` is a script closure.
    fn validate_closure(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
}

/// A runtime's answer to `Validate`.
#[derive(Debug)]
pub enum Validity {
    Valid,
    /// The value is not what the read takes it for.
    Breach(Breach),
    /// The runtime cannot tell from its representation.
    Unknown,
}

/// A breach the adapter detected: X would have undefined behaviour here.
#[derive(Debug)]
pub struct Breach {
    pub what: String,
}

/// The adapter over the trusting runtime `X`.
pub struct Adapter<X> {
    inner: X,
    borrows: Mutex<Borrows>,
}

impl<X: Runtime + Validate> Adapter<X> {
    pub fn new(inner: X) -> Self {
        Adapter { inner, borrows: Mutex::new(Borrows::new()) }
    }

    pub fn inner(&self) -> &X {
        &self.inner
    }
}

/// Lent storage, tracked by address after Tree Borrows: each lend is a node
/// over its address range, a part borrowed from a lend is its child, and an
/// access or a lend is checked against the nodes over the range it touches.
/// A range no node covers is untracked, and nothing is checked there.
struct Borrows {}

/// One node of `Borrows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tag(u64);

impl Borrows {
    fn new() -> Self {
        todo!("an empty tree")
    }

    /// A lend of `range` for a guard.
    fn lend(&mut self, range: Range<usize>, access: Mutability) -> Result<Tag, Breach> {
        todo!("a root node over range, checked against the nodes it overlaps")
    }

    /// A part borrowed from a lend, for a reference value.
    fn reborrow(&mut self, range: Range<usize>, access: Mutability) -> Result<Option<Tag>, Breach> {
        todo!("a child of the node whose range holds it; None where no node covers the range")
    }

    /// A guard dropped.
    fn end(&mut self, tag: Tag) {
        todo!("the node and its children end")
    }
}

/// A guard of X, tracked while it lends.
pub struct Tracked<'a, G> {
    inner: G,
    tag: Tag,
    borrows: &'a Mutex<Borrows>,
}

impl<G: Deref> Deref for Tracked<'_, G> {
    type Target = G::Target;

    fn deref(&self) -> &G::Target {
        &self.inner
    }
}

impl<G: DerefMut> DerefMut for Tracked<'_, G> {
    fn deref_mut(&mut self) -> &mut G::Target {
        &mut self.inner
    }
}

impl<G> Drop for Tracked<'_, G> {
    fn drop(&mut self) {
        todo!("end the tag")
    }
}

impl<X: Runtime + Validate> Runtime for Adapter<X> {
    type Value = X::Value;
    type Tuple = X::Tuple;
    type Object = X::Object;
    type Variant = X::Variant;
    type Shared<'a, T: Crosses + 'a> = Tracked<'a, X::Shared<'a, T>>;
    type Exclusive<'a, T: Crosses + 'a> = Tracked<'a, X::Exclusive<'a, T>>;
    type SharedSlice<'a, T: Crosses + 'a> = Tracked<'a, X::SharedSlice<'a, T>>;
    type ExclusiveSlice<'a, T: Crosses + 'a> = Tracked<'a, X::ExclusiveSlice<'a, T>>;
    type SharedStr<'a> = Tracked<'a, X::SharedStr<'a>>;

    fn take<T: Crosses>(&self, v: Self::Value, at: Settled<'_>) -> T {
        todo!("validate at T, then X's take")
    }
    fn give<T: Crosses>(&self, value: T, at: Settled<'_>) -> Self::Value {
        todo!("X's give")
    }
    fn shared<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::Shared<'a, T> {
        todo!("validate at T, X's guard, lend its range shared")
    }
    fn exclusive<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::Exclusive<'a, T> {
        todo!("validate at T, X's guard, lend its range exclusive")
    }
    fn shared_slice<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::SharedSlice<'a, T> {
        todo!("validate the run, X's guard, lend its range shared")
    }
    fn exclusive_slice<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::ExclusiveSlice<'a, T> {
        todo!("validate the run, X's guard, lend its range exclusive")
    }
    fn shared_str<'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::SharedStr<'a> {
        todo!("validate the text, X's guard, lend its range shared")
    }
    fn take_tuple(&self, v: Self::Value, at: Settled<'_>) -> Self::Tuple {
        todo!("X's take_tuple")
    }
    fn take_object(&self, v: Self::Value, at: Settled<'_>) -> Self::Object {
        todo!("X's take_object")
    }
    fn take_variant(&self, v: Self::Value, at: Settled<'_>) -> Self::Variant {
        todo!("X's take_variant")
    }
    fn make_tuple(&self, parts: impl ExactSizeIterator<Item = Self::Value>, at: Settled<'_>) -> Self::Value {
        todo!("X's make_tuple")
    }
    fn make_object(&self, fields: impl ExactSizeIterator<Item = Self::Value>, at: Settled<'_>) -> Self::Value {
        todo!("X's make_object")
    }
    fn make_variant(&self, tag: u32, payload: Option<Self::Value>, at: Settled<'_>) -> Self::Value {
        todo!("X's make_variant")
    }
    fn lent_part<'a>(&'a self, v: Self::Value, i: u32, at: Settled<'a>) -> Self::Value {
        todo!("X's lent_part")
    }
    fn lent_tag(&self, v: Self::Value, at: Settled<'_>) -> u32 {
        todo!("X's lent_tag")
    }
    fn reference<'a, P: ?Sized>(&'a self, target: RefTarget<'a, P>, at: Settled<'a>) -> Self::Value {
        todo!("reborrow the part's range from the lend that holds it, then X's reference")
    }
    fn interner(&self) -> &Interner {
        self.inner.interner()
    }
    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send {
        self.inner.sleep(d)
    }
}

impl<X: Instances + Validate> Instances for Adapter<X> {
    fn instance<R, F, K>(&self, entry: InstanceEntry<R, F, K>) -> Self::Value
    where
        R: Runtime<Value = Self::Value>,
        F: Family,
        K: TaskKind,
    {
        self.inner.instance(entry)
    }
    fn entry<R, F, K>(&self, v: Self::Value, at: Settled<'_>) -> &InstanceEntry<R, F, K>
    where
        R: Runtime<Value = Self::Value>,
        F: Family,
        K: TaskKind,
    {
        todo!("validate v as an impl value, then X's entry")
    }
    fn task_of(&self, v: Self::Value, at: Settled<'_>) -> Task {
        todo!("validate v as an impl value, then X's task_of")
    }
}

impl<X: CallsBack + Validate> CallsBack for Adapter<X> {
    fn call_now<R, A, U>(&self, closure: Self::Value, args: A::Lent<'_>, at: Settled<'_>) -> U
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: Crosses,
    {
        todo!("validate the closure, then X's call_now")
    }
    fn call_later<R, A, U>(&self, closure: Self::Value, args: A::Moved, at: Settled<'_>) -> impl Future<Output = U> + Send
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: Crosses,
    {
        async { todo!("validate the closure, then X's call_later") }
    }
}
