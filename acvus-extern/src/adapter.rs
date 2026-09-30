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
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::callback::{CallArgs, CallsBack, RustFn, RustFns};
use crate::impls::task::TaskKind;
use crate::boundary::instance::InstanceEntry;
use crate::impls::{Family, Instances, Task};
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

/// The tracker's lock. Nothing panics while holding it: every breach is
/// returned out of `Borrows` and raised after the lock is released.
fn lock(borrows: &Mutex<Borrows>) -> MutexGuard<'_, Borrows> {
    borrows.lock().expect("the borrow tracker's lock is poisoned")
}

/// Lent storage, tracked by address after Tree Borrows: each lend is a node
/// over its address range, a part borrowed from a lend is its child, and an
/// access or a lend is checked against the nodes over the range it touches.
/// A range no node covers is untracked, and nothing is checked there.
struct Borrows {
    nodes: Vec<Node>,
    next: u64,
}

/// One node of `Borrows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tag(u64);

struct Node {
    tag: Tag,
    range: Range<usize>,
    access: Mutability,
    parent: Option<Tag>,
    depth: u32,
}

impl Node {
    fn overlaps(&self, range: &Range<usize>) -> bool {
        self.range.start < range.end && range.start < self.range.end
    }

    fn holds(&self, range: &Range<usize>) -> bool {
        self.range.start <= range.start && range.end <= self.range.end
    }
}

/// Two lends of one address conflict unless both are shared.
fn conflict(a: Mutability, b: Mutability) -> bool {
    a == Mutability::Exclusive || b == Mutability::Exclusive
}

impl Borrows {
    fn new() -> Self {
        Borrows { nodes: Vec::new(), next: 0 }
    }

    fn fresh(&mut self) -> Tag {
        let tag = Tag(self.next);
        self.next += 1;
        tag
    }

    /// A lend of `range` for a guard.
    fn lend(&mut self, range: Range<usize>, access: Mutability) -> Result<Tag, Breach> {
        if let Some(held) = self.nodes.iter().find(|n| n.overlaps(&range) && conflict(n.access, access)) {
            return Err(Breach {
                what: format!(
                    "a {access:?} lend of {range:#x?} overlaps a live {:?} lend of {:#x?}",
                    held.access, held.range
                ),
            });
        }
        let tag = self.fresh();
        self.nodes.push(Node { tag, range, access, parent: None, depth: 0 });
        Ok(tag)
    }

    /// A part borrowed from a lend, for a reference value.
    ///
    /// Only the parent's access is checked. A part lies inside its parent, so
    /// a live lend of another guard that overlaps the part overlaps the
    /// parent too, and `lend` already refused whichever of the two came
    /// second. Two parts of one lend are Rust borrows of one guard, which
    /// Rust's borrow checker already orders.
    fn reborrow(&mut self, range: Range<usize>, access: Mutability) -> Result<Option<Tag>, Breach> {
        let Some(parent) = self.nodes.iter().filter(|n| n.holds(&range)).max_by_key(|n| n.depth) else {
            return Ok(None);
        };
        if parent.access == Mutability::Shared && access == Mutability::Exclusive {
            return Err(Breach {
                what: format!(
                    "a writable reference to {range:#x?} is made from a shared lend of {:#x?}",
                    parent.range
                ),
            });
        }
        let (parent, depth) = (parent.tag, parent.depth + 1);
        let tag = self.fresh();
        self.nodes.push(Node { tag, range, access, parent: Some(parent), depth });
        Ok(Some(tag))
    }

    /// A guard dropped.
    fn end(&mut self, tag: Tag) {
        let mut ended = vec![tag];
        let mut next = 0;
        while let Some(&tag) = ended.get(next) {
            ended.extend(self.nodes.iter().filter(|n| n.parent == Some(tag)).map(|n| n.tag));
            next += 1;
        }
        self.nodes.retain(|n| !ended.contains(&n.tag));
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
        lock(self.borrows).end(self.tag);
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

#[cfg(test)]
mod tests {
    use super::Borrows;
    use crate::runtime::Mutability::{Exclusive, Shared};

    #[test]
    fn shared_lends_of_one_range_coexist() {
        let mut borrows = Borrows::new();

        borrows.lend(0..8, Shared).unwrap();

        assert!(borrows.lend(4..12, Shared).is_ok());
    }

    #[test]
    fn an_exclusive_lend_over_a_live_shared_lend_is_a_breach() {
        let mut borrows = Borrows::new();
        borrows.lend(0..8, Shared).unwrap();

        assert!(borrows.lend(4..12, Exclusive).is_err());
        assert!(borrows.lend(8..16, Exclusive).is_ok());
    }

    #[test]
    fn a_shared_lend_over_a_live_exclusive_lend_is_a_breach() {
        let mut borrows = Borrows::new();
        borrows.lend(0..8, Exclusive).unwrap();

        assert!(borrows.lend(7..8, Shared).is_err());
        assert!(borrows.lend(8..9, Shared).is_ok());
    }

    #[test]
    fn a_part_inside_a_lend_is_its_child() {
        let mut borrows = Borrows::new();
        let whole = borrows.lend(0..24, Exclusive).unwrap();

        let part = borrows.reborrow(8..16, Exclusive).unwrap().unwrap();

        let node = borrows.nodes.iter().find(|n| n.tag == part).unwrap();
        assert_eq!(node.parent, Some(whole));
    }

    #[test]
    fn a_writable_part_of_a_shared_lend_is_a_breach() {
        let mut borrows = Borrows::new();
        borrows.lend(0..24, Shared).unwrap();

        assert!(borrows.reborrow(8..16, Exclusive).is_err());
        assert!(borrows.reborrow(8..16, Shared).unwrap().is_some());
    }

    #[test]
    fn a_lend_that_ended_no_longer_conflicts() {
        let mut borrows = Borrows::new();
        let exclusive = borrows.lend(0..8, Exclusive).unwrap();
        assert!(borrows.lend(0..8, Shared).is_err());

        borrows.end(exclusive);

        assert!(borrows.lend(0..8, Shared).is_ok());
    }

    #[test]
    fn a_lend_ends_with_its_parts() {
        let mut borrows = Borrows::new();
        let whole = borrows.lend(0..24, Exclusive).unwrap();
        borrows.reborrow(8..16, Exclusive).unwrap().unwrap();
        borrows.reborrow(8..12, Shared).unwrap().unwrap();

        borrows.end(whole);

        assert!(borrows.nodes.is_empty());
        assert!(borrows.lend(8..12, Exclusive).is_ok());
    }

    #[test]
    fn a_range_no_lend_covers_is_untracked() {
        let mut borrows = Borrows::new();
        borrows.lend(0..24, Shared).unwrap();

        assert!(borrows.reborrow(100..108, Exclusive).unwrap().is_none());
        assert!(borrows.reborrow(16..32, Exclusive).unwrap().is_none());
        assert_eq!(borrows.nodes.len(), 1);
    }
}

impl<X: RustFns + Validate> RustFns for Adapter<X> {
    fn rust_fn<R>(&self, f: RustFn<R>) -> Self::Value
    where
        R: Runtime<Value = Self::Value>,
    {
        self.inner.rust_fn(f)
    }
}
