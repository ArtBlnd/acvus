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

use std::collections::HashMap;
use std::ops::{Deref, DerefMut, Range};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::entry::Settled;
use crate::callback::{CallArgs, CallsBack, LoanForm, RustCallee, RustFns};
use crate::impls::task::TaskKind;
use crate::boundary::instance::InstanceEntry;
use crate::impls::{Family, Instances, Task};
use crate::register::{ExternFn, Refused, Registers};
use crate::runtime::{Crosses, Mutability, RefTarget, Runtime};

/// What a runtime answers, from its own representation, about a value the
/// crossing is about to read at a type. It is not part of the contract: only
/// the adapter asks it, and a runtime run without the adapter is never asked.
pub trait Validate: Runtime {
    /// Whether `v` is a value of `T`.
    fn validate<T: Crosses>(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
    /// Whether `v` is an impl value of the generic function `F` at task `K`.
    fn validate_instance<F: Family, K: TaskKind>(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
    /// Whether `v` is an impl value at all, of whatever generic function and
    /// task.
    fn validate_impl(&self, v: &Self::Value, at: Settled<'_>) -> Validity;
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

impl Breach {
    fn detected(self) -> ! {
        panic!("the adapter detected a breach: {}", self.what)
    }
}

impl Validity {
    fn pass_or_panic(self) {
        match self {
            Validity::Valid | Validity::Unknown => {}
            Validity::Breach(breach) => breach.detected(),
        }
    }
}

/// The adapter over the trusting runtime `X`.
pub struct Adapter<X: Runtime + Validate> {
    inner: X,
    borrows: Mutex<Borrows>,
    /// The functions registered at the adapter, their dispatch monomorphised
    /// at it.
    registered: HashMap<QualifiedRef, Box<dyn ExternFn<Adapter<X>>>>,
}

impl<X: Runtime + Validate> Adapter<X> {
    pub fn new(inner: X) -> Self {
        Adapter { inner, borrows: Mutex::new(Borrows::new()), registered: HashMap::new() }
    }

    pub fn inner(&self) -> &X {
        &self.inner
    }

    /// Tracks `guard` as a lend of the storage its target occupies.
    fn track<'a, G>(&'a self, guard: G, access: Mutability) -> Tracked<'a, G>
    where
        G: Deref,
    {
        let lent = lock(&self.borrows).lend(range_of(&*guard), access);
        match lent {
            Ok(tag) => Tracked { inner: guard, tag, borrows: &self.borrows },
            Err(breach) => breach.detected(),
        }
    }
}

/// The tracker's lock. Nothing panics while holding it: every breach is
/// returned out of `Borrows` and raised after the lock is released.
fn lock(borrows: &Mutex<Borrows>) -> MutexGuard<'_, Borrows> {
    borrows.lock().expect("the borrow tracker's lock is poisoned")
}

/// The addresses a borrow's target occupies.
fn range_of<P: ?Sized>(target: &P) -> Range<usize> {
    let start = std::ptr::from_ref(target).addr();
    start..start + size_of_val(target)
}

/// Lent storage, tracked by address after Tree Borrows: each lend is a node
/// over its address range, a part borrowed from a lend is its child, and an
/// access or a lend is checked against the nodes over the range it touches.
/// A lend made through a reference value is that reference's child, as a
/// pointer derived from a tag is the tag's child; a reference value is known
/// only by the range it names, so a lend inside a live reference's range is
/// taken to be made through it. A range no node covers is untracked, and
/// nothing is checked there.
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
    made: Made,
    parent: Option<Tag>,
    depth: u32,
}

/// What made a node: a guard's lend, or a reference value borrowed from one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Made {
    Lend,
    Reference,
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

    /// A lend of `range` for a guard: a child of the reference it is made
    /// through, if any, and checked against every other node over its range.
    fn lend(&mut self, range: Range<usize>, access: Mutability) -> Result<Tag, Breach> {
        let through = self
            .nodes
            .iter()
            .filter(|n| n.made == Made::Reference && n.holds(&range))
            .max_by_key(|n| n.depth);
        let parent = match through {
            Some(reference) if reference.access == Mutability::Shared && access == Mutability::Exclusive => {
                return Err(Breach {
                    what: format!(
                        "an Exclusive lend of {range:#x?} is made through a shared reference to {:#x?}",
                        reference.range
                    ),
                });
            }
            Some(reference) => Some((reference.tag, reference.depth)),
            None => None,
        };
        let lineage = self.lineage(parent.map(|(tag, _)| tag));
        let held = self
            .nodes
            .iter()
            .find(|n| !lineage.contains(&n.tag) && n.overlaps(&range) && conflict(n.access, access));
        if let Some(held) = held {
            return Err(Breach {
                what: format!(
                    "a {access:?} lend of {range:#x?} overlaps a live {:?} lend of {:#x?}",
                    held.access, held.range
                ),
            });
        }
        let tag = self.fresh();
        let (parent, depth) = match parent {
            Some((parent, depth)) => (Some(parent), depth + 1),
            None => (None, 0),
        };
        self.nodes.push(Node { tag, range, access, made: Made::Lend, parent, depth });
        Ok(tag)
    }

    /// `tag` and its ancestors.
    fn lineage(&self, tag: Option<Tag>) -> Vec<Tag> {
        let mut lineage = Vec::new();
        let mut next = tag;
        while let Some(tag) = next {
            lineage.push(tag);
            next = self.nodes.iter().find(|n| n.tag == tag).and_then(|n| n.parent);
        }
        lineage
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
        self.nodes.push(Node { tag, range, access, made: Made::Reference, parent: Some(parent), depth });
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
    type Shared<'a, T: Crosses + 'a> = Tracked<'a, X::Shared<'a, T>>;
    type Exclusive<'a, T: Crosses + 'a> = Tracked<'a, X::Exclusive<'a, T>>;

    fn take<T: Crosses>(&self, v: Self::Value, at: Settled<'_>) -> T {
        self.inner.validate::<T>(&v, at).pass_or_panic();
        self.inner.take(v, at)
    }

    fn give<T: Crosses>(&self, value: T, at: Settled<'_>) -> Self::Value {
        self.inner.give(value, at)
    }

    fn shared<'a, T: Crosses + 'a>(&'a self, v: &'a Self::Value, at: Settled<'a>) -> Self::Shared<'a, T> {
        self.inner.validate::<T>(v, at).pass_or_panic();
        self.track(self.inner.shared(v, at), Mutability::Shared)
    }

    fn exclusive<'a, T: Crosses + 'a>(&'a self, v: &'a mut Self::Value, at: Settled<'a>) -> Self::Exclusive<'a, T> {
        self.inner.validate::<T>(v, at).pass_or_panic();
        self.track(self.inner.exclusive(v, at), Mutability::Exclusive)
    }

    fn reference<'a, P: ?Sized>(&'a self, target: RefTarget<'a, P>, at: Settled<'a>) -> Self::Value {
        let (range, access) = match &target {
            RefTarget::Shared(part) => (range_of(*part), Mutability::Shared),
            RefTarget::Exclusive(part) => (range_of(&**part), Mutability::Exclusive),
        };
        let reborrowed = lock(&self.borrows).reborrow(range, access);
        if let Err(breach) = reborrowed {
            breach.detected();
        }
        self.inner.reference(target, at)
    }

    fn target<'l, P: ?Sized>(&'l self, v: Self::Value, at: Settled<'l>) -> &'l P {
        self.inner.target(v, at)
    }

    fn target_mut<'l, P: ?Sized>(&'l self, v: Self::Value, at: Settled<'l>) -> &'l mut P {
        self.inner.target_mut(v, at)
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
        self.inner.validate_instance::<F, K>(&v, at).pass_or_panic();
        self.inner.entry(v, at)
    }

    fn task_of(&self, v: Self::Value, at: Settled<'_>) -> Task {
        self.inner.validate_impl(&v, at).pass_or_panic();
        self.inner.task_of(v, at)
    }
}

impl<X: CallsBack + Validate> CallsBack for Adapter<X> {
    fn call_now<'l, R, A, U>(&self, rt: &'l R, closure: Self::Value, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: LoanForm,
    {
        self.inner.validate_closure(&closure, at).pass_or_panic();
        self.inner.call_now::<R, A, U>(rt, closure, args, at)
    }

    fn call_later<'a, R, A, U>(
        &self,
        rt: &'a R,
        closure: Self::Value,
        args: A::Moved,
        at: Settled<'a>,
    ) -> impl Future<Output = U> + Send
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: Crosses,
    {
        self.inner.validate_closure(&closure, at).pass_or_panic();
        self.inner.call_later::<R, A, U>(rt, closure, args, at)
    }
}

impl<X: RustFns + Validate> RustFns for Adapter<X> {
    fn rust_fn<R>(&self, f: RustCallee<R>) -> Self::Value
    where
        R: Runtime<Value = Self::Value>,
    {
        self.inner.rust_fn(f)
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
    fn a_lend_through_a_live_reference_is_its_child() {
        let mut borrows = Borrows::new();
        let whole = borrows.lend(0..24, Exclusive).unwrap();
        let reference = borrows.reborrow(0..24, Exclusive).unwrap().unwrap();

        let through = borrows.lend(0..24, Exclusive).unwrap();

        let node = borrows.nodes.iter().find(|n| n.tag == through).unwrap();
        assert_eq!(node.parent, Some(reference));
        borrows.end(whole);
        assert!(borrows.nodes.is_empty());
    }

    #[test]
    fn a_lend_beside_a_live_exclusive_lend_is_still_a_breach() {
        let mut borrows = Borrows::new();
        borrows.lend(0..24, Exclusive).unwrap();
        borrows.reborrow(8..16, Exclusive).unwrap().unwrap();

        assert!(borrows.lend(0..24, Shared).is_err());
        assert!(borrows.lend(4..12, Exclusive).is_err());
    }

    #[test]
    fn an_exclusive_lend_through_a_shared_reference_is_a_breach() {
        let mut borrows = Borrows::new();
        borrows.lend(0..24, Shared).unwrap();
        borrows.reborrow(8..16, Shared).unwrap().unwrap();

        assert!(borrows.lend(8..16, Exclusive).is_err());
        assert!(borrows.lend(8..16, Shared).is_ok());
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

impl<X: Runtime + Validate> Registers for Adapter<X> {
    fn register<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self>,
    {
        let path = f.path(self.inner.interner());
        if self.registered.contains_key(&path) {
            return Err(Refused::PathTaken(path));
        }
        self.registered.insert(path, Box::new(f));
        Ok(())
    }

    fn registered(&self, path: QualifiedRef) -> Option<&dyn ExternFn<Self>> {
        self.registered.get(&path).map(|f| &**f)
    }
}
