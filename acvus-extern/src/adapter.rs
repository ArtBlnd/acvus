//! The checking adapter (runtime contract §3, K15). Only a trusting runtime
//! implements the contract; for every such runtime X, `Adapter<X>` implements
//! it again with X's own values and X's own operations, and checks what X
//! trusts. The host chooses whether to run X or `Adapter<X>`, and an extern
//! cannot tell which.
//!
//! The detection is partial and grows. Lent storage is tracked by address in a
//! model after Tree Borrows (`Borrows`). No tracked model is
//! known to cover every breach, so a breach the tracking does not cover passes
//! as it would under X, and a use the tracking cannot yet tell apart from a
//! breach may be refused.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;
use std::ops::{Deref, DerefMut, Range};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::entry::Settled;
use crate::callback::{CallArgs, CallsBack, LoanForm, MovedArgs};
use crate::impls::task::TaskKind;
use crate::boundary::instance::InstanceEntry;
use crate::impls::{Family, Instances, Task};
use crate::boundary::entry::SiteShape;
use crate::done::Done;
use crate::register::{AwaitedFn, Declared, ExternFn, Refused, Registers};
use crate::boundary::uniform::Primitive;
use crate::runtime::{
    Crosses, Mutability, Opaque, OpaqueKind, OptionKind, PrimitiveKind, RefKind, RefTarget, ResultKind, Runtime,
    StringKind, TupleKind,
};

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

/// The adapter over the trusting runtime `X`.
pub struct Adapter<X: Runtime> {
    inner: X,
    borrows: Mutex<Borrows>,
    /// The functions registered at the adapter, their dispatch monomorphised
    /// at it.
    registered: HashMap<QualifiedRef, Registered<Adapter<X>>>,
}

/// A function registered at the adapter: returning, or awaited, whose call's
/// future the adapter boxes, checking being its purpose rather than speed.
enum Registered<R: Runtime> {
    Returning(Box<dyn ExternFn<R>>),
    Awaited(Box<dyn Boxed<R>>),
}

/// An awaited function whose call's future is boxed.
trait Boxed<R: Runtime>: Send + Sync {
    fn call<'c>(
        &'c self,
        runtime: &'c R,
        args: &'c [R::Value],
        site: &'c SiteShape<R>,
    ) -> Pin<Box<dyn Future<Output = Done<R>> + Send + 'c>>;
}

impl<R: Runtime, F: AwaitedFn<R>> Boxed<R> for F {
    fn call<'c>(
        &'c self,
        runtime: &'c R,
        args: &'c [R::Value],
        site: &'c SiteShape<R>,
    ) -> Pin<Box<dyn Future<Output = Done<R>> + Send + 'c>> {
        Box::pin(AwaitedFn::call(self, runtime, args, site))
    }
}

impl<X: Runtime> Adapter<X> {
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

impl<X: Runtime> Runtime for Adapter<X> {
    type Value = X::Value;

    type Opaque = Adapted<X::Opaque>;
    type Primitive = Adapted<X::Primitive>;
    type String = Adapted<X::String>;
    type Option = Adapted<X::Option>;
    type Result = Adapted<X::Result>;
    type Tuple = Adapted<X::Tuple>;
    type Ref = Adapted<X::Ref>;

    fn interner(&self) -> &Interner {
        self.inner.interner()
    }

    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send {
        self.inner.sleep(d)
    }
}

/// X's kind `K` at the adapter: X's own operations over X's values, a lend of
/// storage tracked while its guard lives, and a reference value checked
/// against the lends it is made from.
pub struct Adapted<K>(PhantomData<fn() -> K>);

impl<X: Runtime, K: OpaqueKind<X>> OpaqueKind<Adapter<X>> for Adapted<K> {
    type Shared<'a, T: Opaque + 'a> = Tracked<'a, K::Shared<'a, T>>;
    type Exclusive<'a, T: Opaque + 'a> = Tracked<'a, K::Exclusive<'a, T>>;

    fn give<T: Opaque>(rt: &Adapter<X>, value: T, at: Settled<'_>) -> X::Value {
        K::give(&rt.inner, value, at)
    }

    fn take<T: Opaque>(rt: &Adapter<X>, v: X::Value, at: Settled<'_>) -> T {
        K::take(&rt.inner, v, at)
    }

    fn shared<'a, T: Opaque + 'a>(rt: &'a Adapter<X>, place: &'a X::Value, at: Settled<'a>) -> Self::Shared<'a, T> {
        rt.track(K::shared(&rt.inner, place, at), Mutability::Shared)
    }

    fn exclusive<'a, T: Opaque + 'a>(
        rt: &'a Adapter<X>,
        place: &'a mut X::Value,
        at: Settled<'a>,
    ) -> Self::Exclusive<'a, T> {
        rt.track(K::exclusive(&rt.inner, place, at), Mutability::Exclusive)
    }

    fn shared_ref<'a, T: Opaque + 'a>(rt: &'a Adapter<X>, reference: X::Value, at: Settled<'a>) -> Self::Shared<'a, T> {
        rt.track(K::shared_ref(&rt.inner, reference, at), Mutability::Shared)
    }

    fn exclusive_ref<'a, T: Opaque + 'a>(
        rt: &'a Adapter<X>,
        reference: X::Value,
        at: Settled<'a>,
    ) -> Self::Exclusive<'a, T> {
        rt.track(K::exclusive_ref(&rt.inner, reference, at), Mutability::Exclusive)
    }
}

impl<X: Runtime, K: PrimitiveKind<X>> PrimitiveKind<Adapter<X>> for Adapted<K> {
    type Shared<'a, P: Primitive> = Tracked<'a, K::Shared<'a, P>>;
    type Exclusive<'a, P: Primitive> = Tracked<'a, K::Exclusive<'a, P>>;

    fn give<P: Primitive>(rt: &Adapter<X>, value: P, at: Settled<'_>) -> X::Value {
        K::give(&rt.inner, value, at)
    }

    fn take<P: Primitive>(rt: &Adapter<X>, v: X::Value, at: Settled<'_>) -> P {
        K::take(&rt.inner, v, at)
    }

    fn shared<'a, P: Primitive>(rt: &'a Adapter<X>, place: &'a X::Value, at: Settled<'a>) -> Self::Shared<'a, P> {
        rt.track(K::shared(&rt.inner, place, at), Mutability::Shared)
    }

    fn exclusive<'a, P: Primitive>(rt: &'a Adapter<X>, place: &'a mut X::Value, at: Settled<'a>) -> Self::Exclusive<'a, P> {
        rt.track(K::exclusive(&rt.inner, place, at), Mutability::Exclusive)
    }

    fn shared_ref<'a, P: Primitive>(rt: &'a Adapter<X>, reference: X::Value, at: Settled<'a>) -> Self::Shared<'a, P> {
        rt.track(K::shared_ref(&rt.inner, reference, at), Mutability::Shared)
    }

    fn exclusive_ref<'a, P: Primitive>(
        rt: &'a Adapter<X>,
        reference: X::Value,
        at: Settled<'a>,
    ) -> Self::Exclusive<'a, P> {
        rt.track(K::exclusive_ref(&rt.inner, reference, at), Mutability::Exclusive)
    }
}

impl<X: Runtime, K: StringKind<X>> StringKind<Adapter<X>> for Adapted<K> {
    type Shared<'a> = Tracked<'a, K::Shared<'a>>;
    type Exclusive<'a> = Tracked<'a, K::Exclusive<'a>>;

    fn give(rt: &Adapter<X>, value: String, at: Settled<'_>) -> X::Value {
        K::give(&rt.inner, value, at)
    }

    fn take(rt: &Adapter<X>, v: X::Value, at: Settled<'_>) -> String {
        K::take(&rt.inner, v, at)
    }

    fn shared<'a>(rt: &'a Adapter<X>, place: &'a X::Value, at: Settled<'a>) -> Self::Shared<'a> {
        rt.track(K::shared(&rt.inner, place, at), Mutability::Shared)
    }

    fn exclusive<'a>(rt: &'a Adapter<X>, place: &'a mut X::Value, at: Settled<'a>) -> Self::Exclusive<'a> {
        rt.track(K::exclusive(&rt.inner, place, at), Mutability::Exclusive)
    }

    fn shared_ref<'a>(rt: &'a Adapter<X>, reference: X::Value, at: Settled<'a>) -> Self::Shared<'a> {
        rt.track(K::shared_ref(&rt.inner, reference, at), Mutability::Shared)
    }

    fn exclusive_ref<'a>(rt: &'a Adapter<X>, reference: X::Value, at: Settled<'a>) -> Self::Exclusive<'a> {
        rt.track(K::exclusive_ref(&rt.inner, reference, at), Mutability::Exclusive)
    }
}

impl<X: Runtime, K: OptionKind<X>> OptionKind<Adapter<X>> for Adapted<K> {
    fn give(rt: &Adapter<X>, value: Option<X::Value>, at: Settled<'_>) -> X::Value {
        K::give(&rt.inner, value, at)
    }

    fn take(rt: &Adapter<X>, v: X::Value, at: Settled<'_>) -> Option<X::Value> {
        K::take(&rt.inner, v, at)
    }

    fn shared<'a>(rt: &'a Adapter<X>, place: &'a X::Value, at: Settled<'a>) -> Option<&'a X::Value> {
        K::shared(&rt.inner, place, at)
    }

    fn exclusive<'a>(rt: &'a Adapter<X>, place: &'a mut X::Value, at: Settled<'a>) -> Option<&'a mut X::Value> {
        K::exclusive(&rt.inner, place, at)
    }

    fn shared_ref<'a>(rt: &'a Adapter<X>, reference: X::Value, at: Settled<'a>) -> Option<&'a X::Value> {
        K::shared_ref(&rt.inner, reference, at)
    }

    fn exclusive_ref<'a>(rt: &'a Adapter<X>, reference: X::Value, at: Settled<'a>) -> Option<&'a mut X::Value> {
        K::exclusive_ref(&rt.inner, reference, at)
    }
}

impl<X: Runtime, K: ResultKind<X>> ResultKind<Adapter<X>> for Adapted<K> {
    fn give(rt: &Adapter<X>, value: Result<X::Value, X::Value>, at: Settled<'_>) -> X::Value {
        K::give(&rt.inner, value, at)
    }

    fn take(rt: &Adapter<X>, v: X::Value, at: Settled<'_>) -> Result<X::Value, X::Value> {
        K::take(&rt.inner, v, at)
    }

    fn shared<'a>(rt: &'a Adapter<X>, place: &'a X::Value, at: Settled<'a>) -> Result<&'a X::Value, &'a X::Value> {
        K::shared(&rt.inner, place, at)
    }

    fn exclusive<'a>(
        rt: &'a Adapter<X>,
        place: &'a mut X::Value,
        at: Settled<'a>,
    ) -> Result<&'a mut X::Value, &'a mut X::Value> {
        K::exclusive(&rt.inner, place, at)
    }

    fn shared_ref<'a>(
        rt: &'a Adapter<X>,
        reference: X::Value,
        at: Settled<'a>,
    ) -> Result<&'a X::Value, &'a X::Value> {
        K::shared_ref(&rt.inner, reference, at)
    }

    fn exclusive_ref<'a>(
        rt: &'a Adapter<X>,
        reference: X::Value,
        at: Settled<'a>,
    ) -> Result<&'a mut X::Value, &'a mut X::Value> {
        K::exclusive_ref(&rt.inner, reference, at)
    }
}

impl<X: Runtime, K: TupleKind<X>> TupleKind<Adapter<X>> for Adapted<K> {
    fn give<const N: usize>(rt: &Adapter<X>, parts: [X::Value; N], at: Settled<'_>) -> X::Value {
        K::give(&rt.inner, parts, at)
    }

    fn take<const N: usize>(rt: &Adapter<X>, v: X::Value, at: Settled<'_>) -> [X::Value; N] {
        K::take(&rt.inner, v, at)
    }

    fn shared<'a, const N: usize>(rt: &'a Adapter<X>, place: &'a X::Value, at: Settled<'a>) -> &'a [X::Value; N] {
        K::shared(&rt.inner, place, at)
    }

    fn exclusive<'a, const N: usize>(
        rt: &'a Adapter<X>,
        place: &'a mut X::Value,
        at: Settled<'a>,
    ) -> &'a mut [X::Value; N] {
        K::exclusive(&rt.inner, place, at)
    }

    fn shared_ref<'a, const N: usize>(rt: &'a Adapter<X>, reference: X::Value, at: Settled<'a>) -> &'a [X::Value; N] {
        K::shared_ref(&rt.inner, reference, at)
    }

    fn exclusive_ref<'a, const N: usize>(
        rt: &'a Adapter<X>,
        reference: X::Value,
        at: Settled<'a>,
    ) -> &'a mut [X::Value; N] {
        K::exclusive_ref(&rt.inner, reference, at)
    }
}

impl<X: Runtime, K: RefKind<X>> RefKind<Adapter<X>> for Adapted<K> {
    fn reference<'a, P: ?Sized>(rt: &'a Adapter<X>, target: RefTarget<'a, P>, at: Settled<'a>) -> X::Value {
        let (range, access) = match &target {
            RefTarget::Shared(part) => (range_of(*part), Mutability::Shared),
            RefTarget::Exclusive(part) => (range_of(&**part), Mutability::Exclusive),
        };
        let reborrowed = lock(&rt.borrows).reborrow(range, access);
        if let Err(breach) = reborrowed {
            breach.detected();
        }
        K::reference(&rt.inner, target, at)
    }

    fn target<'l, P: ?Sized>(rt: &'l Adapter<X>, v: X::Value, at: Settled<'l>) -> &'l P {
        K::target(&rt.inner, v, at)
    }

    fn target_mut<'l, P: ?Sized>(rt: &'l Adapter<X>, v: X::Value, at: Settled<'l>) -> &'l mut P {
        K::target_mut(&rt.inner, v, at)
    }
}

impl<X: Instances> Instances for Adapter<X> {
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
        self.inner.entry(v, at)
    }

    fn task_of(&self, v: Self::Value, at: Settled<'_>) -> Task {
        self.inner.task_of(v, at)
    }
}

impl<X: CallsBack> CallsBack for Adapter<X> {
    fn call_now<'l, R, A, U>(&self, rt: &'l R, closure: Self::Value, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: LoanForm<R>,
        A: 'l,
        U: 'l,
    {
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
        A: MovedArgs<R>,
        U: Crosses<R>,
    {
        self.inner.call_later::<R, A, U>(rt, closure, args, at)
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

impl<X: Runtime> Adapter<X> {
    /// Registers `f` at `path`, refused where a function is registered there.
    fn enter(&mut self, path: QualifiedRef, f: Registered<Adapter<X>>) -> Result<(), Refused> {
        if self.registered.contains_key(&path) {
            return Err(Refused::PathTaken(path));
        }
        self.registered.insert(path, f);
        Ok(())
    }
}

impl<X: Runtime> Registers for Adapter<X> {
    fn register<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self> + Declared<Self>,
    {
        let path = Declared::path(&f, self.inner.interner());
        self.enter(path, Registered::Returning(Box::new(f)))
    }

    fn register_awaited<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: AwaitedFn<Self> + Declared<Self>,
    {
        let path = Declared::path(&f, self.inner.interner());
        self.enter(path, Registered::Awaited(Box::new(f)))
    }

    fn registered(&self, path: QualifiedRef) -> Option<&dyn ExternFn<Self>> {
        match self.registered.get(&path)? {
            Registered::Returning(f) => Some(&**f),
            Registered::Awaited(_) => None,
        }
    }

    fn call_awaited<'c>(
        &'c self,
        path: QualifiedRef,
        args: &'c [Self::Value],
        site: &'c SiteShape<Self>,
    ) -> Option<impl Future<Output = Done<Self>> + Send + 'c> {
        match self.registered.get(&path)? {
            Registered::Awaited(f) => Some(f.call(self, args, site)),
            Registered::Returning(_) => None,
        }
    }
}
