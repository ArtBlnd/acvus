//! The classes of one kind: a union-find over their ids, each root holding
//! its class, and every bound recorded between them that is not finished yet.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::mem;

use crate::primitive::ItemId;
use crate::primitive::ScopeId;
use crate::primitive::class::{
    Bounds, Call, Class, ClassKind, Constraints, Decided, Held, Origin, Part, Pending, Side, State, Unplaced,
};
use crate::primitive::domain::Domain;
use crate::primitive::id::Arena;
use crate::primitive::refusal::Refusal;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{PartKey, Shape};

/// Every class of the kind `K` made, by id.
///
/// An id stays valid forever, as a handle, but once classes are unified
/// several ids name one class, and only one of them, the root, holds it.
/// [`Classes::find`] gives the root as a [`Root`], and a class is read, and
/// two classes compared, through roots alone.
#[derive(Clone, Debug)]
pub struct Classes<K>
where
    K: ClassKind,
{
    /// Open to the rest of the store for [`Classes::verify`] alone.
    pub(super) slots: Arena<K::Id, Slot<K>>,
    /// Every bound recorded and not finished, from its recording until whoever
    /// judges or makes it finishes it. Open to the rest of the store for
    /// [`Classes::verify`] and for copying a template.
    pub(super) bounds: Bounds<K>,
    /// Every bound finished. A bound is recorded once: what connects parts by
    /// name comes to the same pair again, and recording it again does nothing.
    finished: Bounds<K>,
    /// The bounds to look at: each recorded and not looked at yet, and each that
    /// waits on a class that has changed since.
    woken: Bounds<K>,
}

// Written out, as a derive would ask an id to have a default, and none has.
impl<K> Default for Classes<K>
where
    K: ClassKind,
{
    fn default() -> Self {
        Self {
            slots: Arena::default(),
            bounds: Bounds::<K>::new(),
            finished: Bounds::<K>::new(),
            woken: Bounds::<K>::new(),
        }
    }
}

/// The class itself where the id is a root, or the id it was unified into.
#[derive(Clone, Debug)]
pub(super) enum Slot<K>
where
    K: ClassKind,
{
    Root(Class<K>),
    Link(K::Id),
}

/// The id that holds a class now, as [`Classes::find`] gives it.
///
/// It borrows the classes, so no union can happen while it is held, and it
/// never names a class that has since been unified into another:
///
/// ```compile_fail
/// use acvus_typeck_model::primitive::ItemId;
/// use acvus_typeck_model::primitive::class::Live;
/// use acvus_typeck_model::primitive::id::Arena;
/// use acvus_typeck_model::primitive::scope::Scopes;
/// use acvus_typeck_model::store::classes::Classes;
///
/// let scopes = Scopes::new();
/// let mut items: Arena<ItemId, ()> = Arena::default();
/// let mut classes = Classes::<Live>::default();
/// let a = classes.undecided(scopes.root());
/// let b = classes.undecided(scopes.root());
/// let root = classes.find(b);
/// classes.unify(&scopes, a, b, items.push(()));
/// classes.get(root);
/// ```
///
/// A root is found again after a union instead:
///
/// ```
/// use acvus_typeck_model::primitive::ItemId;
/// use acvus_typeck_model::primitive::class::Live;
/// use acvus_typeck_model::primitive::id::Arena;
/// use acvus_typeck_model::primitive::scope::Scopes;
/// use acvus_typeck_model::store::classes::Classes;
///
/// let scopes = Scopes::new();
/// let mut items: Arena<ItemId, ()> = Arena::default();
/// let mut classes = Classes::<Live>::default();
/// let a = classes.undecided(scopes.root());
/// let b = classes.undecided(scopes.root());
/// classes.unify(&scopes, a, b, items.push(()));
/// let root = classes.find(b);
/// assert_eq!(root, classes.find(a));
/// classes.get(root);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Root<'a, K>
where
    K: ClassKind,
{
    id: K::Id,
    classes: PhantomData<&'a Classes<K>>,
}

impl<K> Root<'_, K>
where
    K: ClassKind,
{
    /// The root's id, as a handle to be stored.
    pub fn id(self) -> K::Id {
        self.id
    }
}

/// The one way a class is decided by solving.
///
/// A class is decided once, and only when nothing can still be merged into it:
/// its head is the join of what enters it, and a merge brings more to enter. So a
/// class is held, and not decided, while a bound guards it ([`Classes::guard`]),
/// and while it is named as a part on an undecided class that it does not hold in
/// turn. What is held is found once, when this is made, and this borrows the
/// store for as long as it lives: nothing is recorded and nothing is merged
/// between that finding and the decisions, and the classes it decides are decided
/// together.
///
/// ```compile_fail
/// use acvus_typeck_model::primitive::ItemId;
/// use acvus_typeck_model::primitive::bound::Bound;
/// use acvus_typeck_model::primitive::class::Live;
/// use acvus_typeck_model::primitive::id::Arena;
/// use acvus_typeck_model::primitive::scope::Scopes;
/// use acvus_typeck_model::store::classes::Classes;
///
/// let scopes = Scopes::new();
/// let mut items: Arena<ItemId, ()> = Arena::default();
/// let mut classes = Classes::<Live>::default();
/// let a = classes.undecided(scopes.root());
/// let b = classes.undecided(scopes.root());
/// let deciding = classes.deciding();
/// classes.bound(Bound::Change { lower: a, upper: b }, items.push(()));
/// deciding.is_held(a);
/// ```
#[derive(Debug)]
pub struct Deciding<'a, K>
where
    K: ClassKind,
{
    classes: &'a mut Classes<K>,
    held: BTreeSet<K::Id>,
}

impl<K> Deciding<'_, K>
where
    K: ClassKind,
{
    /// The store, to read what a decision goes by.
    pub fn classes(&self) -> &Classes<K> {
        self.classes
    }

    /// Whether `class` may not be decided now.
    pub fn is_held(&self, class: K::Id) -> bool {
        self.held.contains(&self.classes.find(class).id)
    }

    /// Decides `class` as the head `shape` gives, by `at`. It gives none, and does
    /// nothing, where the class is held or is decided already.
    ///
    /// Only the constructor and members of `shape` are taken. The parts of the
    /// head are the class's own: the part already named on it, or a class made
    /// anew where none is. A part of what gave the head is another class, to be
    /// made one type with this one as any two parts are, and not put in its place.
    pub fn decide<P>(&mut self, scopes: &Scopes, class: K::Id, shape: &Shape<P>, by: BTreeSet<ItemId>, at: ItemId) -> Option<Unified<K>> {
        if self.is_held(class) || self.classes.is_decided(class) {
            return None;
        }
        let root = self.classes.find(class).id;
        let scope = self.classes.root(root).scope;
        let mut count = 0;
        let numbered = shape.map(|_| {
            count += 1;
            count - 1
        });
        let keys: BTreeMap<usize, PartKey> = numbered.places().into_iter().map(|(key, number)| (*number, key)).collect();
        let own = numbered.map(|number| match self.classes.root(root).state.part(&keys[number]) {
            Some(named) => named,
            None => self.classes.undecided(scope),
        });
        let head = self.classes.decided(scope, own, by);
        Some(self.classes.unify(scopes, class, head, at))
    }
}

/// What a union, or a record on a decided class, leaves to its caller.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Unified<K>
where
    K: ClassKind,
{
    pub refusals: Vec<Refusal>,
    pub pending: Vec<Pending<K>>,
    pub unplaced: Vec<Unplaced<K>>,
}

impl<K> Unified<K>
where
    K: ClassKind,
{
    /// Takes in what another union left.
    pub fn absorb(&mut self, other: Self) {
        self.refusals.extend(other.refusals);
        self.pending.extend(other.pending);
        self.unplaced.extend(other.unplaced);
    }
}

impl<K> Classes<K>
where
    K: ClassKind,
{
    pub fn undecided(&mut self, scope: ScopeId) -> K::Id {
        self.slots.push(Slot::Root(Class {
            scope,
            state: State::Undecided(Constraints::default()),
            held: Held::default(),
        }))
    }

    /// A class decided from the start, as the text writes it or solving decides
    /// it.
    pub fn decided(&mut self, scope: ScopeId, shape: Shape<K::Id>, by: BTreeSet<ItemId>) -> K::Id {
        self.decided_from(scope, shape, by, Origin::Free)
    }

    /// A class a declaration or the host writes.
    pub fn external(&mut self, scope: ScopeId, shape: Shape<K::Id>, by: BTreeSet<ItemId>) -> K::Id {
        self.decided_from(scope, shape, by, Origin::External)
    }

    fn decided_from(&mut self, scope: ScopeId, shape: Shape<K::Id>, by: BTreeSet<ItemId>, origin: Origin) -> K::Id {
        self.slots.push(Slot::Root(Class {
            scope,
            state: State::Decided(Decided { shape, by, origin }),
            held: Held::default(),
        }))
    }

    pub fn find(&self, class: K::Id) -> Root<'_, K> {
        let mut id = class;
        while let Slot::Link(next) = self.slots.get(id) {
            id = *next;
        }
        Root {
            id,
            classes: PhantomData,
        }
    }

    pub fn get(&self, root: Root<'_, K>) -> &Class<K> {
        self.root(root.id)
    }

    /// Every class, by its root, in the order of their ids.
    pub fn roots(&self) -> impl Iterator<Item = Root<'_, K>> {
        self.slots.iter().filter_map(|(id, slot)| {
            matches!(slot, Slot::Root(_)).then_some(Root {
                id,
                classes: PhantomData,
            })
        })
    }

    /// Whether the class `class` names is a part of itself, at any depth (O5).
    pub fn is_own_part(&self, class: K::Id) -> bool {
        self.occurs(self.find(class).id)
    }

    /// Whether the class `class` names is decided.
    pub fn is_decided(&self, class: K::Id) -> bool {
        self.root(self.find(class).id).state.is_decided()
    }

    /// Whether the class `class` names is one a declaration or the host writes.
    pub fn is_external(&self, class: K::Id) -> bool {
        matches!(
            &self.root(self.find(class).id).state,
            State::Decided(Decided {
                origin: Origin::External,
                ..
            })
        )
    }

    /// Records `bound`, by `at`, and says whether it was not recorded before. The
    /// store holds it until it is finished.
    pub fn bound(&mut self, bound: K::Bound, at: ItemId) -> bool {
        let recorded = (bound, at);
        if self.finished.contains(&recorded) || !self.bounds.insert(recorded.clone()) {
            return false;
        }
        self.woken.insert(recorded);
        true
    }

    /// The bounds to look at, which are then to look at no longer until a class
    /// one of them waits on is merged or decided.
    pub fn take_woken(&mut self) -> Bounds<K> {
        mem::take(&mut self.woken)
    }

    /// States anew which classes `bound`, recorded by `at`, waits on: it is to be
    /// looked at again when one of them is merged with another or decided.
    pub fn wait(&mut self, bound: &K::Bound, at: ItemId, on: &BTreeSet<K::Id>) {
        let recorded = (bound.clone(), at);
        for root in self.root_ids() {
            self.root_mut(root).held.waiters.remove(&recorded);
        }
        if !self.bounds.contains(&recorded) {
            return;
        }
        for class in on {
            let root = self.find(*class).id;
            self.root_mut(root).held.waiters.insert(recorded.clone());
        }
    }

    fn root_ids(&self) -> Vec<K::Id> {
        self.roots().map(|root| root.id).collect()
    }

    /// Wakes what waits on the class held at `root`, which has been merged or
    /// decided.
    fn wake(&mut self, root: K::Id) {
        let waiters = self.root(root).held.waiters.clone();
        self.woken.extend(waiters);
    }

    /// Gives each part of the class held at `root`, and each part below, the
    /// guards that reach it: those of the class above it, and those the class
    /// above it holds apart, but for a guard the part holds apart itself.
    fn push_down(&mut self, root: K::Id) {
        let mut stack = vec![root];
        while let Some(class) = stack.pop() {
            let class = self.find(class).id;
            let above = &self.root(class).held;
            let reaching: Bounds<K> = above.guards.union(&above.apart).cloned().collect();
            if reaching.is_empty() {
                continue;
            }
            for part in self.root(class).state.parts() {
                let part = self.find(part).id;
                let held = &mut self.root_mut(part).held;
                let new: Vec<(K::Bound, ItemId)> = reaching
                    .iter()
                    .filter(|guard| !held.apart.contains(guard) && !held.guards.contains(guard))
                    .cloned()
                    .collect();
                if !new.is_empty() {
                    held.guards.extend(new);
                    stack.push(part);
                }
            }
        }
    }

    /// Every bound recorded and not finished.
    pub fn bounds(&self) -> &Bounds<K> {
        &self.bounds
    }

    /// Finishes `bound`, recorded by `at`: it has been judged or made, and the
    /// store holds it no longer. It says whether the store held it.
    pub fn finish(&mut self, bound: &K::Bound, at: ItemId) -> bool {
        let recorded = (bound.clone(), at);
        let held = self.bounds.remove(&recorded);
        if held {
            self.lift(&recorded);
            for root in self.root_ids() {
                self.root_mut(root).held.waiters.remove(&recorded);
            }
            self.woken.remove(&recorded);
            self.finished.insert(recorded);
        }
        held
    }

    /// States anew what `bound`, recorded by `at`, guards: every class below one
    /// of `apart`, at any depth, but for the classes of `apart` themselves. Those
    /// are the classes it may still merge with another before it is finished, and
    /// what it guarded before and does not now is free of it.
    ///
    /// A class of `apart` is an end of the bound, or a class it reads and never
    /// merges. The guard goes down to each part such a class comes to have, as
    /// structure grows, without the bound being looked at again. The store reads
    /// no more of a bound here than anywhere: whoever knows what the bound means
    /// says what it holds apart, and says it again as that changes.
    pub fn guard(&mut self, bound: &K::Bound, at: ItemId, apart: &BTreeSet<K::Id>) {
        let recorded = (bound.clone(), at);
        self.lift(&recorded);
        if !self.bounds.contains(&recorded) {
            return;
        }
        let apart: BTreeSet<K::Id> = apart.iter().map(|class| self.find(*class).id).collect();
        for root in &apart {
            self.root_mut(*root).held.apart.insert(recorded.clone());
        }
        for root in apart {
            self.push_down(root);
        }
    }

    fn lift(&mut self, recorded: &(K::Bound, ItemId)) {
        for root in self.root_ids() {
            let held = &mut self.root_mut(root).held;
            held.guards.remove(recorded);
            held.apart.remove(recorded);
        }
    }

    /// Every class below `classes`: their parts, and the parts of those, at any
    /// depth, by root. A class of `classes` is among them only where it is a part
    /// of one of them.
    pub fn below<I>(&self, classes: I) -> BTreeSet<K::Id>
    where
        I: IntoIterator<Item = K::Id>,
    {
        let mut found = BTreeSet::new();
        let mut stack: Vec<K::Id> = classes
            .into_iter()
            .flat_map(|class| self.root(self.find(class).id).state.parts())
            .collect();
        while let Some(class) = stack.pop() {
            let root = self.find(class).id;
            if found.insert(root) {
                stack.extend(self.root(root).state.parts());
            }
        }
        found
    }

    /// The one way a class is decided by solving: see [`Deciding`].
    pub fn deciding(&mut self) -> Deciding<'_, K> {
        let mut held: BTreeSet<K::Id> = self
            .roots()
            .filter(|root| !self.get(*root).held.guards.is_empty())
            .map(|root| root.id)
            .collect();
        // A part is decided no earlier than the class it is named on. Where the
        // part holds that class in turn, the order asks nothing of the two, and
        // they are decided together.
        for owner in self.roots() {
            let State::Undecided(constraints) = &self.get(owner).state else {
                continue;
            };
            for part in constraints.parts.values() {
                let part = self.find(part.class).id;
                if !self.below([part]).contains(&owner.id) && part != owner.id {
                    held.insert(part);
                }
            }
        }
        Deciding { classes: self, held }
    }


    /// The class of the part `key` of `class`, named on `side` by `at`.
    ///
    /// An undecided class keeps the name, and makes the part a class where
    /// nothing has named it yet. A decided class gives the part in its place,
    /// and gives none where its shape has no place for it.
    pub fn part(&mut self, class: K::Id, key: PartKey, side: Side, at: ItemId) -> Option<K::Id> {
        let root = self.find(class).id;
        let owner = self.root(root);
        if owner.state.is_decided() {
            return owner.state.part(&key);
        }
        let part_class = match owner.state.part(&key) {
            Some(part_class) => part_class,
            None => {
                let scope = owner.scope;
                self.undecided(scope)
            }
        };
        let State::Undecided(constraints) = &mut self.root_mut(root).state else {
            unreachable!("a decided class returned above, and making a class decides none");
        };
        let named_anew = !constraints.parts.contains_key(&key);
        let part = constraints.parts.entry(key).or_insert_with(|| Part {
            class: part_class,
            lower: BTreeSet::new(),
            upper: BTreeSet::new(),
            carried: BTreeSet::new(),
        });
        let items = match side {
            Side::Lower => &mut part.lower,
            Side::Upper => &mut part.upper,
            Side::Carried => &mut part.carried,
        };
        items.insert(at);
        if named_anew {
            self.push_down(root);
        }
        Some(part_class)
    }

    /// Records a call on `class`. A decided class does not keep it, and
    /// gives it back to be checked against its shape.
    pub fn call(&mut self, class: K::Id, call: Call<K>, at: ItemId) -> Option<Pending<K>> {
        let root = self.find(class).id;
        match &mut self.root_mut(root).state {
            State::Undecided(constraints) => {
                constraints.calls.insert((call, at));
                None
            }
            State::Decided(_) => Some(Pending {
                class: root,
                calls: BTreeSet::from([(call, at)]),
                domains: BTreeSet::new(),
            }),
        }
    }

    /// Records a domain on `class`. A decided class does not keep it, and
    /// gives it back to be checked against its shape.
    pub fn domain(&mut self, class: K::Id, domain: Domain, at: ItemId) -> Option<Pending<K>> {
        let root = self.find(class).id;
        match &mut self.root_mut(root).state {
            State::Undecided(constraints) => {
                constraints.domains.insert((domain, at));
                None
            }
            State::Decided(_) => Some(Pending {
                class: root,
                calls: BTreeSet::new(),
                domains: BTreeSet::from([(domain, at)]),
            }),
        }
    }

    /// Makes `a` and `b` one class, by `at`, and with them each pair of their
    /// parts of one name, at any depth.
    ///
    /// Of two ids the smaller is kept as the root, and the two classes are
    /// merged there ([`Class::merge`]). Deciding a class is unifying it with a
    /// decided class.
    ///
    /// A union may make a class a part of itself. That is a fact of structure,
    /// asked of [`Classes::is_own_part`], and refusing it is the caller's: which
    /// union closes a cycle depends on the order unions are made in.
    pub fn unify(&mut self, scopes: &Scopes, a: K::Id, b: K::Id, at: ItemId) -> Unified<K> {
        let mut unified = Unified::default();
        let mut pairs = vec![(a, b)];
        while let Some((a, b)) = pairs.pop() {
            let (a, b) = (self.find(a).id, self.find(b).id);
            if a == b {
                continue;
            }
            let (kept, gone) = (a.min(b), a.max(b));
            let Slot::Root(gone_class) = mem::replace(self.slots.get_mut(gone), Slot::Link(kept)) else {
                unreachable!("find gives a root");
            };
            // The kept slot holds an empty class only while the two merge,
            // which reads nothing of the store.
            let scope = self.root(kept).scope;
            let placeholder = Slot::Root(Class {
                scope,
                state: State::Undecided(Constraints::default()),
                held: Held::default(),
            });
            let Slot::Root(kept_class) = mem::replace(self.slots.get_mut(kept), placeholder) else {
                unreachable!("find gives a root");
            };
            let merge = kept_class.merge(gone_class, kept, scopes, at);
            *self.slots.get_mut(kept) = Slot::Root(merge.class);
            self.push_down(kept);
            self.wake(kept);
            pairs.extend(merge.pairs);
            unified.refusals.extend(merge.refusals);
            unified.pending.extend(merge.pending);
            unified.unplaced.extend(merge.unplaced);
        }
        unified
    }

    fn root(&self, id: K::Id) -> &Class<K> {
        let Slot::Root(class) = self.slots.get(id) else {
            unreachable!("{id:?} is not a root");
        };
        class
    }

    fn root_mut(&mut self, id: K::Id) -> &mut Class<K> {
        let Slot::Root(class) = self.slots.get_mut(id) else {
            unreachable!("{id:?} is not a root");
        };
        class
    }

    /// Whether the class held at `root` is a part of itself, at any depth.
    fn occurs(&self, root: K::Id) -> bool {
        let mut seen = BTreeSet::new();
        let mut stack = self.root(root).state.parts();
        while let Some(class) = stack.pop() {
            let class = self.find(class).id;
            if class == root {
                return true;
            }
            if seen.insert(class) {
                stack.extend(self.root(class).state.parts());
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::refusal::Cause;
    use crate::primitive::ClassId;
    use crate::primitive::Name;
    use crate::primitive::bound::Bound;
    use crate::primitive::class::Live;
    use crate::primitive::ty::{Mutability, Scalar};

    struct World {
        classes: Classes<Live>,
        scopes: Scopes,
        items: Arena<ItemId, &'static str>,
    }

    impl World {
        fn new() -> Self {
            Self {
                classes: Classes::default(),
                scopes: Scopes::new(),
                items: Arena::default(),
            }
        }

        fn item(&mut self, at: &'static str) -> ItemId {
            self.items.push(at)
        }

        fn undecided(&mut self) -> ClassId {
            self.classes.undecided(self.scopes.root())
        }

        fn decided(&mut self, shape: Shape<ClassId>) -> (ClassId, ItemId) {
            let by = self.item("written");
            let class = self.classes.decided(self.scopes.root(), shape, BTreeSet::from([by]));
            self.verify();
            (class, by)
        }

        fn part(&mut self, class: ClassId, key: PartKey, side: Side) -> (ClassId, ItemId) {
            let at = self.item("projection");
            let part = self.classes.part(class, key, side, at).expect("the part is named");
            self.verify();
            (part, at)
        }

        fn unify(&mut self, a: ClassId, b: ClassId) -> (Unified<Live>, ItemId) {
            let at = self.item("unify");
            let unified = self.classes.unify(&self.scopes, a, b, at);
            self.verify();
            (unified, at)
        }

        fn bound(&mut self, lower: ClassId, upper: ClassId) -> ItemId {
            let at = self.item("bound");
            self.classes.bound(Bound::Enter { lower, upper }, at);
            self.verify();
            at
        }

        fn verify(&self) {
            assert_eq!(self.classes.verify(), Ok(()));
        }

        fn root(&self, class: ClassId) -> ClassId {
            self.classes.find(class).id()
        }

        fn same(&self, a: ClassId, b: ClassId) -> bool {
            self.classes.find(a) == self.classes.find(b)
        }

        fn state(&self, class: ClassId) -> &State<Live> {
            &self.classes.get(self.classes.find(class)).state
        }
    }

    /// Every test ends with the store verified, whatever it changed directly.
    impl Drop for World {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                self.verify();
            }
        }
    }

    fn field(name: &str) -> PartKey {
        PartKey::Field(Name::new(name))
    }

    fn some() -> PartKey {
        PartKey::Payload(Name::new("Some"))
    }

    #[test]
    fn a_union_keeps_the_smaller_id_as_its_root() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let (unified, _) = world.unify(b, a);
        assert_eq!(unified, Unified::default());
        assert_eq!(world.classes.find(b).id(), a);
        assert!(world.same(a, b));
    }

    #[test]
    fn parts_of_one_name_become_one_class() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let (read, read_at) = world.part(a, field("x"), Side::Upper);
        let (written, written_at) = world.part(b, field("x"), Side::Lower);
        assert!(!world.same(read, written));
        world.unify(a, b);
        assert!(world.same(read, written));
        let State::Undecided(constraints) = world.state(a) else {
            panic!("two undecided classes stay undecided");
        };
        let part = &constraints.parts[&field("x")];
        assert_eq!(part.lower, BTreeSet::from([written_at]));
        assert_eq!(part.upper, BTreeSet::from([read_at]));
    }

    #[test]
    fn naming_a_part_twice_gives_one_class() {
        let mut world = World::new();
        let a = world.undecided();
        let (first, _) = world.part(a, field("x"), Side::Upper);
        let (second, _) = world.part(a, field("x"), Side::Lower);
        assert_eq!(first, second);
    }

    #[test]
    fn decided_classes_of_one_layer_unify_their_parts() {
        let mut world = World::new();
        let p = world.undecided();
        let q = world.undecided();
        let (a, _) = world.decided(Shape::Option(p));
        let (b, _) = world.decided(Shape::Option(q));
        let (unified, _) = world.unify(a, b);
        assert_eq!(unified, Unified::default());
        assert!(world.same(p, q));
    }

    #[test]
    fn decided_classes_of_two_layers_are_refused() {
        let mut world = World::new();
        let target = world.undecided();
        let (shared, shared_by) = world.decided(Shape::Ref(Mutability::Shared, target));
        let (exclusive, exclusive_by) = world.decided(Shape::Ref(Mutability::Exclusive, target));
        let (unified, at) = world.unify(shared, exclusive);
        assert_eq!(
            unified.refusals,
            vec![Refusal {
                cause: Cause::Mismatch,
                items: BTreeSet::from([shared_by, exclusive_by, at]),
            }]
        );
    }

    #[test]
    fn an_undecided_class_gives_its_parts_to_the_places_of_the_shape() {
        let mut world = World::new();
        let u = world.undecided();
        let (payload, _) = world.part(u, some(), Side::Upper);
        let p = world.undecided();
        let (d, _) = world.decided(Shape::Option(p));
        let (unified, _) = world.unify(u, d);
        assert_eq!(unified, Unified::default());
        assert!(world.same(payload, p));
        assert!(matches!(world.state(u), State::Decided(_)));
    }

    #[test]
    fn a_part_the_shape_has_no_place_for_goes_to_the_caller() {
        let mut world = World::new();
        let u = world.undecided();
        let (part, named_at) = world.part(u, field("x"), Side::Upper);
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let (unified, at) = world.unify(u, d);
        assert_eq!(unified.refusals, Vec::new());
        assert_eq!(
            unified.unplaced,
            vec![Unplaced {
                class: world.root(u),
                key: field("x"),
                part: Part {
                    class: part,
                    lower: BTreeSet::new(),
                    upper: BTreeSet::from([named_at]),
                    carried: BTreeSet::new(),
                },
                at,
            }]
        );
    }

    #[test]
    fn a_decided_class_gives_a_part_in_its_place_or_gives_none() {
        let mut world = World::new();
        let p = world.undecided();
        let (d, _) = world.decided(Shape::Option(p));
        let at = world.item("projection");
        assert_eq!(world.classes.part(d, some(), Side::Upper, at), Some(p));
        assert_eq!(world.classes.part(d, field("x"), Side::Upper, at), None);
    }

    #[test]
    fn a_class_that_becomes_its_own_part_is_found_and_not_refused() {
        let mut world = World::new();
        let a = world.undecided();
        let (reference, _) = world.decided(Shape::Ref(Mutability::Shared, a));
        assert!(!world.classes.is_own_part(reference));
        let (unified, _) = world.unify(a, reference);
        assert_eq!(unified.refusals, Vec::new());
        assert!(world.classes.is_own_part(a));
        assert!(world.classes.is_own_part(reference));
    }

    #[test]
    fn calls_and_domains_on_a_class_that_becomes_decided_go_to_the_caller() {
        let mut world = World::new();
        let u = world.undecided();
        let registered = world.item("literal");
        let domain = Domain(BTreeSet::from([Scalar::U8, Scalar::I32]));
        assert_eq!(world.classes.domain(u, domain.clone(), registered), None);
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let (unified, _) = world.unify(u, d);
        assert_eq!(
            unified.pending,
            vec![Pending {
                class: world.classes.find(d).id(),
                calls: BTreeSet::new(),
                domains: BTreeSet::from([(domain, registered)]),
            }]
        );
    }

    #[test]
    fn a_domain_recorded_on_a_decided_class_goes_to_the_caller() {
        let mut world = World::new();
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let at = world.item("literal");
        let domain = Domain(BTreeSet::from([Scalar::U8]));
        let pending = world.classes.domain(d, domain.clone(), at);
        assert_eq!(
            pending,
            Some(Pending {
                class: d,
                calls: BTreeSet::new(),
                domains: BTreeSet::from([(domain, at)]),
            })
        );
    }

    #[test]
    fn a_bound_is_held_until_it_is_finished_and_no_longer() {
        let mut world = World::new();
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let (e, _) = world.decided(Shape::Scalar(Scalar::U8));
        let at = world.bound(d, e);
        let bound = Bound::Enter { lower: d, upper: e };
        assert_eq!(world.classes.bounds(), &BTreeSet::from([(bound, at)]));

        assert!(world.classes.finish(&bound, at));
        assert_eq!(world.classes.bounds(), &BTreeSet::new());
        assert!(!world.classes.finish(&bound, at));
    }

    #[test]
    fn bounds_of_two_kinds_on_one_pair_of_classes_stay_two() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let at = world.item("bound");
        let enter = Bound::Enter { lower: a, upper: b };
        let change = Bound::Change { lower: a, upper: b };
        world.classes.bound(enter, at);
        world.classes.bound(change, at);
        assert_eq!(world.classes.bounds(), &BTreeSet::from([(enter, at), (change, at)]));
    }

    #[test]
    fn what_a_declaration_writes_stays_external_whatever_it_is_made_one_with() {
        let mut world = World::new();
        let root = world.scopes.root();
        let at = world.item("declared");
        let u8_head = Shape::Scalar(Scalar::U8);
        let hole = world.undecided();
        let declared = world.classes.external(root, u8_head.clone(), BTreeSet::from([at]));
        assert!(!world.classes.is_external(hole));
        world.unify(hole, declared);
        assert!(world.classes.is_external(hole));

        let (solved, _) = world.decided(u8_head.clone());
        assert!(!world.classes.is_external(solved));
        let declared = world.classes.external(root, u8_head, BTreeSet::from([at]));
        world.unify(solved, declared);
        assert!(world.classes.is_external(solved));
    }

    #[test]
    fn a_union_belongs_to_the_outer_scope() {
        let mut world = World::new();
        let body = world.scopes.child(world.scopes.root());
        let inner = world.classes.undecided(body);
        let outer = world.undecided();
        world.unify(inner, outer);
        assert_eq!(world.classes.get(world.classes.find(inner)).scope, world.scopes.root());
    }
}
