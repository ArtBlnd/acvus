//! The classes of one kind: a union-find over their ids, each root holding
//! its class, and every constraint recorded between them that is not finished yet.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::mem;

use crate::primitive::components::components;
use crate::primitive::constraint::ConstraintOf;

use crate::primitive::ItemId;
use crate::primitive::ScopeId;
use crate::primitive::class::{
    Call, Class, ClassKind, Connections, Constraints, Decided, Held, Origin, Part, Pending, Side, State, Undecided, Unplaced,
};
use crate::primitive::domain::Domain;
use crate::primitive::id::Arena;
use crate::primitive::refusal::Refused;
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
    /// Open to the rest of the store for [`Classes::verify`] and for copying a
    /// template.
    pub(super) slots: Arena<K::Id, Slot<K>>,
    /// Every constraint recorded and not finished, from its recording until whoever
    /// judges or makes it finishes it. Open to the rest of the store for
    /// [`Classes::verify`] and for copying a template.
    pub(super) constraints: Constraints<K>,
    /// Every constraint finished. A constraint is recorded once: what connects parts by
    /// name comes to the same pair again, and recording it again does nothing.
    finished: Constraints<K>,
    /// The constraints to look at: each recorded and not looked at yet, and each that
    /// waits on a class that has changed since.
    woken: Constraints<K>,
    /// For each constraint not finished, the classes it was put on, to wait on or
    /// to guard: where it is held is looked for among these and not among every
    /// class. An id here may since have been merged into another, and a class may
    /// hold the constraint no longer.
    pub(super) put_on: BTreeMap<(K::Constraint, ItemId), BTreeSet<K::Id>>,
    /// For each class, by its root, the constraints not finished that have it as
    /// an end.
    pub(super) of_end: BTreeMap<K::Id, Constraints<K>>,
}

// Written out, as a derive would ask an id to have a default, and none has.
impl<K> Default for Classes<K>
where
    K: ClassKind,
{
    fn default() -> Self {
        Self {
            slots: Arena::default(),
            constraints: Constraints::<K>::new(),
            finished: Constraints::<K>::new(),
            woken: Constraints::<K>::new(),
            put_on: BTreeMap::new(),
            of_end: BTreeMap::new(),
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
/// class is held, and not decided, while a constraint guards it ([`Classes::guard`]),
/// and while it is named as a part on an undecided class that it does not hold in
/// turn. What is held is found once, when this is made, and this borrows the
/// store for as long as it lives: nothing is recorded and nothing is merged
/// between that finding and the decisions, and the classes it decides are decided
/// together.
///
/// ```compile_fail
/// use acvus_typeck_model::primitive::ItemId;
/// use acvus_typeck_model::primitive::constraint::Bound;
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
/// classes.constraint(Bound::Change { lower: a, upper: b }, items.push(()));
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
        Some(self.classes.unite(scopes, class, head, at))
    }
}

/// What a union, or a record on a decided class, leaves to its caller.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Unified<K>
where
    K: ClassKind,
{
    pub refusals: Vec<Refused<K>>,
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
            state: State::Undecided(Undecided::default()),
            held: Held::default(),
            connections: Connections::<K>::new(),
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
            connections: Connections::<K>::new(),
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

    /// Whether the class `class` names is a part of itself, at any depth.
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

    /// Records `constraint`, by `at`, and says whether it was not recorded before. The
    /// store holds it until it is finished.
    pub fn constrain(&mut self, constraint: K::Constraint, at: ItemId) -> bool {
        let recorded = (constraint, at);
        if self.finished.contains(&recorded) || !self.constraints.insert(recorded.clone()) {
            return false;
        }
        for end in self.ends(&recorded.0) {
            self.of_end.entry(end).or_default().insert(recorded.clone());
        }
        self.woken.insert(recorded);
        true
    }

    /// The constraints to look at, which are then to look at no longer until a class
    /// one of them waits on is merged or decided.
    pub fn take_woken(&mut self) -> Constraints<K> {
        mem::take(&mut self.woken)
    }

    /// States anew which classes `constraint`, recorded by `at`, waits on: it is to be
    /// looked at again when one of them is merged with another or decided.
    pub fn wait(&mut self, constraint: &K::Constraint, at: ItemId, on: &BTreeSet<K::Id>) {
        let recorded = (constraint.clone(), at);
        for root in self.holding(&recorded) {
            self.root_mut(root).held.waiters.remove(&recorded);
        }
        if !self.constraints.contains(&recorded) {
            return;
        }
        for class in on {
            let root = self.find(*class).id;
            self.root_mut(root).held.waiters.insert(recorded.clone());
            self.put_on.entry(recorded.clone()).or_default().insert(root);
        }
    }

    /// The two ends of `constraint`, each as the class it is now.
    fn ends(&self, constraint: &K::Constraint) -> [K::Id; 2] {
        [self.find(constraint.lower()).id, self.find(constraint.upper()).id]
    }

    /// The constraints not finished that have `class` as an end.
    pub fn constraints_of(&self, class: K::Id) -> impl Iterator<Item = &(K::Constraint, ItemId)> {
        self.of_end.get(&self.find(class).id).into_iter().flatten()
    }

    /// The classes `recorded` was put on, each as the class it is now.
    pub(super) fn holding(&self, recorded: &(K::Constraint, ItemId)) -> BTreeSet<K::Id> {
        self.put_on
            .get(recorded)
            .into_iter()
            .flatten()
            .map(|class| self.find(*class).id)
            .collect()
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
            let reaching: Constraints<K> = above.guards.union(&above.apart).cloned().collect();
            if reaching.is_empty() {
                continue;
            }
            for part in self.root(class).state.parts() {
                let part = self.find(part).id;
                let held = &mut self.root_mut(part).held;
                let new: Vec<(K::Constraint, ItemId)> = reaching
                    .iter()
                    .filter(|guard| !held.apart.contains(guard) && !held.guards.contains(guard))
                    .cloned()
                    .collect();
                if !new.is_empty() {
                    held.guards.extend(new.iter().cloned());
                    for guard in new {
                        self.put_on.entry(guard).or_default().insert(part);
                    }
                    stack.push(part);
                }
            }
        }
    }

    /// Every constraint recorded and not finished.
    pub fn constraints(&self) -> &Constraints<K> {
        &self.constraints
    }

    /// Finishes `constraint`, recorded by `at`: it has been judged or made, and the
    /// store holds it no longer. It says whether the store held it.
    pub fn finish(&mut self, constraint: &K::Constraint, at: ItemId) -> bool {
        let recorded = (constraint.clone(), at);
        let held = self.constraints.remove(&recorded);
        if held {
            self.lift(&recorded);
            for root in self.holding(&recorded) {
                self.root_mut(root).held.waiters.remove(&recorded);
            }
            self.put_on.remove(&recorded);
            for end in self.ends(&recorded.0) {
                if let Some(of_end) = self.of_end.get_mut(&end) {
                    of_end.remove(&recorded);
                }
            }
            self.woken.remove(&recorded);
            self.finished.insert(recorded);
        }
        held
    }

    /// States anew what `constraint`, recorded by `at`, guards: every class below one
    /// of `apart`, at any depth, but for the classes of `apart` themselves. Those
    /// are the classes it may still merge with another before it is finished, and
    /// what it guarded before and does not now is free of it.
    ///
    /// A class of `apart` is an end of the constraint, or a class it reads and never
    /// merges. The guard goes down to each part such a class comes to have, as
    /// structure grows, without the constraint being looked at again. The store reads
    /// no more of a constraint here than anywhere: whoever knows what the constraint means
    /// says what it holds apart, and says it again as that changes.
    pub fn guard(&mut self, constraint: &K::Constraint, at: ItemId, apart: &BTreeSet<K::Id>) {
        let recorded = (constraint.clone(), at);
        self.lift(&recorded);
        if !self.constraints.contains(&recorded) {
            return;
        }
        let apart: BTreeSet<K::Id> = apart.iter().map(|class| self.find(*class).id).collect();
        for root in &apart {
            self.root_mut(*root).held.apart.insert(recorded.clone());
            self.put_on.entry(recorded.clone()).or_default().insert(*root);
        }
        for root in apart {
            self.push_down(root);
        }
    }

    fn lift(&mut self, recorded: &(K::Constraint, ItemId)) {
        for root in self.holding(recorded) {
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
        // they are decided together: a part holds its owner where the two reach
        // one another through parts.
        let parts: BTreeMap<K::Id, Vec<K::Id>> = self
            .roots()
            .map(|owner| (owner.id, self.get(owner).state.parts().into_iter().map(|part| self.find(part).id).collect()))
            .collect();
        let component = components(&parts);
        for (owner, parts) in &parts {
            if self.root(*owner).state.is_decided() {
                continue;
            }
            held.extend(parts.iter().filter(|part| component[*part] != component[owner]));
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
        let State::Undecided(undecided) = &mut self.root_mut(root).state else {
            unreachable!("a decided class returned above, and making a class decides none");
        };
        let named_anew = !undecided.parts.contains_key(&key);
        let part = undecided.parts.entry(key).or_insert_with(|| Part {
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
            State::Undecided(undecided) => {
                undecided.calls.insert((call, at));
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
            State::Undecided(undecided) => {
                undecided.domains.insert((domain, at));
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
    ///
    /// The class keeps the connection, whether or not the two were one class
    /// already: which of several connections merges two classes depends on the
    /// order they are made in, and that they were all made does not.
    pub fn unify(&mut self, scopes: &Scopes, a: K::Id, b: K::Id, at: ItemId) -> Unified<K> {
        let unified = self.unite(scopes, a, b, at);
        let root = self.find(a).id;
        self.root_mut(root).connections.insert((a, b, at));
        unified
    }

    /// Every connection that made `class` the class it is: those that made it
    /// one class, and those that made one class of a class that holds it as a
    /// part, in turn. Parts of one name are one class because the classes that
    /// hold them are, so what made those one made these one.
    pub fn connections(&self, class: K::Id) -> Connections<K> {
        let mut holders: BTreeMap<K::Id, Vec<K::Id>> = BTreeMap::new();
        for holder in self.roots() {
            for part in self.get(holder).state.parts() {
                holders.entry(self.find(part).id).or_default().push(holder.id);
            }
        }
        let mut found = Connections::<K>::new();
        let mut seen = BTreeSet::new();
        let mut stack = vec![self.find(class).id];
        while let Some(class) = stack.pop() {
            if !seen.insert(class) {
                continue;
            }
            found.extend(self.root(class).connections.iter().copied());
            stack.extend(holders.get(&class).into_iter().flatten());
        }
        found
    }

    /// Makes `a` and `b` one class, as [`Classes::unify`] does, and keeps no
    /// connection: a decision is one, and connects nothing.
    fn unite(&mut self, scopes: &Scopes, a: K::Id, b: K::Id, at: ItemId) -> Unified<K> {
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
                state: State::Undecided(Undecided::default()),
                held: Held::default(),
            connections: Connections::<K>::new(),
            });
            let Slot::Root(kept_class) = mem::replace(self.slots.get_mut(kept), placeholder) else {
                unreachable!("find gives a root");
            };
            let merge = kept_class.merge(gone_class, kept, scopes, at);
            *self.slots.get_mut(kept) = Slot::Root(merge.class);
            if let Some(of_gone) = self.of_end.remove(&gone) {
                self.of_end.entry(kept).or_default().extend(of_gone);
            }
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
    use crate::primitive::constraint::{Bound, Constraint};
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

        fn enter(&mut self, lower: ClassId, upper: ClassId) -> ItemId {
            let at = self.item("bound");
            self.classes.constrain(Bound::Enter { lower, upper }.into(), at);
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
        PartKey::Payload(crate::primitive::ty::VariantOf::Option, Name::new("Some"))
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
        let State::Undecided(undecided) = world.state(a) else {
            panic!("two undecided classes stay undecided");
        };
        let part = &undecided.parts[&field("x")];
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
        let (shared, _) = world.decided(Shape::Ref(Mutability::Shared, target));
        let (exclusive, exclusive_by) = world.decided(Shape::Ref(Mutability::Exclusive, target));
        let (unified, at) = world.unify(shared, exclusive);
        assert_eq!(
            unified.refusals,
            vec![Refused {
                cause: Cause::Mismatch,
                items: BTreeSet::from([exclusive_by, at]),
                of: vec![shared],
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
    fn a_constraint_is_held_until_it_is_finished_and_no_longer() {
        let mut world = World::new();
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let (e, _) = world.decided(Shape::Scalar(Scalar::U8));
        let at = world.enter(d, e);
        let constraint = Constraint::Bound(Bound::Enter { lower: d, upper: e });
        assert_eq!(world.classes.constraints(), &BTreeSet::from([(constraint, at)]));

        assert!(world.classes.finish(&constraint, at));
        assert_eq!(world.classes.constraints(), &BTreeSet::new());
        assert!(!world.classes.finish(&constraint, at));
    }

    #[test]
    fn bounds_of_two_kinds_on_one_pair_of_classes_stay_two() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let at = world.item("bound");
        let enter = Constraint::Bound(Bound::Enter { lower: a, upper: b });
        let change = Constraint::Bound(Bound::Change { lower: a, upper: b });
        world.classes.constrain(enter, at);
        world.classes.constrain(change, at);
        assert_eq!(world.classes.constraints(), &BTreeSet::from([(enter, at), (change, at)]));
    }

    #[test]
    fn a_class_keeps_every_connection_made_of_it_whichever_merged_it() {
        let mut world = World::new();
        let (a, b, c) = (world.undecided(), world.undecided(), world.undecided());
        let (_, first) = world.unify(a, b);
        let (_, second) = world.unify(b, c);
        // The three are one class already, and this is kept as the others are.
        let (_, third) = world.unify(a, c);
        assert_eq!(world.classes.connections(c), BTreeSet::from([(a, b, first), (b, c, second), (a, c, third)]));
    }

    #[test]
    fn what_made_a_class_one_made_its_parts_one() {
        let mut world = World::new();
        let (a, b) = (world.undecided(), world.undecided());
        let (x, _) = world.part(a, field("x"), Side::Upper);
        let (y, _) = world.part(b, field("x"), Side::Upper);
        let (_, at) = world.unify(a, b);
        assert!(world.same(x, y));
        assert_eq!(world.classes.connections(y), BTreeSet::from([(a, b, at)]));
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
