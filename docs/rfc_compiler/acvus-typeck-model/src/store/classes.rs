//! The classes of one kind: a union-find over their ids, each root holding
//! its class, and every bound recorded between them that is not finished yet.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::mem;

use crate::primitive::ItemId;
use crate::primitive::ScopeId;
use crate::primitive::bound::BoundOf;
use crate::primitive::class::{
    Bounds, Call, Class, ClassKind, Constraints, Decided, Origin, Part, Pending, Side, State, Unplaced,
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

    /// The bounds not finished whose upper end is `root`, by the class at their
    /// lower end, `root` itself among those where it is its own lower bound. None
    /// is given for a decided class, which nothing more decides.
    pub fn lowers(&self, root: Root<'_, K>) -> BTreeMap<Root<'_, K>, Bounds<K>> {
        self.side(root, |bound| (bound.upper(), bound.lower()))
    }

    /// The bounds not finished whose lower end is `root`, by the class at their
    /// upper end, as [`Classes::lowers`] gives the others.
    pub fn uppers(&self, root: Root<'_, K>) -> BTreeMap<Root<'_, K>, Bounds<K>> {
        self.side(root, |bound| (bound.lower(), bound.upper()))
    }

    /// Records `bound`, by `at`, and says whether it was not recorded before. The
    /// store holds it until it is finished.
    pub fn bound(&mut self, bound: K::Bound, at: ItemId) -> bool {
        let recorded = (bound, at);
        !self.finished.contains(&recorded) && self.bounds.insert(recorded)
    }

    /// Every bound recorded and not finished.
    pub fn bounds(&self) -> &Bounds<K> {
        &self.bounds
    }

    /// The bounds not finished both of whose ends are decided: those that can
    /// be judged, and the changes that are not made yet.
    pub fn between_decided(&self) -> Bounds<K> {
        self.bounds
            .iter()
            .filter(|(bound, _)| self.is_decided(bound.lower()) && self.is_decided(bound.upper()))
            .cloned()
            .collect()
    }

    /// Finishes `bound`, recorded by `at`: it has been judged or made, and the
    /// store holds it no longer. It says whether the store held it.
    pub fn finish(&mut self, bound: &K::Bound, at: ItemId) -> bool {
        let recorded = (bound.clone(), at);
        let held = self.bounds.remove(&recorded);
        if held {
            self.finished.insert(recorded);
        }
        held
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
            });
            let Slot::Root(kept_class) = mem::replace(self.slots.get_mut(kept), placeholder) else {
                unreachable!("find gives a root");
            };
            let merge = kept_class.merge(gone_class, kept, scopes, at);
            *self.slots.get_mut(kept) = Slot::Root(merge.class);
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

    /// The bounds whose end `ends` gives first is `root`, by the root of the end
    /// it gives second.
    fn side<F>(&self, root: Root<'_, K>, ends: F) -> BTreeMap<Root<'_, K>, Bounds<K>>
    where
        F: Fn(&K::Bound) -> (K::Id, K::Id),
    {
        if self.get(root).state.is_decided() {
            return BTreeMap::new();
        }
        let mut found: BTreeMap<Root<'_, K>, Bounds<K>> = BTreeMap::new();
        for (bound, at) in &self.bounds {
            let (this, other) = ends(bound);
            if self.find(this).id == root.id {
                found.entry(self.find(other)).or_default().insert((bound.clone(), *at));
            }
        }
        found
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

/// What the bounds of a class relate it to, by root and by item alone, for a
/// test that does not ask what kind they are, or which ids recorded them.
#[cfg(test)]
impl<K> Classes<K>
where
    K: ClassKind,
{
    pub(crate) fn lower_items(&self, class: K::Id) -> BTreeMap<K::Id, BTreeSet<ItemId>> {
        Self::items(self.lowers(self.find(class)))
    }

    pub(crate) fn upper_items(&self, class: K::Id) -> BTreeMap<K::Id, BTreeSet<ItemId>> {
        Self::items(self.uppers(self.find(class)))
    }

    fn items(found: BTreeMap<Root<'_, K>, Bounds<K>>) -> BTreeMap<K::Id, BTreeSet<ItemId>> {
        found
            .into_iter()
            .map(|(other, bounds)| (other.id(), bounds.into_iter().map(|(_, at)| at).collect()))
            .collect()
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

        fn lowers(&self, class: ClassId) -> BTreeMap<ClassId, BTreeSet<ItemId>> {
            self.classes.lower_items(class)
        }

        fn uppers(&self, class: ClassId) -> BTreeMap<ClassId, BTreeSet<ItemId>> {
            self.classes.upper_items(class)
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
    fn a_bound_is_read_from_both_of_its_ends() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let at = world.bound(a, b);
        assert_eq!(world.classes.between_decided(), BTreeSet::new());
        assert_eq!(world.lowers(b), BTreeMap::from([(a, BTreeSet::from([at]))]));
        assert_eq!(world.uppers(a), BTreeMap::from([(b, BTreeSet::from([at]))]));
        assert_eq!(world.lowers(a), BTreeMap::new());
        assert_eq!(world.uppers(b), BTreeMap::new());
    }

    #[test]
    fn a_class_can_be_its_own_lower_bound() {
        let mut world = World::new();
        let a = world.undecided();
        let written = world.bound(a, a);
        assert_eq!(world.lowers(a), BTreeMap::from([(a, BTreeSet::from([written]))]));

        let b = world.undecided();
        let c = world.undecided();
        let b_into_c = world.bound(b, c);
        world.unify(b, c);
        let bc = world.root(b);
        assert_eq!(world.lowers(bc), BTreeMap::from([(bc, BTreeSet::from([b_into_c]))]));
        assert_eq!(world.uppers(bc), BTreeMap::from([(bc, BTreeSet::from([b_into_c]))]));
    }

    #[test]
    fn bounds_are_read_through_the_roots_of_their_classes() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let c = world.undecided();
        let a_into_c = world.bound(a, c);
        let b_into_c = world.bound(b, c);
        world.unify(a, b);
        let ab = world.root(a);
        assert_eq!(world.lowers(c), BTreeMap::from([(ab, BTreeSet::from([a_into_c, b_into_c]))]));
        assert_eq!(world.uppers(ab), BTreeMap::from([(c, BTreeSet::from([a_into_c, b_into_c]))]));
    }

    #[test]
    fn a_bound_between_decided_classes_can_be_judged_at_once() {
        let mut world = World::new();
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let (e, _) = world.decided(Shape::Scalar(Scalar::U8));
        let at = world.bound(d, e);
        let bound = Bound::Enter { lower: d, upper: e };
        assert_eq!(world.classes.between_decided(), BTreeSet::from([(bound, at)]));
    }

    #[test]
    fn a_bound_can_be_judged_once_neither_of_its_ends_is_undecided() {
        let mut world = World::new();
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let u = world.undecided();
        let at = world.bound(d, u);
        assert_eq!(world.classes.between_decided(), BTreeSet::new());
        assert_eq!(world.lowers(u), BTreeMap::from([(d, BTreeSet::from([at]))]));

        let (e, _) = world.decided(Shape::Scalar(Scalar::U8));
        world.unify(u, e);
        let bound = Bound::Enter { lower: d, upper: u };
        assert_eq!(world.classes.between_decided(), BTreeSet::from([(bound, at)]));
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
        let expected = BTreeSet::from([(enter, at), (change, at)]);
        let b_root = world.classes.find(b);
        assert_eq!(world.classes.lowers(b_root)[&world.classes.find(a)], expected);
        let a_root = world.classes.find(a);
        assert_eq!(world.classes.uppers(a_root)[&world.classes.find(b)], expected);

        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        let (e, _) = world.decided(Shape::Scalar(Scalar::U8));
        world.unify(a, d);
        assert_eq!(world.classes.between_decided(), BTreeSet::new());
        world.unify(b, e);
        assert_eq!(world.classes.between_decided(), expected);
    }

    #[test]
    fn a_bound_with_an_undecided_end_is_not_yet_to_be_judged() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let at = world.bound(a, b);
        let (d, _) = world.decided(Shape::Scalar(Scalar::U8));
        world.unify(a, d);
        assert_eq!(world.classes.between_decided(), BTreeSet::new());
        assert_eq!(world.lowers(b), BTreeMap::from([(world.root(d), BTreeSet::from([at]))]));
        assert_eq!(world.uppers(a), BTreeMap::new());
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
