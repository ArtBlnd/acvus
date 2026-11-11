//! The ledger: every constraint not finished, and what each has to do with
//! which class.
//!
//! ```text
//! recorded ──► awake ──(looked at)──► finished
//!                ▲                └──► asleep, waiting on some classes and
//!                │                     holding some apart
//!                └──(a class it waits on is merged or decided)──┘
//! ```
//!
//! A constraint has to do with a class in four ways (`Role`). The ledger keeps
//! each both ways round, by constraint and by class, and the two are written
//! together in one place (`Ledger::put`, `Ledger::take`): nothing else writes
//! either, so neither can say what the other does not.
//!
//! The ledger knows no class but by its id. Whoever holds the classes tells it
//! when two are merged and when one comes to have a part, and gives it the parts
//! of a class when it asks.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::ItemId;
use crate::primitive::class::{ClassKind, Constraints};

/// A constraint, with the item that recorded it.
pub type Recorded<K> = (<K as ClassKind>::Constraint, ItemId);

/// What a class is to a constraint not finished.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Role {
    /// One of its two ends.
    End,
    /// The constraint waits on it: it is looked at again when the class is
    /// merged with another or decided.
    Waited,
    /// The constraint holds it apart: it may still merge what lies below the
    /// class, though not the class itself.
    Apart,
    /// The constraint guards it: it lies below a class held apart, so what
    /// enters it is not all known, and it is not decided.
    Guarded,
}

pub(super) const ROLES: [Role; 4] = [Role::End, Role::Waited, Role::Apart, Role::Guarded];

/// Four sets, one for each role.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct ByRole<T> {
    ends: BTreeSet<T>,
    waited: BTreeSet<T>,
    apart: BTreeSet<T>,
    guarded: BTreeSet<T>,
}

impl<T> Default for ByRole<T> {
    fn default() -> Self {
        Self {
            ends: BTreeSet::new(),
            waited: BTreeSet::new(),
            apart: BTreeSet::new(),
            guarded: BTreeSet::new(),
        }
    }
}

impl<T> ByRole<T> {
    pub(super) fn of(&self, role: Role) -> &BTreeSet<T> {
        match role {
            Role::End => &self.ends,
            Role::Waited => &self.waited,
            Role::Apart => &self.apart,
            Role::Guarded => &self.guarded,
        }
    }

    fn of_mut(&mut self, role: Role) -> &mut BTreeSet<T> {
        match role {
            Role::End => &mut self.ends,
            Role::Waited => &mut self.waited,
            Role::Apart => &mut self.apart,
            Role::Guarded => &mut self.guarded,
        }
    }
}

/// A constraint taken out to be looked at. It is given back finished
/// (`Classes::finish`) or asleep (`Classes::sleep`), and by nothing else: only
/// the ledger makes one, and it is not copied.
#[must_use = "a constraint looked at is finished or put to sleep"]
#[derive(Debug)]
pub struct Awake<K>
where
    K: ClassKind,
{
    recorded: Recorded<K>,
}

impl<K> Awake<K>
where
    K: ClassKind,
{
    pub fn constraint(&self) -> &K::Constraint {
        &self.recorded.0
    }

    /// The item that recorded it.
    pub fn at(&self) -> ItemId {
        self.recorded.1
    }
}

/// Each constraint not finished, with the classes it has to do with.
pub(super) type ClassesOf<K> = BTreeMap<Recorded<K>, ByRole<<K as ClassKind>::Id>>;

/// Each class, with the constraints that have to do with it.
pub(super) type ConstraintsOf<K> = BTreeMap<<K as ClassKind>::Id, ByRole<Recorded<K>>>;

#[derive(Clone, Debug)]
pub struct Ledger<K>
where
    K: ClassKind,
{
    classes_of: ClassesOf<K>,
    /// The same the other way round.
    constraints_of: ConstraintsOf<K>,
    /// The constraints to look at: each recorded and not looked at yet, and each
    /// that waits on a class that has changed since.
    awake: Constraints<K>,
    /// Every constraint finished. A constraint is recorded once: what connects
    /// parts by name comes to the same pair again, and recording it again does
    /// nothing.
    finished: Constraints<K>,
}

impl<K> Default for Ledger<K>
where
    K: ClassKind,
{
    fn default() -> Self {
        Self {
            classes_of: BTreeMap::new(),
            constraints_of: BTreeMap::new(),
            awake: Constraints::<K>::new(),
            finished: Constraints::<K>::new(),
        }
    }
}

impl<K> Ledger<K>
where
    K: ClassKind,
{
    /// Says `class` is `role` to `recorded`, and whether it was not so already.
    /// With [`Ledger::take`], the only place either side is written.
    fn put(&mut self, recorded: &Recorded<K>, role: Role, class: K::Id) -> bool {
        let classes = self.classes_of.get_mut(recorded).expect("a constraint is given a role while it is held");
        if !classes.of_mut(role).insert(class) {
            return false;
        }
        self.constraints_of.entry(class).or_default().of_mut(role).insert(recorded.clone());
        true
    }

    /// Says `class` is `role` to `recorded` no longer.
    fn take(&mut self, recorded: &Recorded<K>, role: Role, class: K::Id) {
        if let Some(classes) = self.classes_of.get_mut(recorded) {
            classes.of_mut(role).remove(&class);
        }
        if let Some(constraints) = self.constraints_of.get_mut(&class) {
            constraints.of_mut(role).remove(recorded);
        }
    }

    /// The classes that are `role` to `recorded`.
    fn classes(&self, recorded: &Recorded<K>, role: Role) -> Vec<K::Id> {
        self.classes_of
            .get(recorded)
            .into_iter()
            .flat_map(|classes| classes.of(role).iter().copied())
            .collect()
    }

    /// The constraints `class` is `role` to.
    pub(super) fn constraints(&self, class: K::Id, role: Role) -> impl Iterator<Item = &Recorded<K>> {
        self.constraints_of.get(&class).into_iter().flat_map(move |constraints| constraints.of(role))
    }

    /// Records `constraint`, whose ends are the classes `ends`, by `at`, and says
    /// whether it was not recorded before. It is awake.
    pub(super) fn record(&mut self, constraint: K::Constraint, at: ItemId, ends: [K::Id; 2]) -> bool {
        let recorded = (constraint, at);
        if self.finished.contains(&recorded) || self.classes_of.contains_key(&recorded) {
            return false;
        }
        self.classes_of.insert(recorded.clone(), ByRole::default());
        for end in ends {
            self.put(&recorded, Role::End, end);
        }
        self.awake.insert(recorded);
        true
    }

    /// Takes out every constraint that is awake, to be looked at.
    pub(super) fn awake(&mut self) -> Vec<Awake<K>> {
        std::mem::take(&mut self.awake).into_iter().map(|recorded| Awake { recorded }).collect()
    }

    /// Finishes a constraint looked at: it has been judged, and is held no
    /// longer.
    pub(super) fn finish(&mut self, looked: Awake<K>) {
        let recorded = looked.recorded;
        for role in ROLES {
            for class in self.classes(&recorded, role) {
                self.take(&recorded, role, class);
            }
        }
        self.classes_of.remove(&recorded);
        self.awake.remove(&recorded);
        self.finished.insert(recorded);
    }

    /// Puts a constraint looked at to sleep: it waits on the classes `on`, and
    /// holds the classes `apart` apart, so it guards every class below one of
    /// those. What it waited on and held apart before, it does no longer.
    pub(super) fn sleep<P>(&mut self, looked: Awake<K>, on: &BTreeSet<K::Id>, apart: &BTreeSet<K::Id>, parts: P)
    where
        P: Fn(K::Id) -> Vec<K::Id>,
    {
        let recorded = looked.recorded;
        for role in [Role::Waited, Role::Apart, Role::Guarded] {
            for class in self.classes(&recorded, role) {
                self.take(&recorded, role, class);
            }
        }
        for class in on {
            self.put(&recorded, Role::Waited, *class);
        }
        for class in apart {
            self.put(&recorded, Role::Apart, *class);
        }
        for class in apart {
            self.guard_below(&recorded, *class, &parts);
        }
    }

    /// Guards, by `recorded`, every class below `class` that it does not hold
    /// apart.
    fn guard_below<P>(&mut self, recorded: &Recorded<K>, class: K::Id, parts: &P)
    where
        P: Fn(K::Id) -> Vec<K::Id>,
    {
        let mut above = vec![class];
        while let Some(class) = above.pop() {
            for part in parts(class) {
                let held_apart = self.classes_of[recorded].of(Role::Apart).contains(&part);
                if !held_apart && self.put(recorded, Role::Guarded, part) {
                    above.push(part);
                }
            }
        }
    }

    /// `class` has come to have a part: each constraint that guards the class or
    /// holds it apart guards the part too, and what lies below it.
    pub(super) fn grew<P>(&mut self, class: K::Id, parts: P)
    where
        P: Fn(K::Id) -> Vec<K::Id>,
    {
        let reaching: Vec<Recorded<K>> = self
            .constraints(class, Role::Apart)
            .chain(self.constraints(class, Role::Guarded))
            .cloned()
            .collect();
        for recorded in reaching {
            self.guard_below(&recorded, class, &parts);
        }
    }

    /// The class `gone` has been merged into `kept`: what `gone` was to each
    /// constraint, `kept` is. Each constraint that waited on either is awake, and
    /// what guards the class guards its parts.
    pub(super) fn merged<P>(&mut self, kept: K::Id, gone: K::Id, parts: P)
    where
        P: Fn(K::Id) -> Vec<K::Id>,
    {
        if let Some(of_gone) = self.constraints_of.remove(&gone) {
            for role in ROLES {
                for recorded in of_gone.of(role) {
                    if let Some(classes) = self.classes_of.get_mut(recorded) {
                        classes.of_mut(role).remove(&gone);
                    }
                    self.put(recorded, role, kept);
                }
            }
        }
        // A class held apart is not guarded by what holds it apart.
        let both: Vec<Recorded<K>> = self
            .constraints(kept, Role::Apart)
            .filter(|recorded| self.classes_of[*recorded].of(Role::Guarded).contains(&kept))
            .cloned()
            .collect();
        for recorded in both {
            self.take(&recorded, Role::Guarded, kept);
        }
        let waiting: Vec<Recorded<K>> = self.constraints(kept, Role::Waited).cloned().collect();
        self.awake.extend(waiting);
        self.grew(kept, parts);
    }

    /// Every constraint not finished.
    pub(super) fn held(&self) -> impl Iterator<Item = &Recorded<K>> {
        self.classes_of.keys()
    }

    /// Whether a constraint guards `class`.
    pub(super) fn guards(&self, class: K::Id) -> bool {
        self.constraints(class, Role::Guarded).next().is_some()
    }

    /// Each constraint not finished with the classes it has to do with, and each
    /// class with the constraints that have to do with it, for these to be
    /// checked against one another and against the classes.
    pub(super) fn both_ways(&self) -> (&ClassesOf<K>, &ConstraintsOf<K>) {
        (&self.classes_of, &self.constraints_of)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::ClassId;
    use crate::primitive::class::Live;
    use crate::primitive::constraint::Bound;
    use crate::primitive::id::Arena;

    /// A ledger, with classes that are ids and parts alone.
    #[derive(Default)]
    struct World {
        ledger: Ledger<Live>,
        ids: Arena<ClassId, ()>,
        items: Arena<ItemId, ()>,
        parts: BTreeMap<ClassId, Vec<ClassId>>,
    }

    impl World {
        fn class(&mut self) -> ClassId {
            self.ids.push(())
        }

        /// A class that is a part of `of`, which the ledger is told of.
        fn part(&mut self, of: ClassId) -> ClassId {
            let part = self.class();
            self.parts.entry(of).or_default().push(part);
            let parts = &self.parts;
            self.ledger.grew(of, |class| parts.get(&class).cloned().unwrap_or_default());
            part
        }

        fn record(&mut self, lower: ClassId, upper: ClassId) -> Recorded<Live> {
            let recorded = (Bound::Enter { lower, upper }.into(), self.items.push(()));
            assert!(self.ledger.record(recorded.0, recorded.1, [lower, upper]));
            recorded
        }

        /// Looks at the one constraint awake, and puts it to sleep.
        fn sleep(&mut self, on: &[ClassId], apart: &[ClassId]) {
            let mut awake = self.ledger.awake();
            let looked = awake.pop().expect("a constraint is awake");
            assert!(awake.is_empty(), "one constraint is awake");
            let parts = &self.parts;
            let (on, apart) = (on.iter().copied().collect(), apart.iter().copied().collect());
            self.ledger.sleep(looked, &on, &apart, |class| parts.get(&class).cloned().unwrap_or_default());
        }

        /// Merges `gone` into `kept`: the parts of both are those of `kept`.
        fn merge(&mut self, kept: ClassId, gone: ClassId) {
            let of_gone = self.parts.remove(&gone).unwrap_or_default();
            self.parts.entry(kept).or_default().extend(of_gone);
            for parts in self.parts.values_mut() {
                for part in parts.iter_mut().filter(|part| **part == gone) {
                    *part = kept;
                }
            }
            let parts = &self.parts;
            self.ledger.merged(kept, gone, |class| parts.get(&class).cloned().unwrap_or_default());
        }

        fn awake(&mut self) -> Vec<Recorded<Live>> {
            self.ledger.awake().into_iter().map(|looked| looked.recorded).collect()
        }
    }

    #[test]
    fn a_constraint_recorded_is_awake_once_and_finished_is_recorded_no_more() {
        let mut world = World::default();
        let (a, b) = (world.class(), world.class());
        let recorded = world.record(a, b);
        assert!(!world.ledger.record(recorded.0, recorded.1, [a, b]));
        let mut awake = world.ledger.awake();
        assert_eq!(awake.len(), 1);
        assert!(world.ledger.awake().is_empty());
        world.ledger.finish(awake.pop().expect("one is awake"));
        assert_eq!(world.ledger.held().count(), 0);
        assert_eq!(world.ledger.constraints(a, Role::End).count(), 0);
        assert!(!world.ledger.record(recorded.0, recorded.1, [a, b]));
    }

    #[test]
    fn a_constraint_asleep_guards_what_lies_below_what_it_holds_apart_and_not_that() {
        let mut world = World::default();
        let (a, b) = (world.class(), world.class());
        let part = world.part(a);
        let below = world.part(part);
        world.record(a, b);
        world.sleep(&[], &[a]);
        assert!(!world.ledger.guards(a));
        assert!(world.ledger.guards(part));
        assert!(world.ledger.guards(below));
        assert!(!world.ledger.guards(b));
    }

    #[test]
    fn a_part_that_comes_later_is_guarded_as_one_that_was_there() {
        let mut world = World::default();
        let (a, b) = (world.class(), world.class());
        world.record(a, b);
        world.sleep(&[], &[a]);
        let part = world.part(a);
        let below = world.part(part);
        assert!(world.ledger.guards(part));
        assert!(world.ledger.guards(below));
    }

    #[test]
    fn a_constraint_wakes_when_a_class_it_waits_on_is_merged() {
        let mut world = World::default();
        let (a, b, c) = (world.class(), world.class(), world.class());
        let recorded = world.record(a, b);
        world.sleep(&[b], &[]);
        assert_eq!(world.awake(), []);
        world.merge(a, c);
        assert_eq!(world.awake(), []);
        world.merge(b, c);
        assert_eq!(world.awake(), [recorded]);
    }

    #[test]
    fn a_class_merged_is_what_the_class_gone_was_and_its_parts_are_guarded() {
        let mut world = World::default();
        let (a, b, other) = (world.class(), world.class(), world.class());
        let guarded = world.part(a);
        let part_of_other = world.part(other);
        let recorded = world.record(a, b);
        world.sleep(&[], &[a]);
        assert!(!world.ledger.guards(part_of_other));
        world.merge(other, guarded);
        assert!(world.ledger.guards(other));
        assert!(world.ledger.guards(part_of_other));
        assert_eq!(world.ledger.constraints(guarded, Role::Guarded).count(), 0);
        world.merge(a, b);
        assert_eq!(world.ledger.constraints(a, Role::End).collect::<Vec<_>>(), [&recorded]);
    }

    #[test]
    fn a_class_held_apart_is_not_guarded_by_what_holds_it_apart() {
        let mut world = World::default();
        let (a, b) = (world.class(), world.class());
        let part = world.part(a);
        world.record(a, b);
        world.sleep(&[], &[a]);
        assert!(world.ledger.guards(part));
        world.merge(a, part);
        assert!(!world.ledger.guards(a));
    }

    #[test]
    fn a_constraint_put_to_sleep_again_waits_and_guards_as_it_then_says() {
        let mut world = World::default();
        let (a, b, c) = (world.class(), world.class(), world.class());
        let part = world.part(a);
        world.record(a, b);
        world.sleep(&[b], &[a]);
        world.merge(b, c);
        world.sleep(&[a], &[]);
        assert!(!world.ledger.guards(part));
        assert_eq!(world.ledger.constraints(b, Role::Waited).count(), 0);
        assert_eq!(world.ledger.constraints(a, Role::Waited).count(), 1);
    }
}
