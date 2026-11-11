//! The invariants the store keeps, checked.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::constraint::ConstraintOf;
use crate::primitive::class::{Class, ClassKind, State};
use crate::primitive::id::Id;
use crate::primitive::ty::PartKey;

use super::classes::{Classes, Slot};

/// An invariant of the store that does not hold.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Violation<K>
where
    K: ClassKind,
{
    /// A link that does not point to a smaller id, so that `find` might not
    /// end.
    Link { from: K::Id, to: K::Id },
    /// An id that `holder` holds and that names no class made.
    Dangling { holder: K::Id, id: K::Id },
    /// A part of an undecided class that no item names.
    Unnamed { class: K::Id, key: PartKey },
    /// A constraint the store holds, an end of which names no class made.
    Unanchored { constraint: K::Constraint, by: ItemId },
    /// A class holding something of a constraint the store does not hold: one
    /// finished, or one never recorded.
    Guard { class: K::Id, constraint: K::Constraint, by: ItemId },
    /// A class that a constraint guards and should not, or does not and should: what
    /// a constraint guards is every class below one it holds apart, but for those it
    /// holds apart.
    Guarded { class: K::Id, constraint: K::Constraint, by: ItemId },
    /// A class holding something of a constraint that is not among the classes
    /// the store keeps the constraint as put on, so that it would be left there.
    Unlisted { class: K::Id, constraint: K::Constraint, by: ItemId },
    /// A constraint the store keeps as having `class` as an end and that has not,
    /// or one that has and is not kept so.
    Misfiled { class: K::Id, constraint: K::Constraint, by: ItemId },
}

impl<K> Classes<K>
where
    K: ClassKind,
{
    /// Checks every invariant of the store.
    ///
    /// The links and the ids held are checked first: where one fails, `find`
    /// cannot be trusted, and the rest is left unchecked.
    pub fn verify(&self) -> Result<(), Vec<Violation<K>>> {
        let mut violations = Vec::new();
        for (id, slot) in self.slots.iter() {
            match slot {
                Slot::Link(to) if *to >= id => violations.push(Violation::Link { from: id, to: *to }),
                Slot::Link(_) => {}
                Slot::Root(class) => {
                    let dangling = class
                        .state
                        .held()
                        .into_iter()
                        .chain(class.connections.iter().flat_map(|(a, b, _)| [*a, *b]))
                        .filter(|held| held.index() >= self.slots.len())
                        .map(|held| Violation::Dangling { holder: id, id: held });
                    violations.extend(dangling);
                }
            }
        }
        for (constraint, by) in &self.constraints {
            if [constraint.lower(), constraint.upper()].iter().any(|end| end.index() >= self.slots.len()) {
                violations.push(Violation::Unanchored {
                    constraint: constraint.clone(),
                    by: *by,
                });
            }
        }
        if !violations.is_empty() {
            return Err(violations);
        }

        // The constraints kept by end, against those of each class found anew.
        for root in self.roots() {
            let kept: BTreeSet<_> = self.constraints_of(root.id()).cloned().collect();
            let found: BTreeSet<_> = self
                .constraints
                .iter()
                .filter(|(constraint, _)| self.find(constraint.lower()) == root || self.find(constraint.upper()) == root)
                .cloned()
                .collect();
            for (constraint, by) in kept.symmetric_difference(&found) {
                violations.push(Violation::Misfiled {
                    class: root.id(),
                    constraint: constraint.clone(),
                    by: *by,
                });
            }
        }

        for (id, slot) in self.slots.iter() {
            let Slot::Root(class) = slot else {
                continue;
            };
            let held = &class.held;
            for guard in held.guards.iter().chain(&held.apart).chain(&held.waiters) {
                if !self.constraints.contains(guard) {
                    violations.push(Violation::Guard {
                        class: id,
                        constraint: guard.0.clone(),
                        by: guard.1,
                    });
                }
                if !self.holding(guard).contains(&id) {
                    violations.push(Violation::Unlisted {
                        class: id,
                        constraint: guard.0.clone(),
                        by: guard.1,
                    });
                }
            }
        }
        // What each constraint guards, found anew by walking down from what it holds
        // apart, against the guards the store kept as structure grew.
        for recorded in &self.constraints {
            let apart: BTreeSet<K::Id> = self
                .roots()
                .filter(|root| self.get(*root).held.apart.contains(recorded))
                .map(|root| root.id())
                .collect();
            let below = self.below(apart.iter().copied());
            for root in self.roots() {
                let guarded = self.get(root).held.guards.contains(recorded);
                let expected = below.contains(&root.id()) && !apart.contains(&root.id());
                if guarded != expected {
                    violations.push(Violation::Guarded {
                        class: root.id(),
                        constraint: recorded.0.clone(),
                        by: recorded.1,
                    });
                }
            }
        }

        for (id, slot) in self.slots.iter() {
            let Slot::Root(Class {
                state: State::Undecided(undecided),
                ..
            }) = slot
            else {
                continue;
            };
            for (key, part) in &undecided.parts {
                if part.lower.is_empty() && part.upper.is_empty() && part.carried.is_empty() {
                    violations.push(Violation::Unnamed {
                        class: id,
                        key: key.clone(),
                    });
                }
            }
        }

        if violations.is_empty() { Ok(()) } else { Err(violations) }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::primitive::class::{Live, Part, Side, Template, Undecided};
    use crate::primitive::constraint::{Bound, Constraint};
    use crate::primitive::id::Arena;
    use crate::primitive::scope::Scopes;
    use crate::primitive::{ClassId, Name};

    struct World {
        classes: Classes<Live>,
        scopes: Scopes,
        items: Arena<ItemId, ()>,
    }

    impl World {
        fn new() -> Self {
            Self {
                classes: Classes::default(),
                scopes: Scopes::new(),
                items: Arena::default(),
            }
        }

        fn undecided(&mut self) -> ClassId {
            self.classes.undecided(self.scopes.root())
        }

        fn undecided_of(&mut self, class: ClassId) -> &mut Undecided<Live> {
            let Slot::Root(Class {
                state: State::Undecided(undecided),
                ..
            }) = self.classes.slots.get_mut(class)
            else {
                panic!("{class:?} is an undecided root");
            };
            undecided
        }
    }

    #[test]
    fn a_store_kept_by_its_own_operations_verifies() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let at = world.items.push(());
        world.classes.constrain(Bound::Enter { lower: a, upper: b }.into(), at);
        world
            .classes
            .part(a, PartKey::Field(Name::new("x")), Side::Upper, at)
            .expect("an undecided class names any part");
        world.classes.unify(&world.scopes, a, b, at);
        assert_eq!(world.classes.verify(), Ok(()));
    }

    #[test]
    fn a_template_store_is_kept_by_the_same_operations() {
        let scopes = Scopes::new();
        let mut items: Arena<ItemId, ()> = Arena::default();
        let mut templates = Classes::<Template>::default();
        let hole = templates.undecided(scopes.root());
        let at = items.push(());
        let reference = templates.decided(
            scopes.root(),
            crate::primitive::ty::Shape::Ref(crate::primitive::ty::Mutability::Shared, hole),
            BTreeSet::from([at]),
        );
        let other = templates.undecided(scopes.root());
        templates.constrain(Bound::Enter { lower: other, upper: hole }.into(), at);
        templates.unify(&scopes, other, reference, at);
        assert_eq!(templates.verify(), Ok(()));
    }

    #[test]
    fn a_constraint_to_no_class_made_is_found() {
        let mut world = World::new();
        let a = world.undecided();
        let mut elsewhere = Classes::<Live>::default();
        let stray = (0..4).map(|_| elsewhere.undecided(world.scopes.root())).last().expect("four classes were made");
        let at = world.items.push(());
        let constraint = Constraint::Bound(Bound::Enter { lower: a, upper: stray });
        world.classes.constraints.insert((constraint, at));
        assert_eq!(world.classes.verify(), Err(vec![Violation::Unanchored { constraint, by: at }]));
    }

    #[test]
    fn a_part_no_item_names_is_found() {
        let mut world = World::new();
        let a = world.undecided();
        let part = world.undecided();
        let key = PartKey::Field(Name::new("x"));
        world.undecided_of(a).parts.insert(
            key.clone(),
            Part {
                class: part,
                lower: BTreeSet::new(),
                upper: BTreeSet::new(),
                carried: BTreeSet::new(),
            },
        );
        assert_eq!(world.classes.verify(), Err(vec![Violation::Unnamed { class: a, key }]));
    }

    #[test]
    fn a_link_up_to_a_larger_id_is_found() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        *world.classes.slots.get_mut(a) = Slot::Link(b);
        assert_eq!(world.classes.verify(), Err(vec![Violation::Link { from: a, to: b }]));
    }

    #[test]
    fn an_id_of_no_class_made_is_found() {
        let mut world = World::new();
        let a = world.undecided();
        let mut elsewhere = Classes::<Live>::default();
        let stray = (0..4).map(|_| elsewhere.undecided(world.scopes.root())).last().expect("four classes were made");
        let at = world.items.push(());
        world.undecided_of(a).parts.insert(
            PartKey::Target,
            Part {
                class: stray,
                lower: BTreeSet::from([at]),
                upper: BTreeSet::new(),
                carried: BTreeSet::new(),
            },
        );
        assert_eq!(world.classes.verify(), Err(vec![Violation::Dangling { holder: a, id: stray }]));
    }
}
