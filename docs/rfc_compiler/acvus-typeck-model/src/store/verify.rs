//! The invariants the store keeps, checked.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::ItemId;
use crate::primitive::components::components;
use crate::primitive::constraint::ConstraintOf;
use crate::primitive::class::{Class, ClassKind, State};
use crate::primitive::id::Id;
use crate::primitive::ty::PartKey;

use super::classes::{Classes, Slot, parts_in};
use super::ledger::{ROLES, Recorded, Role};

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
    /// A class that a constraint guards and should not, or does not and should: what
    /// a constraint guards is every class below one it holds apart, but for those it
    /// holds apart.
    Guarded { class: K::Id, constraint: K::Constraint, by: ItemId },
    /// What the ledger says `class` is to a constraint by the constraint and by
    /// the class do not agree, or the class is no root, or it is said to be an
    /// end of a constraint and is not, or is one and is not said to be.
    Misfiled { class: K::Id, constraint: K::Constraint, by: ItemId },
    /// `part` is a part of the undecided class `owner` and the owners do not
    /// say so both ways, or they say so and it is not.
    Owned { owner: K::Id, part: K::Id },
    /// An undecided class that is held and should not be, or is not and should:
    /// what is held is found from the ledger and the owners, and here anew
    /// from every class, a part and its owner being decided together where they
    /// are of one strongly connected component of the parts.
    Held { class: K::Id },
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
                        .parts()
                        .into_iter()
                        .chain(class.connections.iter().flat_map(|(a, b, _)| [*a, *b]))
                        .filter(|held| held.index() >= self.slots.len())
                        .map(|held| Violation::Dangling { holder: id, id: held });
                    violations.extend(dangling);
                }
            }
        }
        if !violations.is_empty() {
            return Err(violations);
        }

        // The ledger says what each constraint has to do with which class twice,
        // by constraint and by class: the two agree, name roots alone, and have
        // the ends the constraint has.
        let (classes_of, constraints_of) = self.ledger().both_ways();
        let misfiled = |class: K::Id, recorded: &Recorded<K>| Violation::Misfiled {
            class,
            constraint: recorded.0.clone(),
            by: recorded.1,
        };
        for (recorded, classes) in classes_of {
            for role in ROLES {
                for class in classes.of(role) {
                    let is_root = class.index() < self.slots.len() && self.find(*class).id() == *class;
                    let both = constraints_of.get(class).is_some_and(|constraints| constraints.of(role).contains(recorded));
                    if !is_root || !both {
                        violations.push(misfiled(*class, recorded));
                    }
                }
            }
            let ends = BTreeSet::from([self.find(recorded.0.lower()).id(), self.find(recorded.0.upper()).id()]);
            for class in ends.symmetric_difference(classes.of(Role::End)) {
                violations.push(misfiled(*class, recorded));
            }
        }
        for (class, constraints) in constraints_of {
            for role in ROLES {
                for recorded in constraints.of(role) {
                    if !classes_of.get(recorded).is_some_and(|classes| classes.of(role).contains(class)) {
                        violations.push(misfiled(*class, recorded));
                    }
                }
            }
        }
        // What each constraint guards, found anew by walking down from what it holds
        // apart, against what the ledger kept as structure grew.
        for (recorded, classes) in classes_of {
            let apart = classes.of(Role::Apart);
            let below = self.below(apart.iter().copied());
            for root in self.roots() {
                let guarded = classes.of(Role::Guarded).contains(&root.id());
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

        // What each undecided class owns, found anew from its parts, against what
        // the owners kept both ways.
        let owned: BTreeSet<(K::Id, K::Id)> = self
            .roots()
            .flat_map(|owner| self.owned_by(owner.id()).into_iter().map(move |part| (owner.id(), part)))
            .collect();
        let (parts_of, owners_of) = self.owners.both_ways();
        let by_owner: BTreeSet<(K::Id, K::Id)> = parts_of.iter().flat_map(|(owner, parts)| parts.iter().map(|part| (*owner, *part))).collect();
        let by_part: BTreeSet<(K::Id, K::Id)> = owners_of.iter().flat_map(|(part, owners)| owners.iter().map(|owner| (*owner, *part))).collect();
        for (owner, part) in owned.iter().chain(&by_owner).chain(&by_part).collect::<BTreeSet<_>>() {
            if [&owned, &by_owner, &by_part].iter().any(|said| !said.contains(&(*owner, *part))) {
                violations.push(Violation::Owned {
                    owner: *owner,
                    part: *part,
                });
            }
        }

        // What is held, found anew from every class.
        let parts: BTreeMap<K::Id, Vec<K::Id>> = self.roots().map(|owner| (owner.id(), parts_in(&self.slots, owner.id()))).collect();
        let component = components(&parts);
        let mut expected: BTreeSet<K::Id> = self.roots().map(|root| root.id()).filter(|root| self.ledger().guards(*root)).collect();
        for (owner, parts) in &parts {
            if !self.is_decided(*owner) {
                expected.extend(parts.iter().filter(|part| component[*part] != component[owner]));
            }
        }
        let held = self.held();
        for root in self.roots() {
            let class = root.id();
            if !self.is_decided(class) && held.contains(&class) != expected.contains(&class) {
                violations.push(Violation::Held { class });
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
    use crate::primitive::constraint::Bound;
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

    /// The part is written past the store's own operations: no item names it,
    /// the owners were not told of it, and so it is not held as a part is.
    #[test]
    fn a_part_written_past_the_store_is_found_unnamed_unowned_and_not_held() {
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
        assert_eq!(
            world.classes.verify(),
            Err(vec![Violation::Unnamed { class: a, key }, Violation::Owned { owner: a, part }, Violation::Held { class: part }])
        );
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
