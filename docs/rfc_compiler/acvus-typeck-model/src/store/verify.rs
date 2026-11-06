//! The invariants the store keeps, checked.

use std::collections::BTreeSet;

use crate::primitive::class::{Bound, Class, State};
use crate::primitive::id::Id;
use crate::primitive::ty::PartKey;
use crate::primitive::{ClassId, ItemId};

use super::classes::{Classes, Slot};

/// An invariant of the store that does not hold.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Violation {
    /// A link that does not point to a smaller id, so that `find` might not
    /// end.
    Link { from: ClassId, to: ClassId },
    /// An id that `holder` holds and that names no class made.
    Dangling { holder: ClassId, id: ClassId },
    /// A part of an undecided class that no item names.
    Unnamed { class: ClassId, key: PartKey },
    /// A bound between two undecided classes that only one of them holds.
    OneSided { bound: Bound, by: ItemId },
}

impl Classes {
    /// Checks every invariant of the store.
    ///
    /// The links and the ids held are checked first: where one fails, `find`
    /// cannot be trusted, and the rest is left unchecked.
    pub fn verify(&self) -> Result<(), Vec<Violation>> {
        let mut violations = Vec::new();
        for (id, slot) in self.slots.iter() {
            match slot {
                Slot::Link(to) if *to >= id => violations.push(Violation::Link { from: id, to: *to }),
                Slot::Link(_) => {}
                Slot::Root(class) => {
                    let dangling = held(class)
                        .into_iter()
                        .filter(|held| held.index() >= self.slots.len())
                        .map(|held| Violation::Dangling { holder: id, id: held });
                    violations.extend(dangling);
                }
            }
        }
        if !violations.is_empty() {
            return Err(violations);
        }

        let mut held_as_upper = BTreeSet::new();
        let mut held_as_lower = BTreeSet::new();
        for (id, slot) in self.slots.iter() {
            let Slot::Root(Class {
                state: State::Undecided(constraints),
                ..
            }) = slot
            else {
                continue;
            };
            for (key, part) in &constraints.parts {
                if part.lower.is_empty() && part.upper.is_empty() {
                    violations.push(Violation::Unnamed {
                        class: id,
                        key: key.clone(),
                    });
                }
            }
            for (upper, items) in &constraints.upper {
                for by in items {
                    held_as_upper.insert((self.rooted(Bound { lower: id, upper: *upper }), *by));
                }
            }
            for (lower, items) in &constraints.lower {
                for by in items {
                    held_as_lower.insert((self.rooted(Bound { lower: *lower, upper: id }), *by));
                }
            }
        }
        let one_sided = held_as_upper
            .iter()
            .filter(|(bound, by)| self.is_undecided(bound.upper) && !held_as_lower.contains(&(*bound, *by)))
            .chain(
                held_as_lower
                    .iter()
                    .filter(|(bound, by)| self.is_undecided(bound.lower) && !held_as_upper.contains(&(*bound, *by))),
            )
            .map(|(bound, by)| Violation::OneSided { bound: *bound, by: *by });
        violations.extend(one_sided);

        if violations.is_empty() { Ok(()) } else { Err(violations) }
    }

    fn rooted(&self, bound: Bound) -> Bound {
        Bound {
            lower: self.find(bound.lower).id(),
            upper: self.find(bound.upper).id(),
        }
    }

    fn is_undecided(&self, class: ClassId) -> bool {
        matches!(self.get(self.find(class)).state, State::Undecided(_))
    }
}

/// Every id a class holds: its parts, the other classes of its bounds, the
/// classes of its calls, and the parts of its shape.
fn held(class: &Class) -> Vec<ClassId> {
    match &class.state {
        State::Undecided(constraints) => constraints
            .parts
            .values()
            .map(|part| part.class)
            .chain(constraints.lower.keys().copied())
            .chain(constraints.upper.keys().copied())
            .chain(
                constraints
                    .calls
                    .iter()
                    .flat_map(|(call, _)| call.args.iter().copied().chain([call.result])),
            )
            .collect(),
        State::Decided(decided) => decided.shape.places().into_iter().map(|(_, class)| *class).collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::primitive::Name;
    use crate::primitive::class::{Part, Side};
    use crate::primitive::id::Arena;
    use crate::store::scopes::Scopes;

    struct World {
        classes: Classes,
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

        fn constraints(&mut self, class: ClassId) -> &mut crate::primitive::class::Constraints {
            let Slot::Root(Class {
                state: State::Undecided(constraints),
                ..
            }) = self.classes.slots.get_mut(class)
            else {
                panic!("{class:?} is an undecided root");
            };
            constraints
        }
    }

    #[test]
    fn a_store_kept_by_its_own_operations_verifies() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let at = world.items.push(());
        world.classes.bound(Bound { lower: a, upper: b }, at);
        world
            .classes
            .part(a, PartKey::Field(Name::new("x")), Side::Upper, at)
            .expect("an undecided class names any part");
        world.classes.unify(&world.scopes, a, b, at);
        assert_eq!(world.classes.verify(), Ok(()));
    }

    #[test]
    fn a_bound_only_one_undecided_class_holds_is_found() {
        let mut world = World::new();
        let a = world.undecided();
        let b = world.undecided();
        let at = world.items.push(());
        world.classes.bound(Bound { lower: a, upper: b }, at);
        world.constraints(b).lower.clear();
        assert_eq!(
            world.classes.verify(),
            Err(vec![Violation::OneSided {
                bound: Bound { lower: a, upper: b },
                by: at,
            }])
        );
    }

    #[test]
    fn a_part_no_item_names_is_found() {
        let mut world = World::new();
        let a = world.undecided();
        let part = world.undecided();
        let key = PartKey::Field(Name::new("x"));
        world.constraints(a).parts.insert(
            key.clone(),
            Part {
                class: part,
                lower: BTreeSet::new(),
                upper: BTreeSet::new(),
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
        let mut elsewhere = Classes::default();
        let stray = (0..4).map(|_| elsewhere.undecided(world.scopes.root())).last().expect("four classes were made");
        let at = world.items.push(());
        world.constraints(a).parts.insert(
            PartKey::Target,
            Part {
                class: stray,
                lower: BTreeSet::from([at]),
                upper: BTreeSet::new(),
            },
        );
        assert_eq!(world.classes.verify(), Err(vec![Violation::Dangling { holder: a, id: stray }]));
    }
}
