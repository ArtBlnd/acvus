//! The invariants the store keeps, checked.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::class::{Bound, Class, ClassKind, State};
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
    /// A bound between two undecided classes that only one of them holds.
    OneSided { bound: Bound<K>, by: ItemId },
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

    fn rooted(&self, bound: Bound<K>) -> Bound<K> {
        Bound {
            lower: self.find(bound.lower).id(),
            upper: self.find(bound.upper).id(),
        }
    }

    fn is_undecided(&self, class: K::Id) -> bool {
        !self.get(self.find(class)).state.is_decided()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::primitive::class::{Constraints, Live, Part, Side, Template};
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

        fn constraints(&mut self, class: ClassId) -> &mut Constraints<Live> {
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
        templates.bound(Bound { lower: other, upper: hole }, at);
        templates.unify(&scopes, other, reference, at);
        assert_eq!(templates.verify(), Ok(()));
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
        let mut elsewhere = Classes::<Live>::default();
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
