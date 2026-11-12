//! Templates: a store of classes written once from a type or a text, and
//! copied whole at each use.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::constraint::ConstraintInto;
use crate::primitive::class::{Class, ClassKind};
use crate::primitive::ty::Ty;
use crate::primitive::{ItemId, ScopeId};

use super::classes::{Classes, Slot};

impl<K> Classes<K>
where
    K: ClassKind,
{
    /// Writes `ty` as classes in `scope`, by `by`: each node an external class,
    /// and each hole an undecided one.
    ///
    /// `written` holds the classes already written for one signature, so that
    /// one type, a hole included, is one class wherever the signature writes
    /// it.
    pub fn write(&mut self, scope: ScopeId, ty: &Ty, by: ItemId, written: &mut BTreeMap<Ty, K::Id>) -> K::Id {
        if let Some(&class) = written.get(ty) {
            return class;
        }
        let class = match ty {
            Ty::Hole(_) => self.undecided(scope),
            Ty::Node(shape) => {
                let shape = shape.map(|part| self.write(scope, part, by, written));
                self.external(scope, shape, BTreeSet::from([by]))
            }
        };
        written.insert(ty.clone(), class);
        class
    }

    /// Copies every class of `template` into this store, in `scope`, and gives
    /// every id of the template with the class it became.
    ///
    /// Each class is made anew holding what its original holds, every id in
    /// it given as the class its original became, and each constraint the template
    /// holds is recorded between the copies of its ends. The template is not
    /// changed.
    pub fn copy_from<S>(&mut self, template: &Classes<S>, scope: ScopeId) -> BTreeMap<S::Id, K::Id>
    where
        S: ClassKind,
        S::Constraint: ConstraintInto<S, K>,
    {
        let mut copies = BTreeMap::new();
        for (id, slot) in template.slots.iter() {
            let Slot::Root(_) = slot else {
                continue;
            };
            copies.insert(id, self.undecided(scope));
        }

        for (id, slot) in template.slots.iter() {
            let Slot::Root(class) = slot else {
                continue;
            };
            let Class { state, connections, .. } = class.clone().translate::<K, _>(|held| copies[&template.find(held).id()]);
            *self.slots.get_mut(copies[&id]) = Slot::Root(Class { scope, state, connections });
            self.owners.written(copies[&id], self.owned_by(copies[&id]));
        }
        for (id, slot) in template.slots.iter() {
            if let Slot::Link(_) = slot {
                let copy = copies[&template.find(id).id()];
                copies.insert(id, copy);
            }
        }
        for (constraint, by) in template.constraints() {
            let copied = constraint.clone().translate(|held| copies[&template.find(held).id()]);
            self.constrain(copied, *by);
        }
        copies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::Name;
    use crate::primitive::constraint::{Bound, ConstraintOf};
    use crate::primitive::class::{Live, Side, State, Template};
    use crate::primitive::id::Arena;
    use crate::primitive::scope::Scopes;
    use crate::primitive::ty::{Mutability, PartKey, Shape};
    use crate::primitive::{ClassId, TemplateId};

    struct World {
        live: Classes<Live>,
        scopes: Scopes,
        items: Arena<ItemId, &'static str>,
    }

    impl World {
        fn new() -> Self {
            Self {
                live: Classes::default(),
                scopes: Scopes::new(),
                items: Arena::default(),
            }
        }

        /// `push<T>(&mut Option<T>, T)` as a template, with its two
        /// parameters.
        fn push(&mut self) -> (Classes<Template>, [TemplateId; 2]) {
            let declared = self.items.push("extern fn push");
            let t = Ty::Hole(Name::new("T"));
            let first = node(Shape::Ref(Mutability::Exclusive, node(Shape::Option(t.clone()))));
            let mut template = Classes::default();
            let mut written = BTreeMap::new();
            let root = self.scopes.root();
            let first = template.write(root, &first, declared, &mut written);
            let second = template.write(root, &t, declared, &mut written);
            (template, [first, second])
        }

        fn copy(&mut self, template: &Classes<Template>) -> BTreeMap<TemplateId, ClassId> {
            self.live.copy_from(template, self.scopes.root())
        }

        fn shape(&self, class: ClassId) -> &Shape<ClassId> {
            let State::Decided(decided) = &self.live.get(self.live.find(class)).state else {
                panic!("{class:?} is decided");
            };
            &decided.shape
        }
    }

    impl Drop for World {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(self.live.verify(), Ok(()));
            }
        }
    }

    fn node(shape: Shape<Ty>) -> Ty {
        Ty::Node(Box::new(shape))
    }

    /// A copy is written whole and not through the naming of a part, so the
    /// owners are told of it there (`World` verifies the store when it ends).
    #[test]
    fn a_copy_of_an_undecided_class_owns_the_copies_of_its_parts() {
        let mut world = World::new();
        let named = world.items.push("a.x");
        let mut template: Classes<Template> = Classes::default();
        let owner = template.undecided(world.scopes.root());
        let part = template.part(owner, PartKey::Field(Name::new("x")), Side::Upper, named).expect("an undecided class names any part");
        let copied = world.copy(&template);
        assert_eq!(world.live.owned_by(copied[&owner]), [copied[&part]]);
        assert!(world.live.held().contains(&copied[&part]));
    }

    #[test]
    fn a_signature_writes_one_class_for_each_type_it_writes() {
        let mut world = World::new();
        let (template, [first, second]) = world.push();
        assert_eq!(template.verify(), Ok(()));
        let State::Decided(reference) = &template.get(template.find(first)).state else {
            panic!("a node is written decided");
        };
        let Shape::Ref(Mutability::Exclusive, option) = reference.shape else {
            panic!("the first parameter is `&mut Option<T>`");
        };
        let State::Decided(option) = &template.get(template.find(option)).state else {
            panic!("a node is written decided");
        };
        assert_eq!(option.shape, Shape::Option(second));
        assert!(!template.get(template.find(second)).state.is_decided());
        assert!(template.is_external(first));
    }

    #[test]
    fn a_copy_keeps_what_a_declaration_writes_external_and_a_hole_free() {
        let mut world = World::new();
        let (template, [first, second]) = world.push();
        let copied = world.copy(&template);
        assert!(world.live.is_external(copied[&first]));
        assert!(!world.live.is_external(copied[&second]));
    }

    #[test]
    fn a_copy_makes_every_class_anew_and_keeps_one_class_one() {
        let mut world = World::new();
        let (template, [first, second]) = world.push();
        let copied = world.copy(&template);
        let (first, second) = (copied[&first], copied[&second]);
        let Shape::Ref(Mutability::Exclusive, option) = *world.shape(first) else {
            panic!("the copy of `&mut Option<T>` is a reference");
        };
        assert_eq!(*world.shape(option), Shape::Option(second));
        assert!(!world.live.get(world.live.find(second)).state.is_decided());
    }

    #[test]
    fn each_use_makes_its_own_copy() {
        let mut world = World::new();
        let (template, [_, t]) = world.push();
        let at_first_use = world.copy(&template)[&t];
        let at_second_use = world.copy(&template)[&t];
        assert_ne!(world.live.find(at_first_use), world.live.find(at_second_use));
    }

    /// `|x| last = Some(x)`, whose text writes into the capture `last`.
    /// The classes at the lower end of the constraints whose upper end is `upper`,
    /// each with the items that recorded those constraints.
    fn entering<K>(classes: &Classes<K>, upper: K::Id) -> BTreeMap<K::Id, BTreeSet<ItemId>>
    where
        K: ClassKind,
    {
        let mut found = BTreeMap::<K::Id, BTreeSet<ItemId>>::new();
        for (constraint, at) in classes.constraints() {
            if classes.find(constraint.upper()) == classes.find(upper) {
                found.entry(classes.find(constraint.lower()).id()).or_default().insert(*at);
            }
        }
        found
    }

    fn remember(world: &mut World) -> (Classes<Template>, TemplateId, TemplateId, ItemId) {
        let root = world.scopes.root();
        let mut template = Classes::<Template>::default();
        let last = template.undecided(root);
        let x = template.undecided(root);
        let wrote = world.items.push("Some(x)");
        let some = template.decided(root, Shape::Option(x), BTreeSet::from([wrote]));
        let meets = world.items.push("last = Some(x)");
        template.constrain(Bound::Enter { lower: some, upper: last }.into(), meets);
        (template, last, some, meets)
    }

    #[test]
    fn a_template_is_not_changed_by_its_copies() {
        let mut world = World::new();
        let (template, stand_in, some, meets) = remember(&mut world);
        let before = template.clone();
        world.copy(&template);
        assert_eq!(template.verify(), Ok(()));
        assert_eq!(
            entering(&template, stand_in),
            BTreeMap::from([(template.find(some).id(), BTreeSet::from([meets]))])
        );
        assert_eq!(format!("{template:?}"), format!("{before:?}"));
    }

    #[test]
    fn a_bound_between_two_copies_is_recorded_between_them() {
        let mut world = World::new();
        let root = world.scopes.root();
        let mut template = Classes::<Template>::default();
        let a = template.undecided(root);
        let b = template.undecided(root);
        let at = world.items.push("a into b");
        template.constrain(Bound::Change { lower: a, upper: b }.into(), at);
        let copied = world.copy(&template);
        let (a, b) = (copied[&a], copied[&b]);
        assert_eq!(world.live.constraints().cloned().collect::<BTreeSet<_>>(), BTreeSet::from([(Bound::Change { lower: a, upper: b }.into(), at)]));
    }

    #[test]
    fn a_template_copies_into_another_template() {
        let mut world = World::new();
        let root = world.scopes.root();
        // A call of `remember` within the text of an enclosing body: the copy
        // is a template too.
        let (inner, stand_in, some, meets) = remember(&mut world);
        let mut enclosing = Classes::<Template>::default();
        let copied = enclosing.copy_from(&inner, root);
        assert_eq!(enclosing.verify(), Ok(()));
        let some = enclosing.find(copied[&some]).id();
        assert_eq!(entering(&enclosing, copied[&stand_in]), BTreeMap::from([(some, BTreeSet::from([meets]))]));
    }
}
