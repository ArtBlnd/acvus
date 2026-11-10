//! Templates: a store of classes written once from a type or a text, and
//! copied whole at each use.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::bound::BoundInto;
use crate::primitive::class::{Class, ClassKind};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::Ty;
use crate::primitive::{ItemId, ScopeId};

use super::classes::{Classes, Root, Slot, Unified};

/// What becomes of a class of a template in a copy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fate<J> {
    /// It is made anew.
    Anew,
    /// It stands for this class of the store copied into: it is made anew and
    /// then made one with it, so that whatever the template records on it is
    /// recorded on this class.
    Keep(J),
}

/// A template copied.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Copied<S, K>
where
    S: ClassKind,
    K: ClassKind,
{
    /// Every id of the template, with the class it became.
    pub classes: BTreeMap<S::Id, K::Id>,
    /// What making the classes kept one with their copies left.
    pub unified: Unified<K>,
}

impl<K> Classes<K>
where
    K: ClassKind,
{
    /// Writes `ty` as classes in `scope`, by `by`: each node an external class,
    /// and each hole an undecided one (O13).
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

    /// Copies every class of `template` into this store, in `scope`, by `at`,
    /// as `fate` gives each.
    ///
    /// Each class is made anew holding what its original holds, every id in
    /// it given as the class its original became, and each bound the template
    /// holds is recorded between the copies of its ends. Then each class kept is
    /// made one with its copy. The template is not changed.
    pub fn copy_from<S, F>(
        &mut self,
        template: &Classes<S>,
        scopes: &Scopes,
        scope: ScopeId,
        at: ItemId,
        mut fate: F,
    ) -> Copied<S, K>
    where
        S: ClassKind,
        S::Bound: BoundInto<S, K>,
        F: FnMut(Root<'_, S>) -> Fate<K::Id>,
    {
        let mut copies = BTreeMap::new();
        let mut kept = Vec::new();
        for (id, slot) in template.slots.iter() {
            let Slot::Root(_) = slot else {
                continue;
            };
            if let Fate::Keep(target) = fate(template.find(id)) {
                kept.push((id, target));
            }
            copies.insert(id, self.undecided(scope));
        }

        for (id, slot) in template.slots.iter() {
            let Slot::Root(class) = slot else {
                continue;
            };
            let Class {
                state, held, connections, ..
            } = class.clone().translate::<K, _>(|held| copies[&template.find(held).id()]);
            *self.slots.get_mut(copies[&id]) = Slot::Root(Class {
                scope,
                state,
                held,
                connections,
            });
        }
        for (id, slot) in template.slots.iter() {
            if let Slot::Link(_) = slot {
                let copy = copies[&template.find(id).id()];
                copies.insert(id, copy);
            }
        }
        for (bound, by) in &template.bounds {
            let copied = bound.clone().translate(|held| copies[&template.find(held).id()]);
            self.bound(copied, *by);
        }

        let mut unified = Unified::default();
        for (id, target) in kept {
            unified.absorb(self.unify(scopes, copies[&id], target, at));
            copies.insert(id, target);
        }
        Copied {
            classes: copies,
            unified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::Name;
    use crate::primitive::bound::{Bound, BoundOf};
    use crate::primitive::class::{Live, Side, State, Template};
    use crate::primitive::id::Arena;
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

        fn copy<F>(&mut self, template: &Classes<Template>, fate: F) -> Copied<Template, Live>
        where
            F: FnMut(Root<'_, Template>) -> Fate<ClassId>,
        {
            let at = self.items.push("use");
            self.live.copy_from(template, &self.scopes, self.scopes.root(), at, fate)
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
        let copied = world.copy(&template, |_| Fate::Anew);
        assert!(world.live.is_external(copied.classes[&first]));
        assert!(!world.live.is_external(copied.classes[&second]));
    }

    #[test]
    fn a_copy_makes_every_class_anew_and_keeps_one_class_one() {
        let mut world = World::new();
        let (template, [first, second]) = world.push();
        let copied = world.copy(&template, |_| Fate::Anew);
        assert_eq!(copied.unified, Unified::default());
        let (first, second) = (copied.classes[&first], copied.classes[&second]);
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
        let at_first_use = world.copy(&template, |_| Fate::Anew).classes[&t];
        let at_second_use = world.copy(&template, |_| Fate::Anew).classes[&t];
        assert_ne!(world.live.find(at_first_use), world.live.find(at_second_use));
    }

    /// `|x| last = Some(x)`, whose text writes into the capture `last`.
    /// The classes at the lower end of the bounds whose upper end is `upper`,
    /// each with the items that recorded those bounds.
    fn entering<K>(classes: &Classes<K>, upper: K::Id) -> BTreeMap<K::Id, BTreeSet<ItemId>>
    where
        K: ClassKind,
    {
        let mut found = BTreeMap::<K::Id, BTreeSet<ItemId>>::new();
        for (bound, at) in classes.bounds() {
            if classes.find(bound.upper()) == classes.find(upper) {
                found.entry(classes.find(bound.lower()).id()).or_default().insert(*at);
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
        template.bound(Bound::Enter { lower: some, upper: last }, meets);
        (template, last, some, meets)
    }

    #[test]
    fn a_class_kept_takes_what_the_template_records_on_it() {
        let mut world = World::new();
        let (template, stand_in, some, meets) = remember(&mut world);
        let last = world.live.undecided(world.scopes.root());

        let copied = world.copy(&template, |reached| {
            if reached.id() == stand_in { Fate::Keep(last) } else { Fate::Anew }
        });

        assert_eq!(copied.unified, Unified::default());
        assert_eq!(copied.classes[&stand_in], last);
        let some = world.live.find(copied.classes[&some]).id();
        assert_eq!(entering(&world.live, last), BTreeMap::from([(some, BTreeSet::from([meets]))]));
    }

    #[test]
    fn a_template_is_not_changed_by_its_copies() {
        let mut world = World::new();
        let (template, stand_in, some, meets) = remember(&mut world);
        let before = template.clone();
        let last = world.live.undecided(world.scopes.root());
        world.copy(&template, |reached| {
            if reached.id() == stand_in { Fate::Keep(last) } else { Fate::Anew }
        });
        assert_eq!(template.verify(), Ok(()));
        assert_eq!(
            entering(&template, stand_in),
            BTreeMap::from([(template.find(some).id(), BTreeSet::from([meets]))])
        );
        assert_eq!(format!("{template:?}"), format!("{before:?}"));
    }

    #[test]
    fn a_part_the_template_names_on_a_class_kept_is_named_on_it() {
        let mut world = World::new();
        let root = world.scopes.root();
        let mut template = Classes::<Template>::default();
        let stand_in = template.undecided(root);
        let named = world.items.push("last.x");
        let x = PartKey::Field(Name::new("x"));
        let field = template
            .part(stand_in, x.clone(), Side::Upper, named)
            .expect("an undecided class names any part");
        let last = world.live.undecided(root);

        let copied = world.copy(&template, |reached| {
            if reached.id() == stand_in { Fate::Keep(last) } else { Fate::Anew }
        });

        let last = world.live.find(last);
        assert_eq!(world.live.get(last).state.part(&x).map(|part| world.live.find(part)), Some(world.live.find(copied.classes[&field])));
    }

    #[test]
    fn a_bound_between_two_copies_is_recorded_between_them() {
        let mut world = World::new();
        let root = world.scopes.root();
        let mut template = Classes::<Template>::default();
        let a = template.undecided(root);
        let b = template.undecided(root);
        let at = world.items.push("a into b");
        template.bound(Bound::Change { lower: a, upper: b }, at);
        let copied = world.copy(&template, |_| Fate::Anew);
        let (a, b) = (copied.classes[&a], copied.classes[&b]);
        assert_eq!(world.live.bounds(), &BTreeSet::from([(Bound::Change { lower: a, upper: b }, at)]));
    }

    #[test]
    fn a_template_copies_into_another_template() {
        let mut world = World::new();
        let root = world.scopes.root();
        // A call of `remember` within the text of an enclosing body: the copy
        // is a template too, and its stand-in is a class of the enclosing one.
        let (inner, stand_in, some, meets) = remember(&mut world);
        let mut enclosing = Classes::<Template>::default();
        let last = enclosing.undecided(root);
        let at = world.items.push("remember(..)");
        let copied = enclosing.copy_from(&inner, &world.scopes, root, at, |reached| {
            if reached.id() == stand_in { Fate::Keep(last) } else { Fate::Anew }
        });
        assert_eq!(enclosing.verify(), Ok(()));
        let some = enclosing.find(copied.classes[&some]).id();
        assert_eq!(entering(&enclosing, last), BTreeMap::from([(some, BTreeSet::from([meets]))]));
    }
}
