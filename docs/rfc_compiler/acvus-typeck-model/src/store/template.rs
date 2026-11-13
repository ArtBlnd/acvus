//! Templates: a store of classes written once from a type or a text, and
//! copied whole at each use; and a body of the store being solved, made anew
//! where it lies ([`anew`]).

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::constraint::{ConstraintInto, ConstraintOf};
use crate::primitive::class::{Class, ClassKind, State};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{Shape, Ty};
use crate::primitive::{BrandId, ClosureId, ItemId, ScopeId};

use super::Store;
use super::classes::{Classes, Slot};
use super::closures::{Closure, Copy};

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

    /// Copies each class that lies in a scope `placed` names into the scope it
    /// gives for that one, and gives every id of such a class with the class it
    /// became.
    ///
    /// Each is made anew holding what its original holds, an id of a class
    /// copied given as its copy and any other id as it is, a brand it has as
    /// its head given as `brands` gives it, and a closure its head names that
    /// `named` holds given as `named` gives it. Each constraint with an end
    /// copied is recorded between what its ends became. No class that was there
    /// is changed.
    fn copy_within(&mut self, placed: &BTreeMap<ScopeId, ScopeId>, brands: &BTreeMap<BrandId, BrandId>, named: &BTreeMap<ClosureId, ClosureId>) -> BTreeMap<K::Id, K::Id>
    where
        K::Constraint: ConstraintInto<K, K>,
    {
        let lying_there: Vec<(K::Id, Class<K>)> = self
            .slots
            .iter()
            .filter_map(|(id, slot)| match slot {
                Slot::Root(class) if placed.contains_key(&class.scope) => Some((id, class.clone())),
                Slot::Root(_) | Slot::Link(_) => None,
            })
            .collect();
        let merged_into_one: Vec<K::Id> = self.slots.iter().filter_map(|(id, slot)| matches!(slot, Slot::Link(_)).then_some(id)).collect();

        let mut copies = BTreeMap::new();
        for (id, class) in &lying_there {
            copies.insert(*id, self.undecided(placed[&class.scope]));
        }
        for (id, class) in lying_there {
            let scope = placed[&class.scope];
            let Class { state, connections, .. } = class.translate::<K, _>(|held| copies.get(&self.find(held).id()).copied().unwrap_or(held));
            let state = match state {
                State::Decided(mut decided) => {
                    match &mut decided.shape {
                        Shape::Brand(brand) => *brand = brands[brand],
                        Shape::FnValue(closures) => *closures = closures.iter().map(|closure| named.get(closure).copied().unwrap_or(*closure)).collect(),
                        _ => {}
                    }
                    State::Decided(decided)
                }
                undecided @ State::Undecided(_) => undecided,
            };
            *self.slots.get_mut(copies[&id]) = Slot::Root(Class { scope, state, connections });
            self.owners.written(copies[&id], self.owned_by(copies[&id]));
        }
        for id in merged_into_one {
            if let Some(copy) = copies.get(&self.find(id).id()).copied() {
                copies.insert(id, copy);
            }
        }

        let of_a_copied_class: Vec<(K::Constraint, ItemId)> = self
            .constraints()
            .filter(|(constraint, _)| [constraint.lower(), constraint.upper()].iter().any(|end| copies.contains_key(end)))
            .cloned()
            .collect();
        for (constraint, by) in of_a_copied_class {
            let copied = constraint.translate(|held| copies.get(&held).copied().unwrap_or(held));
            self.constrain(copied, by);
        }
        copies
    }
}

/// A body made anew: the scope the copy lies in, and every id of a class of the
/// body with the class it became.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Anew<K>
where
    K: ClassKind,
{
    pub scope: ScopeId,
    pub classes: BTreeMap<K::Id, K::Id>,
}

/// Makes the body `body` of `store` anew within the scope `within`.
///
/// | Of the body | In what is made |
/// |---|---|
/// | a class that lies in it, at any depth | a class made anew, holding what the original holds |
/// | a class that lies outside it | that same class |
/// | a constraint with an end in it | one recorded between what its ends became |
/// | a scope within it | a scope within the copy's, in the same place |
/// | a brand a class of it has as its head | a brand made anew, by the same item |
/// | a closure whose body lies within it | a closure named anew, of what its body became |
/// | a copy said of a scope within it | the same said of what that scope became |
///
/// The body is not changed. The closure the body itself is of is not named
/// anew, and nothing is said of the copy's own scope: whoever makes the copy
/// says what it is a copy of.
pub fn anew<K>(store: &mut Store<K>, body: ScopeId, within: ScopeId) -> Anew<K>
where
    K: ClassKind,
    K::Constraint: ConstraintInto<K, K>,
{
    let Store { classes, scopes, closures, brands } = store;
    let mut placed = BTreeMap::new();
    let mut made = BTreeMap::new();
    for root in classes.roots() {
        let class = classes.get(root);
        if !scopes.lies_within(class.scope, body) {
            continue;
        }
        place(scopes, &mut placed, body, within, class.scope);
        if let State::Decided(decided) = &class.state
            && let Shape::Brand(brand) = decided.shape
        {
            made.entry(brand).or_insert_with(|| brands.push(*brands.get(brand)));
        }
    }
    let scope = place(scopes, &mut placed, body, within, body);

    // Each closure of a body within is named anew, of the classes it was of:
    // they are given as what they became once the classes are copied.
    let within_it: Vec<(ClosureId, Closure<K>)> = closures
        .iter()
        .filter(|(_, closure)| closure.body != body && scopes.lies_within(closure.body, body))
        .map(|(id, closure)| (id, closure.clone()))
        .collect();
    let mut named = BTreeMap::new();
    for (id, closure) in &within_it {
        let becomes = place(scopes, &mut placed, body, within, closure.body);
        named.insert(*id, closures.name(Closure { body: becomes, ..closure.clone() }));
    }
    let said: Vec<(ScopeId, Copy<K>)> = closures
        .copies()
        .filter(|(of, _)| *of != body && scopes.lies_within(*of, body))
        .map(|(of, copy)| (of, copy.clone()))
        .collect();
    for (of, _) in &said {
        place(scopes, &mut placed, body, within, *of);
    }

    let copies = classes.copy_within(&placed, &made, &named);
    let became = |class: &K::Id| copies.get(class).copied().unwrap_or(*class);
    for (id, closure) in within_it {
        let anew = closures.get_mut(named[&id]);
        anew.parameters = closure.parameters.iter().map(became).collect();
        anew.result = became(&closure.result);
    }
    for (of, copy) in said {
        let copy = Copy {
            closure: named.get(&copy.closure).copied().unwrap_or(copy.closure),
            parameters: copy.parameters.iter().map(became).collect(),
            result: became(&copy.result),
        };
        closures.copied(placed[&of], copy);
    }
    Anew { scope, classes: copies }
}

/// The scope that `scope`, which lies within `body`, is placed as where `body`
/// is made anew within `within`: made where `placed` does not hold it yet, after
/// the one the scope it lies in is placed as.
fn place(scopes: &mut Scopes, placed: &mut BTreeMap<ScopeId, ScopeId>, body: ScopeId, within: ScopeId, scope: ScopeId) -> ScopeId {
    if let Some(known) = placed.get(&scope) {
        return *known;
    }
    let lies_in = match scope == body {
        true => within,
        false => {
            let holding = scopes.parent(scope).expect("a scope within a body lies in a scope");
            place(scopes, placed, body, within, holding)
        }
    };
    let made = scopes.child(lies_in);
    placed.insert(scope, made);
    made
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::Name;
    use crate::primitive::constraint::{Bound, Constraint, ConstraintOf};
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

    // A body of the store being solved, made anew where it lies.

    fn shape_in(store: &Store<Live>, class: ClassId) -> Shape<ClassId> {
        let State::Decided(decided) = &store.classes.get(store.classes.find(class)).state else {
            panic!("{class:?} is decided");
        };
        decided.shape.clone()
    }

    fn scope_in(store: &Store<Live>, class: ClassId) -> ScopeId {
        store.classes.get(store.classes.find(class)).scope
    }

    fn enters(lower: ClassId, upper: ClassId) -> Constraint<Live> {
        Bound::Enter { lower, upper }.into()
    }

    /// `|x| { a = Some(x); x }`, `a` lying outside the body: the body's own
    /// classes are made anew, and what its constraints name outside it is the
    /// same class in the copy.
    #[test]
    fn a_body_made_anew_makes_its_classes_anew_and_keeps_what_lies_outside_it() {
        let mut store = Store::<Live>::default();
        let mut items: Arena<ItemId, &'static str> = Arena::default();
        let root = store.scopes.root();
        let body = store.scopes.child(root);
        let outside = store.classes.undecided(root);
        let (parameter, result) = (store.classes.undecided(body), store.classes.undecided(body));
        let written = store.classes.decided(body, Shape::Option(parameter), BTreeSet::from([items.push("Some(x)")]));
        let (into_outside, into_result) = (items.push("a = Some(x)"), items.push("x"));
        store.classes.constrain(enters(written, outside), into_outside);
        store.classes.constrain(enters(parameter, result), into_result);
        let before: BTreeSet<_> = store.classes.constraints().cloned().collect();

        let made = anew(&mut store, body, root);

        assert_ne!(made.classes[&parameter], parameter);
        assert!(!made.classes.contains_key(&outside));
        assert_eq!(shape_in(&store, made.classes[&written]), Shape::Option(made.classes[&parameter]));
        assert_eq!(store.scopes.parent(made.scope), Some(root));
        assert_eq!(scope_in(&store, made.classes[&parameter]), made.scope);

        let after: BTreeSet<_> = store.classes.constraints().cloned().collect();
        let recorded: BTreeSet<_> = after.difference(&before).cloned().collect();
        let expected = BTreeSet::from([
            (enters(made.classes[&written], outside), into_outside),
            (enters(made.classes[&parameter], made.classes[&result]), into_result),
        ]);
        assert_eq!(recorded, expected);

        // The body is as it was.
        assert_eq!(shape_in(&store, written), Shape::Option(parameter));
        assert_eq!(scope_in(&store, parameter), body);
        assert_eq!(store.classes.verify(), Ok(()));
    }

    #[test]
    fn a_scope_within_a_body_is_placed_within_the_copy_and_ids_of_one_class_are_one_copy() {
        let mut store = Store::<Live>::default();
        let mut items: Arena<ItemId, &'static str> = Arena::default();
        let root = store.scopes.root();
        let body = store.scopes.child(root);
        let within = store.scopes.child(body);
        let nested = store.classes.undecided(within);
        let (first, second) = (store.classes.undecided(body), store.classes.undecided(body));
        store.classes.unify(&store.scopes, first, second, items.push("one class"));

        let made = anew(&mut store, body, root);

        assert_eq!(store.scopes.parent(scope_in(&store, made.classes[&nested])), Some(made.scope));
        assert_eq!(made.classes[&first], made.classes[&second]);
        assert_ne!(store.classes.find(made.classes[&first]), store.classes.find(first));
        assert_eq!(store.classes.verify(), Ok(()));
    }

    #[test]
    fn each_copy_of_a_body_makes_the_brands_of_the_body_anew() {
        let mut store = Store::<Live>::default();
        let mut items: Arena<ItemId, &'static str> = Arena::default();
        let root = store.scopes.root();
        let body = store.scopes.child(root);
        let by = items.push("open()");
        let (of_the_body, of_outside) = (store.brands.push(by), store.brands.push(by));
        let branded = store.classes.decided(body, Shape::Brand(of_the_body), BTreeSet::from([by]));
        let kept = store.classes.decided(root, Shape::Brand(of_outside), BTreeSet::from([by]));
        let holder = store.classes.decided(body, Shape::Tuple(vec![branded, kept]), BTreeSet::from([by]));

        let brand_of = |store: &Store<Live>, class| match shape_in(store, class) {
            Shape::Brand(brand) => brand,
            other => panic!("{other:?} is a brand"),
        };
        let first = anew(&mut store, body, root);
        let second = anew(&mut store, body, root);

        let (in_first, in_second) = (brand_of(&store, first.classes[&branded]), brand_of(&store, second.classes[&branded]));
        assert_ne!(in_first, of_the_body);
        assert_ne!(in_first, in_second);
        assert_eq!(*store.brands.get(in_first), by);
        assert_eq!(shape_in(&store, first.classes[&holder]), Shape::Tuple(vec![first.classes[&branded], kept]));
        assert_eq!(brand_of(&store, kept), of_outside);
        assert_eq!(store.classes.verify(), Ok(()));
    }
}
