//! Templates: a signature written as classes ([`Classes::write`]), and a body of
//! the store made anew where it lies ([`anew`]).

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::constraint::ConstraintOf;
use crate::primitive::class::{Class, State};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{Shape, Ty};
use crate::primitive::{BrandId, ClassId, ClosureId, ItemId, ScopeId};

use super::Store;
use super::classes::{Classes, Slot};
use super::closures::{Closure, Copy};

impl<C> Classes<C>
where
    C: ConstraintOf,
{
    /// Writes `ty` as classes in `scope`, by `by`: each node an external class,
    /// and each hole an undecided one.
    ///
    /// `written` holds the classes already written for one signature, so that
    /// one type, a hole included, is one class wherever the signature writes
    /// it.
    pub fn write(&mut self, scope: ScopeId, ty: &Ty, by: ItemId, written: &mut BTreeMap<Ty, ClassId>) -> ClassId {
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
    fn copy_within(&mut self, placed: &BTreeMap<ScopeId, ScopeId>, brands: &BTreeMap<BrandId, BrandId>, named: &BTreeMap<ClosureId, ClosureId>) -> BTreeMap<ClassId, ClassId> {
        let lying_there: Vec<(ClassId, Class)> = self
            .slots
            .iter()
            .filter_map(|(id, slot)| match slot {
                Slot::Root(class) if placed.contains_key(&class.scope) => Some((id, class.clone())),
                Slot::Root(_) | Slot::Link(_) => None,
            })
            .collect();
        let merged_into_one: Vec<ClassId> = self.slots.iter().filter_map(|(id, slot)| matches!(slot, Slot::Link(_)).then_some(id)).collect();

        let mut copies = BTreeMap::new();
        for (id, class) in &lying_there {
            copies.insert(*id, self.undecided(placed[&class.scope]));
        }
        for (id, class) in lying_there {
            let scope = placed[&class.scope];
            let Class { state, connections, .. } = class.renamed(|held| copies.get(&self.find(held).id()).copied().unwrap_or(held));
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

        let of_a_copied_class: Vec<(C, ItemId)> = self
            .constraints()
            .filter(|(constraint, _)| [constraint.lower(), constraint.upper()].iter().any(|end| copies.contains_key(end)))
            .cloned()
            .collect();
        for (constraint, by) in of_a_copied_class {
            let copied = constraint.renamed(|held| copies.get(&held).copied().unwrap_or(held));
            self.constrain(copied, by);
        }
        copies
    }
}

/// A body made anew: the scope the copy lies in, and every id of a class of the
/// body with the class it became.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Anew {
    pub scope: ScopeId,
    pub classes: BTreeMap<ClassId, ClassId>,
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
pub fn anew<C>(store: &mut Store<C>, body: ScopeId, within: ScopeId) -> Anew
where
    C: ConstraintOf,
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
    let within_it: Vec<(ClosureId, Closure)> = closures
        .iter()
        .filter(|(_, closure)| closure.body != body && scopes.lies_within(closure.body, body))
        .map(|(id, closure)| (id, closure.clone()))
        .collect();
    let mut named = BTreeMap::new();
    for (id, closure) in &within_it {
        let becomes = place(scopes, &mut placed, body, within, closure.body);
        named.insert(*id, closures.name(Closure { body: becomes, ..closure.clone() }));
    }
    let said: Vec<(ScopeId, Copy)> = closures
        .copies()
        .filter(|(of, _)| *of != body && scopes.lies_within(*of, body))
        .map(|(of, copy)| (of, copy.clone()))
        .collect();
    for (of, _) in &said {
        place(scopes, &mut placed, body, within, *of);
    }

    let copies = classes.copy_within(&placed, &made, &named);
    let became = |class: &ClassId| copies.get(class).copied().unwrap_or(*class);
    for (id, closure) in within_it {
        let anew = closures.get_mut(named[&id]);
        anew.parameters = closure.parameters.iter().map(became).collect();
        anew.result = became(&closure.result);
        anew.makes = closure.makes.iter().map(became).collect();
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
    use crate::primitive::ClassId;
    use crate::primitive::Name;
    use crate::primitive::constraint::{Bound, Constraint};
    use crate::primitive::id::Arena;
    use crate::primitive::class::Side;
    use crate::primitive::ty::{Mutability, PartKey};

    fn node(shape: Shape<Ty>) -> Ty {
        Ty::Node(Box::new(shape))
    }

    /// `push<T>(&mut Option<T>, T)` written as the body `body`, with its two
    /// parameters.
    fn push(store: &mut Store<Constraint>, body: ScopeId, items: &mut Arena<ItemId, &'static str>) -> [ClassId; 2] {
        let declared = items.push("extern fn push");
        let t = Ty::Hole(Name::new("T"));
        let first = node(Shape::Ref(Mutability::Exclusive, node(Shape::Option(t.clone()))));
        let mut written = BTreeMap::new();
        let first = store.classes.write(body, &first, declared, &mut written);
        let second = store.classes.write(body, &t, declared, &mut written);
        [first, second]
    }

    #[test]
    fn a_signature_writes_one_class_for_each_type_it_writes() {
        let mut store = Store::<Constraint>::default();
        let mut items = Arena::default();
        let body = store.scopes.child(store.scopes.root());
        let [first, second] = push(&mut store, body, &mut items);
        let Shape::Ref(Mutability::Exclusive, option) = shape_in(&store, first) else {
            panic!("the first parameter is `&mut Option<T>`");
        };
        assert_eq!(shape_in(&store, option), Shape::Option(second));
        assert!(!store.classes.is_decided(second));
        assert!(store.classes.is_external(first));
        assert_eq!(store.classes.verify(), Ok(()));
    }

    #[test]
    fn a_signature_made_anew_keeps_what_the_declaration_writes_external_and_a_hole_free() {
        let mut store = Store::<Constraint>::default();
        let mut items = Arena::default();
        let root = store.scopes.root();
        let body = store.scopes.child(root);
        let [first, second] = push(&mut store, body, &mut items);
        let (one, other) = (anew(&mut store, body, root), anew(&mut store, body, root));
        assert!(store.classes.is_external(one.classes[&first]));
        assert!(!store.classes.is_decided(one.classes[&second]));
        assert_ne!(one.classes[&second], other.classes[&second]);
        let Shape::Ref(Mutability::Exclusive, option) = shape_in(&store, one.classes[&first]) else {
            panic!("the copy of `&mut Option<T>` is a reference");
        };
        assert_eq!(shape_in(&store, option), Shape::Option(one.classes[&second]));
        assert_eq!(store.classes.verify(), Ok(()));
    }

    /// A copy is written whole and not through the naming of a part, so the
    /// owners are told of it there.
    #[test]
    fn a_copy_of_an_undecided_class_owns_the_copies_of_its_parts() {
        let mut store = Store::<Constraint>::default();
        let mut items: Arena<ItemId, &'static str> = Arena::default();
        let root = store.scopes.root();
        let body = store.scopes.child(root);
        let owner = store.classes.undecided(body);
        let part = store
            .classes
            .part(owner, PartKey::Field(Name::new("x")), Side::Upper, items.push("a.x"))
            .expect("an undecided class names any part");
        let made = anew(&mut store, body, root);
        assert_eq!(store.classes.owned_by(made.classes[&owner]), [made.classes[&part]]);
        assert!(store.classes.held().contains(&made.classes[&part]));
        assert_eq!(store.classes.verify(), Ok(()));
    }

    // A body of the store being solved, made anew where it lies.

    fn shape_in(store: &Store<Constraint>, class: ClassId) -> Shape<ClassId> {
        let State::Decided(decided) = &store.classes.get(store.classes.find(class)).state else {
            panic!("{class:?} is decided");
        };
        decided.shape.clone()
    }

    fn scope_in(store: &Store<Constraint>, class: ClassId) -> ScopeId {
        store.classes.get(store.classes.find(class)).scope
    }

    fn enters(lower: ClassId, upper: ClassId) -> Constraint {
        Bound::Enter { lower, upper }.into()
    }

    /// `|x| { a = Some(x); x }`, `a` lying outside the body: the body's own
    /// classes are made anew, and what its constraints name outside it is the
    /// same class in the copy.
    #[test]
    fn a_body_made_anew_makes_its_classes_anew_and_keeps_what_lies_outside_it() {
        let mut store = Store::<Constraint>::default();
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
        let mut store = Store::<Constraint>::default();
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
        let mut store = Store::<Constraint>::default();
        let mut items: Arena<ItemId, &'static str> = Arena::default();
        let root = store.scopes.root();
        let body = store.scopes.child(root);
        let by = items.push("open()");
        let (of_the_body, of_outside) = (store.brands.push(by), store.brands.push(by));
        let branded = store.classes.decided(body, Shape::Brand(of_the_body), BTreeSet::from([by]));
        let kept = store.classes.decided(root, Shape::Brand(of_outside), BTreeSet::from([by]));
        let holder = store.classes.decided(body, Shape::Tuple(vec![branded, kept]), BTreeSet::from([by]));

        let brand_of = |store: &Store<Constraint>, class| match shape_in(store, class) {
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
