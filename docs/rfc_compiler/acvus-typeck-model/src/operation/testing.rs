//! What the tests of these operations write programs with.

use std::collections::BTreeSet;

use crate::primitive::bound::Bound;
use crate::primitive::class::Live;
use crate::primitive::id::Arena;
use crate::primitive::scope::Scopes;
use crate::primitive::decl::Conversion;
use crate::primitive::ty::{Mutability, Scalar, Shape, Ty};
use crate::primitive::{ClassId, ItemId, Name};
use crate::store::classes::Classes;

use super::Outcome;
use super::bound::shape_of;
use super::change::Conversions;
use super::settle::Settled;

pub(super) const SHARED: Mutability = Mutability::Shared;
pub(super) const EXCLUSIVE: Mutability = Mutability::Exclusive;
pub(super) const U8: Shape<ClassId> = Shape::Scalar(Scalar::U8);

pub(super) struct World {
    pub(super) classes: Classes<Live>,
    pub(super) scopes: Scopes,
    pub(super) items: Arena<ItemId, &'static str>,
    pub(super) outcome: Outcome<Live>,
    pub(super) conversions: Conversions,
    /// The classes made here, in the order settling visits them, ahead of any
    /// other class.
    pub(super) open: Vec<ClassId>,
}

impl World {
    pub(super) fn new() -> Self {
        Self {
            classes: Classes::default(),
            scopes: Scopes::new(),
            items: Arena::default(),
            outcome: Outcome::default(),
            conversions: Conversions::default(),
            open: Vec::new(),
        }
    }

    pub(super) fn undecided(&mut self) -> ClassId {
        let class = self.classes.undecided(self.scopes.root());
        self.open.push(class);
        class
    }

    /// The class of an expression, whose lower bound is the head the text
    /// writes.
    pub(super) fn text(&mut self, shape: Shape<ClassId>) -> ClassId {
        let by = self.items.push("written");
        let written = self.classes.decided(self.scopes.root(), shape, BTreeSet::from([by]));
        let class = self.undecided();
        self.enter(written, class);
        class
    }

    pub(super) fn external(&mut self, shape: Shape<ClassId>) -> ClassId {
        let by = self.items.push("declared");
        self.classes.external(self.scopes.root(), shape, BTreeSet::from([by]))
    }

    pub(super) fn enter(&mut self, lower: ClassId, upper: ClassId) -> ItemId {
        self.record(Bound::Enter { lower, upper })
    }

    pub(super) fn record(&mut self, bound: Bound<Live>) -> ItemId {
        let at = self.items.push("bound");
        self.outcome.bounds.extend(self.classes.bound(bound, at));
        at
    }

    /// `&place` or `&mut place`: the class of the expression, whose lower bound is
    /// the reference the text writes with `place` as its target, and the item by
    /// which that reference enters it.
    pub(super) fn lend(&mut self, place: ClassId, mutability: Mutability) -> (ClassId, ItemId) {
        let by = self.items.push("written");
        let written = self
            .classes
            .decided(self.scopes.root(), Shape::Ref(mutability, place), BTreeSet::from([by]));
        let class = self.undecided();
        let at = self.enter(written, class);
        (class, at)
    }

    /// Declares the conversion `name: from -> to`.
    pub(super) fn declare(&mut self, name: &str, from: Ty, to: Ty) {
        let by = self.items.push("declared");
        self.conversions
            .declare(Name::new(name), &Conversion { from, to }, &self.scopes, by)
            .expect("the conversion writes a class node at both its ends");
    }

    /// `argument` passed to a parameter of the type `upper`: the argument's class
    /// enters the boundary's, which goes to the parameter's (O2).
    pub(super) fn pass(&mut self, argument: ClassId, upper: ClassId) -> (ClassId, ItemId) {
        let boundary = self.meets(argument);
        let at = self.record(Bound::Change {
            lower: boundary,
            upper,
        });
        (boundary, at)
    }

    /// `let name = lower`, or a branch of an `if`.
    pub(super) fn meets(&mut self, lower: ClassId) -> ClassId {
        let place = self.undecided();
        self.enter(lower, place);
        place
    }

    pub(super) fn decide(&mut self, class: ClassId, shape: Shape<ClassId>) {
        let at = self.items.push("decision");
        self.outcome
            .decide(&mut self.classes, &self.scopes, class, shape, BTreeSet::from([at]), at);
    }

    fn visiting(open: &[ClassId], classes: &Classes<Live>) -> Vec<ClassId> {
        open.iter()
            .copied()
            .chain(classes.roots().map(|root| root.id()))
            .collect()
    }

    /// The two rungs that wait for no fixpoint of what is above them: connecting
    /// structure, and deciding what its lower bounds have given.
    pub(super) fn settle(&mut self) {
        let at = self.items.push("settling");
        let open = self.open.clone();
        self.outcome
            .settle_lower(&mut self.classes, &self.scopes, &self.conversions, at, &|classes: &Classes<Live>| {
                Self::visiting(&open, classes)
            });
    }

    /// Every rung there is.
    pub(super) fn settle_fully(&mut self) -> Settled {
        let at = self.items.push("settling");
        let open = self.open.clone();
        self.outcome
            .settle_visiting(&mut self.classes, &self.scopes, &self.conversions, at, &|classes: &Classes<Live>| {
                Self::visiting(&open, classes)
            })
    }

    pub(super) fn shape(&self, class: ClassId) -> Shape<ClassId> {
        shape_of(&self.classes, class).expect("the class is decided").clone()
    }

    pub(super) fn same(&self, a: ClassId, b: ClassId) -> bool {
        self.classes.find(a) == self.classes.find(b)
    }

    pub(super) fn target(&self, class: ClassId) -> (Mutability, ClassId) {
        let Shape::Ref(mutability, target) = self.shape(class) else {
            panic!("{class:?} is a reference");
        };
        (mutability, target)
    }

    pub(super) fn no_refusals(&self) {
        assert_eq!(self.outcome.refusals, Vec::new());
    }
}

impl Drop for World {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert_eq!(self.classes.verify(), Ok(()));
        }
    }
}
