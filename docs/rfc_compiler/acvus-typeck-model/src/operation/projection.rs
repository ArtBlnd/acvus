//! Projections: a constraint suspended on one class's head.
//!
//! `p.f` is recorded before `p` has a head, and gives it none (O7). It waits as
//! a part named on `p`. Once `p` has a head it branches on it:
//!
//! | Head | What the projection does |
//! |---|---|
//! | one with a place for the part | connects to that place |
//! | a reference | connects to the part of its target, and records a deref |
//! | any other | is refused |
//!
//! `*r` names the one part a reference has a place for, so it never derefs.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::{ClassKind, Part, Side, State, Unplaced};
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{PartKey, Shape};
use crate::store::classes::{Classes, Unified};

use super::Outcome;

impl<K> Outcome<K>
where
    K: ClassKind,
{
    /// The class of the part `key` of `operand`, named on `side` by `at`.
    ///
    /// Where `operand` is decided, the projection branches at once, as it does
    /// on release.
    pub fn project(
        &mut self,
        classes: &mut Classes<K>,
        operand: K::Id,
        key: PartKey,
        side: Side,
        at: ItemId,
    ) -> Result<K::Id, Refusal> {
        if let Some(part) = classes.part(operand, key.clone(), side, at) {
            return Ok(part);
        }
        if let Some(target) = target_of(classes, operand)
            && let Some(part) = classes.part(target, key, side, at)
        {
            self.adjustments.entry(at).or_default().push(Adjustment::Deref);
            return Ok(part);
        }
        Err(not_held(classes, operand, BTreeSet::from([at])))
    }

    /// Takes in what a union left, and releases each part it left with no
    /// place, along with those the unions this makes leave in turn.
    pub fn release(&mut self, classes: &mut Classes<K>, scopes: &Scopes, unified: Unified<K>) {
        let mut unplaced = self.take_in(unified);
        while let Some(Unplaced { class, key, part, at }) = unplaced.pop() {
            let named = target_of(classes, class).and_then(|target| name_again(classes, target, &key, &part));
            let mut projections: BTreeSet<ItemId> = part.lower.iter().chain(&part.upper).copied().collect();
            let Some(named) = named else {
                // A name only carried here leaves nothing to do. The class a
                // projection named it on holds that part too, and connects it
                // or refuses it there.
                if !projections.is_empty() {
                    projections.insert(at);
                    self.refusals.push(not_held(classes, class, projections));
                }
                continue;
            };
            for item in projections {
                self.adjustments.entry(item).or_default().push(Adjustment::Deref);
            }
            let unified = classes.unify(scopes, named, part.class, at);
            unplaced.extend(self.take_in(unified));
        }
    }
}

/// The target of `class`, where it is decided as a reference.
fn target_of<K>(classes: &Classes<K>, class: K::Id) -> Option<K::Id>
where
    K: ClassKind,
{
    match &classes.get(classes.find(class)).state {
        State::Decided(decided) => match decided.shape {
            Shape::Ref(_, target) => Some(target),
            _ => None,
        },
        State::Undecided(_) => None,
    }
}

/// Names `key` on `target` by every projection that named `part`, on the side
/// each named it. It gives none where `target` is decided with no place for it,
/// and where no projection named the part.
fn name_again<K>(classes: &mut Classes<K>, target: K::Id, key: &PartKey, part: &Part<K>) -> Option<K::Id>
where
    K: ClassKind,
{
    let mut named = None;
    for (side, items) in [(Side::Lower, &part.lower), (Side::Upper, &part.upper)] {
        for at in items {
            named = Some(classes.part(target, key.clone(), side, *at)?);
        }
    }
    named
}

/// The refusal of a part of `class`, naming `items` and what decided `class`
/// and, where it is a reference, its target.
fn not_held<K>(classes: &Classes<K>, class: K::Id, mut items: BTreeSet<ItemId>) -> Refusal
where
    K: ClassKind,
{
    let target = target_of(classes, class);
    for held in [Some(class), target].into_iter().flatten() {
        if let State::Decided(decided) = &classes.get(classes.find(held)).state {
            items.extend(&decided.by);
        }
    }
    Refusal {
        cause: Cause::NotHeld,
        items,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::primitive::adjustment::Adjustments;
    use crate::primitive::bound::Bound;
    use crate::primitive::class::Live;
    use crate::primitive::id::Arena;
    use crate::primitive::ty::{Mutability, Scalar};
    use crate::primitive::{ClassId, Name};

    struct World {
        classes: Classes<Live>,
        scopes: Scopes,
        items: Arena<ItemId, &'static str>,
        outcome: Outcome<Live>,
    }

    impl World {
        fn new() -> Self {
            Self {
                classes: Classes::default(),
                scopes: Scopes::new(),
                items: Arena::default(),
                outcome: Outcome::default(),
            }
        }

        fn undecided(&mut self) -> ClassId {
            self.classes.undecided(self.scopes.root())
        }

        fn written(&mut self, shape: Shape<ClassId>) -> (ClassId, ItemId) {
            let by = self.items.push("written");
            (self.classes.decided(self.scopes.root(), shape, BTreeSet::from([by])), by)
        }

        fn project(&mut self, operand: ClassId, key: PartKey) -> (ClassId, ItemId) {
            let at = self.items.push("projection");
            let part = self
                .outcome
                .project(&mut self.classes, operand, key, Side::Upper, at)
                .expect("the projection is held");
            (part, at)
        }

        /// Decides `class` as what `decided` holds, and releases what that
        /// leaves.
        fn decide(&mut self, class: ClassId, decided: ClassId) -> ItemId {
            let at = self.items.push("decision");
            let unified = self.classes.unify(&self.scopes, class, decided, at);
            self.outcome.release(&mut self.classes, &self.scopes, unified);
            at
        }

        fn same(&self, a: ClassId, b: ClassId) -> bool {
            self.classes.find(a) == self.classes.find(b)
        }

        fn object_with_x(&mut self, field: ClassId) -> ClassId {
            let fields = BTreeMap::from([(Name::new("x"), field)]);
            self.written(Shape::Object(fields)).0
        }
    }

    impl Drop for World {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(self.classes.verify(), Ok(()));
            }
        }
    }

    fn x() -> PartKey {
        PartKey::Field(Name::new("x"))
    }

    fn deref_of<I>(items: I) -> Adjustments
    where
        I: IntoIterator<Item = ItemId>,
    {
        items.into_iter().map(|item| (item, vec![Adjustment::Deref])).collect()
    }

    #[test]
    fn a_projection_gives_its_operand_no_head() {
        let mut world = World::new();
        let operand = world.undecided();
        world.project(operand, x());
        assert!(!world.classes.is_decided(operand));
        assert_eq!(world.outcome, Outcome::default());
    }

    #[test]
    fn a_part_with_a_place_connects_to_it_and_takes_no_adjustment() {
        let mut world = World::new();
        let operand = world.undecided();
        let (part, _) = world.project(operand, x());
        let field = world.undecided();
        let object = world.object_with_x(field);
        world.decide(operand, object);
        assert!(world.same(part, field));
        assert_eq!(world.outcome, Outcome::default());
    }

    #[test]
    fn a_part_on_a_reference_connects_to_its_target_with_one_deref() {
        for mutability in [Mutability::Shared, Mutability::Exclusive] {
            let mut world = World::new();
            let operand = world.undecided();
            let (part, at) = world.project(operand, x());
            let field = world.undecided();
            let object = world.object_with_x(field);
            let (reference, _) = world.written(Shape::Ref(mutability, object));
            world.decide(operand, reference);
            assert!(world.same(part, field));
            assert_eq!(world.outcome.adjustments, deref_of([at]));
            assert_eq!(world.outcome.refusals, Vec::new());
        }
    }

    #[test]
    fn a_projection_recorded_after_the_head_is_decided_comes_out_the_same() {
        let mut world = World::new();
        let operand = world.undecided();
        let field = world.undecided();
        let object = world.object_with_x(field);
        let (reference, _) = world.written(Shape::Ref(Mutability::Shared, object));
        world.decide(operand, reference);
        let (part, at) = world.project(operand, x());
        assert!(world.same(part, field));
        assert_eq!(world.outcome.adjustments, deref_of([at]));
    }

    #[test]
    fn a_deref_written_in_the_text_takes_no_adjustment() {
        let mut world = World::new();
        let operand = world.undecided();
        let (part, _) = world.project(operand, PartKey::Target);
        let target = world.undecided();
        let (reference, _) = world.written(Shape::Ref(Mutability::Shared, target));
        world.decide(operand, reference);
        assert!(world.same(part, target));
        assert_eq!(world.outcome, Outcome::default());
    }

    #[test]
    fn a_part_on_an_undecided_target_waits_there() {
        let mut world = World::new();
        let operand = world.undecided();
        let (part, at) = world.project(operand, x());
        let target = world.undecided();
        let (reference, _) = world.written(Shape::Ref(Mutability::Shared, target));
        world.decide(operand, reference);
        let named = world
            .classes
            .get(world.classes.find(target))
            .state
            .part(&x())
            .expect("the part waits on the target");
        assert!(world.same(named, part));
        assert_eq!(world.outcome.adjustments, deref_of([at]));

        let field = world.undecided();
        let object = world.object_with_x(field);
        world.decide(target, object);
        assert!(world.same(part, field));
        assert_eq!(world.outcome.adjustments, deref_of([at]));
    }

    #[test]
    fn two_projections_of_one_name_are_one_class_and_each_records_a_deref() {
        let mut world = World::new();
        let operand = world.undecided();
        let (first, first_at) = world.project(operand, x());
        let (second, second_at) = world.project(operand, x());
        assert_eq!(first, second);
        let field = world.undecided();
        let object = world.object_with_x(field);
        let (reference, _) = world.written(Shape::Ref(Mutability::Shared, object));
        world.decide(operand, reference);
        assert_eq!(world.outcome.adjustments, deref_of([first_at, second_at]));
    }

    /// `p.x.y`, where `p` is `&{x: &{y: _}}`: the release of the first deref
    /// leaves the second.
    #[test]
    fn a_chain_of_projections_is_released_in_turn() {
        let mut world = World::new();
        let p = world.undecided();
        let (x_part, x_at) = world.project(p, x());
        let y = PartKey::Field(Name::new("y"));
        let (y_part, y_at) = world.project(x_part, y);

        let field = world.undecided();
        let inner = world.written(Shape::Object(BTreeMap::from([(Name::new("y"), field)]))).0;
        let (inner_reference, _) = world.written(Shape::Ref(Mutability::Shared, inner));
        let outer = world.object_with_x(inner_reference);
        let (outer_reference, _) = world.written(Shape::Ref(Mutability::Shared, outer));
        world.decide(p, outer_reference);

        assert!(world.same(y_part, field));
        assert_eq!(world.outcome.adjustments, deref_of([x_at, y_at]));
        assert_eq!(world.outcome.refusals, Vec::new());
    }

    /// `r[i]`, where `r` is `&[T]`: the element goes to the target, and the
    /// index meets `usize` in a class of its own, which never sees `r`'s head.
    #[test]
    fn an_index_derefs_its_element_and_leaves_its_index_apart() {
        let mut world = World::new();
        let r = world.undecided();
        let (elem, elem_at) = world.project(r, PartKey::Elem);
        let i = world.undecided();
        let index = world.undecided();
        let (usize_class, _) = world.written(Shape::Scalar(Scalar::Usize));
        let usize_enters = Bound::Enter { lower: usize_class, upper: index };
        let i_enters = Bound::Enter { lower: i, upper: index };
        world.classes.bound(usize_enters, elem_at);
        world.classes.bound(i_enters, elem_at);

        let t = world.undecided();
        let (slice, _) = world.written(Shape::Slice(t));
        let (reference, _) = world.written(Shape::Ref(Mutability::Shared, slice));
        world.decide(r, reference);

        assert!(world.same(elem, t));
        assert_eq!(world.outcome.adjustments, deref_of([elem_at]));
        assert!(!world.classes.is_decided(index));
        assert!(!world.classes.is_decided(i));
        assert_eq!(world.classes.between_decided(), BTreeSet::new());
    }

    #[test]
    fn a_part_with_no_place_is_refused_naming_what_registered_it() {
        let mut world = World::new();
        let operand = world.undecided();
        let (_, named_at) = world.project(operand, x());
        let (scalar, written) = world.written(Shape::Scalar(Scalar::U8));
        let decided_at = world.decide(operand, scalar);
        assert_eq!(
            world.outcome.refusals,
            vec![Refusal {
                cause: Cause::NotHeld,
                items: BTreeSet::from([named_at, written, decided_at]),
            }]
        );
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// `let a; a.x = 0;`: nothing gives `a` a head, so it falls back to `!`,
    /// which holds no part.
    #[test]
    fn a_projection_on_a_class_that_falls_back_to_never_is_refused() {
        let mut world = World::new();
        let operand = world.undecided();
        let (_, named_at) = world.project(operand, x());
        let (never, fallback) = world.written(Shape::Never);
        let decided_at = world.decide(operand, never);
        assert_eq!(
            world.outcome.refusals,
            vec![Refusal {
                cause: Cause::NotHeld,
                items: BTreeSet::from([named_at, fallback, decided_at]),
            }]
        );
    }

    #[test]
    fn a_reference_whose_target_has_no_place_is_refused_and_records_no_deref() {
        let mut world = World::new();
        let operand = world.undecided();
        let (_, named_at) = world.project(operand, x());
        let (scalar, scalar_by) = world.written(Shape::Scalar(Scalar::U8));
        let (reference, reference_by) = world.written(Shape::Ref(Mutability::Shared, scalar));
        let decided_at = world.decide(operand, reference);
        assert_eq!(
            world.outcome.refusals,
            vec![Refusal {
                cause: Cause::NotHeld,
                items: BTreeSet::from([named_at, scalar_by, reference_by, decided_at]),
            }]
        );
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    #[test]
    fn a_projection_on_a_decided_class_with_no_place_is_refused_at_once() {
        let mut world = World::new();
        let (scalar, written) = world.written(Shape::Scalar(Scalar::U8));
        let at = world.items.push("projection");
        let refused = world.outcome.project(&mut world.classes, scalar, x(), Side::Upper, at);
        assert_eq!(
            refused,
            Err(Refusal {
                cause: Cause::NotHeld,
                items: BTreeSet::from([written, at]),
            })
        );
    }

    #[test]
    fn a_bound_a_release_gives_back_reaches_the_caller() {
        let mut world = World::new();
        let operand = world.undecided();
        let (part, _) = world.project(operand, x());
        let (literal, _) = world.written(Shape::Scalar(Scalar::U8));
        let at = world.items.push("meeting");
        let enters = Bound::Enter { lower: literal, upper: part };
        world.classes.bound(enters, at);

        let (field, _) = world.written(Shape::Scalar(Scalar::U8));
        let object = world.object_with_x(field);
        let (reference, _) = world.written(Shape::Ref(Mutability::Shared, object));
        world.decide(operand, reference);

        assert_eq!(world.classes.between_decided(), BTreeSet::from([(enters, at)]));
    }
}
