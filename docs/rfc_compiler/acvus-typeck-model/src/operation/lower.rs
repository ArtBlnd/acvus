//! What enters a class: the join of its lower bounds, and the judgment of each
//! lower bound once both of its ends are decided.
//!
//! Deciding a class and connecting parts are two steps. The join gives the head
//! alone. Each bound that entered then connects the parts it holds to the head's
//! when the store gives it back, so a part is connected the same way whichever
//! class was decided first.

use std::collections::BTreeSet;
use std::mem;

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::bound::{Bound, BoundOf};
use crate::primitive::class::{Bounds, ClassKind, State};
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{Mutability, Shape};
use crate::store::classes::Classes;

use super::Outcome;

/// What the lower bounds of a class give it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Join<K>
where
    K: ClassKind,
{
    /// No lower bound enters the class, so another dimension decides it (O17).
    Nothing,
    /// A lower bound has not given its head (O18).
    Waits,
    Head {
        shape: Shape<K::Id>,
        by: BTreeSet<ItemId>,
    },
    /// Every lower bound has given its head, and they are of two head classes.
    Mismatch(Refusal),
}

/// The head `bound` gives its upper end, where it gives one and its lower end
/// is decided. A change and an equality decide no head.
fn given<K>(classes: &Classes<K>, bound: &Bound<K>) -> Option<Shape<K::Id>>
where
    K: ClassKind<Bound = Bound<K>>,
{
    match bound {
        Bound::Enter { lower, .. } => shape_of(classes, *lower).cloned(),
        Bound::Lend { lower, mutability, .. } => Some(match shape_of(classes, *lower)? {
            Shape::Ref(of_place, target) => Shape::Ref(mutability.join(*of_place), *target),
            _ => Shape::Ref(*mutability, *lower),
        }),
        Bound::Change { .. } | Bound::Equal { .. } => None,
    }
}

fn shape_of<K>(classes: &Classes<K>, class: K::Id) -> Option<&Shape<K::Id>>
where
    K: ClassKind,
{
    match &classes.get(classes.find(class)).state {
        State::Decided(decided) => Some(&decided.shape),
        State::Undecided(_) => None,
    }
}

fn decided_by<K>(classes: &Classes<K>, class: K::Id) -> BTreeSet<ItemId>
where
    K: ClassKind,
{
    match &classes.get(classes.find(class)).state {
        State::Decided(decided) => decided.by.clone(),
        State::Undecided(_) => BTreeSet::new(),
    }
}

/// The join of the heads the lower bounds of `class` give (O6).
///
/// It waits for all of them, so its outcome does not depend on the order they
/// are decided in. A class's entering itself gives nothing to wait for.
pub fn join_lowers<K>(classes: &Classes<K>, class: K::Id) -> Join<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    let root = classes.find(class);
    let mut entering = Vec::new();
    for (lower, bounds) in classes.lowers(root) {
        for (bound, at) in bounds {
            let gives_a_head = match bound {
                Bound::Enter { .. } => lower != root,
                Bound::Lend { .. } => true,
                Bound::Change { .. } | Bound::Equal { .. } => false,
            };
            if gives_a_head {
                entering.push((bound, at));
            }
        }
    }
    if entering.is_empty() {
        return Join::Nothing;
    }
    let Some(heads) = entering
        .iter()
        .map(|(bound, _)| given(classes, bound))
        .collect::<Option<Vec<_>>>()
    else {
        return Join::Waits;
    };
    let by: BTreeSet<ItemId> = entering.iter().map(|(_, at)| *at).collect();
    let mut heads = heads.into_iter();
    let first = heads.next().expect("something enters the class");
    match heads.try_fold(first, |joined, head| joined.join(&head)) {
        Some(shape) => Join::Head { shape, by },
        None => {
            let mut items = by;
            for (bound, _) in &entering {
                items.extend(decided_by(classes, bound.lower()));
            }
            Join::Mismatch(Refusal {
                cause: Cause::Mismatch,
                items,
            })
        }
    }
}

impl<K> Outcome<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// Decides `class` as `shape`, and takes what that gives back as far as
    /// the operations here reach.
    pub fn decide(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        class: K::Id,
        shape: Shape<K::Id>,
        by: BTreeSet<ItemId>,
        at: ItemId,
    ) {
        let scope = classes.get(classes.find(class)).scope;
        let head = classes.decided(scope, shape, by);
        let unified = classes.unify(scopes, class, head, at);
        self.release(classes, scopes, unified);
        self.judge(classes, scopes);
    }

    /// Judges each lower bound given back, along with those its judgment gives
    /// back in turn. Other bounds stay given back to the caller.
    pub fn judge(&mut self, classes: &mut Classes<K>, scopes: &Scopes) {
        let mut queue = mem::take(&mut self.bounds);
        let mut others = Bounds::<K>::new();
        while let Some((bound, at)) = queue.pop_first() {
            match bound {
                Bound::Enter { .. } | Bound::Lend { .. } => {
                    self.judge_lower(classes, scopes, bound, at);
                    queue.append(&mut self.bounds);
                }
                Bound::Change { .. } | Bound::Equal { .. } => {
                    others.insert((bound, at));
                }
            }
        }
        self.bounds = others;
    }

    fn judge_lower(&mut self, classes: &mut Classes<K>, scopes: &Scopes, bound: Bound<K>, at: ItemId) {
        let (Some(head), Some(upper)) = (given(classes, &bound), shape_of(classes, bound.upper()).cloned()) else {
            unreachable!("a bound is given back once both of its ends are decided");
        };
        if !head.enters(&upper) {
            let mut items = decided_by(classes, bound.lower());
            items.extend(decided_by(classes, bound.upper()));
            items.insert(at);
            self.refusals.push(Refusal {
                cause: Cause::Mismatch,
                items,
            });
            return;
        }
        if let Bound::Lend { lower, .. } = bound
            && matches!(shape_of(classes, lower), Some(Shape::Ref(..)))
        {
            self.adjustments.entry(at).or_default().push(Adjustment::Deref);
        }
        if matches!(
            (&head, &upper),
            (Shape::Ref(Mutability::Exclusive, _), Shape::Ref(Mutability::Shared, _))
        ) {
            self.adjustments.entry(at).or_default().push(Adjustment::Reborrow);
        }
        for (key, part) in head.places() {
            let Some(corresponding) = upper.place(&key) else {
                unreachable!("a head that enters another has its every place there");
            };
            self.one_type(classes, scopes, *part, *corresponding, at);
        }
    }

    /// Makes `part` one type with the part that corresponds to it, as the
    /// state of that part tells (O9).
    fn one_type(&mut self, classes: &mut Classes<K>, scopes: &Scopes, part: K::Id, corresponding: K::Id, at: ItemId) {
        if classes.find(part) == classes.find(corresponding) {
            return;
        }
        if classes.is_external(corresponding) {
            let equal = Bound::Equal {
                lower: part,
                upper: corresponding,
            };
            self.bounds.extend(classes.bound(equal, at));
        } else {
            let unified = classes.unify(scopes, part, corresponding, at);
            self.release(classes, scopes, unified);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::primitive::adjustment::Adjustments;
    use crate::primitive::class::{Live, Side};
    use crate::primitive::id::Arena;
    use crate::primitive::ty::{PartKey, Scalar};
    use crate::primitive::{ClassId, Name};

    const SHARED: Mutability = Mutability::Shared;
    const EXCLUSIVE: Mutability = Mutability::Exclusive;
    const U8: Shape<ClassId> = Shape::Scalar(Scalar::U8);

    struct World {
        classes: Classes<Live>,
        scopes: Scopes,
        items: Arena<ItemId, &'static str>,
        outcome: Outcome<Live>,
        /// Every undecided class made here, in the order settling visits them.
        open: Vec<ClassId>,
    }

    impl World {
        fn new() -> Self {
            Self {
                classes: Classes::default(),
                scopes: Scopes::new(),
                items: Arena::default(),
                outcome: Outcome::default(),
                open: Vec::new(),
            }
        }

        fn undecided(&mut self) -> ClassId {
            let class = self.classes.undecided(self.scopes.root());
            self.open.push(class);
            class
        }

        /// The class of an expression, whose lower bound is the head the text
        /// writes.
        fn text(&mut self, shape: Shape<ClassId>) -> ClassId {
            let by = self.items.push("written");
            let written = self.classes.decided(self.scopes.root(), shape, BTreeSet::from([by]));
            let class = self.undecided();
            self.enter(written, class);
            class
        }

        fn external(&mut self, shape: Shape<ClassId>) -> ClassId {
            let by = self.items.push("declared");
            self.classes.external(self.scopes.root(), shape, BTreeSet::from([by]))
        }

        fn enter(&mut self, lower: ClassId, upper: ClassId) -> ItemId {
            self.record(Bound::Enter { lower, upper })
        }

        fn record(&mut self, bound: Bound<Live>) -> ItemId {
            let at = self.items.push("bound");
            self.outcome.bounds.extend(self.classes.bound(bound, at));
            at
        }

        fn lend(&mut self, place: ClassId, mutability: Mutability) -> (ClassId, ItemId) {
            let reference = self.undecided();
            let at = self.record(Bound::Lend {
                lower: place,
                upper: reference,
                mutability,
            });
            (reference, at)
        }

        /// `let name = lower`, or a branch of an `if`.
        fn meets(&mut self, lower: ClassId) -> ClassId {
            let place = self.undecided();
            self.enter(lower, place);
            place
        }

        fn decide(&mut self, class: ClassId, shape: Shape<ClassId>) {
            let at = self.items.push("decision");
            self.outcome
                .decide(&mut self.classes, &self.scopes, class, shape, BTreeSet::from([at]), at);
        }

        /// The lower dimension of settling alone: decides each class that a
        /// lower bound enters, once all of them have given their heads, until
        /// nothing changes.
        fn settle(&mut self) {
            self.outcome.judge(&mut self.classes, &self.scopes);
            loop {
                let mut changed = false;
                for class in self.open.clone() {
                    if self.classes.is_decided(class) {
                        continue;
                    }
                    match join_lowers(&self.classes, class) {
                        Join::Head { shape, by } => {
                            let at = self.items.push("decision");
                            self.outcome.decide(&mut self.classes, &self.scopes, class, shape, by, at);
                            changed = true;
                        }
                        Join::Mismatch(refusal) => {
                            if !self.outcome.refusals.contains(&refusal) {
                                self.outcome.refusals.push(refusal);
                            }
                        }
                        Join::Nothing | Join::Waits => {}
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        fn shape(&self, class: ClassId) -> Shape<ClassId> {
            shape_of(&self.classes, class).expect("the class is decided").clone()
        }

        fn same(&self, a: ClassId, b: ClassId) -> bool {
            self.classes.find(a) == self.classes.find(b)
        }

        fn target(&self, class: ClassId) -> (Mutability, ClassId) {
            let Shape::Ref(mutability, target) = self.shape(class) else {
                panic!("{class:?} is a reference");
            };
            (mutability, target)
        }

        fn no_refusals(&self) {
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

    fn adjusted(entries: &[(ItemId, &[Adjustment])]) -> Adjustments {
        entries.iter().map(|(item, kinds)| (*item, kinds.to_vec())).collect()
    }

    /// Example 2: `let a = if c { Some(1u8) } else { None };`
    #[test]
    fn an_if_with_information_on_one_side_unifies_the_payloads() {
        let mut world = World::new();
        let one = world.text(U8);
        let some = world.text(Shape::Option(one));
        let nothing_enters = world.undecided();
        let none = world.text(Shape::Option(nothing_enters));
        let if_class = world.meets(some);
        world.enter(none, if_class);
        let a = world.meets(if_class);

        world.settle();

        world.no_refusals();
        let Shape::Option(payload) = world.shape(a) else {
            panic!("`a` is an Option");
        };
        assert!(world.same(payload, one));
        assert!(world.same(payload, nothing_enters));
        assert_eq!(world.shape(payload), U8);
    }

    /// Example 3: `let r = if c { &mut x } else { &y };`
    #[test]
    fn a_reference_that_meets_a_shared_one_records_a_reborrow() {
        let mut world = World::new();
        let x = world.text(U8);
        let y = world.text(U8);
        let (exclusive, _) = world.lend(x, EXCLUSIVE);
        let (shared, _) = world.lend(y, SHARED);
        let if_class = world.undecided();
        let exclusive_meets = world.enter(exclusive, if_class);
        world.enter(shared, if_class);
        let r = world.meets(if_class);

        world.settle();

        world.no_refusals();
        assert_eq!(world.target(r).0, SHARED);
        assert_eq!(world.target(exclusive).0, EXCLUSIVE);
        assert!(world.same(x, y));
        assert_eq!(world.outcome.adjustments, adjusted(&[(exclusive_meets, &[Adjustment::Reborrow])]));
    }

    /// Example 24: `let a = if c { return 0u8 } else { 1u8 };`
    #[test]
    fn a_branch_that_does_not_complete_leaves_the_other_branchs_head() {
        let mut world = World::new();
        let returns = world.text(Shape::Never);
        let one = world.text(U8);
        let if_class = world.meets(returns);
        world.enter(one, if_class);
        let a = world.meets(if_class);

        world.settle();

        world.no_refusals();
        assert_eq!(world.shape(a), U8);
        assert_eq!(world.shape(returns), Shape::Never);
    }

    /// Example 25, whichever class is visited first.
    #[test]
    fn a_value_that_meets_in_a_place_keeps_its_own_class() {
        for reversed in [false, true] {
            let mut world = World::new();
            let x = world.text(U8);
            let y = world.text(U8);
            let (lent_x, _) = world.lend(x, EXCLUSIVE);
            let p = world.meets(lent_x);
            let (lent_y, _) = world.lend(y, SHARED);
            let if_class = world.undecided();
            let p_meets = world.enter(p, if_class);
            world.enter(lent_y, if_class);
            let r = world.meets(if_class);
            // `touch(p)`, with `touch(&mut u8)`.
            let declared_u8 = world.external(U8);
            let parameter = world.external(Shape::Ref(EXCLUSIVE, declared_u8));
            let boundary = world.meets(p);
            let changes = world.record(Bound::Change {
                lower: boundary,
                upper: parameter,
            });
            if reversed {
                world.open.reverse();
            }

            world.settle();

            world.no_refusals();
            assert_eq!(world.target(p).0, EXCLUSIVE);
            assert_eq!(world.target(r).0, SHARED);
            assert_eq!(world.target(boundary).0, EXCLUSIVE);
            assert!(world.same(x, y));
            assert_eq!(world.outcome.adjustments, adjusted(&[(p_meets, &[Adjustment::Reborrow])]));
            assert_eq!(
                world.outcome.bounds,
                BTreeSet::from([(
                    Bound::Change {
                        lower: boundary,
                        upper: parameter
                    },
                    changes
                )])
            );
        }
    }

    /// `let a = &ₘ x; let r = &ₙ a;`, for each `m` and `n` (O23): no reference
    /// to a reference, the join of the two for its mutability, and `a` as it
    /// was.
    #[test]
    fn a_lend_through_a_reference_takes_it_off_and_joins_its_mutability() {
        let cases = [
            (SHARED, SHARED, SHARED),
            (SHARED, EXCLUSIVE, SHARED),
            (EXCLUSIVE, SHARED, SHARED),
            (EXCLUSIVE, EXCLUSIVE, EXCLUSIVE),
        ];
        for (of_a, of_r, expected) in cases {
            let mut world = World::new();
            let x = world.text(U8);
            let (lent_x, _) = world.lend(x, of_a);
            let a = world.meets(lent_x);
            let (lent_a, lend_at) = world.lend(a, of_r);
            let r = world.meets(lent_a);

            world.settle();

            world.no_refusals();
            assert_eq!(world.target(r).0, expected);
            assert!(world.same(world.target(r).1, x));
            assert_eq!(world.target(a).0, of_a);
            assert_eq!(world.outcome.adjustments, adjusted(&[(lend_at, &[Adjustment::Deref])]));
        }
    }

    #[test]
    fn a_lend_of_what_is_no_reference_targets_it_and_takes_no_adjustment() {
        let mut world = World::new();
        let x = world.text(U8);
        let (lent, _) = world.lend(x, EXCLUSIVE);
        let r = world.meets(lent);

        world.settle();

        world.no_refusals();
        assert_eq!(world.target(r).0, EXCLUSIVE);
        assert!(world.same(world.target(r).1, x));
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// `let a; let r = &a;`: the reference waits for `a` by the rule every
    /// lower bound waits by, and nothing else in this module decides `a`.
    #[test]
    fn a_lend_waits_for_its_place() {
        let mut world = World::new();
        let a = world.undecided();
        let (lent, _) = world.lend(a, SHARED);

        world.settle();
        assert_eq!(join_lowers(&world.classes, lent), Join::Waits);
        assert!(!world.classes.is_decided(lent));

        world.decide(a, Shape::Never);
        world.settle();
        assert_eq!(world.target(lent), (SHARED, world.classes.find(a).id()));
        world.no_refusals();
    }

    /// `let x = if c { {x: 1u8} } else { {x: 2u8, y: 3u8} };`: members are
    /// united, and which are initialized is another check's.
    #[test]
    fn objects_that_meet_unite_their_fields() {
        let mut world = World::new();
        let (first_x, second_x, y) = (world.text(U8), world.text(U8), world.text(U8));
        let first = world.text(Shape::Object(BTreeMap::from([(Name::new("x"), first_x)])));
        let second = world.text(Shape::Object(BTreeMap::from([
            (Name::new("x"), second_x),
            (Name::new("y"), y),
        ])));
        let if_class = world.meets(first);
        world.enter(second, if_class);

        world.settle();

        world.no_refusals();
        let Shape::Object(fields) = world.shape(if_class) else {
            panic!("the `if` is an object");
        };
        assert_eq!(fields.keys().cloned().collect::<Vec<_>>(), vec![Name::new("x"), Name::new("y")]);
        assert!(world.same(first_x, second_x));
        assert!(world.same(fields[&Name::new("y")], y));
    }

    #[test]
    fn a_join_of_two_head_classes_is_refused_naming_what_registered_them() {
        let mut world = World::new();
        let (one, letter) = (world.text(U8), world.text(Shape::Scalar(Scalar::Char)));
        let if_class = world.undecided();
        let one_meets = world.enter(one, if_class);
        let letter_meets = world.enter(letter, if_class);

        world.settle();

        assert_eq!(world.outcome.refusals.len(), 1);
        let refusal = &world.outcome.refusals[0];
        assert_eq!(refusal.cause, Cause::Mismatch);
        assert!(refusal.items.is_superset(&BTreeSet::from([one_meets, letter_meets])));
        assert!(!world.classes.is_decided(if_class));
    }

    #[test]
    fn a_class_waits_for_every_lower_bound_and_not_for_itself() {
        let mut world = World::new();
        let one = world.text(U8);
        let nothing_enters = world.undecided();
        let if_class = world.meets(one);
        world.enter(if_class, if_class);
        let waiting = world.meets(if_class);
        world.enter(nothing_enters, waiting);

        world.settle();

        world.no_refusals();
        assert_eq!(world.shape(if_class), U8);
        assert_eq!(join_lowers(&world.classes, waiting), Join::Waits);
    }

    #[test]
    fn a_bound_that_decides_no_head_gives_none() {
        let mut world = World::new();
        let (below, above) = (world.undecided(), world.undecided());
        world.record(Bound::Change {
            lower: below,
            upper: above,
        });
        world.record(Bound::Equal {
            lower: below,
            upper: above,
        });
        assert_eq!(join_lowers(&world.classes, above), Join::Nothing);
    }

    /// O9: a part entered by `u16` and corresponding to a declared `u8` keeps
    /// its own head, and the two stay apart for `Equal` to judge.
    #[test]
    fn a_part_corresponding_to_an_external_one_takes_it_as_an_upper_bound() {
        let mut world = World::new();
        let declared_u8 = world.external(U8);
        let declared = world.external(Shape::Option(declared_u8));
        let sixteen = world.text(Shape::Scalar(Scalar::U16));
        let some = world.text(Shape::Option(sixteen));
        let meets = world.enter(some, declared);

        world.settle();

        world.no_refusals();
        assert!(!world.same(sixteen, declared_u8));
        assert_eq!(world.shape(sixteen), Shape::Scalar(Scalar::U16));
        assert_eq!(
            world.outcome.bounds,
            BTreeSet::from([(
                Bound::Equal {
                    lower: sixteen,
                    upper: declared_u8
                },
                meets
            )])
        );
    }

    /// `let y = f();`, with `f() -> Option<u8>`: the join gives `y` the parts of
    /// what entered it, so there is nothing to judge between them.
    #[test]
    fn a_class_that_takes_the_parts_of_what_enters_it_has_nothing_to_compare() {
        let mut world = World::new();
        let declared_u8 = world.external(U8);
        let result = world.external(Shape::Option(declared_u8));
        let y = world.meets(result);

        world.settle();

        world.no_refusals();
        assert_eq!(world.shape(y), Shape::Option(declared_u8));
        assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
    }

    /// O9 goes by where a class comes from and not by whether it is decided
    /// yet: a hole is unified with its part whichever is decided first.
    #[test]
    fn a_hole_is_unified_with_its_part_whether_or_not_it_is_decided_yet() {
        for hole_first in [false, true] {
            let mut world = World::new();
            let hole = world.undecided();
            let declared = world.external(Shape::Option(hole));
            let one = world.text(U8);
            let some = world.text(Shape::Option(one));
            if hole_first {
                world.decide(hole, U8);
            }
            world.enter(some, declared);

            world.settle();

            world.no_refusals();
            assert!(world.same(hole, one));
            assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
        }
    }

    #[test]
    fn a_head_that_does_not_enter_a_written_head_is_refused() {
        let cases: [(Shape<ClassId>, Shape<ClassId>); 2] = [(Shape::Scalar(Scalar::U16), U8), (U8, Shape::Never)];
        for (lower_head, declared_head) in cases {
            let mut world = World::new();
            let declared = world.external(declared_head);
            let lower = world.text(lower_head);
            let meets = world.enter(lower, declared);

            world.settle();

            let refusal = &world.outcome.refusals[0];
            assert_eq!(refusal.cause, Cause::Mismatch);
            assert!(refusal.items.contains(&meets));
        }
    }

    #[test]
    fn a_shared_reference_does_not_enter_an_exclusive_one() {
        let mut world = World::new();
        let declared_u8 = world.external(U8);
        let declared = world.external(Shape::Ref(EXCLUSIVE, declared_u8));
        let x = world.text(U8);
        let (lent, _) = world.lend(x, SHARED);
        let meets = world.enter(lent, declared);

        world.settle();

        let refusal = &world.outcome.refusals[0];
        assert_eq!(refusal.cause, Cause::Mismatch);
        assert!(refusal.items.contains(&meets));
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    fn field(name: &str) -> PartKey {
        PartKey::Field(Name::new(name))
    }

    fn project(world: &mut World, operand: ClassId, key: PartKey) -> (ClassId, ItemId) {
        let at = world.items.push("projection");
        let part = world
            .outcome
            .project(&mut world.classes, operand, key, Side::Upper, at)
            .expect("an undecided class names any part");
        (part, at)
    }

    /// `p.x.y`, where what enters `p` holds a reference at `x`: the union that
    /// decides `p` connects `x`, and leaves `y` with no place on the reference.
    #[test]
    fn a_decision_releases_the_projections_its_union_leaves() {
        let mut world = World::new();
        let p = world.undecided();
        let (x_part, _) = project(&mut world, p, field("x"));
        let (y_part, y_at) = project(&mut world, x_part, field("y"));

        let inner_y = world.text(U8);
        let inner = world.text(Shape::Object(BTreeMap::from([(Name::new("y"), inner_y)])));
        let (lent, _) = world.lend(inner, SHARED);
        let written = world.text(Shape::Object(BTreeMap::from([(Name::new("x"), lent)])));
        world.enter(written, p);

        world.settle();

        world.no_refusals();
        assert!(world.same(y_part, inner_y));
        assert_eq!(world.outcome.adjustments, adjusted(&[(y_at, &[Adjustment::Deref])]));
    }

    /// `let v = q.x; let a = if c { Some(q) } else { Some(&obj) };`: the join
    /// keeps `q` as the payload, and only judging the second branch makes `q`
    /// the reference, which leaves `x` with no place on it.
    #[test]
    fn a_judgment_releases_the_projections_its_union_leaves() {
        let mut world = World::new();
        let q = world.undecided();
        let (x_part, x_at) = project(&mut world, q, field("x"));
        let first = world.text(Shape::Option(q));
        let field_class = world.text(U8);
        let obj = world.text(Shape::Object(BTreeMap::from([(Name::new("x"), field_class)])));
        let (lent, _) = world.lend(obj, SHARED);
        let second = world.text(Shape::Option(lent));
        let if_class = world.meets(first);
        world.enter(second, if_class);

        world.settle();

        world.no_refusals();
        assert!(world.same(q, lent));
        assert!(world.same(x_part, field_class));
        assert_eq!(world.outcome.adjustments, adjusted(&[(x_at, &[Adjustment::Deref])]));
    }

    /// `let v = q.y; let a = if c { Some(q) } else { Some(obj) };` where `obj`
    /// is `{ w: _ }`: judging the second branch gives `q` a head with no place
    /// for what was named on it, and a judgment's refusal reaches the caller.
    #[test]
    fn a_judgment_refuses_a_projection_its_union_leaves_with_no_place() {
        let mut world = World::new();
        let q = world.undecided();
        let (_, y_at) = project(&mut world, q, field("y"));
        let first = world.text(Shape::Option(q));
        let w = world.text(U8);
        let obj = world.text(Shape::Object(BTreeMap::from([(Name::new("w"), w)])));
        let second = world.text(Shape::Option(obj));
        let if_class = world.meets(first);
        world.enter(second, if_class);

        world.settle();

        assert_eq!(world.outcome.refusals.len(), 1);
        assert_eq!(world.outcome.refusals[0].cause, Cause::NotHeld);
        assert!(world.outcome.refusals[0].items.contains(&y_at));
    }

    /// A bound a judgment's union gives back is judged in the same turn:
    /// joining the branches makes the payloads one, and that decides a class
    /// another lower bound was held on.
    #[test]
    fn a_bound_a_judgment_gives_back_is_judged_in_turn() {
        let mut world = World::new();
        // `hole` holds the bound from `written`, where `written` is `&mut u8`
        // and `hole` becomes `&u8` only by being made one with `shared`.
        let hole = world.undecided();
        let x = world.text(U8);
        let (written, _) = world.lend(x, EXCLUSIVE);
        let meets = world.enter(written, hole);
        let target = world.text(U8);
        let shared = world
            .classes
            .decided(world.scopes.root(), Shape::Ref(SHARED, target), BTreeSet::new());
        let first = world.text(Shape::Option(hole));
        let second = world.text(Shape::Option(shared));
        let if_class = world.meets(second);
        world.enter(first, if_class);
        // Taken out of settling, so that only the judgment can decide `hole`.
        world.open.retain(|class| *class != hole);

        world.settle();

        world.no_refusals();
        assert!(world.same(hole, shared));
        assert_eq!(world.outcome.adjustments, adjusted(&[(meets, &[Adjustment::Reborrow])]));
        assert!(world.same(x, target));
    }
}
