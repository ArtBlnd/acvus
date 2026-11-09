//! What bounds do: the join of a class's lower bounds, and the judgment of each
//! bound once both of its ends are decided.
//!
//! Deciding a class and connecting parts are two steps. The join gives the head
//! alone, and each bound connects the parts it holds to the head's when the store
//! gives it back. That a class is decided only once nothing more can enter it is
//! the order settling takes these in (`settle`), not anything here.
//!
//! Classes that enter one another would wait for one another, so they are made
//! one class first (O6). A bound that went between them is then the class's bound
//! on itself: `x ≥ x ⊔ y` is a class holding `Enter`s with both ends in it, and
//! one from `y`. It keeps its item for a refusal to name, and gives no head.

use std::collections::{BTreeMap, BTreeSet};
use std::mem;

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::record::Coercion;
use crate::primitive::bound::{Bound, BoundOf};
use crate::primitive::class::{Bounds, ClassKind, Pending, State};
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
pub(super) fn given<K>(classes: &Classes<K>, bound: &Bound<K>) -> Option<Shape<K::Id>>
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

pub(super) fn shape_of<K>(classes: &Classes<K>, class: K::Id) -> Option<&Shape<K::Id>>
where
    K: ClassKind,
{
    match &classes.get(classes.find(class)).state {
        State::Decided(decided) => Some(&decided.shape),
        State::Undecided(_) => None,
    }
}

pub(super) fn decided_by<K>(classes: &Classes<K>, class: K::Id) -> BTreeSet<ItemId>
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

/// Each `Enter` both of whose ends lie on one cycle of `Enter`s, with the item
/// that recorded it.
///
/// A bound between two decided classes is not among them, as no class holds it
/// any more. Each direction of it has been judged, so their heads are equal, and
/// a cycle through them decides the same heads without their being one class.
fn entering_on_cycles<K>(classes: &Classes<K>) -> Vec<(K::Id, K::Id, ItemId)>
where
    K: ClassKind<Bound = Bound<K>>,
{
    let mut entered: BTreeMap<K::Id, BTreeSet<(K::Id, ItemId)>> = BTreeMap::new();
    for root in classes.roots() {
        for (bound, at) in classes.get(root).state.bounds().into_iter().flatten() {
            if let Bound::Enter { lower, upper } = bound {
                let (lower, upper) = (classes.find(*lower).id(), classes.find(*upper).id());
                if lower != upper {
                    entered.entry(lower).or_default().insert((upper, *at));
                }
            }
        }
    }
    entered
        .iter()
        .flat_map(|(lower, uppers)| uppers.iter().map(|(upper, at)| (*lower, *upper, *at)))
        .filter(|(lower, upper, _)| reaches(&entered, *upper, *lower))
        .collect()
}

fn reaches<I>(entered: &BTreeMap<I, BTreeSet<(I, ItemId)>>, from: I, to: I) -> bool
where
    I: Copy + Ord,
{
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(class) = stack.pop() {
        if class == to {
            return true;
        }
        if seen.insert(class) {
            stack.extend(entered.get(&class).into_iter().flatten().map(|(next, _)| *next));
        }
    }
    false
}

impl<K> Outcome<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// Makes classes that enter one another one class (O6), each bound on a
    /// cycle making its two ends one by the item that recorded it, and says
    /// whether it found any. A union of parts may close another cycle, which the
    /// next call finds.
    pub fn merge_cycles(&mut self, classes: &mut Classes<K>, scopes: &Scopes) -> bool {
        let on_cycles = entering_on_cycles(classes);
        let merged = !on_cycles.is_empty();
        for (lower, upper, at) in on_cycles {
            let unified = classes.unify(scopes, lower, upper, at);
            self.release(classes, scopes, unified);
        }
        self.judge(classes, scopes);
        merged
    }

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
    /// back in turn. A change stays given back to the caller, which settling makes
    /// with the conversions declared.
    pub fn judge(&mut self, classes: &mut Classes<K>, scopes: &Scopes) {
        let mut queue = mem::take(&mut self.bounds);
        let mut others = Bounds::<K>::new();
        // A class refused as its own part stays cyclic, so an `Equal` going
        // down two such classes comes back to itself. A second judgment of one
        // bound would say nothing the first did not.
        let mut judged = Bounds::<K>::new();
        while let Some((bound, at)) = queue.pop_first() {
            if !judged.insert((bound, at)) {
                continue;
            }
            match bound {
                Bound::Enter { .. } | Bound::Lend { .. } => self.judge_lower(classes, scopes, bound, at),
                Bound::Equal { .. } => self.judge_equal(classes, scopes, bound, at),
                Bound::Change { .. } => {
                    others.insert((bound, at));
                }
            }
            queue.append(&mut self.bounds);
        }
        self.bounds = others;
        self.judge_domains(classes);
    }

    /// A domain holds the head decided for its class (O8), and is done with. A
    /// call stays with the caller, as nothing here judges one yet.
    fn judge_domains(&mut self, classes: &Classes<K>) {
        for pending in mem::take(&mut self.pending) {
            let held = shape_of(classes, pending.class);
            for (domain, by) in &pending.domains {
                if !matches!(held, Some(Shape::Scalar(scalar)) if domain.0.contains(scalar)) {
                    let mut items = decided_by(classes, pending.class);
                    items.insert(*by);
                    self.refuse_once(Refusal {
                        cause: Cause::EmptyDomain,
                        items,
                    });
                }
            }
            if !pending.calls.is_empty() {
                self.pending.push(Pending {
                    class: pending.class,
                    calls: pending.calls,
                    domains: BTreeSet::new(),
                });
            }
        }
    }

    /// No change is made at a part (O9): the two heads are equal, members
    /// included, and each part below is one type with the corresponding part.
    /// One layer at a time gives what one comparison of the two types would, so
    /// it waits for nothing below this layer.
    fn judge_equal(&mut self, classes: &mut Classes<K>, scopes: &Scopes, bound: Bound<K>, at: ItemId) {
        let (Some(lower), Some(upper)) = (
            shape_of(classes, bound.lower()).cloned(),
            shape_of(classes, bound.upper()).cloned(),
        ) else {
            unreachable!("a bound is given back once both of its ends are decided");
        };
        if lower.layer() != upper.layer() {
            self.refuse_mismatch(classes, &bound, at);
            return;
        }
        self.connect_parts(classes, scopes, &lower, &upper, at);
    }

    fn judge_lower(&mut self, classes: &mut Classes<K>, scopes: &Scopes, bound: Bound<K>, at: ItemId) {
        let (Some(head), Some(upper)) = (given(classes, &bound), shape_of(classes, bound.upper()).cloned()) else {
            unreachable!("a bound is given back once both of its ends are decided");
        };
        if !head.enters(&upper) {
            self.refuse_mismatch(classes, &bound, at);
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
            self.adjustments.entry(at).or_default().push(Adjustment::Coerce(Coercion::Reborrow));
        }
        self.connect_parts(classes, scopes, &head, &upper, at);
    }

    fn refuse_mismatch(&mut self, classes: &Classes<K>, bound: &Bound<K>, at: ItemId) {
        let mut items = decided_by(classes, bound.lower());
        items.extend(decided_by(classes, bound.upper()));
        items.insert(at);
        self.refusals.push(Refusal {
            cause: Cause::Mismatch,
            items,
        });
    }

    /// Makes each part of `head` one type with the part `upper` has in its
    /// place. `head` has already been found to reach `upper`.
    fn connect_parts(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &Scopes,
        head: &Shape<K::Id>,
        upper: &Shape<K::Id>,
        at: ItemId,
    ) {
        for (key, part) in head.places() {
            let Some(corresponding) = upper.place(&key) else {
                unreachable!("a head that reaches another has its every place there");
            };
            self.one_type(classes, scopes, *part, *corresponding, at);
        }
    }

    /// Makes `part` one type with the part that corresponds to it, as the
    /// state of that part tells (O9).
    pub(super) fn one_type(&mut self, classes: &mut Classes<K>, scopes: &Scopes, part: K::Id, corresponding: K::Id, at: ItemId) {
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
    use crate::operation::testing::{EXCLUSIVE, SHARED, U8, World};
    use crate::primitive::class::{Live, Side};
    use crate::primitive::ty::{PartKey, Scalar};
    use crate::primitive::{ClassId, Name};

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
        assert_eq!(world.outcome.adjustments, adjusted(&[(exclusive_meets, &[Adjustment::Coerce(Coercion::Reborrow)])]));
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
            world.record(Bound::Change {
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
            assert_eq!(world.outcome.adjustments, adjusted(&[(p_meets, &[Adjustment::Coerce(Coercion::Reborrow)])]));
            assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
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
    /// its own head, stays apart from the declared one, and is refused there.
    #[test]
    fn a_part_corresponding_to_an_external_one_takes_it_as_an_upper_bound() {
        let mut world = World::new();
        let declared_u8 = world.external(U8);
        let declared = world.external(Shape::Option(declared_u8));
        let sixteen = world.text(Shape::Scalar(Scalar::U16));
        let some = world.text(Shape::Option(sixteen));
        let meets = world.enter(some, declared);

        world.settle();

        assert!(!world.same(sixteen, declared_u8));
        assert_eq!(world.shape(sixteen), Shape::Scalar(Scalar::U16));
        assert_eq!(world.outcome.refusals.len(), 1);
        assert_eq!(world.outcome.refusals[0].cause, Cause::Mismatch);
        assert!(world.outcome.refusals[0].items.contains(&meets));
        assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
    }

    #[test]
    fn an_equal_of_one_head_is_accepted_and_leaves_two_classes() {
        let mut world = World::new();
        let declared = world.external(U8);
        let lower = world.text(U8);
        world.record(Bound::Equal { lower, upper: declared });

        world.settle();

        world.no_refusals();
        assert!(!world.same(lower, declared));
        assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
    }

    /// No change is made at a part, so none of what a boundary or a meeting
    /// allows is allowed here: `take(Some(&mut x))` with `take(Option<&T>)` is
    /// refused (example 12), and `!` is no least head.
    #[test]
    fn an_equal_of_two_heads_is_refused_naming_what_decided_them() {
        let mut world = World::new();
        let target = world.external(U8);
        let object = |fields: &[&str]| Shape::Object(fields.iter().map(|field| (Name::new(*field), target)).collect());
        let cases = [
            (Shape::Scalar(Scalar::U16), U8),
            (Shape::Ref(EXCLUSIVE, target), Shape::Ref(SHARED, target)),
            (Shape::Ref(SHARED, target), Shape::Ref(EXCLUSIVE, target)),
            (Shape::Never, U8),
            (object(&["x", "y"]), object(&["x"])),
            (object(&["x"]), object(&["x", "y"])),
        ];
        for (lower_head, upper_head) in cases {
            let refused = world.outcome.refusals.len();
            let lower = world.text(lower_head);
            let upper = world.external(upper_head);
            let at = world.record(Bound::Equal { lower, upper });

            world.settle();

            assert_eq!(world.outcome.refusals.len(), refused + 1);
            let refusal = &world.outcome.refusals[refused];
            assert_eq!(refusal.cause, Cause::Mismatch);
            assert!(refusal.items.contains(&at));
            let decided: Vec<BTreeSet<ItemId>> = [lower, upper]
                .into_iter()
                .map(|class| decided_by(&world.classes, class))
                .collect();
            assert!(decided.iter().all(|by| !by.is_empty() && by.is_subset(&refusal.items)));
        }
    }

    #[test]
    fn an_equal_goes_down_to_each_part() {
        for (head, refused) in [(U8, false), (Shape::Scalar(Scalar::U16), true)] {
            let mut world = World::new();
            let declared_u8 = world.external(U8);
            let declared = world.external(Shape::Option(declared_u8));
            let payload = world.text(head);
            let lower = world.text(Shape::Option(payload));
            world.record(Bound::Equal { lower, upper: declared });

            world.settle();

            assert!(!world.same(payload, declared_u8));
            assert_eq!(world.outcome.refusals.is_empty(), !refused);
            assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
        }
    }

    /// `Vec<T>` going to where `T` is a hole: the hole is free, so the two are
    /// one class, whichever is decided first.
    #[test]
    fn an_equal_unifies_a_part_with_the_hole_it_corresponds_to() {
        for hole_first in [false, true] {
            let mut world = World::new();
            let hole = world.undecided();
            let declared = world.external(Shape::Option(hole));
            let payload = world.text(U8);
            let lower = world.text(Shape::Option(payload));
            if hole_first {
                world.decide(hole, U8);
            }
            world.record(Bound::Equal { lower, upper: declared });

            world.settle();

            world.no_refusals();
            assert!(world.same(payload, hole));
        }
    }

    /// A class refused as its own part stays cyclic, as solving goes on after
    /// a refusal, and an `Equal` makes no two classes one on its way down.
    #[test]
    fn an_equal_of_two_classes_that_hold_themselves_ends() {
        let mut world = World::new();
        let (first, second) = (world.undecided(), world.undecided());
        let first_option = world.external(Shape::Option(first));
        let second_option = world.external(Shape::Option(second));
        let at = world.items.push("cycle");
        let occurs = [
            world.classes.unify(&world.scopes, first, first_option, at),
            world.classes.unify(&world.scopes, second, second_option, at),
        ];
        assert!(occurs.iter().all(|unified| unified.refusals[0].cause == Cause::Occurs));
        world.record(Bound::Equal {
            lower: first_option,
            upper: second_option,
        });

        world.settle();

        world.no_refusals();
        assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
    }

    /// `&?a` against `&u8` needs nothing of `?a` to be judged: `?a` takes the
    /// declared `u8` as an upper bound, and is judged against it once decided.
    #[test]
    fn an_equal_of_references_does_not_wait_for_the_target() {
        for (head, refused) in [(U8, false), (Shape::Scalar(Scalar::U16), true)] {
            let mut world = World::new();
            let declared_u8 = world.external(U8);
            let declared = world.external(Shape::Ref(SHARED, declared_u8));
            let target = world.undecided();
            let lower = world.text(Shape::Ref(SHARED, target));
            let at = world.record(Bound::Equal { lower, upper: declared });

            world.settle();

            world.no_refusals();
            assert!(!world.classes.is_decided(target));
            assert_eq!(world.outcome.bounds, Bounds::<Live>::new());
            let held = world.classes.upper_items(target);
            assert_eq!(held, BTreeMap::from([(world.classes.find(declared_u8).id(), BTreeSet::from([at]))]));

            world.decide(target, head);

            assert_eq!(world.outcome.refusals.is_empty(), !refused);
            assert!(!world.same(target, declared_u8));
        }
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

    /// `x = if c { x } else { 1u8 };`: `x ≥ x ⊔ u8`, whichever class is visited
    /// first.
    #[test]
    fn a_class_that_enters_itself_through_an_if_takes_what_else_enters_it() {
        for reversed in [false, true] {
            let mut world = World::new();
            let one = world.text(U8);
            let x = world.undecided();
            let if_class = world.undecided();
            world.enter(x, if_class);
            world.enter(one, if_class);
            world.enter(if_class, x);
            if reversed {
                world.open.reverse();
            }

            world.settle();

            world.no_refusals();
            assert!(world.same(x, if_class));
            assert_eq!(world.shape(x), U8);
        }
    }

    /// What `x ≥ x ⊔ y` is in the store: one class with the bounds that went
    /// between its classes on itself, and what else enters it beside them.
    #[test]
    fn a_class_merged_from_a_cycle_is_its_own_lower_bound_beside_what_else_enters_it() {
        let mut world = World::new();
        let y = world.undecided();
        let x = world.undecided();
        let if_class = world.undecided();
        let x_meets = world.enter(x, if_class);
        let y_meets = world.enter(y, if_class);
        let assigned = world.enter(if_class, x);

        assert!(world.outcome.merge_cycles(&mut world.classes, &world.scopes));

        let (root, y_root) = (world.classes.find(x).id(), world.classes.find(y).id());
        assert_eq!(
            world.classes.lower_items(x),
            BTreeMap::from([
                (root, BTreeSet::from([x_meets, assigned])),
                (y_root, BTreeSet::from([y_meets])),
            ])
        );
        assert_eq!(join_lowers(&world.classes, x), Join::Waits);
        assert!(!world.outcome.merge_cycles(&mut world.classes, &world.scopes));

        world.decide(y, U8);
        world.settle();
        world.no_refusals();
        assert_eq!(world.shape(x), U8);
    }

    /// `let mut x = 1u8; x = id(x);`, with `id(T) -> T`.
    #[test]
    fn a_class_passed_through_a_hole_back_to_itself_is_one_with_the_hole() {
        let mut world = World::new();
        let one = world.text(U8);
        let x = world.meets(one);
        let hole = world.meets(x);
        world.enter(hole, x);

        world.settle();

        world.no_refusals();
        assert!(world.same(x, hole));
        assert_eq!(world.shape(hole), U8);
    }

    /// `let mut x; x = x;`, and the same through two other places: what is only
    /// its own lower bound has no lower bound to wait for, so another dimension
    /// decides it (O17).
    #[test]
    fn a_class_that_only_enters_itself_has_nothing_entering_it() {
        let mut world = World::new();
        let alone = world.undecided();
        world.enter(alone, alone);
        let (a, b, c) = (world.undecided(), world.undecided(), world.undecided());
        world.enter(a, b);
        world.enter(b, c);
        world.enter(c, a);

        world.settle();

        world.no_refusals();
        assert!(world.same(a, b) && world.same(b, c));
        assert_eq!(join_lowers(&world.classes, alone), Join::Nothing);
        assert_eq!(join_lowers(&world.classes, a), Join::Nothing);
    }

    #[test]
    fn classes_that_only_one_enters_the_other_stay_apart() {
        let mut world = World::new();
        let (top, left, right, bottom) = (
            world.undecided(),
            world.undecided(),
            world.undecided(),
            world.undecided(),
        );
        world.enter(top, left);
        world.enter(top, right);
        world.enter(left, bottom);
        world.enter(right, bottom);

        assert!(!world.outcome.merge_cycles(&mut world.classes, &world.scopes));
        assert!(!world.same(left, right));
        assert!(!world.same(top, bottom));
    }

    /// Only entering makes one class (O6). A boundary between two classes that
    /// meet the other way round must still make its change.
    #[test]
    fn a_cycle_through_any_other_bound_stays_two_classes() {
        let kinds: [fn(ClassId, ClassId) -> Bound<Live>; 3] = [
            |lower, upper| Bound::Change { lower, upper },
            |lower, upper| Bound::Equal { lower, upper },
            |lower, upper| Bound::Lend {
                lower,
                upper,
                mutability: SHARED,
            },
        ];
        for kind in kinds {
            let mut world = World::new();
            let (a, b) = (world.undecided(), world.undecided());
            world.enter(a, b);
            world.record(kind(b, a));

            assert!(!world.outcome.merge_cycles(&mut world.classes, &world.scopes));
            assert!(!world.same(a, b));
        }
    }

    /// `let mut x = &y; x = &x;`. Neither enters the other's way round: the
    /// reference waits for `x`, and `x` for the reference, so both are still
    /// undecided. This is where the model stands, not what is settled to be right.
    #[test]
    fn a_lend_of_the_class_it_enters_waits_as_that_class_does() {
        let mut world = World::new();
        let y = world.text(U8);
        let (lent_y, _) = world.lend(y, SHARED);
        let x = world.meets(lent_y);
        let (lent_x, _) = world.lend(x, SHARED);
        world.enter(lent_x, x);

        world.settle();

        world.no_refusals();
        assert_eq!(join_lowers(&world.classes, x), Join::Waits);
        assert_eq!(join_lowers(&world.classes, lent_x), Join::Waits);
    }

    /// Two cycles, one entering the other: `a` and `b` are one class, `c` and
    /// `d` another, and the head goes from the first to the second.
    #[test]
    fn a_cycle_that_enters_another_stays_a_class_of_its_own() {
        let mut world = World::new();
        let one = world.text(U8);
        let (a, b, c, d) = (
            world.undecided(),
            world.undecided(),
            world.undecided(),
            world.undecided(),
        );
        world.enter(one, a);
        world.enter(a, b);
        world.enter(b, a);
        world.enter(b, c);
        world.enter(c, d);
        world.enter(d, c);

        world.settle();

        world.no_refusals();
        assert!(world.same(a, b) && world.same(c, d));
        assert!(!world.same(a, c));
        assert_eq!(world.shape(d), U8);
    }

    /// `p` enters `r`, `r` enters `q`, and joining `Some(p)` with `Some(q)` makes
    /// `p` and `q` one: the union closes a cycle that was not there before it.
    #[test]
    fn a_cycle_a_union_of_parts_closes_is_merged_too() {
        let mut world = World::new();
        let (p, q, r) = (world.undecided(), world.undecided(), world.undecided());
        world.enter(p, r);
        world.enter(r, q);
        let first = world.text(Shape::Option(p));
        let second = world.text(Shape::Option(q));
        let if_class = world.meets(first);
        world.enter(second, if_class);

        world.settle();

        world.no_refusals();
        assert!(world.same(p, q));
        assert!(world.same(r, p));
    }

    /// `let mut a = &mut x; let mut b = &y; a = b; b = a;`: one class, shared,
    /// which the `&mut x` that entered it reaches by the reborrow.
    #[test]
    fn a_reference_that_enters_a_cycle_reaches_its_join_by_the_reborrow() {
        for reversed in [false, true] {
            let mut world = World::new();
            let (x, y) = (world.text(U8), world.text(U8));
            let (exclusive, _) = world.lend(x, EXCLUSIVE);
            let a = world.undecided();
            let exclusive_meets = world.enter(exclusive, a);
            let (shared, _) = world.lend(y, SHARED);
            let b = world.meets(shared);
            world.enter(b, a);
            world.enter(a, b);
            if reversed {
                world.open.reverse();
            }

            world.settle();

            world.no_refusals();
            assert!(world.same(a, b));
            assert_eq!(world.target(a).0, SHARED);
            assert_eq!(world.target(exclusive).0, EXCLUSIVE);
            assert_eq!(world.outcome.adjustments, adjusted(&[(exclusive_meets, &[Adjustment::Coerce(Coercion::Reborrow)])]));
        }
    }

    #[test]
    fn a_cycle_through_what_a_declaration_writes_takes_it_and_stays_external() {
        let mut world = World::new();
        let x = world.undecided();
        let declared = world.external(U8);
        world.enter(x, declared);
        world.enter(declared, x);

        world.settle();

        world.no_refusals();
        assert!(world.same(x, declared));
        assert!(world.classes.is_external(x));
    }

    /// What the union gives back is judged: nothing else would decide the two
    /// classes here, so only the merge can.
    #[test]
    fn a_merge_judges_what_it_gives_back() {
        let mut world = World::new();
        let x = world.undecided();
        let declared = world.external(U8);
        let other = world.external(Shape::Scalar(Scalar::U16));
        let other_meets = world.enter(other, x);
        world.enter(x, declared);
        world.enter(declared, x);

        assert!(world.outcome.merge_cycles(&mut world.classes, &world.scopes));

        assert_eq!(world.outcome.refusals.len(), 1);
        assert_eq!(world.outcome.refusals[0].cause, Cause::Mismatch);
        assert!(world.outcome.refusals[0].items.contains(&other_meets));
    }

    #[test]
    fn a_cycle_through_two_heads_is_refused_naming_what_it_went_by() {
        let mut world = World::new();
        let first = world.external(U8);
        let second = world.external(Shape::Scalar(Scalar::U16));
        let (between, back) = (world.undecided(), world.undecided());
        world.enter(first, between);
        world.enter(between, second);
        world.enter(second, back);
        world.enter(back, first);

        world.settle();

        assert_eq!(world.outcome.refusals.len(), 1);
        assert_eq!(world.outcome.refusals[0].cause, Cause::Mismatch);
        assert!(world.same(first, second));
    }

    /// A bound between two decided classes was given back when it was recorded,
    /// so no class shows the cycle it closes. Judging each bound on the way
    /// round still decides one head, and refuses two.
    #[test]
    fn a_cycle_through_a_bound_between_decided_classes_is_judged_all_the_way_round() {
        for (second_head, refused) in [(U8, false), (Shape::Scalar(Scalar::U16), true)] {
            let mut world = World::new();
            let first = world.external(U8);
            let second = world.external(second_head.clone());
            let between = world.undecided();
            world.enter(first, second);
            world.enter(second, between);
            world.enter(between, first);

            assert!(!world.outcome.merge_cycles(&mut world.classes, &world.scopes));
            world.settle();

            assert_eq!(world.shape(between), second_head);
            assert_eq!(!world.outcome.refusals.is_empty(), refused);
        }
    }

    /// `x = Some(x);`: the class is no cycle of entering, as `Some(x)` is another
    /// class, and its head holds it as a part (O5).
    #[test]
    fn a_class_that_enters_itself_as_a_part_is_refused_as_its_own_part() {
        let mut world = World::new();
        let x = world.undecided();
        let some = world.text(Shape::Option(x));
        world.enter(some, x);

        world.settle();

        assert!(world.outcome.refusals.iter().any(|refusal| refusal.cause == Cause::Occurs));
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

    /// `let v = q.x; Some(q)` entering what is decided already as `Some(&{x})`.
    /// Nothing is carried to a decided class, so only judging the bound makes `q`
    /// the reference, which leaves `x` with no place on it.
    #[test]
    fn a_judgment_against_a_decided_class_releases_what_its_union_leaves() {
        let mut world = World::new();
        let q = world.undecided();
        let (x_part, x_at) = project(&mut world, q, field("x"));
        let first = world.text(Shape::Option(q));
        let field_class = world.text(U8);
        let by = world.items.push("written");
        let root = world.scopes.root();
        let fields = BTreeMap::from([(Name::new("x"), field_class)]);
        let object = world.classes.decided(root, Shape::Object(fields), BTreeSet::from([by]));
        let reference = world.classes.decided(root, Shape::Ref(SHARED, object), BTreeSet::from([by]));
        let decided = world.classes.decided(root, Shape::Option(reference), BTreeSet::from([by]));
        world.enter(first, decided);

        world.settle();

        world.no_refusals();
        assert!(world.same(q, reference));
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
        assert_eq!(world.outcome.adjustments, adjusted(&[(meets, &[Adjustment::Coerce(Coercion::Reborrow)])]));
        assert!(world.same(x, target));
    }
}
