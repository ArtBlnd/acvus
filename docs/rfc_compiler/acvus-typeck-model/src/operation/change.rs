//! Changes: what a boundary's class reaches its upper bound by (O11).
//!
//! A change goes by class nodes and members, so it waits for both class nodes to
//! be decided, which for a reference takes its target's head too. It changes
//! nothing below the class node: each part there is one type with the part it
//! corresponds to (O9).
//!
//! Only the stage of coercion is made here. No conversion is made yet, so what
//! no coercion reaches is refused, and a function value going to a function type
//! waits, as its instances are not recorded yet (O30).

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::bound::{Bound, BoundOf};
use crate::primitive::class::ClassKind;
use crate::primitive::record::Coercion;
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{ClassNode, Constructor, Mutability, Shape};
use crate::store::classes::Classes;

use super::Outcome;
use super::bound::{decided_by, shape_of};

/// The class node of `class`, once it is decided. No class node is partly
/// decided: a reference has none until its target has a head.
pub(super) fn class_node<K>(classes: &Classes<K>, class: K::Id) -> Option<ClassNode>
where
    K: ClassKind,
{
    match shape_of(classes, class)? {
        Shape::Ref(mutability, target) => Some(ClassNode::Ref(*mutability, shape_of(classes, *target)?.constructor()?)),
        shape => Some(ClassNode::Plain(shape.constructor()?)),
    }
}

/// What a stage of coercion comes to.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Stage<I> {
    /// It goes by these coercions, none being the empty change, and these pairs
    /// of parts below the class node are one type, the second corresponding to
    /// the first.
    Goes {
        coercions: BTreeSet<Coercion>,
        parts: Vec<(I, I)>,
    },
    /// A function value type going to a function type, whose instances are not
    /// recorded yet.
    ToFnType,
    NoChange,
}

/// The parts of `fewer` with the part `more` has in the same place, where `more`
/// has a place for each.
fn in_places_of<P>(fewer: &Shape<P>, more: &Shape<P>) -> Option<Vec<(P, P)>>
where
    P: Copy,
{
    fewer
        .places()
        .into_iter()
        .map(|(key, part)| more.place(&key).map(|other| (*part, *other)))
        .collect()
}

/// The one stage of coercion that goes from `from` to `to`, which are decided,
/// with their class nodes (Changes).
fn stage<K>(classes: &Classes<K>, from: &Shape<K::Id>, to: &Shape<K::Id>, nodes: (&ClassNode, &ClassNode)) -> Stage<K::Id>
where
    K: ClassKind,
{
    let goes = |coercions: &[Coercion], parts| Stage::Goes {
        coercions: coercions.iter().copied().collect(),
        parts,
    };
    match nodes {
        (ClassNode::Plain(Constructor::Never), ClassNode::Plain(Constructor::Never)) => goes(&[], Vec::new()),
        (ClassNode::Plain(Constructor::Never), _) => goes(&[Coercion::Never], Vec::new()),
        (ClassNode::Ref(from_mutability, from_target), ClassNode::Ref(to_mutability, to_target)) => {
            let (Shape::Ref(_, from_class), Shape::Ref(_, to_class)) = (from, to) else {
                unreachable!("a class node is a reference's where the head is one");
            };
            let mut coercions = BTreeSet::new();
            match (from_mutability, to_mutability) {
                (Mutability::Exclusive, Mutability::Shared) => {
                    coercions.insert(Coercion::Reborrow);
                }
                (Mutability::Shared, Mutability::Exclusive) => return Stage::NoChange,
                _ => {}
            }
            let parts = match (from_target, to_target) {
                (Constructor::Array(_), Constructor::Slice) => {
                    coercions.insert(Coercion::Unsize);
                    let (Some(Shape::Array(_, from_elem)), Some(Shape::Slice(to_elem))) =
                        (shape_of(classes, *from_class), shape_of(classes, *to_class))
                    else {
                        unreachable!("a class node holds the constructor of its target's head");
                    };
                    vec![(*from_elem, *to_elem)]
                }
                (same, other) if same == other => vec![(*from_class, *to_class)],
                _ => return Stage::NoChange,
            };
            Stage::Goes { coercions, parts }
        }
        (ClassNode::Plain(Constructor::FnValue), ClassNode::Plain(Constructor::Fn(_))) => Stage::ToFnType,
        (ClassNode::Plain(same), ClassNode::Plain(other)) if same == other => {
            let widened = from.layer() != to.layer();
            match same {
                Constructor::Object => match in_places_of(to, from) {
                    Some(fields) if widened => goes(&[Coercion::DropFields], flipped(fields)),
                    Some(fields) => goes(&[], flipped(fields)),
                    None => Stage::NoChange,
                },
                Constructor::Enum(_) | Constructor::Error => match in_places_of(from, to) {
                    Some(variants) if widened => goes(&[Coercion::AddVariants], variants),
                    Some(variants) => goes(&[], variants),
                    None => Stage::NoChange,
                },
                Constructor::FnValue => match (from, to) {
                    (Shape::FnValue(fewer), Shape::FnValue(more)) if !fewer.is_subset(more) => Stage::NoChange,
                    _ if widened => goes(&[Coercion::AddClosures], Vec::new()),
                    _ => goes(&[], Vec::new()),
                },
                _ => match in_places_of(from, to) {
                    Some(parts) => goes(&[], parts),
                    None => Stage::NoChange,
                },
            }
        }
        _ => Stage::NoChange,
    }
}

fn flipped<P>(pairs: Vec<(P, P)>) -> Vec<(P, P)> {
    pairs.into_iter().map(|(first, second)| (second, first)).collect()
}

impl<K> Outcome<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// Makes the change `bound` asks for, and says whether it is done with. It
    /// is not while a class node is undecided, or a function type waits for its
    /// instances.
    pub(super) fn judge_change(&mut self, classes: &mut Classes<K>, scopes: &Scopes, bound: Bound<K>, at: ItemId) -> bool {
        let (Some(from_node), Some(to_node)) = (class_node(classes, bound.lower()), class_node(classes, bound.upper())) else {
            return false;
        };
        let (Some(from), Some(to)) = (
            shape_of(classes, bound.lower()).cloned(),
            shape_of(classes, bound.upper()).cloned(),
        ) else {
            unreachable!("a class node is of a decided class");
        };
        match stage(classes, &from, &to, (&from_node, &to_node)) {
            Stage::ToFnType => false,
            Stage::NoChange => {
                let mut items = decided_by(classes, bound.lower());
                items.extend(decided_by(classes, bound.upper()));
                items.insert(at);
                self.refusals.push(Refusal {
                    cause: Cause::NoChange,
                    items,
                });
                true
            }
            Stage::Goes { coercions, parts } => {
                let adjustments = coercions.into_iter().map(Adjustment::Coerce);
                if adjustments.len() > 0 {
                    self.adjustments.entry(at).or_default().extend(adjustments);
                }
                for (part, corresponding) in parts {
                    self.one_type(classes, scopes, part, corresponding, at);
                }
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::operation::settle::Settled;
    use crate::operation::testing::{EXCLUSIVE, SHARED, U8, World};
    use crate::primitive::adjustment::Adjustments;
    use crate::primitive::class::Live;
    use crate::primitive::id::Arena;
    use crate::primitive::ty::Scalar;
    use crate::primitive::{ClassId, ClosureId, Name};

    const U16: Shape<ClassId> = Shape::Scalar(Scalar::U16);

    fn object(world: &mut World, fields: &[&str], declared: bool) -> ClassId {
        let mut held = BTreeMap::new();
        for field in fields {
            let part = if declared { world.external(U8) } else { world.text(U8) };
            held.insert(Name::new(*field), part);
        }
        if declared {
            world.external(Shape::Object(held))
        } else {
            world.text(Shape::Object(held))
        }
    }

    fn coerced(at: ItemId, coercions: &[Coercion]) -> Adjustments {
        let adjustments = coercions.iter().copied().map(Adjustment::Coerce).collect();
        BTreeMap::from([(at, adjustments)])
    }

    fn refused_at(world: &World, cause: Cause, at: ItemId) {
        assert_eq!(world.outcome.refusals.len(), 1, "{:?}", world.outcome.refusals);
        assert_eq!(world.outcome.refusals[0].cause, cause);
        assert!(world.outcome.refusals[0].items.contains(&at));
    }

    #[test]
    fn the_empty_change_records_no_adjustment() {
        let mut world = World::new();
        let argument = world.text(U8);
        let parameter = world.external(U8);
        world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, Adjustments::new());
        assert_eq!(world.outcome.bounds.len(), 0);
    }

    /// `take(return 0);`, with `take(u8)`.
    #[test]
    fn never_goes_to_any_type() {
        let mut world = World::new();
        let argument = world.text(Shape::Never);
        let parameter = world.external(U8);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Never]));
    }

    #[test]
    fn never_going_to_never_is_the_empty_change() {
        let mut world = World::new();
        let argument = world.text(Shape::Never);
        let parameter = world.external(Shape::Never);
        world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    #[test]
    fn nothing_but_never_goes_to_never() {
        let mut world = World::new();
        let argument = world.text(U8);
        let parameter = world.external(Shape::Never);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    #[test]
    fn a_reference_goes_from_exclusive_to_shared_and_not_back() {
        let mut world = World::new();
        let x = world.text(U8);
        let (exclusive, _) = world.lend(x, EXCLUSIVE);
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(SHARED, declared));
        let (_, at) = world.pass(exclusive, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Reborrow]));

        let mut world = World::new();
        let x = world.text(U8);
        let (shared, _) = world.lend(x, SHARED);
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(EXCLUSIVE, declared));
        let (_, at) = world.pass(shared, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// `&mut [T; n]` to `&[T]` takes both in the one stage, and the element below
    /// is one type.
    #[test]
    fn an_array_behind_a_reference_goes_to_a_slice() {
        for (mutability, expected) in [
            (SHARED, &[Coercion::Unsize][..]),
            (EXCLUSIVE, &[Coercion::Reborrow, Coercion::Unsize][..]),
        ] {
            let mut world = World::new();
            let elem = world.text(U8);
            let array = world.text(Shape::Array(2, elem));
            let (lent, _) = world.lend(array, mutability);
            let hole = world.undecided();
            let slice = world.external(Shape::Slice(hole));
            let parameter = world.external(Shape::Ref(SHARED, slice));
            let (_, at) = world.pass(lent, parameter);

            assert_eq!(world.settle_fully(), Settled::Finished);

            world.no_refusals();
            assert_eq!(world.outcome.adjustments, coerced(at, expected));
            assert!(world.same(elem, hole));
        }
    }

    /// An array by value has no unsizing: its size is part of the value.
    #[test]
    fn an_array_not_behind_a_reference_does_not_go_to_a_slice() {
        let mut world = World::new();
        let elem = world.text(U8);
        let array = world.text(Shape::Array(2, elem));
        let hole = world.undecided();
        let slice = world.external(Shape::Slice(hole));
        let (_, at) = world.pass(array, slice);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    #[test]
    fn a_slice_behind_a_reference_does_not_go_to_an_array() {
        let mut world = World::new();
        let elem = world.undecided();
        let slice = world.text(Shape::Slice(elem));
        let (lent, _) = world.lend(slice, SHARED);
        let declared = world.external(U8);
        let array = world.external(Shape::Array(2, declared));
        let parameter = world.external(Shape::Ref(SHARED, array));
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    #[test]
    fn an_object_goes_to_one_with_only_some_of_its_fields() {
        let mut world = World::new();
        let argument = object(&mut world, &["x", "y"], false);
        let parameter = object(&mut world, &["x"], true);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::DropFields]));
    }

    #[test]
    fn an_object_does_not_go_to_one_with_a_field_it_lacks() {
        let mut world = World::new();
        let argument = object(&mut world, &["x"], false);
        let parameter = object(&mut world, &["x", "y"], true);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// The fields that stay are parts below the class node, so no change alters
    /// them: `u16` does not reach a declared `u8` there.
    #[test]
    fn a_field_that_stays_is_one_type_with_the_field_it_goes_to() {
        let mut world = World::new();
        let sixteen = world.text(U16);
        let y = world.text(U8);
        let argument = world.text(Shape::Object(BTreeMap::from([
            (Name::new("x"), sixteen),
            (Name::new("y"), y),
        ])));
        let parameter = object(&mut world, &["x"], true);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
    }

    /// Members of a target are not in a reference's class node, so what lies
    /// behind it is one type: the fields of what is lent cannot be dropped.
    #[test]
    fn an_object_behind_a_reference_keeps_its_fields() {
        let mut world = World::new();
        let argument = object(&mut world, &["x", "y"], false);
        let (lent, _) = world.lend(argument, SHARED);
        let target = object(&mut world, &["x"], true);
        let parameter = world.external(Shape::Ref(SHARED, target));
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    /// The same holds of variants: a reference points at the value as it is, and
    /// no change makes another value for it to point at.
    #[test]
    fn an_enum_behind_a_reference_keeps_its_variants() {
        let mut world = World::new();
        let payload = world.external(U8);
        let one = BTreeMap::from([(Name::new("A"), payload)]);
        let two = BTreeMap::from([(Name::new("A"), payload), (Name::new("B"), payload)]);
        let argument = world.external(Shape::Enum(Name::new("E"), one));
        let (lent, _) = world.lend(argument, SHARED);
        let target = world.external(Shape::Enum(Name::new("E"), two));
        let parameter = world.external(Shape::Ref(SHARED, target));
        let (_, at) = world.pass(lent, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
        assert_eq!(world.outcome.adjustments, Adjustments::new());
    }

    #[test]
    fn an_enum_or_an_error_goes_to_one_with_more_variants_and_not_fewer() {
        let enumerate = |variants: &[&str], part: ClassId| -> BTreeMap<Name, ClassId> {
            variants.iter().map(|variant| (Name::new(*variant), part)).collect()
        };
        type Kind = fn(BTreeMap<Name, ClassId>) -> Shape<ClassId>;
        let kinds: [Kind; 2] = [|variants| Shape::Enum(Name::new("E"), variants), Shape::Error];
        for kind in kinds {
            let mut world = World::new();
            let payload = world.external(U8);
            let argument = world.external(kind(enumerate(&["A"], payload)));
            let parameter = world.external(kind(enumerate(&["A", "B"], payload)));
            let (_, at) = world.pass(argument, parameter);
            assert_eq!(world.settle_fully(), Settled::Finished);
            world.no_refusals();
            assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::AddVariants]));

            let mut world = World::new();
            let payload = world.external(U8);
            let argument = world.external(kind(enumerate(&["A", "B"], payload)));
            let parameter = world.external(kind(enumerate(&["A"], payload)));
            let (_, at) = world.pass(argument, parameter);
            assert_eq!(world.settle_fully(), Settled::Finished);
            refused_at(&world, Cause::NoChange, at);
        }
    }

    #[test]
    fn enums_of_two_names_do_not_change_into_one_another() {
        let mut world = World::new();
        let argument = world.external(Shape::Enum(Name::new("E"), BTreeMap::new()));
        let parameter = world.external(Shape::Enum(Name::new("F"), BTreeMap::new()));
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    #[test]
    fn a_function_value_goes_to_one_with_more_closures_and_not_fewer() {
        let mut closures = Arena::<ClosureId, ()>::default();
        let (first, second) = (closures.push(()), closures.push(()));
        let values = |ids: &[ClosureId]| Shape::<ClassId>::FnValue(ids.iter().copied().collect());

        let mut world = World::new();
        let argument = world.external(values(&[first]));
        let parameter = world.external(values(&[first, second]));
        let (_, at) = world.pass(argument, parameter);
        assert_eq!(world.settle_fully(), Settled::Finished);
        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::AddClosures]));

        let mut world = World::new();
        let argument = world.external(values(&[first, second]));
        let parameter = world.external(values(&[first]));
        let (_, at) = world.pass(argument, parameter);
        assert_eq!(world.settle_fully(), Settled::Finished);
        refused_at(&world, Cause::NoChange, at);
    }

    /// Its instances are not recorded yet (O30), so it is not made, and settling
    /// says so.
    #[test]
    fn a_function_value_going_to_a_function_type_waits() {
        let mut closures = Arena::<ClosureId, ()>::default();
        let closure = closures.push(());
        let mut world = World::new();
        let argument = world.external(Shape::FnValue(BTreeSet::from([closure])));
        let result = world.external(U8);
        let parameter = world.external(Shape::Fn(Vec::new(), result));
        let (boundary, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::AwaitsInstances);

        world.no_refusals();
        assert_eq!(
            world.outcome.bounds,
            BTreeSet::from([(
                Bound::Change {
                    lower: boundary,
                    upper: parameter
                },
                at
            )])
        );
    }

    #[test]
    fn heads_of_two_constructors_do_not_change_into_one_another() {
        let mut world = World::new();
        let argument = world.text(U16);
        let parameter = world.external(U8);
        let (_, at) = world.pass(argument, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::NoChange, at);
    }

    /// `let a; g(&a);`, with `g(&[u8])`: the boundary's head is `&`, and what its
    /// target is is not known, so no change can be chosen. The head is given by hand
    /// here, to one upper bounds would not give: it is the wait that is looked at.
    #[test]
    fn a_change_waits_for_the_target_of_a_reference() {
        let mut world = World::new();
        let a = world.undecided();
        let (lent, _) = world.lend(a, SHARED);
        let declared = world.external(U8);
        let slice = world.external(Shape::Slice(declared));
        let parameter = world.external(Shape::Ref(SHARED, slice));
        let (boundary, at) = world.pass(lent, parameter);

        world.settle();

        world.no_refusals();
        assert!(!world.classes.is_decided(boundary));

        let elem = world.external(U8);
        world.decide(a, Shape::Array(2, elem));
        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.outcome.adjustments, coerced(at, &[Coercion::Unsize]));
    }

    /// `a` is decided as a reference, and its target is not: a class node is not
    /// partly decided, however much of it would do.
    #[test]
    fn no_class_node_is_partly_decided() {
        let mut world = World::new();
        let target = world.undecided();
        let argument = world.external(Shape::Ref(SHARED, target));
        let declared = world.external(U8);
        let parameter = world.external(Shape::Ref(SHARED, declared));
        let boundary = world.external(Shape::Ref(SHARED, target));
        let at = world.record(Bound::Change {
            lower: boundary,
            upper: parameter,
        });
        world.enter(argument, boundary);

        world.settle();

        assert_eq!(
            world.outcome.bounds,
            BTreeSet::from([(
                Bound::Change {
                    lower: boundary,
                    upper: parameter
                },
                at
            )])
        );
    }

    /// Whatever the coercion, a part takes the declared part it corresponds to as
    /// an upper bound, and stays undecided for what enters it to decide. Taken the
    /// other way round, the two would be one class, and the part decided already.
    #[test]
    fn a_part_takes_the_declared_part_as_an_upper_bound_under_every_coercion() {
        type Heads = fn(ClassId, ClassId) -> (Shape<ClassId>, Shape<ClassId>);
        fn named(entries: &[(&str, ClassId)]) -> BTreeMap<Name, ClassId> {
            entries.iter().map(|(name, part)| (Name::new(*name), *part)).collect()
        }
        let kinds: [Heads; 5] = [
            |part, declared| (Shape::Option(part), Shape::Option(declared)),
            |part, declared| (Shape::Tuple(vec![part]), Shape::Tuple(vec![declared])),
            |part, declared| {
                (
                    Shape::Object(named(&[("A", part), ("B", part)])),
                    Shape::Object(named(&[("A", declared)])),
                )
            },
            |part, declared| {
                (
                    Shape::Enum(Name::new("E"), named(&[("A", part)])),
                    Shape::Enum(Name::new("E"), named(&[("A", declared), ("B", declared)])),
                )
            },
            |part, declared| {
                (
                    Shape::Error(named(&[("A", part)])),
                    Shape::Error(named(&[("A", declared), ("B", declared)])),
                )
            },
        ];
        for kind in kinds {
            let mut world = World::new();
            let part = world.undecided();
            let declared = world.external(U8);
            let (from, to) = kind(part, declared);
            let argument = world.text(from);
            let parameter = world.external(to);
            world.pass(argument, parameter);

            world.settle();

            world.no_refusals();
            assert!(!world.classes.is_decided(part));
            assert!(world.classes.upper_items(part).contains_key(&world.classes.find(declared).id()));

            assert_eq!(world.settle_fully(), Settled::Finished);

            world.no_refusals();
            assert_eq!(world.shape(part), U8);
        }
    }

    /// The same behind a reference, where the element is what lies below the
    /// class node.
    #[test]
    fn an_element_takes_the_declared_element_as_an_upper_bound_when_unsized() {
        let mut world = World::new();
        let elem = world.undecided();
        let array = world.text(Shape::Array(2, elem));
        let (lent, _) = world.lend(array, SHARED);
        let declared = world.external(U8);
        let slice = world.external(Shape::Slice(declared));
        let parameter = world.external(Shape::Ref(SHARED, slice));
        world.pass(lent, parameter);

        world.settle();

        world.no_refusals();
        assert!(!world.classes.is_decided(elem));
        assert!(world.classes.upper_items(elem).contains_key(&world.classes.find(declared).id()));

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(elem), U8);
    }

    /// Example 12: `take(Some(&mut x));`, with `take(Option<&T>)`. The change
    /// between the two `Option`s is empty, and alters no part.
    #[test]
    fn a_part_is_not_changed_at_a_boundary() {
        let mut world = World::new();
        let x = world.text(U8);
        let (exclusive, _) = world.lend(x, EXCLUSIVE);
        let some = world.text(Shape::Option(exclusive));
        let hole = world.undecided();
        let reference = world.external(Shape::Ref(SHARED, hole));
        let parameter = world.external(Shape::Option(reference));
        let (_, at) = world.pass(some, parameter);

        assert_eq!(world.settle_fully(), Settled::Finished);

        refused_at(&world, Cause::Mismatch, at);
    }

    /// Example 6: `let mut buf = new(); push(&mut buf, 1u16); consume(buf);`, with
    /// `new() -> Option<T>`, `push(&mut Option<T>, T)` and `consume(Option<u8>)`.
    /// The boundary does not change what enters a part: `buf` is `Option<u16>`,
    /// and it is `consume` that refuses it, whichever class is visited first.
    #[test]
    fn a_boundary_does_not_change_what_enters_a_part() {
        for reversed in [false, true] {
            let mut world = World::new();
            let new_hole = world.undecided();
            let result = world.external(Shape::Option(new_hole));
            let buf = world.meets(result);

            let push_hole = world.undecided();
            let push_option = world.external(Shape::Option(push_hole));
            let push_first = world.external(Shape::Ref(EXCLUSIVE, push_option));
            let (lent, _) = world.lend(buf, EXCLUSIVE);
            world.pass(lent, push_first);
            let sixteen = world.text(U16);
            world.enter(sixteen, push_hole);

            let declared_u8 = world.external(U8);
            let consume = world.external(Shape::Option(declared_u8));
            let (_, consume_at) = world.pass(buf, consume);
            if reversed {
                world.open.reverse();
            }

            assert_eq!(world.settle_fully(), Settled::Finished);

            refused_at(&world, Cause::Mismatch, consume_at);
            assert!(world.same(new_hole, push_hole));
            assert_eq!(world.shape(buf), Shape::Option(new_hole));
            assert_eq!(world.shape(new_hole), U16);
        }
    }

    /// Example 4: `let a = new(); fill_all(&mut a);`,
    /// with `new() -> Option<T>` and `fill_all(&mut Option<u8>)`. The part has the
    /// upper bound `u8`, and takes it once nothing else moves.
    #[test]
    fn a_part_a_boundary_reaches_has_the_part_it_corresponds_to_as_an_upper_bound() {
        let mut world = World::new();
        let hole = world.undecided();
        let result = world.external(Shape::Option(hole));
        let a = world.meets(result);
        let (lent, _) = world.lend(a, EXCLUSIVE);
        let declared_u8 = world.external(U8);
        let declared = world.external(Shape::Option(declared_u8));
        let parameter = world.external(Shape::Ref(EXCLUSIVE, declared));
        world.pass(lent, parameter);

        world.settle();

        world.no_refusals();
        assert!(!world.classes.is_decided(hole));
        let uppers: Vec<ClassId> = world.classes.upper_items(hole).into_keys().collect();
        assert_eq!(uppers, vec![world.classes.find(declared_u8).id()]);
        assert_eq!(world.outcome.bounds, crate::primitive::class::Bounds::<Live>::new());

        assert_eq!(world.settle_fully(), Settled::Finished);

        world.no_refusals();
        assert_eq!(world.shape(hole), U8);
    }
}
