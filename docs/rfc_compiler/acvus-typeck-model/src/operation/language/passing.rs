//! Being passed: an argument's type may be used as the type it is passed to.
//!
//! The type given is read as far as it is written, and the argument only as far
//! as that. Where the type given writes a hole, the argument is not read there:
//! what stands at that place is one type with the hole. Where it writes a head,
//! the argument's head there must be that head with each index of it loosened or
//! left as it is:
//!
//! | Index of a head | Loosened |
//! |---|---|
//! | a reference's mutability | exclusive to shared |
//! | an object's fields | to fewer, where no reference is above it |
//! | an enum's or an `Error`'s variants | to more |
//! | an array's length, behind a reference | to none: a slice |
//! | the count of references above a type | to one, each below the first being shared |
//!
//! Each is a family the language alone can give, one rule for each value of the
//! index, and they are taken together in the one look. A function value's
//! closures are no such index: they carry the bounds of their parameters, and a
//! function value is passed only to one of the same closures. `!` is passed to
//! any type.
//!
//! An index is loosened only where the two do not match as they are written, so
//! `&&u8` passed to `&T` leaves `T` as `&u8`, and passed to `&u8` it is read
//! through.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::ClassKind;
use crate::primitive::record::Coercion;
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::ty::{Mutability, Shape};

use super::super::law::{Looked, Pair};
use super::{Told, View, decided_by};

/// What the language gives between an argument and the type it is passed to.
enum Given<K>
where
    K: ClassKind,
{
    /// The argument goes: these parts are one type, and these are taken.
    Goes { pairs: Vec<Pair<K>>, adjustments: Vec<Adjustment> },
    /// It cannot be told until this class has a head, which will be held against
    /// this one, where the type given writes one there.
    Reads(K::Id, Option<Shape<K::Id>>),
    /// The language gives nothing between the two.
    Nothing,
}

/// What `argument` being passed to `given` comes to.
///
/// While a class it has to read has no head, it waits on that class. Until it is
/// told, it may still make what lies below its two ends one type with a hole, so
/// it holds apart its ends and each class it has read or has yet to read: those
/// it only compares, and what lies below them is not decided.
pub(super) fn look<K>(view: &View<'_, K>, argument: K::Id, given: K::Id, at: ItemId) -> Told<K>
where
    K: ClassKind,
{
    let classes = view.classes;
    let mut read = BTreeSet::from([argument, given]);
    let looked = match language(view, argument, given, &mut read) {
        Given::Goes { pairs, adjustments } => Looked::Holds { pairs, adjustments },
        Given::Reads(class, against) => {
            let looked = Looked::Waits {
                on: BTreeSet::from([class]),
                apart: read,
                pairs: Vec::new(),
            };
            let asks = against.map(|head| (class, head)).into_iter().collect();
            return Told { looked, asks };
        }
        // A declared function is looked for only where the language gives nothing.
        // None is declared in this law yet, so none goes.
        Given::Nothing => {
            let mut items = decided_by(classes, argument);
            items.extend(decided_by(classes, given));
            items.insert(at);
            Looked::Refused(Refusal {
                cause: Cause::NoChange,
                items,
            })
        }
    };
    Told::done(looked)
}

fn goes<K>(pairs: Vec<Pair<K>>, adjustments: Vec<Adjustment>) -> Given<K>
where
    K: ClassKind,
{
    Given::Goes { pairs, adjustments }
}

fn loosened(by: Coercion, where_loosened: bool) -> Vec<Adjustment> {
    where_loosened.then_some(Adjustment::Coerce(by)).into_iter().collect()
}

/// What the language gives between the heads of `argument` and `given`.
fn language<K>(view: &View<'_, K>, argument: K::Id, given: K::Id, read: &mut BTreeSet<K::Id>) -> Given<K>
where
    K: ClassKind,
{
    let Some(to) = view.head(given) else {
        return Given::Reads(given, None);
    };
    let Some(from) = view.head(argument) else {
        return Given::Reads(argument, Some(to.clone()));
    };
    let by_name = |parts: Vec<(&K::Id, &K::Id)>| parts.into_iter().map(|(a, b)| Pair::Classes(*a, *b)).collect();
    match (from, to) {
        (Shape::Never, Shape::Never) => goes(Vec::new(), Vec::new()),
        (Shape::Never, _) => goes(Vec::new(), vec![Adjustment::Coerce(Coercion::Never)]),
        (Shape::Ref(of_argument, target), Shape::Ref(of_given, given_target)) => {
            if (*of_argument, *of_given) == (Mutability::Shared, Mutability::Exclusive) {
                return Given::Nothing;
            }
            let reborrowed = loosened(Coercion::Reborrow, of_argument != of_given);
            match behind_a_reference(view, *target, *given_target, read) {
                Given::Goes { pairs, adjustments } => goes(pairs, adjustments.into_iter().chain(reborrowed).collect()),
                other => other,
            }
        }
        (Shape::Object(fields), Shape::Object(given_fields)) => {
            let Some(kept) = given_fields
                .iter()
                .map(|(name, part)| fields.get(name).map(|field| (field, part)))
                .collect::<Option<Vec<_>>>()
            else {
                return Given::Nothing;
            };
            goes(by_name(kept), loosened(Coercion::DropFields, given_fields.len() < fields.len()))
        }
        (Shape::Enum(name, variants), Shape::Enum(given_name, given_variants)) if name == given_name => {
            let Some(held) = variants
                .iter()
                .map(|(variant, part)| given_variants.get(variant).map(|other| (part, other)))
                .collect::<Option<Vec<_>>>()
            else {
                return Given::Nothing;
            };
            goes(by_name(held), loosened(Coercion::AddVariants, variants.len() < given_variants.len()))
        }
        (Shape::Error(variants), Shape::Error(given_variants)) => {
            let Some(held) = variants
                .iter()
                .map(|(variant, part)| given_variants.get(variant).map(|other| (part, other)))
                .collect::<Option<Vec<_>>>()
            else {
                return Given::Nothing;
            };
            goes(by_name(held), loosened(Coercion::AddVariants, variants.len() < given_variants.len()))
        }
        (Shape::FnValue(_), Shape::Fn(..)) => todo!("a function value passed to a function type: its instances"),
        _ if from.layer() == to.layer() => {
            let parts = from.places().into_iter().zip(to.places()).map(|((_, a), (_, b))| (a, b)).collect();
            goes(by_name(parts), Vec::new())
        }
        _ => Given::Nothing,
    }
}

/// What the language gives between the target of the argument's reference and
/// the target the type given writes.
///
/// A hole takes what comes, as it is. A head written is matched as it is where
/// the argument's target has that head, with every part below one type. Else an
/// array's length is loosened to a slice, or one reference more is read through,
/// which must be a shared one, as it is copied out. No field is dropped here: a
/// reference points at the value as it lies.
fn behind_a_reference<K>(view: &View<'_, K>, mut target: K::Id, given: K::Id, read: &mut BTreeSet<K::Id>) -> Given<K>
where
    K: ClassKind,
{
    if !view.classes.is_external(given) {
        return goes(vec![Pair::Classes(target, given)], Vec::new());
    }
    let Some(to) = view.head(given) else {
        unreachable!("a class a declaration writes is decided");
    };
    let mut read_through = Vec::new();
    loop {
        read.insert(target);
        let Some(from) = view.head(target) else {
            return Given::Reads(target, Some(to.clone()));
        };
        if from.layer() == to.layer() {
            return goes(vec![Pair::Classes(target, given)], read_through);
        }
        match (from, to) {
            (Shape::Array(_, element), Shape::Slice(given_element)) => {
                read_through.push(Adjustment::Coerce(Coercion::Unsize));
                return goes(vec![Pair::Classes(*element, *given_element)], read_through);
            }
            (Shape::Ref(Mutability::Shared, inner), _) => {
                read_through.push(Adjustment::Deref);
                target = *inner;
            }
            _ => return Given::Nothing,
        }
    }
}
