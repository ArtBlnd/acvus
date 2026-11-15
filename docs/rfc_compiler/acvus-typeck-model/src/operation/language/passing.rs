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
//! any type. An index is loosened only where the two do not match as they are
//! written, so `&&u8` passed to `&T` leaves `T` as `&u8`, and passed to `&u8` it
//! is read through.
//!
//! Where the language gives nothing between the two, the argument does not go.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::adjustment::{Adjustment, Subtyping};
use crate::primitive::constraint::Constraint;
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{Mutability, Shape};
use crate::primitive::{ClassId, ItemId, Name};

use super::super::law::{Looked, Pair};
use super::{Told, View};

/// What the language gives between an argument and a type given.
enum Given {
    /// The argument goes: these parts are one type, and these are taken.
    Goes {
        pairs: Vec<(ClassId, ClassId)>,
        adjustments: Vec<Adjustment>,
    },
    /// It cannot be told yet: this part of the argument is undecided, and the
    /// head it comes to have will be held against this one.
    Undecided(ClassId, Shape<ClassId>),
    /// The argument does not go: the language gives nothing between the two.
    DoesNotGo,
}

fn loosened(by: Subtyping, where_loosened: bool) -> Vec<Adjustment> {
    where_loosened.then_some(Adjustment::Subtype(by)).into_iter().collect()
}

/// What the language gives between `argument` and `given`. `read` takes each
/// part of the argument that is read.
fn by_the_language(view: &View<'_>, argument: ClassId, given: ClassId, read: &mut BTreeSet<ClassId>) -> Given {
    let goes = |pairs, adjustments| Given::Goes { pairs, adjustments };
    // A hole takes what comes, as it comes: a class nothing says the head of.
    let Some(wanted) = view.head(given) else {
        return goes(vec![(argument, given)], Vec::new());
    };
    read.insert(argument);
    let Some(head) = view.head(argument) else {
        return Given::Undecided(argument, wanted.clone());
    };
    let paired = |parts: Vec<(&ClassId, &ClassId)>| parts.into_iter().map(|(a, g)| (*a, *g)).collect();
    match (head, wanted) {
        (Shape::Never, Shape::Never) => goes(Vec::new(), Vec::new()),
        (Shape::Never, _) => goes(Vec::new(), vec![Adjustment::Subtype(Subtyping::Never)]),
        (Shape::Ref(of_argument, target), Shape::Ref(of_given, given_target)) => {
            if (*of_argument, *of_given) == (Mutability::Shared, Mutability::Exclusive) {
                return Given::DoesNotGo;
            }
            let reborrowed = loosened(Subtyping::Reborrow, of_argument != of_given);
            match behind_a_reference(view, *target, *given_target, *of_given, read) {
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
                return Given::DoesNotGo;
            };
            goes(paired(kept), loosened(Subtyping::DropFields, given_fields.len() < fields.len()))
        }
        (Shape::Enum(name, variants), Shape::Enum(given_name, given_variants)) if name == given_name => {
            into_more_variants(variants, given_variants)
        }
        (Shape::Error(variants), Shape::Error(given_variants)) => into_more_variants(variants, given_variants),
        _ if head.layer() == wanted.layer() => {
            let parts = head.places().into_iter().zip(wanted.places()).map(|((_, a), (_, g))| (a, g)).collect();
            goes(paired(parts), Vec::new())
        }
        _ => Given::DoesNotGo,
    }
}

/// What the language gives between variants and the variants given: each variant
/// of the argument is one given, its payload one type with that one's, and more
/// may be given besides.
fn into_more_variants(variants: &BTreeMap<Name, ClassId>, given: &BTreeMap<Name, ClassId>) -> Given {
    let Some(pairs) = variants
        .iter()
        .map(|(variant, payload)| given.get(variant).map(|given_payload| (*payload, *given_payload)))
        .collect::<Option<Vec<_>>>()
    else {
        return Given::DoesNotGo;
    };
    Given::Goes {
        pairs,
        adjustments: loosened(Subtyping::AddVariants, variants.len() < given.len()),
    }
}

/// What the language gives between the target of the argument's reference and
/// the target the type given writes.
///
/// A hole takes what comes, as it is. A head written is matched as it is where
/// the argument's target has that head, with every part below one type. Else an
/// array's length is loosened to a slice, or one reference more is read through,
/// which must be a shared one, as it is copied out, and so only where the type
/// given is a shared reference: what is copied out of a shared reference is not
/// written through. No field is dropped here: a reference points at the value
/// as it lies.
fn behind_a_reference(view: &View<'_>, mut target: ClassId, given: ClassId, of_given: Mutability, read: &mut BTreeSet<ClassId>) -> Given {
    let goes = |pairs, adjustments| Given::Goes { pairs, adjustments };
    let wanted = match view.head(given) {
        Some(wanted) if view.classes.is_external(given) => wanted,
        _ => return goes(vec![(target, given)], Vec::new()),
    };
    let mut read_through = Vec::new();
    loop {
        read.insert(target);
        let Some(head) = view.head(target) else {
            return Given::Undecided(target, wanted.clone());
        };
        if head.layer() == wanted.layer() {
            return goes(vec![(target, given)], read_through);
        }
        match (head, wanted) {
            (Shape::Array(_, element), Shape::Slice(given_element)) => {
                read_through.push(Adjustment::Subtype(Subtyping::Unsize));
                return goes(vec![(*element, *given_element)], read_through);
            }
            (Shape::Ref(Mutability::Shared, inner), _) if of_given == Mutability::Shared => {
                read_through.push(Adjustment::Deref);
                target = *inner;
            }
            _ => return Given::DoesNotGo,
        }
    }
}

/// What `argument` being passed to `given` comes to.
///
/// | The language gives | The passing |
/// |---|---|
/// | that the argument goes | holds: the parts are one type, and what is taken is taken |
/// | nothing yet, a part of the argument being undecided | waits on that part, whose head will be held against the one given |
/// | that the argument does not go | is refused |
///
/// What is passed to a class bound to be one type with a declared class is
/// passed to the declared class. Such a class has no head yet and is to have
/// that one, so it is no hole: a hole is a class nothing says the head of.
pub(super) fn look(view: &View<'_>, argument: ClassId, given: ClassId, at: ItemId) -> Told {
    let given = view.declared_for(given).unwrap_or(given);
    let mut read = BTreeSet::from([argument, given]);
    match by_the_language(view, argument, given, &mut read) {
        Given::Goes { pairs, adjustments } => Told::done(Looked::Holds {
            pairs: pairs.into_iter().map(|(of_argument, of_given)| Pair::Classes(of_argument, of_given)).collect(),
            adjustments,
            constraints: Vec::new(),
        }),
        Given::Undecided(part, against) => Told {
            looked: waiting(BTreeSet::from([part]), read),
            asks: vec![(part, against)],
            then: None,
        },
        Given::DoesNotGo => Told::done(Looked::Refused(Refused {
            cause: Cause::NoChange,
            items: BTreeSet::from([at]),
            of: vec![argument, given],
        })),
    }
}

/// A passing that waits for each class of `on` to have a head, having read
/// `read`.
///
/// Until it is told, it may still make what lies below its two ends one type
/// with a hole. So it holds apart its ends and each part it has read or has yet
/// to read: those it only compares, and what lies below them is not decided.
fn waiting(on: BTreeSet<ClassId>, read: BTreeSet<ClassId>) -> Looked<Constraint> {
    Looked::Waits {
        on,
        apart: read,
        pairs: Vec::new(),
    }
}
