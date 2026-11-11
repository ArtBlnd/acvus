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
//! Where the language gives nothing between the two, the argument goes through a
//! declared function `P -> R`, and that is being passed twice: the argument is
//! passed to `P`, by what the language gives alone, and what the function gives is
//! passed to the type given, by the language or through a function again. Exactly
//! one such way must go. A row of the table is read as any type given is, as far
//! as it is written, so a row that writes a head deeper is told later.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::adjustment::{Adjustment, Subtyping};
use crate::primitive::class::{ClassKind, State, Template};
use crate::primitive::constraint::Constraint;
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{Mutability, Shape};
use crate::primitive::{ItemId, Name, TemplateId};
use crate::store::classes::Classes;

use super::super::law::{Looked, Pair, Scribe};
use super::functions::{Functions, Row};
use super::{Then, Told, View};

/// Types as they are written somewhere: in the store being solved, or in a row of
/// the table, not made into classes of that store yet.
pub(super) trait Written {
    type Id: Copy + Ord;

    /// The head written at `id`, where one is.
    fn head(&self, id: Self::Id) -> Option<&Shape<Self::Id>>;

    /// Whether a declaration writes what is at `id`, which then never changes.
    fn declared(&self, id: Self::Id) -> bool;

    /// Whether `id` is a hole of a signature not made yet. It has no head and
    /// will have none to wait for: it is whatever it is made one type with.
    fn open(&self, id: Self::Id) -> bool;
}

impl<K> Written for View<'_, K>
where
    K: ClassKind,
{
    type Id = K::Id;

    fn head(&self, id: K::Id) -> Option<&Shape<K::Id>> {
        View::head(self, id)
    }

    fn declared(&self, id: K::Id) -> bool {
        self.classes.is_external(id)
    }

    fn open(&self, _: K::Id) -> bool {
        false
    }
}

impl Written for Classes<Template> {
    type Id = TemplateId;

    fn head(&self, id: TemplateId) -> Option<&Shape<TemplateId>> {
        match &self.get(self.find(id)).state {
            State::Decided(decided) => Some(&decided.shape),
            State::Undecided(_) => None,
        }
    }

    fn declared(&self, id: TemplateId) -> bool {
        self.is_decided(id)
    }

    fn open(&self, id: TemplateId) -> bool {
        !self.is_decided(id)
    }
}

/// What the language gives between an argument and a type given.
enum Given<A, G> {
    /// The argument goes: these parts are one type, and these are taken.
    Goes { pairs: Vec<(A, G)>, adjustments: Vec<Adjustment> },
    /// It cannot be told until this part of the argument has a head, which will
    /// be held against this one.
    Reads(A, Shape<G>),
    /// The argument is a function value and the type given a function type: it
    /// goes as each of its closures does, used as that type.
    Calls,
    /// The language gives nothing between the two.
    Nothing,
}

fn loosened(by: Subtyping, where_loosened: bool) -> Vec<Adjustment> {
    where_loosened.then_some(Adjustment::Subtype(by)).into_iter().collect()
}

/// What the language gives between what `from` writes at `argument` and what `to`
/// writes at `given`. `read` takes each part of the argument that is read.
fn by_the_language<A, G>(from: &A, argument: A::Id, to: &G, given: G::Id, read: &mut BTreeSet<A::Id>) -> Given<A::Id, G::Id>
where
    A: Written,
    G: Written,
{
    let goes = |pairs, adjustments| Given::Goes { pairs, adjustments };
    // A hole takes what comes, as it comes.
    let Some(wanted) = to.head(given) else {
        return goes(vec![(argument, given)], Vec::new());
    };
    read.insert(argument);
    let Some(head) = from.head(argument) else {
        return if from.open(argument) {
            goes(vec![(argument, given)], Vec::new())
        } else {
            Given::Reads(argument, wanted.clone())
        };
    };
    let paired = |parts: Vec<(&A::Id, &G::Id)>| parts.into_iter().map(|(a, g)| (*a, *g)).collect();
    match (head, wanted) {
        (Shape::Never, Shape::Never) => goes(Vec::new(), Vec::new()),
        (Shape::Never, _) => goes(Vec::new(), vec![Adjustment::Subtype(Subtyping::Never)]),
        (Shape::Ref(of_argument, target), Shape::Ref(of_given, given_target)) => {
            if (*of_argument, *of_given) == (Mutability::Shared, Mutability::Exclusive) {
                return Given::Nothing;
            }
            let reborrowed = loosened(Subtyping::Reborrow, of_argument != of_given);
            match behind_a_reference(from, *target, to, *given_target, *of_given, read) {
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
            goes(paired(kept), loosened(Subtyping::DropFields, given_fields.len() < fields.len()))
        }
        (Shape::Enum(name, variants), Shape::Enum(given_name, given_variants)) if name == given_name => {
            into_more_variants(variants, given_variants)
        }
        (Shape::Error(variants), Shape::Error(given_variants)) => into_more_variants(variants, given_variants),
        (Shape::FnValue(_), Shape::Fn(..)) => Given::Calls,
        _ if head.layer() == wanted.layer() => {
            let parts = head.places().into_iter().zip(wanted.places()).map(|((_, a), (_, g))| (a, g)).collect();
            goes(paired(parts), Vec::new())
        }
        _ => Given::Nothing,
    }
}

/// What the language gives between variants and the variants given: each variant
/// of the argument is one given, its payload one type with that one's, and more
/// may be given besides.
fn into_more_variants<A, G>(variants: &BTreeMap<Name, A>, given: &BTreeMap<Name, G>) -> Given<A, G>
where
    A: Copy,
    G: Copy,
{
    let Some(pairs) = variants
        .iter()
        .map(|(variant, payload)| given.get(variant).map(|given_payload| (*payload, *given_payload)))
        .collect::<Option<Vec<_>>>()
    else {
        return Given::Nothing;
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
fn behind_a_reference<A, G>(
    from: &A,
    mut target: A::Id,
    to: &G,
    given: G::Id,
    of_given: Mutability,
    read: &mut BTreeSet<A::Id>,
) -> Given<A::Id, G::Id>
where
    A: Written,
    G: Written,
{
    let goes = |pairs, adjustments| Given::Goes { pairs, adjustments };
    let wanted = match to.head(given) {
        Some(wanted) if to.declared(given) => wanted,
        _ => return goes(vec![(target, given)], Vec::new()),
    };
    let mut read_through = Vec::new();
    loop {
        read.insert(target);
        let Some(head) = from.head(target) else {
            return if from.open(target) {
                goes(vec![(target, given)], read_through)
            } else {
                Given::Reads(target, wanted.clone())
            };
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
            _ => return Given::Nothing,
        }
    }
}

/// Whether what `from` writes at `part` is the type `to` writes at `given`.
enum Same<A> {
    Yes,
    /// It cannot be told until this part has a head.
    Reads(A),
    No,
}

/// Whether the part of an argument is the type a row writes for it. Nothing the
/// language gives goes at a part, so the two are of one head wherever the row
/// writes one, at every depth it writes. A hole is any type.
fn same<A, G>(from: &A, part: A::Id, to: &G, given: G::Id, read: &mut BTreeSet<A::Id>) -> Same<A::Id>
where
    A: Written,
    G: Written,
{
    let Some(wanted) = to.head(given) else {
        return Same::Yes;
    };
    read.insert(part);
    let Some(head) = from.head(part) else {
        return if from.open(part) { Same::Yes } else { Same::Reads(part) };
    };
    if head.layer() != wanted.layer() {
        return Same::No;
    }
    let below: Vec<(A::Id, G::Id)> = head
        .places()
        .into_iter()
        .zip(wanted.places())
        .map(|((_, below), (_, given_below))| (*below, *given_below))
        .collect();
    all_same(from, to, &below, read)
}

/// Whether each of `pairs` is of one type as far as `to` writes it.
fn all_same<A, G>(from: &A, to: &G, pairs: &[(A::Id, G::Id)], read: &mut BTreeSet<A::Id>) -> Same<A::Id>
where
    A: Written,
    G: Written,
{
    let mut waits = None;
    for (part, given) in pairs {
        match same(from, *part, to, *given, read) {
            Same::Yes => {}
            Same::Reads(class) => waits = waits.or(Some(class)),
            Same::No => return Same::No,
        }
    }
    waits.map_or(Same::Yes, Same::Reads)
}

/// A row a receiver goes to: the parts of the receiver that are one type with
/// parts of the row's first parameter, and what the receiver takes to go.
pub(in crate::operation) struct Received<K>
where
    K: ClassKind,
{
    pub(super) row: usize,
    pub(super) pairs: Vec<(K::Id, TemplateId)>,
    pub(super) adjustments: Vec<Adjustment>,
}

/// Every row of the name `name` that `receiver` goes to, and each part of the
/// receiver that has to have a head before a row can be told of.
///
/// A receiver goes to the first parameter of a row as any argument is passed,
/// by what the language gives alone, and is read as far as the row writes. One
/// thing more is given a receiver: one that is no reference is lent where the
/// parameter is a reference.
pub(super) fn receivers<K>(
    view: &View<'_, K>,
    functions: &Functions,
    name: &Name,
    receiver: K::Id,
    read: &mut BTreeSet<K::Id>,
) -> (Vec<Received<K>>, BTreeSet<K::Id>)
where
    K: ClassKind,
{
    let mut found = Vec::new();
    let mut waits = BTreeSet::new();
    for (index, row) in functions.rows.iter().enumerate() {
        let (true, Some(first)) = (row.name == *name, row.parameters.first()) else {
            continue;
        };
        match received(view, receiver, &row.template, *first, read) {
            Given::Goes { pairs, adjustments } => match all_same(view, &row.template, &pairs, read) {
                Same::Yes => found.push(Received {
                    row: index,
                    pairs,
                    adjustments,
                }),
                Same::Reads(class) => {
                    waits.insert(class);
                }
                Same::No => {}
            },
            Given::Reads(class, _) => {
                waits.insert(class);
            }
            Given::Calls => todo!("a function value as the receiver of a function type a row writes"),
            Given::Nothing => {}
        }
    }
    (found, waits)
}

/// What the language gives between a receiver and the first parameter of a
/// function called by name: what it gives any argument, and, where the receiver
/// is no reference and the parameter is one, the receiver lent.
fn received<A, G>(from: &A, receiver: A::Id, to: &G, parameter: G::Id, read: &mut BTreeSet<A::Id>) -> Given<A::Id, G::Id>
where
    A: Written,
    G: Written,
{
    let lent = match (from.head(receiver), to.head(parameter)) {
        (Some(Shape::Ref(..)), _) | (None, _) => None,
        (Some(_), Some(Shape::Ref(mutability, target))) => Some((*mutability, *target)),
        (Some(_), _) => None,
    };
    let Some((mutability, target)) = lent else {
        return by_the_language(from, receiver, to, parameter, read);
    };
    match behind_a_reference(from, receiver, to, target, mutability, read) {
        Given::Goes { pairs, adjustments } => Given::Goes {
            pairs,
            adjustments: [Adjustment::Lend(mutability)].into_iter().chain(adjustments).collect(),
        },
        other => other,
    }
}

/// Where a class written in a way lies: in the store being solved, or in the
/// signature of the step of that number.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Site<I> {
    Solved(I),
    Step(usize, TemplateId),
}

/// Two classes written along a way that are one type.
type Joined<K> = (Site<<K as ClassKind>::Id>, Site<<K as ClassKind>::Id>);

/// One way through declared functions: each row taken in turn, and the parts that
/// are one type along it.
pub(in crate::operation) struct Way<K>
where
    K: ClassKind,
{
    rows: Vec<usize>,
    pairs: Vec<Joined<K>>,
    /// What is taken on the way into each row, then on the way out of the last.
    taken: Vec<Vec<Adjustment>>,
}

/// Every way from `argument` to `given` through the rows of `functions`, and each
/// part of the argument that has to have a head before a row can be told of.
///
/// The way into the first row is told by the argument, read as far as that row's
/// parameter writes. Every way after it is told by the rows alone. No row is taken
/// twice on one way.
fn ways<K>(view: &View<'_, K>, functions: &Functions, argument: K::Id, given: K::Id, read: &mut BTreeSet<K::Id>) -> (Vec<Way<K>>, BTreeSet<K::Id>)
where
    K: ClassKind,
{
    let mut found = Vec::new();
    let mut waits = BTreeSet::new();
    for (index, row) in functions.rows.iter().enumerate() {
        let Some(parameter) = row.only_parameter() else {
            continue;
        };
        match by_the_language(view, argument, &row.template, parameter, read) {
            Given::Goes { pairs, adjustments } => {
                // A row is chosen among rows, so the argument is read as far as
                // the row writes: its parts are the types the row writes for them.
                match all_same(view, &row.template, &pairs, read) {
                    Same::Yes => {}
                    Same::Reads(class) => {
                        waits.insert(class);
                        continue;
                    }
                    Same::No => continue,
                }
                let pairs = pairs
                    .into_iter()
                    .map(|(a, p)| (Site::Solved(a), Site::Step(0, p)))
                    .collect();
                let way = Way {
                    rows: vec![index],
                    pairs,
                    taken: vec![adjustments],
                };
                onward(view, functions, row, way, given, &mut found);
            }
            Given::Reads(class, _) => {
                waits.insert(class);
            }
            Given::Calls => todo!("a function value passed to a function type a row writes"),
            Given::Nothing => {}
        }
    }
    (found, waits)
}

/// Each way that goes on from `way`, whose last row is `row`, to `given`: what
/// the row gives is passed to the type given, by the language, or through one
/// more row.
fn onward<K>(view: &View<'_, K>, functions: &Functions, row: &Row, way: Way<K>, given: K::Id, found: &mut Vec<Way<K>>)
where
    K: ClassKind,
{
    let step = way.rows.len() - 1;
    match by_the_language(&row.template, row.result, view, given, &mut BTreeSet::new()) {
        Given::Goes { pairs, adjustments } => {
            if !matches!(all_same(&row.template, view, &pairs, &mut BTreeSet::new()), Same::Yes) {
                return;
            }
            let mut way = way;
            way.pairs
                .extend(pairs.into_iter().map(|(r, g)| (Site::Step(step, r), Site::Solved(g))));
            way.taken.push(adjustments);
            found.push(way);
        }
        Given::Reads(..) => unreachable!("a row's result is written, and has nothing to wait for"),
        Given::Calls => todo!("a function value a row gives, passed to a function type"),
        Given::Nothing => {
            for (index, next) in functions.rows.iter().enumerate() {
                if way.rows.contains(&index) {
                    continue;
                }
                let Some(parameter) = next.only_parameter() else {
                    continue;
                };
                let Given::Goes { pairs, adjustments } =
                    by_the_language(&row.template, row.result, &next.template, parameter, &mut BTreeSet::new())
                else {
                    continue;
                };
                if !matches!(all_same(&row.template, &next.template, &pairs, &mut BTreeSet::new()), Same::Yes) {
                    continue;
                }
                let mut longer = Way {
                    rows: way.rows.clone(),
                    pairs: way.pairs.clone(),
                    taken: way.taken.clone(),
                };
                longer.rows.push(index);
                longer
                    .pairs
                    .extend(pairs.into_iter().map(|(r, p)| (Site::Step(step, r), Site::Step(step + 1, p))));
                longer.taken.push(adjustments);
                onward(view, functions, next, longer, given, found);
            }
        }
    }
}

/// Makes the signature of each row of `way` anew in the store, and gives what the
/// way comes to: the parts that are one type along it, and what is taken, each
/// function called in its turn.
pub(super) fn take<K>(scribe: &mut Scribe<'_, K>, functions: &Functions, way: Way<K>, argument: K::Id, at: ItemId) -> Looked<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let scope = scribe.classes().get(scribe.classes().find(argument)).scope;
    let made: Vec<BTreeMap<TemplateId, K::Id>> = way
        .rows
        .iter()
        .map(|row| scribe.anew(&functions.rows[*row].template, scope, at))
        .collect();
    let solved = |written: Site<K::Id>| match written {
        Site::Solved(class) => class,
        Site::Step(step, class) => made[step][&class],
    };
    let pairs = way.pairs.iter().map(|(a, b)| Pair::Classes(solved(*a), solved(*b))).collect();
    let mut adjustments = Vec::new();
    for (step, taken) in way.taken.into_iter().enumerate() {
        adjustments.extend(taken);
        if let Some(row) = way.rows.get(step) {
            adjustments.push(Adjustment::Row(functions.rows[*row].name.clone()));
        }
    }
    Looked::Holds {
        pairs,
        adjustments,
        constraints: Vec::new(),
    }
}

/// What `argument` being passed to `given` comes to.
///
/// While a part of the argument it has to read has no head, it waits on that
/// part. Until it is told, it may still make what lies below its two ends one type
/// with a hole, so it holds apart its ends and each part it has read or has yet to
/// read: those it only compares, and what lies below them is not decided.
pub(super) fn look<K>(view: &View<'_, K>, functions: &Functions, argument: K::Id, given: K::Id, at: ItemId) -> Told<K>
where
    K: ClassKind,
{
    let mut read = BTreeSet::from([argument, given]);
    let waits = |on: BTreeSet<K::Id>, read: BTreeSet<K::Id>| Looked::Waits {
        on,
        apart: read,
        pairs: Vec::new(),
    };
    match by_the_language(view, argument, view, given, &mut read) {
        Given::Goes { pairs, adjustments } => {
            let pairs = pairs.into_iter().map(|(a, g)| Pair::Classes(a, g)).collect();
            return Told::done(Looked::Holds {
                pairs,
                adjustments,
                constraints: Vec::new(),
            });
        }
        Given::Reads(class, against) => {
            return Told {
                looked: waits(BTreeSet::from([class]), read),
                asks: vec![(class, against)],
                then: None,
            };
        }
        Given::Calls => {
            return Told {
                looked: Looked::holds_alone(),
                asks: Vec::new(),
                then: Some(Then::Calls),
            };
        }
        // A declared function is looked for only where the language gives nothing.
        Given::Nothing => {}
    }
    let (mut found, on) = ways(view, functions, argument, given, &mut read);
    // A row not told of yet may be a second way, so none is taken before all are
    // told. What a row holds the argument against is no head asked of it: a
    // function called gives type inference nothing.
    if !on.is_empty() {
        return Told::done(waits(on, read));
    }
    // Exactly one way must go: none goes nowhere, and of two neither is chosen.
    let (Some(way), true) = (found.pop(), found.is_empty()) else {
        return Told::done(Looked::Refused(Refused {
            cause: Cause::NoChange,
            items: BTreeSet::from([at]),
            of: vec![argument, given],
        }));
    };
    Told {
        looked: Looked::holds_alone(),
        asks: Vec::new(),
        then: Some(Then::Way(way)),
    }
}
