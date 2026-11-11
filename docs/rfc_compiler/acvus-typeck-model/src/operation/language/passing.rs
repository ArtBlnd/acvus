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
//!
//! Where a function is chosen by what goes to it, a row a value goes through or
//! a function a receiver goes to, the argument has to fit the type the function
//! writes (`Fits`): to go to it by what the language gives, each part of it
//! being the type written for it.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::adjustment::{Adjustment, Subtyping};
use crate::primitive::class::{ClassKind, State, Template};
use crate::primitive::constraint::{Bound, Constraint};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{Mutability, Shape};
use crate::primitive::{ItemId, Name, TemplateId};
use crate::store::classes::Classes;

use super::super::law::{Looked, Pair, Scribe};
use super::functions::{Functions, Named, Row, Used};
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
    /// It cannot be told yet: this part of the argument is undecided, and the
    /// head it comes to have will be held against this one.
    Undecided(A, Shape<G>),
    /// The argument is a function, a value or of a function type, and the type
    /// given a function type: passing the one to the other is a call
    /// ([`super::calling`]), and what the call comes to is told there and not by
    /// reading the two.
    Calls(A, G),
    /// The argument does not go: the language gives nothing between the two.
    DoesNotGo,
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
    // A hole takes what comes, as it comes: a class nothing says the head of.
    let Some(wanted) = to.head(given) else {
        return goes(vec![(argument, given)], Vec::new());
    };
    read.insert(argument);
    let Some(head) = from.head(argument) else {
        return if from.open(argument) {
            goes(vec![(argument, given)], Vec::new())
        } else {
            Given::Undecided(argument, wanted.clone())
        };
    };
    let paired = |parts: Vec<(&A::Id, &G::Id)>| parts.into_iter().map(|(a, g)| (*a, *g)).collect();
    match (head, wanted) {
        (Shape::Never, Shape::Never) => goes(Vec::new(), Vec::new()),
        (Shape::Never, _) => goes(Vec::new(), vec![Adjustment::Subtype(Subtyping::Never)]),
        (Shape::Ref(of_argument, target), Shape::Ref(of_given, given_target)) => {
            if (*of_argument, *of_given) == (Mutability::Shared, Mutability::Exclusive) {
                return Given::DoesNotGo;
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
                return Given::DoesNotGo;
            };
            goes(paired(kept), loosened(Subtyping::DropFields, given_fields.len() < fields.len()))
        }
        (Shape::Enum(name, variants), Shape::Enum(given_name, given_variants)) if name == given_name => {
            into_more_variants(variants, given_variants)
        }
        (Shape::Error(variants), Shape::Error(given_variants)) => into_more_variants(variants, given_variants),
        (Shape::FnValue(_) | Shape::Fn(..), Shape::Fn(..)) => Given::Calls(argument, given),
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
                Given::Undecided(target, wanted.clone())
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
            _ => return Given::DoesNotGo,
        }
    }
}

/// Whether what `from` writes at `part` is the type `to` writes at `given`.
enum Same<A> {
    Yes,
    /// It cannot be told yet: this part is undecided.
    Undecided(A),
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
        return if from.open(part) { Same::Yes } else { Same::Undecided(part) };
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
    let mut undecided = None;
    for (part, given) in pairs {
        match same(from, *part, to, *given, read) {
            Same::Yes => {}
            Same::Undecided(below) => undecided = undecided.or(Some(below)),
            Same::No => return Same::No,
        }
    }
    undecided.map_or(Same::Yes, Same::Undecided)
}

/// Whether an argument fits a type a function writes, where the function is
/// chosen by what goes to it: a function a receiver goes to, or a row a value
/// goes through.
enum Fits<A, G> {
    /// It fits, so.
    Yes(Fit<A, G>),
    /// It cannot be told yet: this part of the argument is undecided.
    Undecided(A),
    /// The argument goes by what the language gives, and a part of it is not
    /// the type the function writes for it.
    PartMismatch,
    /// The argument does not go: the language gives nothing between the two.
    DoesNotGo,
}

/// How an argument fits a type a function writes: what holds of the two once
/// the function's signature is made.
#[derive(Clone)]
pub(in crate::operation) struct Fit<A, G> {
    /// Each part of the argument with what the function writes for it: the two
    /// are one type.
    pub(super) pairs: Vec<(A, G)>,
    /// What the argument takes to go.
    pub(super) adjustments: Vec<Adjustment>,
    /// The argument and the type, where they are a function and a function
    /// type: the one is passed to the other.
    pub(super) passed: Vec<(A, G)>,
}

impl<A, G> Fit<A, G> {
    /// The same, each class of the argument given by `of_argument` and each the
    /// function writes by `of_given`.
    fn placed<B, H, P, Q>(self, of_argument: P, of_given: Q) -> Fit<B, H>
    where
        P: Fn(A) -> B,
        Q: Fn(G) -> H,
    {
        let placed = |pairs: Vec<(A, G)>| pairs.into_iter().map(|(a, g)| (of_argument(a), of_given(g))).collect();
        Fit {
            pairs: placed(self.pairs),
            adjustments: self.adjustments,
            passed: placed(self.passed),
        }
    }
}

/// Whether an argument fits a type a function writes, `language` being what the
/// language gives between the two.
///
/// It fits where the language gives that it goes and each part of it is the
/// type the function writes for it: a function is chosen among functions, so
/// the argument is read as far as the function writes. A function fits a
/// function type as it is. Whether it can be used as that type is told by the
/// call and not by reading, so it is passed to the type once the signature is
/// made, and what the call refuses is no reason to choose another function.
fn fits_by<A, G>(language: Given<A::Id, G::Id>, from: &A, to: &G, read: &mut BTreeSet<A::Id>) -> Fits<A::Id, G::Id>
where
    A: Written,
    G: Written,
{
    match language {
        Given::Goes { pairs, adjustments } => match all_same(from, to, &pairs, read) {
            Same::Yes => Fits::Yes(Fit {
                pairs,
                adjustments,
                passed: Vec::new(),
            }),
            Same::Undecided(part) => Fits::Undecided(part),
            Same::No => Fits::PartMismatch,
        },
        Given::Undecided(part, _) => Fits::Undecided(part),
        Given::Calls(function, function_type) => Fits::Yes(Fit {
            pairs: Vec::new(),
            adjustments: Vec::new(),
            passed: vec![(function, function_type)],
        }),
        Given::DoesNotGo => Fits::DoesNotGo,
    }
}

/// Whether what `from` writes at `argument` fits what `to` writes at `given`.
fn fits<A, G>(from: &A, argument: A::Id, to: &G, given: G::Id, read: &mut BTreeSet<A::Id>) -> Fits<A::Id, G::Id>
where
    A: Written,
    G: Written,
{
    let language = by_the_language(from, argument, to, given, read);
    fits_by(language, from, to, read)
}

/// Whether `receiver` fits the first parameter of a function called by name: as
/// an argument fits, with what the language gives a receiver besides.
fn receiver_fits<A, G>(from: &A, receiver: A::Id, to: &G, parameter: G::Id, read: &mut BTreeSet<A::Id>) -> Fits<A::Id, G::Id>
where
    A: Written,
    G: Written,
{
    let language = to_a_receiver(from, receiver, to, parameter, read);
    fits_by(language, from, to, read)
}

/// What the language gives between a receiver and the first parameter of a
/// function called by name: what it gives any argument, and, where the receiver
/// is no reference and the parameter is one, the receiver lent.
fn to_a_receiver<A, G>(from: &A, receiver: A::Id, to: &G, parameter: G::Id, read: &mut BTreeSet<A::Id>) -> Given<A::Id, G::Id>
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

/// A function a receiver goes to, and how the receiver fits its first
/// parameter.
pub(in crate::operation) struct Received<K>
where
    K: ClassKind,
{
    pub(super) function: Named,
    pub(super) fit: Fit<K::Id, TemplateId>,
}

/// Every function of the name `name` that `receiver` goes to, and each
/// undecided part of the receiver that a function cannot be told of without.
///
/// A receiver goes to a function whose first parameter it fits
/// ([`receiver_fits`]).
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
    let mut undecided = BTreeSet::new();
    for (function, template, first) in functions.taking_a_receiver(name) {
        match receiver_fits(view, receiver, template, first, read) {
            Fits::Yes(fit) => found.push(Received { function, fit }),
            Fits::Undecided(part) => {
                undecided.insert(part);
            }
            Fits::PartMismatch | Fits::DoesNotGo => {}
        }
    }
    (found, undecided)
}

/// Where a class written in a way lies: in the store being solved, or in the
/// signature of the step of that number.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Site<I> {
    Solved(I),
    Step(usize, TemplateId),
}

/// A step of a way: how what comes to it fits what it goes to, each class by
/// where it lies.
type Step<K> = Fit<Site<<K as ClassKind>::Id>, Site<<K as ClassKind>::Id>>;

/// One way through declared functions: each row taken in turn, and each step
/// between.
///
/// The argument fits the parameter of the first row, what each row gives fits
/// the parameter of the next, and what the last gives fits the type given. So a
/// way that is found has one step more than it has rows.
#[derive(Clone)]
pub(in crate::operation) struct Way<K>
where
    K: ClassKind,
{
    rows: Vec<usize>,
    steps: Vec<Step<K>>,
}

/// Every way from `argument` to `given` through the rows of `functions`, and each
/// undecided part of the argument that a row cannot be told of without.
///
/// A way begins with a row whose parameter the argument fits, and goes on from
/// what that row gives ([`onward`]). No row is taken twice on one way.
fn ways<K>(view: &View<'_, K>, functions: &Functions, argument: K::Id, given: K::Id, read: &mut BTreeSet<K::Id>) -> (Vec<Way<K>>, BTreeSet<K::Id>)
where
    K: ClassKind,
{
    let mut found = Vec::new();
    let mut undecided = BTreeSet::new();
    for (index, row) in functions.rows.iter().enumerate() {
        let Some(parameter) = row.only_parameter() else {
            continue;
        };
        match fits(view, argument, &row.template, parameter, read) {
            Fits::Yes(fit) => {
                let into_the_row = fit.placed(Site::Solved, |written| Site::Step(0, written));
                let begun = Way {
                    rows: vec![index],
                    steps: vec![into_the_row],
                };
                onward(view, functions, row, begun, given, &mut found);
            }
            Fits::Undecided(part) => {
                undecided.insert(part);
            }
            Fits::PartMismatch | Fits::DoesNotGo => {}
        }
    }
    (found, undecided)
}

/// Each way that goes on from `way`, whose last row is `row`, to `given`.
///
/// | What the row gives and the type given | The way |
/// |---|---|
/// | the one fits the other | ends there |
/// | the one goes to the other by the language, and a part is another type | goes nowhere |
/// | the one does not go to the other by the language | goes on through each row whose parameter what the row gives fits |
///
/// A row's result is written, so no part of it is undecided.
fn onward<K>(view: &View<'_, K>, functions: &Functions, row: &Row, way: Way<K>, given: K::Id, found: &mut Vec<Way<K>>)
where
    K: ClassKind,
{
    let step = way.rows.len() - 1;
    let of_the_row = |written| Site::Step(step, written);
    match fits(&row.template, row.result, view, given, &mut BTreeSet::new()) {
        Fits::Yes(fit) => {
            let mut ended = way;
            ended.steps.push(fit.placed(of_the_row, Site::Solved));
            found.push(ended);
        }
        Fits::Undecided(_) => unreachable!("a row's result is written, and no part of it is undecided"),
        Fits::PartMismatch => {}
        Fits::DoesNotGo => {
            for (index, next) in functions.rows.iter().enumerate() {
                if way.rows.contains(&index) {
                    continue;
                }
                let Some(parameter) = next.only_parameter() else {
                    continue;
                };
                let Fits::Yes(fit) = fits(&row.template, row.result, &next.template, parameter, &mut BTreeSet::new()) else {
                    continue;
                };
                let mut longer = way.clone();
                longer.rows.push(index);
                longer.steps.push(fit.placed(of_the_row, |written| Site::Step(step + 1, written)));
                onward(view, functions, next, longer, given, found);
            }
        }
    }
}

/// Makes the signature of each row of `way` anew in the store, and gives what the
/// way comes to: what holds at each step of it, each function called in its turn.
pub(super) fn take<K>(scribe: &mut Scribe<'_, K>, functions: &Functions, way: Way<K>, argument: K::Id, at: ItemId) -> Looked<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let scope = scribe.classes().get(scribe.classes().find(argument)).scope;
    // The signature of each row on the way, made anew.
    let used: Vec<Used<K>> = way.rows.iter().map(|row| functions.rows[*row].used(scribe, scope, at)).collect();
    let solved = |written: Site<K::Id>| match written {
        Site::Solved(class) => class,
        Site::Step(step, class) => used[step].classes[&class],
    };

    // One type: what each step pairs, and each brand a row makes with the one
    // made for this use of it.
    let of_the_steps = way.steps.iter().flat_map(|step| &step.pairs).map(|(a, b)| Pair::Classes(solved(*a), solved(*b)));
    let brands = used.iter().flat_map(|used| used.brands.iter().cloned());
    let pairs = of_the_steps.chain(brands).collect();

    // Passed: each function a step brings to a function type, now that the
    // signature that writes the type is made.
    let functions_passed = way.steps.iter().flat_map(|step| &step.passed);
    let constraints = functions_passed
        .map(|(function, function_type)| {
            Bound::Change {
                lower: solved(*function),
                upper: solved(*function_type),
            }
            .into()
        })
        .collect();

    // Taken: at each step what the value takes to go, and then the row it goes
    // into, the last step going into none.
    let mut adjustments = Vec::new();
    for (step, fit) in way.steps.iter().enumerate() {
        adjustments.extend(fit.adjustments.iter().cloned());
        if let Some(row) = way.rows.get(step) {
            adjustments.push(Adjustment::Row(functions.rows[*row].name.clone()));
        }
    }
    Looked::Holds {
        pairs,
        adjustments,
        constraints,
    }
}

/// What `argument` being passed to `given` comes to.
///
/// | The language gives | The passing |
/// |---|---|
/// | that the argument goes | holds: the parts are one type, and what is taken is taken |
/// | nothing yet, a part of the argument being undecided | waits on that part, whose head will be held against the one given |
/// | a call | holds, and is the call ([`super::calling`]) |
/// | that the argument does not go | goes through declared functions ([`through_declared_functions`]) |
///
/// What is passed to a class bound to be one type with a declared class is
/// passed to the declared class. Such a class has no head yet and is to have
/// that one, so it is no hole: a hole is a class nothing says the head of.
pub(super) fn look<K>(view: &View<'_, K>, functions: &Functions, argument: K::Id, given: K::Id, at: ItemId) -> Told<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let given = view.declared_for(given).unwrap_or(given);
    let mut read = BTreeSet::from([argument, given]);
    match by_the_language(view, argument, view, given, &mut read) {
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
        Given::Calls(..) => Told {
            looked: Looked::holds_alone(),
            asks: Vec::new(),
            then: Some(Then::Calls(given)),
        },
        // A declared function is looked for only where the argument does not go
        // by the language.
        Given::DoesNotGo => through_declared_functions(view, functions, argument, given, read, at),
    }
}

/// What `argument` being passed to `given` through declared functions comes to,
/// `read` being what has been read of the argument so far.
///
/// Exactly one way must go: none goes nowhere, and of two neither is chosen. A
/// row that cannot be told of yet, a part of the argument being undecided, may
/// be a second way, so none is taken before all are told of. What a row holds
/// the argument against is no head asked of it: a function called gives type
/// inference nothing.
fn through_declared_functions<K>(view: &View<'_, K>, functions: &Functions, argument: K::Id, given: K::Id, mut read: BTreeSet<K::Id>, at: ItemId) -> Told<K>
where
    K: ClassKind,
{
    let (mut found, undecided) = ways(view, functions, argument, given, &mut read);
    if !undecided.is_empty() {
        return Told::done(waiting(undecided, read));
    }
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

/// A passing that waits for each class of `on` to have a head, having read
/// `read`.
///
/// Until it is told, it may still make what lies below its two ends one type
/// with a hole. So it holds apart its ends and each part it has read or has yet
/// to read: those it only compares, and what lies below them is not decided.
fn waiting<K>(on: BTreeSet<K::Id>, read: BTreeSet<K::Id>) -> Looked<K>
where
    K: ClassKind,
{
    Looked::Waits {
        on,
        apart: read,
        pairs: Vec::new(),
    }
}
