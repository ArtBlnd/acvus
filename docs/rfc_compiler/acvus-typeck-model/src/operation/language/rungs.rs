//! The rungs: the head a class may take, each rung only where none before it
//! decides anything.
//!
//! | Rung | A class takes | Where |
//! |---|---|---|
//! | 0 | the join of what enters it | every class something enters, once each of those has a head |
//! | 1 | the one head its upper bounds give it | a class nothing enters |
//! | 2 | the head made from the parts named of it | a class nothing enters and no upper bound gives a head |
//! | 3 | the fallback of its literals' domain | a class nothing enters and no upper bound gives a head |
//! | 4 | `!` | a class nothing else gives a head |
//!
//! `!` says nothing: where the join of what enters a class is `!`, rung 0 gives
//! the head made from the parts named of the class, as rung 2 does where nothing
//! enters.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::{ItemId, Name};
use crate::primitive::constraint::{Bound, Constraint, ConstraintOf};
use crate::primitive::class::{ClassKind, State};
use crate::primitive::domain::Domain;
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{PartKey, Shape, VariantOf};
use crate::store::classes::Classes;

use super::super::law::Offer;
use super::functions::Functions;
use super::{Looked, Told, View, enter, tell};

pub const COUNT: usize = 5;

pub fn offer<K>(classes: &Classes<K>, functions: &Functions, rung: usize, class: K::Id) -> Offer<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    match rung {
        0 => join(classes, class),
        1 => from_above(classes, functions, class),
        2 => from_parts(classes, functions, class),
        3 => domain(classes, functions, class),
        4 => never(classes, functions, class),
        _ => unreachable!("there are {COUNT} rungs"),
    }
}

/// What enters a class by one bound: its head, where it has one yet, the bound,
/// and the item that recorded the bound.
type Entering<K> = (Option<Shape<<K as ClassKind>::Id>>, Constraint<K>, ItemId);

/// The heads that enter `class`, each with the bound it enters by: none where
/// nothing enters, and a head not given yet where what enters is undecided. A
/// class's entering itself gives nothing to wait for.
fn entering<K>(classes: &Classes<K>, class: K::Id) -> Vec<Entering<K>>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let root = classes.find(class);
    let view = View::of(classes);
    classes
        .constraints()
        .iter()
        .filter(|(constraint, _)| classes.find(constraint.upper()) == root && classes.find(constraint.lower()) != root)
        .filter_map(|(constraint, at)| enter::given(&view, constraint).map(|head| (head, *constraint, *at)))
        .collect()
}

/// Rung 0: the join of the heads that enter the class, once each has a head.
fn join<K>(classes: &Classes<K>, class: K::Id) -> Offer<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let entering = entering(classes, class);
    let Some(heads) = entering.iter().map(|(head, ..)| head.clone()).collect::<Option<Vec<_>>>() else {
        return Offer::Nothing;
    };
    let by: BTreeSet<ItemId> = entering.iter().map(|(.., at)| *at).collect();
    let mut heads = heads.into_iter();
    let Some(first) = heads.next() else {
        return Offer::Nothing;
    };
    match heads.try_fold(first, |joined, head| joined.join(&head)) {
        // `!` is the least head, so what the class's parts make is joined with it.
        Some(Shape::Never) => match made_of_parts(classes, class) {
            Some((shape, named)) => Offer::Head {
                shape,
                by: by.into_iter().chain(named).collect(),
            },
            None => Offer::Head { shape: Shape::Never, by },
        },
        Some(shape) => Offer::Head { shape, by },
        None => Offer::Refused(Refused {
            cause: Cause::Mismatch,
            items: by,
            of: [class].into_iter().chain(entering.iter().map(|(_, constraint, _)| constraint.lower())).collect(),
        }),
    }
}

/// Whether nothing enters `class`.
fn nothing_enters<K>(classes: &Classes<K>, class: K::Id) -> bool
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    entering(classes, class).is_empty()
}

/// The classes `class` enters that have no head yet. The head `class` takes
/// enters each, so what is asked of each is asked of `class` too.
fn entered<K>(classes: &Classes<K>, class: K::Id) -> Vec<K::Id>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let root = classes.find(class);
    classes
        .constraints()
        .iter()
        .filter_map(|(constraint, _)| match constraint {
            Constraint::Bound(Bound::Enter { lower, upper })
                if classes.find(*lower) == root && classes.find(*upper) != root && !classes.is_decided(*upper) =>
            {
                Some(classes.find(*upper).id())
            }
            _ => None,
        })
        .collect()
}

/// Every head `class` will be held against: what each constraint that waits on it
/// asks of it, the heads that have entered a class it enters, and what is asked
/// of that class in turn. `!` is the least head and asks nothing of what meets it.
fn asked<K>(classes: &Classes<K>, functions: &Functions, class: K::Id, seen: &mut BTreeSet<K::Id>) -> Vec<Shape<K::Id>>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let root = classes.find(class).id();
    if !seen.insert(root) {
        return Vec::new();
    }
    let view = View::of(classes);
    let mut heads: Vec<Shape<K::Id>> = classes
        .get(classes.find(root))
        .held
        .waiters
        .iter()
        .flat_map(|(constraint, at)| tell(&view, functions, constraint, *at).asks)
        .filter(|(of, _)| classes.find(*of).id() == root)
        .map(|(_, head)| head)
        .collect();
    for upper in entered(classes, root) {
        heads.extend(entering(classes, upper).into_iter().filter_map(|(head, ..)| head));
        heads.extend(asked(classes, functions, upper, seen));
    }
    heads.retain(|head| *head != Shape::Never);
    heads
}

/// Whether every constraint that waits on `class` would have it take `head`: none
/// would be refused, were that its head. A class it enters would then take the
/// join of `head` and what has entered it, and is asked the same of that.
fn taken<K>(view: View<'_, K>, functions: &Functions, class: K::Id, head: &Shape<K::Id>, seen: &mut BTreeSet<K::Id>) -> bool
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let classes = view.classes;
    let root = classes.find(class).id();
    if !seen.insert(root) {
        return true;
    }
    let view = view.supposing(root, head.clone());
    let refused = classes
        .get(classes.find(root))
        .held
        .waiters
        .iter()
        .any(|(constraint, at)| matches!(tell(&view, functions, constraint, *at), Told { looked: Looked::Refused(_), .. }));
    if refused {
        return false;
    }
    entered(classes, root).into_iter().all(|upper| {
        let mut joined = Some(head.clone());
        for (other, ..) in entering(classes, upper) {
            joined = match (joined, other) {
                (Some(joined), Some(other)) => joined.join(&other),
                (joined, None) => joined,
                (None, _) => None,
            };
        }
        joined.is_some_and(|joined| taken(view.clone(), functions, upper, &joined, seen))
    })
}

/// Rung 1: a class nothing enters takes the one head, of those it will be held
/// against, that every constraint waiting on it would have it take. A literal's class
/// takes one of its domain alone.
fn from_above<K>(classes: &Classes<K>, functions: &Functions, class: K::Id) -> Offer<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    if !nothing_enters(classes, class) {
        return Offer::Nothing;
    }
    let mut candidates: BTreeMap<Shape<()>, Shape<K::Id>> = BTreeMap::new();
    for head in asked(classes, functions, class, &mut BTreeSet::new()) {
        candidates.entry(head.layer()).or_insert(head);
    }
    let domain = domain_of(classes, class);
    let mut taken_heads = candidates.into_values().filter(|head| {
        let in_domain = domain
            .as_ref()
            .is_none_or(|(domain, _)| matches!(head, Shape::Scalar(scalar) if domain.0.contains(scalar)));
        in_domain && taken(View::of(classes), functions, class, head, &mut BTreeSet::new())
    });
    match (taken_heads.next(), taken_heads.next()) {
        (Some(shape), None) => Offer::Head {
            shape,
            by: classes.get(classes.find(class)).held.waiters.iter().map(|(_, at)| *at).collect(),
        },
        _ => Offer::Nothing,
    }
}

/// Whether no head is asked of `class` from above. Where one is, the class has
/// something said of it, and no fallback overrules that, even where no one head
/// will do.
fn nothing_asked<K>(classes: &Classes<K>, functions: &Functions, class: K::Id) -> bool
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    asked(classes, functions, class, &mut BTreeSet::new()).is_empty()
}

/// The domain the literals of `class` leave, where any registered one.
fn domain_of<K>(classes: &Classes<K>, class: K::Id) -> Option<(Domain, BTreeSet<ItemId>)>
where
    K: ClassKind,
{
    let State::Undecided(undecided) = &classes.get(classes.find(class)).state else {
        return None;
    };
    let domain = Domain::intersection(undecided.domains.iter().map(|(domain, _)| domain))?;
    Some((domain, undecided.domains.iter().map(|(_, by)| *by).collect()))
}

/// The parts named on `class` and on each class with no head that it enters, in
/// turn, each with the items that named it. The head `class` takes enters each of
/// those, so a part named there is named of `class` too, as a head asked there
/// is asked of it.
fn parts_named<K>(classes: &Classes<K>, class: K::Id, seen: &mut BTreeSet<K::Id>) -> BTreeMap<PartKey, BTreeSet<ItemId>>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let root = classes.find(class).id();
    if !seen.insert(root) {
        return BTreeMap::new();
    }
    let mut named: BTreeMap<PartKey, BTreeSet<ItemId>> = BTreeMap::new();
    if let State::Undecided(undecided) = &classes.get(classes.find(root)).state {
        for (key, part) in &undecided.parts {
            named
                .entry(key.clone())
                .or_default()
                .extend(part.lower.iter().chain(&part.upper).chain(&part.carried));
        }
    }
    for upper in entered(classes, root) {
        for (key, items) in parts_named(classes, upper, seen) {
            named.entry(key).or_default().extend(items);
        }
    }
    named
}

/// The head made from the parts named of `class`, with the items that named
/// them: an object of the fields named, or the type whose variants the payloads
/// named are of. None where no part is named, where the names are not all of one
/// such type, and where a name does not say the whole head: a target does not say
/// which reference, nor a component how many, nor an element what holds it.
///
/// The parts of the head given stand for no class: a head decided has the
/// class's own parts.
fn made_of_parts<K>(classes: &Classes<K>, class: K::Id) -> Option<(Shape<K::Id>, BTreeSet<ItemId>)>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let named = parts_named(classes, class, &mut BTreeSet::new());
    let by = named.values().flatten().copied().collect();
    let shape = match named.keys().next()? {
        PartKey::Field(_) => Shape::Object(
            named
                .keys()
                .map(|key| match key {
                    PartKey::Field(field) => Some((field.clone(), class)),
                    _ => None,
                })
                .collect::<Option<_>>()?,
        ),
        PartKey::Payload(of, _) => {
            let variants: BTreeMap<Name, K::Id> = named
                .keys()
                .map(|key| match key {
                    PartKey::Payload(other, variant) if other == of => Some((variant.clone(), class)),
                    _ => None,
                })
                .collect::<Option<_>>()?;
            match of {
                VariantOf::Option => Shape::Option(class),
                VariantOf::Result => Shape::Result(class, class),
                VariantOf::Error => Shape::Error(variants),
                VariantOf::Enum(name) => Shape::Enum(name.clone(), variants),
            }
        }
        PartKey::Target | PartKey::Component(_) | PartKey::Elem | PartKey::Param(_) | PartKey::Result | PartKey::Arg(_) | PartKey::Brand => {
            return None;
        }
    };
    Some((shape, by))
}

/// Rung 2: a class that neither lower nor upper bounds decide takes the head made
/// from the parts named of it.
fn from_parts<K>(classes: &Classes<K>, functions: &Functions, class: K::Id) -> Offer<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    if !nothing_enters(classes, class) || !nothing_asked(classes, functions, class) {
        return Offer::Nothing;
    }
    match made_of_parts(classes, class) {
        Some((shape, by)) => Offer::Head { shape, by },
        None => Offer::Nothing,
    }
}

/// Rung 3: a literal's class that neither lower nor upper bounds decide takes the
/// fallback type of its domain's row.
fn domain<K>(classes: &Classes<K>, functions: &Functions, class: K::Id) -> Offer<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    if !nothing_enters(classes, class) || !nothing_asked(classes, functions, class) {
        return Offer::Nothing;
    }
    let Some((domain, by)) = domain_of(classes, class) else {
        return Offer::Nothing;
    };
    if domain.0.is_empty() {
        return Offer::Refused(Refused {
            cause: Cause::EmptyDomain,
            items: by,
            of: vec![class],
        });
    }
    match domain.family() {
        Some(family) => Offer::Head {
            shape: Shape::Scalar(family.fallback()),
            by,
        },
        None => Offer::Nothing,
    }
}

/// Rung 4: a class nothing enters, with no literal, takes `!`: nothing gives it
/// a head.
fn never<K>(classes: &Classes<K>, functions: &Functions, class: K::Id) -> Offer<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    if !nothing_enters(classes, class) || !nothing_asked(classes, functions, class) || domain_of(classes, class).is_some() {
        return Offer::Nothing;
    }
    Offer::Head {
        shape: Shape::Never,
        by: BTreeSet::new(),
    }
}

/// What is refused of `class` when nothing more can be done: a class that is a
/// part of itself has no finite type, and a class still undecided has no type.
pub fn end<K>(classes: &Classes<K>, class: K::Id) -> Vec<Refused<K>>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let root = classes.find(class);
    let named: BTreeSet<ItemId> = match &classes.get(root).state {
        State::Undecided(undecided) => undecided
            .parts
            .values()
            .flat_map(|part| part.lower.iter().chain(&part.upper).chain(&part.carried).copied())
            .collect(),
        State::Decided(_) => BTreeSet::new(),
    };
    let mut refusals = Vec::new();
    if classes.is_own_part(class) {
        refusals.push(Refused {
            cause: Cause::Occurs,
            items: named.clone(),
            of: vec![class],
        });
    }
    if let State::Undecided(undecided) = &classes.get(root).state {
        let constraints = classes
            .constraints()
            .iter()
            .filter(|(constraint, _)| classes.find(constraint.lower()) == root || classes.find(constraint.upper()) == root)
            .map(|(_, at)| *at);
        let domains = undecided.domains.iter().map(|(_, at)| *at);
        refusals.push(Refused {
            cause: Cause::UndecidedClass,
            items: constraints.chain(named.iter().copied()).chain(domains).collect(),
            of: vec![class],
        });
    }
    refusals
}
