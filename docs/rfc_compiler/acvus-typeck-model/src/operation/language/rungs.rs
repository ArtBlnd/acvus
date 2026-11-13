//! The rungs: the head a class may take, each rung only where none before it
//! decides anything.
//!
//! | Rung | A class takes | Where |
//! |---|---|---|
//! | 0 | the join of what enters it | every class something enters, once each of those has a head |
//! | 1 | the head of the declared class it is bound to | a class nothing enters |
//! | 2 | the one head its upper bounds give it | a class nothing enters |
//! | 3 | the head made from the parts named on it | a class nothing enters and no upper bound gives a head |
//! | 4 | the fallback of its literals' domain | a class nothing enters and no upper bound gives a head |
//! | 5 | `!` | a class nothing else gives a head |
//!
//! Rungs 0 and 1 are what solving decides as it settles ([`heads`]), and the
//! others what it decides at the fixpoint ([`fixpoint`]).
//!
//! What is written into a class enters it: the fields written make an object,
//! which rung 0 joins with what enters. `!` says nothing: where the join of what
//! enters a class is `!`, rung 0 gives the head the parts named on the class make,
//! read or written, as rung 3 does where nothing enters.

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::{ClassId, ItemId, Name, ScopeId};
use crate::primitive::constraint::{Bound, Constraint, ConstraintOf};
use crate::primitive::class::State;
use crate::primitive::domain::Domain;
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{PartKey, Shape, VariantOf};
use crate::store::classes::Classes;

use super::super::law::{Bodies, Heads, Offer};
use super::functions::Functions;
use super::{Looked, Told, View, admits, enter, tell};

/// The heads classes take as solving settles: those of the first of rungs 0 and
/// 1 that gives any.
pub fn heads(classes: &Classes<Constraint>, bodies: &Bodies<'_>, functions: &Functions, asked: &[ClassId], held: &dyn Fn(ClassId) -> bool) -> Heads {
    let said: BTreeMap<ClassId, Said<'_>> = asked.iter().map(|class| (*class, Said::of(classes, bodies, functions, *class))).collect();
    let ladder: [&dyn Fn(ClassId) -> Offer; 2] = [&|class| join(&said[&class]), &|class| from_the_declared(&said[&class])];
    Heads::of_ladder(ladder, asked, held)
}

/// The heads classes take at the fixpoint: those of the first of rungs 2 to 5
/// that gives any.
pub fn fixpoint(classes: &Classes<Constraint>, bodies: &Bodies<'_>, functions: &Functions, asked: &[ClassId], held: &dyn Fn(ClassId) -> bool) -> Heads {
    // A class of a body that awaits a use is given what it lacks by that use,
    // and takes no head here.
    let asked: Vec<ClassId> = asked.iter().copied().filter(|class| !bodies.awaits(scope_of(classes, *class))).collect();
    let asked = &asked[..];
    let said: BTreeMap<ClassId, Said<'_>> = asked.iter().map(|class| (*class, Said::of(classes, bodies, functions, *class))).collect();
    let ladder: [&dyn Fn(ClassId) -> Offer; 4] = [
        &|class| from_above(&said[&class]),
        &|class| from_parts(&said[&class]),
        &|class| domain(&said[&class]),
        &|class| never(&said[&class]),
    ];
    Heads::of_ladder(ladder, asked, held)
}

/// What the rungs read of one undecided class.
///
/// The rungs are asked of a class one after another, and each reads the same
/// few things of it: what enters it, and what is asked of it from above. The
/// store does not change while heads are found, as it is borrowed. So each
/// is found once, when a rung first reads it, and kept here for the rungs after;
/// a rung that is never reached costs nothing.
struct Said<'a> {
    classes: &'a Classes<Constraint>,
    bodies: &'a Bodies<'a>,
    functions: &'a Functions,
    class: ClassId,
    entering: OnceCell<Vec<Entering>>,
    asked: OnceCell<Vec<Shape<ClassId>>>,
}

impl<'a> Said<'a> {
    fn of(classes: &'a Classes<Constraint>, bodies: &'a Bodies<'a>, functions: &'a Functions, class: ClassId) -> Self {
        Self {
            classes,
            bodies,
            functions,
            class,
            entering: OnceCell::new(),
            asked: OnceCell::new(),
        }
    }

    /// The heads that enter the class: see [`entering`].
    fn entering(&self) -> &[Entering] {
        self.entering.get_or_init(|| entering(self.classes, self.bodies, self.class))
    }

    /// Every head the class will be held against: see [`asked`].
    fn asked(&self) -> &[Shape<ClassId>] {
        self.asked.get_or_init(|| asked(self.classes, self.bodies, self.functions, self.class, &mut BTreeSet::new()))
    }

    /// Those of `heads` the class may take: each of the domain its literals
    /// leave, and every one where it holds no literal.
    fn may_take<'h, I>(&self, heads: I) -> impl Iterator<Item = &'h Shape<ClassId>>
    where
        I: IntoIterator<Item = &'h Shape<ClassId>>,
    {
        let domain = domain_of(self.classes, self.class);
        heads.into_iter().filter(move |head| domain.as_ref().is_none_or(|(domain, _)| admits(head, domain)))
    }

    /// Whether heads are asked of the class and it may take none of them: it
    /// holds a literal, and each head asked is outside the literal's domain.
    fn may_take_none_of_what_is_asked(&self) -> bool {
        !self.asked().is_empty() && self.may_take(self.asked()).next().is_none()
    }

    /// The one head of `candidates` that every constraint waiting on the class
    /// would have it take ([`taken`]), where there is exactly one such.
    ///
    /// Two candidates of one layer are one: which of them gives the head makes
    /// no difference, as only its constructor and members are taken.
    fn the_one_taken<'h, I>(&self, candidates: I) -> Offer
    where
        I: IntoIterator<Item = &'h Shape<ClassId>>,
    {
        let (classes, bodies, functions, class) = (self.classes, self.bodies, self.functions, self.class);
        let mut of_each_layer: BTreeMap<Shape<()>, &Shape<ClassId>> = BTreeMap::new();
        for head in candidates {
            of_each_layer.entry(head.layer()).or_insert(head);
        }
        let would_be_taken = |head: &&Shape<ClassId>| taken(View::of(classes, bodies.closures()), bodies, functions, class, head, &mut BTreeSet::new());
        let mut taken_heads = of_each_layer.into_values().filter(would_be_taken).cloned();
        match (taken_heads.next(), taken_heads.next()) {
            (Some(shape), None) => Offer::Head {
                shape,
                by: classes.waiting_on(class).map(|(_, at)| *at).collect(),
            },
            _ => Offer::Nothing,
        }
    }

    /// Whether neither a lower nor an upper bound says anything of the class:
    /// nothing enters it, and no head is asked of it from above. Where one is
    /// asked, no fallback overrules that, even where no one head will do.
    fn nothing_said(&self) -> bool {
        self.entering().is_empty() && self.asked().is_empty()
    }
}

/// What enters a class by one bound: its head, where it has one yet, the bound,
/// and the item that recorded the bound.
type Entering = (Option<Shape<ClassId>>, Constraint, ItemId);

/// The heads that enter `class`, each with the bound it enters by: none where
/// nothing enters, and a head not given yet where what enters is undecided. A
/// class's entering itself gives nothing to wait for.
///
/// What has no head and lies in a body that awaits a use, which `class` lies
/// outside, is not given to `class`: nothing gives that body what it lacks, a
/// use giving it to a body made anew. So it is left out, and `class` does not
/// wait for it.
fn entering(classes: &Classes<Constraint>, bodies: &Bodies<'_>, class: ClassId) -> Vec<Entering> {
    let root = classes.find(class);
    let view = View::of(classes, bodies.closures());
    classes
        .constraints_of(class)
        .filter(|(constraint, _)| classes.find(constraint.upper()) == root && classes.find(constraint.lower()) != root)
        .filter_map(|(constraint, at)| enter::given(&view, constraint).map(|head| (head, *constraint, *at)))
        .filter(|(head, constraint, _)| head.is_some() || !bodies.apart(scope_of(classes, constraint.lower()), scope_of(classes, class)))
        .collect()
}

/// The scope `class` lies in.
fn scope_of(classes: &Classes<Constraint>, class: ClassId) -> ScopeId {
    classes.get(classes.find(class)).scope
}

/// The object of the fields written into `class`, with the items that wrote
/// them: none where no field is written into it.
///
/// A write `a.x = v` says a field `x` is there, which is all an object's head
/// says, so it writes that head as a lend writes a reference. A component, an
/// element and a target say no whole head, and write none.
fn written(classes: &Classes<Constraint>, class: ClassId) -> Option<(Shape<ClassId>, BTreeSet<ItemId>)> {
    let State::Undecided(undecided) = &classes.get(classes.find(class)).state else {
        return None;
    };
    let mut fields = BTreeMap::new();
    let mut by = BTreeSet::new();
    for (key, part) in &undecided.parts {
        if let (PartKey::Field(field), false) = (key, part.lower.is_empty()) {
            fields.insert(field.clone(), class);
            by.extend(&part.lower);
        }
    }
    (!fields.is_empty()).then_some((Shape::Object(fields), by))
}

/// Rung 0: the join of the heads that enter the class, once each has a head,
/// and of what its own parts say.
///
/// | What enters | The head |
/// |---|---|
/// | nothing | the object of the fields written into the class, where any is |
/// | a head not given yet | none yet |
/// | heads that do not join | none: the program is refused |
/// | heads whose join is an object | that, with the fields written besides its own |
/// | heads whose join is `!`, which says nothing | the head the parts make, read or written, or else `!` |
/// | heads of any other join | the join |
///
/// What is written joins as what enters does. A head that is no object is not
/// joined with what is written: a part written into a reference is written
/// through it, and what it refers to is as it lies.
fn join(said: &Said<'_>) -> Offer {
    let (classes, class) = (said.classes, said.class);
    let entering = said.entering();
    if entering.is_empty() {
        return match written(classes, class) {
            Some((shape, by)) => Offer::Head { shape, by },
            None => Offer::Nothing,
        };
    }
    let Some(heads) = entering.iter().map(|(head, ..)| head.clone()).collect::<Option<Vec<_>>>() else {
        return Offer::Nothing;
    };
    let mut by: BTreeSet<ItemId> = entering.iter().map(|(.., at)| *at).collect();
    let mut heads = heads.into_iter();
    let first = heads.next().expect("something enters");
    let Some(joined) = heads.try_fold(first, |joined, head| joined.join(&head)) else {
        return Offer::Refused(Refused {
            cause: Cause::Mismatch,
            items: by,
            of: [class].into_iter().chain(entering.iter().map(|(_, constraint, _)| constraint.lower())).collect(),
        });
    };
    let of_its_parts = match joined {
        Shape::Never => made_of_parts(classes, class),
        Shape::Object(_) => written(classes, class),
        _ => None,
    };
    let Some((made, named)) = of_its_parts else {
        return Offer::Head { shape: joined, by };
    };
    by.extend(named);
    Offer::Head {
        shape: joined.join(&made).expect("`!` joins with any head, and an object with an object"),
        by,
    }
}

/// The classes `class` enters that have no head yet. The head `class` takes
/// enters each, so what is asked of each is asked of `class` too.
fn entered(classes: &Classes<Constraint>, class: ClassId) -> Vec<ClassId> {
    let root = classes.find(class);
    classes
        .constraints_of(class)
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
fn asked(classes: &Classes<Constraint>, bodies: &Bodies<'_>, functions: &Functions, class: ClassId, seen: &mut BTreeSet<ClassId>) -> Vec<Shape<ClassId>> {
    let root = classes.find(class).id();
    if !seen.insert(root) {
        return Vec::new();
    }
    let view = View::of(classes, bodies.closures());
    let mut heads: Vec<Shape<ClassId>> = classes
        .waiting_on(root)
        .flat_map(|(constraint, at)| tell(&view, functions, constraint, *at).asks)
        .filter(|(of, _)| classes.find(*of).id() == root)
        .map(|(_, head)| head)
        .collect();
    for upper in entered(classes, root) {
        heads.extend(entering(classes, bodies, upper).into_iter().filter_map(|(head, ..)| head));
        heads.extend(asked(classes, bodies, functions, upper, seen));
    }
    heads.retain(|head| *head != Shape::Never);
    heads
}

/// Whether every constraint that waits on `class` would have it take `head`: none
/// would be refused, were that its head. A class it enters would then take the
/// join of `head` and what has entered it, and is asked the same of that.
fn taken(view: View<'_>, bodies: &Bodies<'_>, functions: &Functions, class: ClassId, head: &Shape<ClassId>, seen: &mut BTreeSet<ClassId>) -> bool {
    let classes = view.classes;
    let root = classes.find(class).id();
    if !seen.insert(root) {
        return true;
    }
    let view = view.supposing(root, head.clone());
    let refused = classes
        .waiting_on(root)
        .any(|(constraint, at)| matches!(tell(&view, functions, constraint, *at), Told { looked: Looked::Refused(_), .. }));
    if refused {
        return false;
    }
    for upper in entered(classes, root) {
        let entered_already = entering(classes, bodies, upper).into_iter().filter_map(|(other, ..)| other);
        let Some(joined) = entered_already.into_iter().try_fold(head.clone(), |joined, other| joined.join(&other)) else {
            return false;
        };
        if !taken(view.clone(), bodies, functions, upper, &joined, seen) {
            return false;
        }
    }
    true
}

/// The head of each declared class `class` is bound to be one type with: what
/// each such bound asks of it. `!` asks nothing of what meets it.
fn declared(classes: &Classes<Constraint>, bodies: &Bodies<'_>, functions: &Functions, class: ClassId) -> Vec<Shape<ClassId>> {
    let root = classes.find(class).id();
    let view = View::of(classes, bodies.closures());
    classes
        .waiting_on(root)
        .filter(|(constraint, _)| matches!(constraint, Constraint::Bound(Bound::Equal { .. })))
        .flat_map(|(constraint, at)| tell(&view, functions, constraint, *at).asks)
        .filter(|(of, head)| classes.find(*of).id() == root && *head != Shape::Never)
        .map(|(_, head)| head)
        .collect()
}

/// Rung 1: a class nothing enters takes the head of the declared class it is
/// bound to be one type with, where it may take it and every constraint waiting
/// on it would have it take it.
///
/// Such a bound says what the head is, whatever is still to come to the class,
/// so the head is given as solving settles. What a class is passed to gives it
/// a head only at the fixpoint ([`from_above`]).
fn from_the_declared(said: &Said<'_>) -> Offer {
    if !said.entering().is_empty() {
        return Offer::Nothing;
    }
    let declared = declared(said.classes, said.bodies, said.functions, said.class);
    said.the_one_taken(said.may_take(&declared))
}

/// Rung 2: a class nothing enters takes the one head, of those asked of it
/// that it may take, that every constraint waiting on it would have it take.
fn from_above(said: &Said<'_>) -> Offer {
    if !said.entering().is_empty() {
        return Offer::Nothing;
    }
    said.the_one_taken(said.may_take(said.asked()))
}

/// The domain the literals of `class` leave, where any registered one.
fn domain_of(classes: &Classes<Constraint>, class: ClassId) -> Option<(Domain, BTreeSet<ItemId>)> {
    let State::Undecided(undecided) = &classes.get(classes.find(class)).state else {
        return None;
    };
    let domain = Domain::intersection(undecided.domains.iter().map(|(domain, _)| domain))?;
    Some((domain, undecided.domains.iter().map(|(_, by)| *by).collect()))
}

/// The head made from the parts named on `class`, read or written, with the
/// items that named them: an object of the fields named, or the type whose
/// variants the payloads named are of. None where no part is named, where the
/// names are not all of one such type, and where a name does not say the whole
/// head: a target does not say which reference, nor a component how many, nor an
/// element what holds it.
///
/// The parts of the head given stand for no class: a head decided has the
/// class's own parts.
fn made_of_parts(classes: &Classes<Constraint>, class: ClassId) -> Option<(Shape<ClassId>, BTreeSet<ItemId>)> {
    let State::Undecided(undecided) = &classes.get(classes.find(class)).state else {
        return None;
    };
    let named = &undecided.parts;
    let by = named
        .values()
        .flat_map(|part| part.lower.iter().chain(&part.upper).chain(&part.carried).copied())
        .collect();
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
            let variants: BTreeMap<Name, ClassId> = named
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

/// Rung 3: a class that neither lower nor upper bounds decide takes the head made
/// from the parts named on it.
fn from_parts(said: &Said<'_>) -> Offer {
    if !said.nothing_said() {
        return Offer::Nothing;
    }
    match made_of_parts(said.classes, said.class) {
        Some((shape, by)) => Offer::Head { shape, by },
        None => Offer::Nothing,
    }
}

/// Rung 4: a literal's class that neither lower nor upper bounds decide takes the
/// fallback type of its domain's row.
fn domain(said: &Said<'_>) -> Offer {
    let (classes, class) = (said.classes, said.class);
    if !said.nothing_said() {
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

/// Rung 5: a class nothing enters, with no literal, takes `!`: nothing gives it
/// a head.
fn never(said: &Said<'_>) -> Offer {
    if !said.nothing_said() || domain_of(said.classes, said.class).is_some() {
        return Offer::Nothing;
    }
    Offer::Head {
        shape: Shape::Never,
        by: BTreeSet::new(),
    }
}

/// What is refused of `class` when nothing more can be done: a class that is a
/// part of itself has no finite type, and a class still undecided has no type.
///
/// A class that heads are asked of, none of which it may take, is undecided for
/// that: rung 2 passes over such a head, as another may come, and here none has.
pub fn end(classes: &Classes<Constraint>, bodies: &Bodies<'_>, functions: &Functions, class: ClassId) -> Vec<Refused> {
    // A class of a body that awaits a use has no type before the use, and is
    // refused for none.
    if bodies.awaits(scope_of(classes, class)) {
        return Vec::new();
    }
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
        let constraints = classes.constraints_of(class).map(|(_, at)| *at);
        let domains = undecided.domains.iter().map(|(_, at)| *at);
        let said = Said::of(classes, bodies, functions, class);
        let cause = if said.may_take_none_of_what_is_asked() { Cause::EmptyDomain } else { Cause::UndecidedClass };
        refusals.push(Refused {
            cause,
            items: constraints.chain(named.iter().copied()).chain(domains).collect(),
            of: vec![class],
        });
    }
    refusals
}
