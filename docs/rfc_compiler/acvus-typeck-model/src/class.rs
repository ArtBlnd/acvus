//! Equivalence classes: their names, union-find, type variables and structures.
//!
//! A type variable is one type, represented by an equivalence class.
//! An equivalence class holds type variables and at most one structure, and
//! the domains the record gives it bound the constructors their values may
//! take. Unifying two merges them, the structure the conjunction of the two,
//! and unifies their parts of one name in turn.
//! `Unifying` reports each union it makes, with its cause, and the
//! conflict that stops it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use crate::check::Cause as Refused;
use crate::disjunction::{Decision, Disjunction};
use crate::domain::Domain;
use crate::relation::{Item, Recorded, Recorder, Relation, Role, Standing};
use crate::sig::{DeclaredTypes, SigVar};
use crate::ty::{Constructor, Member, Name, PartKey, Term};

#[cfg(test)]
mod tests;

/// A name an equivalence class keeps whatever the order of solving.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClassName {
    /// A type the text gives, by the name the program gives it.
    Text(Name),
    /// The part `key` of the named class, made with the structure that names
    /// it.
    Part(Rc<ClassName>, PartKey),
    /// A class a use or a lend of this name made anew.
    Disjunction(Rc<Name>, crate::disjunction::DisjunctionAt),
    /// The copy of a class of another binding group's C that the use `at`
    /// made anew, or of a class outside C, whose closed type it writes.
    Copy { at: Rc<Name>, of: Rc<ClassName> },
    /// A class of the signature of the cast at `step` of the path that links
    /// the parts of a relation from `from` to `to`, or of the view
    /// that links the container `from` to its element `to`.
    Link {
        from: Rc<ClassName>,
        to: Rc<ClassName>,
        step: usize,
        at: LinkAt,
    },
}

/// Which class of a cast's signature a link's class is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum LinkAt {
    Param,
    Result,
    Var(SigVar),
}

impl fmt::Display for ClassName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClassName::Text(name) => write!(f, "{name}"),
            ClassName::Part(class, key) => write!(f, "{class}.{key:?}"),
            ClassName::Disjunction(disjunction, at) => write!(f, "{disjunction}.{at:?}"),
            ClassName::Copy { at, of } => write!(f, "{at}/{of}"),
            ClassName::Link { from, to, step, at } => {
                write!(f, "link({from} -> {to})[{step}].{at:?}")
            }
        }
    }
}

impl fmt::Debug for ClassName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A class, by its place in the arena of one solving.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ClassId(usize);

/// The classes of one solving, by name.
#[derive(Clone, Default, Debug)]
pub struct Arena {
    names: Vec<ClassName>,
    by_name: BTreeMap<ClassName, ClassId>,
}

impl Arena {
    /// The class of `name`, made where there is none.
    pub fn intern(&mut self, name: ClassName) -> ClassId {
        if let Some(&id) = self.by_name.get(&name) {
            return id;
        }
        let id = ClassId(self.names.len());
        self.names.push(name.clone());
        self.by_name.insert(name, id);
        id
    }

    pub fn get(&self, name: &ClassName) -> Option<ClassId> {
        self.by_name.get(name).copied()
    }

    pub fn name(&self, id: ClassId) -> &ClassName {
        &self.names[id.0]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Every class, with its name.
    pub fn classes(&self) -> impl Iterator<Item = (ClassId, &ClassName)> + '_ {
        self.names
            .iter()
            .enumerate()
            .map(|(i, name)| (ClassId(i), name))
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// A constraint on a class's structure before it is known.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum StructureConstraint {
    /// A reference to `τ`.
    Target(ClassId),
    /// Has field `f: τ`.
    Field(Name, ClassId),
    /// Has component `i: τ`.
    Component(usize, ClassId),
    /// Indexes `τ`.
    Element(ClassId),
    /// Has variant `V` with payload `τ`; `enum_name` is the enum the text
    /// writes it of, `E` of `E::V`, and none for the language's `Some`,
    /// `None`, `Ok` and `Err`.
    Payload {
        enum_name: Option<Name>,
        variant: Name,
        payload: ClassId,
    },
}

impl StructureConstraint {
    pub fn map<F>(self, mut f: F) -> StructureConstraint
    where
        F: FnMut(ClassId) -> ClassId,
    {
        match self {
            StructureConstraint::Target(class) => StructureConstraint::Target(f(class)),
            StructureConstraint::Field(field, class) => StructureConstraint::Field(field, f(class)),
            StructureConstraint::Component(i, class) => StructureConstraint::Component(i, f(class)),
            StructureConstraint::Element(class) => StructureConstraint::Element(f(class)),
            StructureConstraint::Payload {
                enum_name,
                variant,
                payload,
            } => StructureConstraint::Payload {
                enum_name,
                variant,
                payload: f(payload),
            },
        }
    }

    pub fn class(&self) -> ClassId {
        match self {
            StructureConstraint::Target(class)
            | StructureConstraint::Field(_, class)
            | StructureConstraint::Component(_, class)
            | StructureConstraint::Element(class) => *class,
            StructureConstraint::Payload { payload, .. } => *payload,
        }
    }

    /// The key of the part it names.
    pub fn key(&self) -> PartKey {
        match self {
            StructureConstraint::Target(_) => PartKey::Target,
            StructureConstraint::Field(field, _) => PartKey::Field(field.clone()),
            StructureConstraint::Component(i, _) => PartKey::Component(*i),
            StructureConstraint::Element(_) => PartKey::Element,
            StructureConstraint::Payload { variant, .. } => PartKey::Payload(variant.clone()),
        }
    }
}

/// Where the structure an equivalence class holds came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StructureSource {
    /// A term the text, a declaration or a decided alternative writes, held
    /// through equations.
    Written,
    /// A term solving gives the class: a decomposition's, fallback's, or an
    /// identity made anew.
    Solved,
}

/// Which side of a relation a class it names is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Side {
    /// The value passes from it: the `τ` of `τ ≤ α` and of `τ ≤ₚ α`, the `T`
    /// of `T ⇝ U`, and the error value of a wrap.
    Lower,
    /// The value passes into it: the `α`, the `U`, and the class the error
    /// value is wrapped into.
    Upper,
    /// The relation is on it: a structure, a structure constraint, a domain
    /// or a lambda its Values names.
    Own,
    /// A part a structure writes or a structure constraint names.
    Part,
    /// A side of an equation.
    Equated,
}

/// An occurrence of a class in an item of the record: the item's place, and
/// the role of the class in it.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Occurrence {
    pub at: usize,
    pub role: Role,
}

impl Occurrence {
    /// The side of the item the class is, as the judgment reads it; none for
    /// an item's bookkeeping.
    pub fn side(&self) -> Option<Side> {
        self.role.side()
    }
}

/// What one equivalence class holds: what unification made of it, and every
/// occurrence of its members in the record.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Held {
    pub members: BTreeSet<ClassId>,
    pub structure: Option<(Term<ClassId>, StructureSource)>,
    /// Why it has no value, where it has none: what it holds no type holds,
    /// what enters it has no join, or it reaches a part of its own. The
    /// first cause found is kept; a refusal names the class, not the cause's
    /// order.
    pub no_value: Option<Refused>,
    /// The occurrences of its members, in the order recorded.
    pub occurrences: Vec<Occurrence>,
}

/// Why no type holds the conjunction of what two equivalence classes hold.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conflict {
    /// Two structures of different constructors or members.
    Structures(Constructor, Constructor),
    /// A domain and a constructor it does not admit, or two domains that admit
    /// no constructor in common.
    Domain(Domain, Option<Constructor>),
    /// A structure constraint that the structure does not satisfy.
    StructureConstraint(StructureConstraint, Constructor),
}

/// A structure constraint written with its class's name: `ref T`, `field x: T`,
/// `component i: T`, `index T`,
/// `variant V: T`, `variant E::V: T`.
pub struct NamedStructureConstraint<'a> {
    pub structure_constraint: &'a StructureConstraint,
    pub arena: &'a Arena,
}

impl fmt::Display for NamedStructureConstraint<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = |id: &ClassId| self.arena.name(*id);
        match self.structure_constraint {
            StructureConstraint::Target(target) => write!(f, "ref {}", name(target)),
            StructureConstraint::Field(field, part) => write!(f, "field {field}: {}", name(part)),
            StructureConstraint::Component(i, part) => write!(f, "component {i}: {}", name(part)),
            StructureConstraint::Element(element) => write!(f, "index {}", name(element)),
            StructureConstraint::Payload {
                enum_name: Some(enum_name),
                variant,
                payload,
            } => write!(f, "variant {enum_name}::{variant}: {}", name(payload)),
            StructureConstraint::Payload {
                enum_name: None,
                variant,
                payload,
            } => write!(f, "variant {variant}: {}", name(payload)),
        }
    }
}

/// What no type holds written with its classes' names, `<a> vs <b>`.
pub struct NamedConflict<'a> {
    pub conflict: &'a Conflict,
    pub arena: &'a Arena,
}

impl fmt::Display for NamedConflict<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.conflict {
            Conflict::Structures(a, b) => write!(f, "{a:?} vs {b:?}"),
            Conflict::Domain(domain, Some(constructor)) => write!(f, "{domain} vs {constructor:?}"),
            Conflict::Domain(domain, None) => write!(f, "{domain} vs another domain"),
            Conflict::StructureConstraint(structure_constraint, constructor) => {
                let structure_constraint = NamedStructureConstraint {
                    structure_constraint,
                    arena: self.arena,
                };
                write!(f, "{constructor:?} vs {structure_constraint}")
            }
        }
    }
}

/// The record: every item recorded, in order, and the equivalence classes
/// over one arena, each holding the items that name its members.
#[derive(Clone, Default, Debug)]
pub struct Classes {
    parent: Vec<ClassId>,
    held: BTreeMap<ClassId, Held>,
    items: Vec<Recorded>,
    /// Each disjunctive constraint's place in the record, by its name.
    names: BTreeMap<Name, usize>,
}

/// Why two equivalence classes are unified.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Cause {
    /// They are the two the caller unifies: an equation's two sides.
    Given,
    /// Congruence: they are the parts of one name of the two structures the
    /// union at `parent`, of the same unifying, conjoined.
    Congruence { parent: usize, part: PartKey },
    /// Congruence: they are the parts of one name of the structure `class` held
    /// and the structure it was given.
    StructureGiven { class: ClassId, part: PartKey },
    /// One is the class a structure constraint of `class` names, and the other
    /// the class its structure, or another structure constraint of it, names at that
    /// part.
    StructureConstraint {
        class: ClassId,
        structure_constraint: StructureConstraint,
    },
}

/// Two classes to unify, and why.
struct Pair {
    a: ClassId,
    b: ClassId,
    cause: Cause,
}

/// A union of two equivalence classes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Union {
    pub cause: Cause,
    /// The members of the two classes, each as it was before the union, in
    /// the order the cause gives them.
    pub sides: [BTreeSet<ClassId>; 2],
    /// The structure of the class they make: the conjunction of theirs.
    pub structure: Option<Term<ClassId>>,
}

/// What no type holds, found while unifying.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conflicted {
    Apart(Box<Apart>),
    /// The class of these members holds what no type holds: what it was
    /// given, or what the last union made it hold.
    Holds {
        members: BTreeSet<ClassId>,
        conflict: Conflict,
    },
}

/// Two classes that hold what no type holds together, and stay apart: those
/// `cause` would unify, of these members and holding these structures.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Apart {
    pub cause: Cause,
    pub sides: [BTreeSet<ClassId>; 2],
    pub structures: [Option<Term<ClassId>>; 2],
    pub conflict: Conflict,
}

impl Conflicted {
    pub fn conflict(&self) -> &Conflict {
        match self {
            Conflicted::Apart(apart) => &apart.conflict,
            Conflicted::Holds { conflict, .. } => conflict,
        }
    }
}

/// What unifying did: each union it made, in order, and what no type holds
/// where that stopped it.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Unified {
    pub unions: Vec<Union>,
    pub conflict: Option<Conflicted>,
}

impl Unified {
    /// The conflict that stopped it, the unions it made left out.
    pub fn result(&self) -> Result<(), Conflict> {
        match &self.conflict {
            Some(conflicted) => Err(conflicted.conflict().clone()),
            None => Ok(()),
        }
    }
}

impl Classes {
    /// Makes room for every class of the arena, each alone.
    pub fn grow(&mut self, arena: &Arena) {
        while self.parent.len() < arena.len() {
            let id = ClassId(self.parent.len());
            self.parent.push(id);
            self.held.insert(
                id,
                Held {
                    members: [id].into(),
                    structure: None,
                    no_value: None,
                    occurrences: Vec::new(),
                },
            );
        }
    }

    pub fn find(&self, id: ClassId) -> ClassId {
        let mut id = id;
        while self.parent[id.0] != id {
            id = self.parent[id.0];
        }
        id
    }

    pub fn held(&self, id: ClassId) -> &Held {
        &self.held[&self.find(id)]
    }

    pub fn roots(&self) -> impl Iterator<Item = ClassId> + '_ {
        self.held.keys().copied()
    }

    pub fn same(&self, a: ClassId, b: ClassId) -> bool {
        self.find(a) == self.find(b)
    }

    /// Unifying in these classes under the declared types, each union
    /// reported.
    pub fn unifying<'a>(&'a mut self, types: &'a DeclaredTypes) -> Unifying<'a> {
        Unifying {
            classes: self,
            types,
            unions: Vec::new(),
        }
    }

    /// As `Unifying::give_structure`, with the conflict alone.
    pub fn give_structure(
        &mut self,
        id: ClassId,
        term: Term<ClassId>,
        source: StructureSource,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        self.unifying(types)
            .give_structure(id, term, source)
            .result()
    }

    /// Records the item: it is kept at its place in the record, and each
    /// class occurring in it holds the occurrence. Returns its place. Panics
    /// where a disjunctive constraint of the item's name is recorded already.
    pub fn record(&mut self, item: Item, recorder: Recorder) -> usize {
        let at = self.items.len();
        if let Item::Disjunction { name, .. } = &item
            && self.names.insert(name.clone(), at).is_some()
        {
            panic!("the record names the disjunctive constraint {name} once");
        }
        let occurrences = item.occurrences();
        self.items.push(Recorded {
            item,
            recorder,
            decomposed: false,
        });
        for (class, role) in occurrences {
            self.attach(at, class, role);
        }
        at
    }

    /// Attaches the class to the item at `at` in `role`: the one way a class
    /// comes to participate in an item, as the item is recorded and as the
    /// item makes classes anew afterwards.
    pub fn attach(&mut self, at: usize, class: ClassId, role: Role) {
        let root = self.find(class);
        self.held
            .get_mut(&root)
            .expect("a root holds")
            .occurrences
            .push(Occurrence { at, role });
    }

    /// The class in `role` of the item at `at`, where one is.
    pub fn in_role(&self, at: usize, role: &Role) -> Option<ClassId> {
        self.held.iter().find_map(|(&root, held)| {
            held.occurrences
                .iter()
                .any(|occurrence| occurrence.at == at && occurrence.role == *role)
                .then_some(root)
        })
    }

    /// Every class of the item at `at`, each with its role, in the order
    /// attached.
    pub fn roles_of(&self, at: usize) -> Vec<(ClassId, Role)> {
        let mut found: Vec<(usize, ClassId, Role)> = Vec::new();
        for (&root, held) in &self.held {
            for (i, occurrence) in held.occurrences.iter().enumerate() {
                if occurrence.at == at {
                    found.push((i, root, occurrence.role.clone()));
                }
            }
        }
        found.sort();
        found.into_iter().map(|(_, class, role)| (class, role)).collect()
    }

    /// The standing of the class of `id`, where the record gives it one.
    pub fn standing(&self, id: ClassId) -> Option<&Standing> {
        self.relations_at(id, Side::Own)
            .find_map(|relation| match relation {
                Relation::Standing(_, standing) => Some(standing),
                _ => None,
            })
    }

    /// Every item recorded, in order.
    pub fn items(&self) -> &[Recorded] {
        &self.items
    }

    /// The place in the record of the disjunctive constraint of this name.
    pub fn disjunction_at(&self, name: &Name) -> Option<usize> {
        self.names.get(name).copied()
    }

    /// Every disjunctive constraint of the record, in order: its place,
    /// name, kind and decision.
    pub fn disjunctions(
        &self,
    ) -> impl Iterator<Item = (usize, &Name, &Disjunction, Option<&Decision>)> + '_ {
        self.items.iter().enumerate().filter_map(|(at, recorded)| {
            recorded
                .disjunction()
                .map(|(name, kind, decision)| (at, name, kind, decision))
        })
    }

    /// Gives the disjunctive constraint at `at` its decision, or takes it
    /// back where `decision` is none.
    pub fn decide(&mut self, at: usize, decision: Option<Decision>) {
        let Item::Disjunction { decision: held, .. } = &mut self.items[at].item else {
            panic!("the item at {at} is a disjunctive constraint");
        };
        *held = decision;
    }

    /// The occurrences of the members of the class of `id`, in order.
    pub fn occurrences(&self, id: ClassId) -> &[Occurrence] {
        &self.held(id).occurrences
    }

    /// The relations a member of the class of `id` occurs in on the side
    /// `side`.
    fn relations_at(&self, id: ClassId, side: Side) -> impl Iterator<Item = &Relation> + '_ {
        self.occurrences(id)
            .iter()
            .filter(move |occurrence| occurrence.side() == Some(side))
            .filter_map(|occurrence| self.items[occurrence.at].relation())
    }

    /// The constructors the value of the class of `id` may take: what every
    /// domain on it admits; none where no domain is on it.
    pub fn domain(&self, id: ClassId) -> Option<Domain> {
        self.relations_at(id, Side::Own)
            .filter_map(|relation| match relation {
                Relation::Domain(_, domain) => Some(domain.clone()),
                _ => None,
            })
            .reduce(|met, domain| {
                met.meet(&domain)
                    .unwrap_or_else(|| Domain::Only(BTreeSet::new()))
            })
    }

    /// The structure constraints on the class of `id`.
    pub fn structure_constraints(&self, id: ClassId) -> BTreeSet<StructureConstraint> {
        self.relations_at(id, Side::Own)
            .filter_map(|relation| match relation {
                Relation::StructureConstraint(_, structure_constraint) => {
                    Some(structure_constraint.clone())
                }
                _ => None,
            })
            .collect()
    }

    /// The lambdas the Values of the class of `id` names.
    pub fn values(&self, id: ClassId) -> BTreeSet<Name> {
        self.relations_at(id, Side::Own)
            .filter_map(|relation| match relation {
                Relation::Values(_, lambda) => Some(lambda.clone()),
                _ => None,
            })
            .collect()
    }

    /// Records that the equivalence class of `id` has no value, for `cause`,
    /// where no cause is recorded yet.
    pub fn refuse(&mut self, id: ClassId, cause: Refused) {
        let root = self.find(id);
        let held = self.held.get_mut(&root).expect("a root holds");
        if held.no_value.is_none() {
            held.no_value = Some(cause);
        }
    }

    /// Marks the relation at `at` as decomposed.
    pub fn mark_decomposed(&mut self, at: usize) {
        self.items[at].decomposed = true;
    }


    /// As `Unifying::unify`, with the conflict alone.
    pub fn unify(&mut self, a: ClassId, b: ClassId, types: &DeclaredTypes) -> Result<(), Conflict> {
        self.unifying(types).unify(a, b).result()
    }

    /// The pairs of classes that what the equivalence class holds makes one:
    /// structure constraints of one part name with each other and with its structure's part, and
    /// its structure's constructor checked against its domain.
    fn normalize(&mut self, root: ClassId, types: &DeclaredTypes) -> Result<Vec<Pair>, Conflict> {
        let domain = self.domain(root);
        let structure_constraints = self.structure_constraints(root);
        let held = &self.held[&root];
        if let (Some(domain), Some((structure, _))) = (&domain, &held.structure) {
            let constructor = structure.constructor();
            if !domain.admits(&constructor) {
                return Err(Conflict::Domain(domain.clone(), Some(constructor)));
            }
        }
        let mut pairs = Vec::new();
        let mut by_part: BTreeMap<ConstrainedPart, ClassId> = BTreeMap::new();
        for structure_constraint in &structure_constraints {
            let part = ConstrainedPart::of(structure_constraint);
            let cause = || Cause::StructureConstraint {
                class: root,
                structure_constraint: structure_constraint.clone(),
            };
            match by_part.get(&part) {
                Some(&first) => pairs.push(Pair {
                    a: first,
                    b: structure_constraint.class(),
                    cause: cause(),
                }),
                None => {
                    by_part.insert(part, structure_constraint.class());
                }
            }
            if let Some((structure, _)) = &held.structure {
                if let (Term::Extern { .. }, StructureConstraint::Element(_)) =
                    (structure, structure_constraint)
                {
                    // Its decomposition links it along the container's view.
                    continue;
                }
                if let Term::Never = structure {
                    // `!`, which has no value, satisfies every structure constraint.
                    continue;
                }
                let resolved =
                    resolve(structure, structure_constraint, types).ok_or_else(|| {
                        Conflict::StructureConstraint(
                            structure_constraint.clone(),
                            structure.constructor(),
                        )
                    })?;
                pairs.push(Pair {
                    a: resolved,
                    b: structure_constraint.class(),
                    cause: cause(),
                });
            }
        }
        Ok(pairs)
    }
}

/// Unifying in the equivalence classes, which reports each union
/// it makes and the conflict that stops it.
pub struct Unifying<'a> {
    classes: &'a mut Classes,
    types: &'a DeclaredTypes,
    unions: Vec<Union>,
}

impl Unifying<'_> {
    /// Gives the equivalence class of `id` the structure `term`.
    pub fn give_structure(
        mut self,
        id: ClassId,
        term: Term<ClassId>,
        source: StructureSource,
    ) -> Unified {
        let given = self.structure_given(id, term, source);
        self.finish(given)
    }

    fn structure_given(
        &mut self,
        id: ClassId,
        term: Term<ClassId>,
        source: StructureSource,
    ) -> Result<(), Conflicted> {
        let root = self.classes.find(id);
        let held = self.classes.held.get_mut(&root).expect("a root holds");
        let mut pairs = Vec::new();
        held.structure = Some(match held.structure.take() {
            Some((structure, held_source)) => {
                if let Err(conflict) = agree(&structure, &term) {
                    held.structure = Some((structure, held_source));
                    let members = held.members.clone();
                    return Err(Conflicted::Holds { members, conflict });
                }
                pairs.extend(
                    part_pairs(&structure, &term)
                        .into_iter()
                        .map(|(part, a, b)| Pair {
                            a,
                            b,
                            cause: Cause::StructureGiven { class: root, part },
                        }),
                );
                (structure, joined_source(held_source, source))
            }
            None => (term, source),
        });
        pairs.extend(self.normalize(root)?);
        self.unify_pairs(pairs)
    }

    /// Unifies what the class of `id` holds now that the record has a new
    /// structure constraint on it.
    pub fn constrained(mut self, id: ClassId) -> Unified {
        let root = self.classes.find(id);
        let given = self
            .normalize(root)
            .and_then(|pairs| self.unify_pairs(pairs));
        self.finish(given)
    }

    /// Unifies what the class of `id` holds now that the record has the
    /// domain `domain` on it: its domains admit nothing in common where they
    /// conflict.
    pub fn domain_given(mut self, id: ClassId, domain: &Domain) -> Unified {
        let root = self.classes.find(id);
        let admitted = self.classes.domain(root).is_some_and(|met| match met {
            Domain::Only(admitted) => !admitted.is_empty(),
            Domain::Except(_) => true,
        });
        let given = if admitted {
            self.normalize(root)
                .and_then(|pairs| self.unify_pairs(pairs))
        } else {
            Err(Conflicted::Holds {
                members: self.classes.held[&root].members.clone(),
                conflict: Conflict::Domain(domain.clone(), None),
            })
        };
        self.finish(given)
    }

    /// Merges the two equivalence classes, and their parts of one name in turn.
    pub fn unify(mut self, a: ClassId, b: ClassId) -> Unified {
        let unified = self.unify_pairs(vec![Pair {
            a,
            b,
            cause: Cause::Given,
        }]);
        self.finish(unified)
    }

    fn unify_pairs(&mut self, mut work: Vec<Pair>) -> Result<(), Conflicted> {
        while let Some(Pair { a, b, cause }) = work.pop() {
            let (a, b) = (self.classes.find(a), self.classes.find(b));
            if a == b {
                continue;
            }
            let (keep, gone) = if a < b { (a, b) } else { (b, a) };
            let classes = &self.classes;
            let sides = [
                classes.held[&a].members.clone(),
                classes.held[&b].members.clone(),
            ];
            let apart = |conflict| {
                Conflicted::Apart(Box::new(Apart {
                    cause: cause.clone(),
                    sides: sides.clone(),
                    structures: [a, b].map(|side| {
                        classes.held[&side]
                            .structure
                            .as_ref()
                            .map(|(structure, _)| structure.clone())
                    }),
                    conflict,
                }))
            };
            let parent = self.unions.len();
            let (kept, merged) = (&classes.held[&keep], &classes.held[&gone]);
            let structure =
                match (&kept.structure, &merged.structure) {
                    (Some((structure, s)), Some((other, t))) => {
                        agree(structure, other).map_err(apart)?;
                        work.extend(part_pairs(structure, other).into_iter().map(
                            |(part, a, b)| Pair {
                                a,
                                b,
                                cause: Cause::Congruence { parent, part },
                            },
                        ));
                        Some((structure.clone(), joined_source(*s, *t)))
                    }
                    (structure, other) => structure.clone().or_else(|| other.clone()),
                };
            if let (Some(domain), Some(other)) = (classes.domain(keep), classes.domain(gone)) {
                domain
                    .meet(&other)
                    .ok_or_else(|| apart(Conflict::Domain(other.clone(), None)))?;
            }
            self.classes.parent[gone.0] = keep;
            let merged = self.classes.held.remove(&gone).expect("a root holds");
            let held = self.classes.held.get_mut(&keep).expect("a root holds");
            held.members.extend(merged.members);
            held.occurrences.extend(merged.occurrences);
            held.occurrences.sort();
            held.structure = structure;
            if held.no_value.is_none() {
                held.no_value = merged.no_value;
            }
            let structure = held
                .structure
                .as_ref()
                .map(|(structure, _)| structure.clone());
            self.unions.push(Union {
                cause,
                sides,
                structure,
            });
            work.extend(self.normalize(keep)?);
        }
        Ok(())
    }

    /// `Classes::normalize`, what no type holds named by the class's members.
    fn normalize(&mut self, root: ClassId) -> Result<Vec<Pair>, Conflicted> {
        self.classes
            .normalize(root, self.types)
            .map_err(|conflict| Conflicted::Holds {
                members: self.classes.held[&root].members.clone(),
                conflict,
            })
    }

    fn finish(self, result: Result<(), Conflicted>) -> Unified {
        Unified {
            unions: self.unions,
            conflict: result.err(),
        }
    }
}

/// What part a structure constraint names, so that two structure constraints of one part name one class.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ConstrainedPart {
    Target,
    Field(Name),
    Component(usize),
    Element,
    Variant(Name),
}

impl ConstrainedPart {
    fn of(structure_constraint: &StructureConstraint) -> ConstrainedPart {
        match structure_constraint {
            StructureConstraint::Target(_) => ConstrainedPart::Target,
            StructureConstraint::Field(field, _) => ConstrainedPart::Field(field.clone()),
            StructureConstraint::Component(i, _) => ConstrainedPart::Component(*i),
            StructureConstraint::Element(_) => ConstrainedPart::Element,
            StructureConstraint::Payload { variant, .. } => {
                ConstrainedPart::Variant(variant.clone())
            }
        }
    }
}

/// The part of `structure` the structure constraint names, where the structure satisfies the structure constraint. A
/// struct's field is the part at the field's place in the declared order.
/// An extern-defined container's element is not a part of it.
fn resolve(
    structure: &Term<ClassId>,
    structure_constraint: &StructureConstraint,
    types: &DeclaredTypes,
) -> Option<ClassId> {
    match (structure, structure_constraint) {
        (Term::Ref(_, target), StructureConstraint::Target(_)) => Some(*target),
        (Term::Object(fields), StructureConstraint::Field(field, _)) => fields.get(field).copied(),
        (Term::Struct(module, args), StructureConstraint::Field(field, _)) => {
            let place = types
                .struct_decl(module)?
                .fields
                .iter()
                .position(|declared| declared == field)?;
            args.get(place).copied()
        }
        (Term::Tuple(components), StructureConstraint::Component(i, _)) => {
            components.get(*i).copied()
        }
        (Term::Array(_, element) | Term::Slice(element), StructureConstraint::Element(_)) => {
            Some(*element)
        }
        (
            term,
            StructureConstraint::Payload {
                variant, enum_name, ..
            },
        ) => {
            let constructor = term.constructor();
            if let (Constructor::Enum(name), Some(written)) = (&constructor, enum_name)
                && name != written
            {
                return None;
            }
            if !term.members().contains(&Member::Variant(variant.clone())) {
                return None;
            }
            let key = constructor.payload_part(variant)?;
            term.part(&key).copied()
        }
        _ => None,
    }
}

/// A written structure keeps the equivalence class written whatever else it is
/// unified with.
fn joined_source(a: StructureSource, b: StructureSource) -> StructureSource {
    if a == StructureSource::Written || b == StructureSource::Written {
        StructureSource::Written
    } else {
        StructureSource::Solved
    }
}

/// Two structures that one equivalence class may hold together.
fn agree(a: &Term<ClassId>, b: &Term<ClassId>) -> Result<(), Conflict> {
    if a.constructor() == b.constructor() && a.members() == b.members() {
        Ok(())
    } else {
        Err(Conflict::Structures(a.constructor(), b.constructor()))
    }
}

/// The parts of one name of two structures of one constructor and members, each
/// with its name.
fn part_pairs(a: &Term<ClassId>, b: &Term<ClassId>) -> Vec<(PartKey, ClassId, ClassId)> {
    a.parts()
        .into_iter()
        .zip(b.parts())
        .map(|((key, x), (_, y))| (key, *x, *y))
        .collect()
}
