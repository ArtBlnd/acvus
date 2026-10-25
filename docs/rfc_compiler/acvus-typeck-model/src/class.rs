//! Equivalence classes: their names, union-find, type variables and structures.
//!
//! A type variable is one type, represented by an equivalence class.
//! An equivalence class holds type variables, of which the model keeps the
//! domain their values may take, and at most one structure. Unifying two merges them, the structure
//! the conjunction of the two, and unifies their parts of one name in turn.
//! `Unifying` reports each union it makes, with its cause, and the
//! conflict that stops it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use crate::domain::Domain;
use crate::sig::{DeclaredTypes, SigVar};
use crate::ty::{Constructor, ERR, Member, NONE, Name, OK, PartKey, SOME, Term};

#[cfg(test)]
mod tests;

/// A name an equivalence class keeps whatever the order of solving.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClassName {
    /// A type the text gives, by the name the program gives it.
    Text(Name),
    /// The part `key` of the named class: made anew when its equivalence class
    /// takes a constructor, or written with it.
    Part(Rc<ClassName>, PartKey),
    /// A class a use or a lend of this name made anew.
    Node(Rc<Name>, crate::node::NodeAt),
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
            ClassName::Node(node, at) => write!(f, "{node}.{at:?}"),
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

    /// The member this structure constraint names where the body's text writes it, which is
    /// visible to this body: a field, or a variant of an enum the text
    /// names.
    /// The constructor of the member this structure constraint names, with the members that
    /// constructor holds of it, whether visible or not: an object's
    /// field, an enum's variant the text names, and the language's `Some`,
    /// `None`, `Ok` and `Err`, whose constructor holds both its members.
    pub fn named_member(&self) -> Option<(Constructor, BTreeSet<Member>)> {
        if let Some((constructor, member)) = self.visible_member() {
            return Some((constructor, [member].into()));
        }
        let StructureConstraint::Payload {
            enum_name: None,
            variant,
            ..
        } = self
        else {
            return None;
        };
        let both = |a: &str, b: &str| -> BTreeSet<Member> {
            [a, b]
                .into_iter()
                .map(|name| Member::Variant(Name::new(name)))
                .collect()
        };
        match variant.as_str() {
            SOME | NONE => Some((Constructor::Option, both(SOME, NONE))),
            OK | ERR => Some((Constructor::Result, both(OK, ERR))),
            _ => None,
        }
    }

    pub fn visible_member(&self) -> Option<(Constructor, Member)> {
        match self {
            StructureConstraint::Field(field, _) => {
                Some((Constructor::Object, Member::Field(field.clone())))
            }
            StructureConstraint::Payload {
                enum_name: Some(enum_name),
                variant,
                ..
            } => Some((
                Constructor::Enum(enum_name.clone()),
                Member::Variant(variant.clone()),
            )),
            _ => None,
        }
    }
}

/// Where the structure an equivalence class holds came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StructureSource {
    /// A type the text or a declaration writes, held through equations from
    /// before solving decides it.
    Written,
    /// The equivalence class took it when it was solved.
    Solved,
}

/// What one equivalence class holds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Held {
    pub members: BTreeSet<ClassId>,
    pub structure: Option<(Term<ClassId>, StructureSource)>,
    pub structure_constraints: BTreeSet<StructureConstraint>,
    /// The constructors its value may take, where a literal, a generalization or
    /// a lend gives them.
    pub domain: Option<Domain>,
    /// The lambdas the Values of the function values met in it name, joined.
    pub values: BTreeSet<Name>,
    /// Whether it took its constructor and members, written or joined.
    pub solved: bool,
}

/// Why no type holds the conjunction of what two equivalence classes hold.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conflict {
    /// Two structures of different constructors or members.
    Structures(Constructor, Constructor),
    /// A domain and a constructor it does not admit, or two domains that admit
    /// no constructor in common.
    Domain(Domain, Option<Constructor>),
    /// A structure constraint that the structure does not meet.
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

/// The equivalence classes over one arena.
#[derive(Clone, Default, Debug)]
pub struct Classes {
    parent: Vec<ClassId>,
    held: BTreeMap<ClassId, Held>,
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

/// What no type holds, met while unifying.
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
                    structure_constraints: BTreeSet::new(),
                    domain: None,
                    values: BTreeSet::new(),
                    solved: false,
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

    /// Marks the equivalence class of `id` as having taken its constructor.
    pub fn mark_solved(&mut self, id: ClassId) {
        let root = self.find(id);
        self.held.get_mut(&root).expect("a root holds").solved = true;
    }

    /// As `Unifying::give_structure_constraint`, with the conflict alone.
    pub fn give_structure_constraint(
        &mut self,
        id: ClassId,
        structure_constraint: StructureConstraint,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        self.unifying(types)
            .give_structure_constraint(id, structure_constraint)
            .result()
    }

    /// As `Unifying::give_domain`, with the conflict alone.
    pub fn give_domain(
        &mut self,
        id: ClassId,
        domain: Domain,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        self.unifying(types).give_domain(id, domain).result()
    }

    /// Records a lambda the Values of the class of `id` names.
    pub fn give_value(&mut self, id: ClassId, lambda: Name) {
        let root = self.find(id);
        self.held
            .get_mut(&root)
            .expect("a root holds")
            .values
            .insert(lambda);
    }

    /// As `Unifying::unify`, with the conflict alone.
    pub fn unify(&mut self, a: ClassId, b: ClassId, types: &DeclaredTypes) -> Result<(), Conflict> {
        self.unifying(types).unify(a, b).result()
    }

    /// The pairs of classes that what the equivalence class holds makes one:
    /// structure constraints of one part name with each other and with its structure's part, and
    /// its structure's constructor checked against its domain.
    fn normalize(&mut self, root: ClassId, types: &DeclaredTypes) -> Result<Vec<Pair>, Conflict> {
        let held = self.held.get_mut(&root).expect("a root holds");
        if let (Some(domain), Some((structure, _))) = (&held.domain, &held.structure) {
            let constructor = structure.constructor();
            if !domain.admits(&constructor) {
                return Err(Conflict::Domain(domain.clone(), Some(constructor)));
            }
        }
        let mut pairs = Vec::new();
        let mut by_part: BTreeMap<ConstrainedPart, ClassId> = BTreeMap::new();
        for structure_constraint in &held.structure_constraints {
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
                    // The state links it along the container's view.
                    continue;
                }
                if let Term::Never = structure {
                    // `!`, which has no value, meets every structure constraint.
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

    pub fn give_structure_constraint(
        mut self,
        id: ClassId,
        structure_constraint: StructureConstraint,
    ) -> Unified {
        let root = self.classes.find(id);
        self.classes
            .held
            .get_mut(&root)
            .expect("a root holds")
            .structure_constraints
            .insert(structure_constraint);
        let given = self
            .normalize(root)
            .and_then(|pairs| self.unify_pairs(pairs));
        self.finish(given)
    }

    /// Narrows the domain of the equivalence class of `id` to what it and
    /// `domain` both admit.
    pub fn give_domain(mut self, id: ClassId, domain: Domain) -> Unified {
        let given = self.domain_given(id, domain);
        self.finish(given)
    }

    fn domain_given(&mut self, id: ClassId, domain: Domain) -> Result<(), Conflicted> {
        let root = self.classes.find(id);
        {
            let held = self.classes.held.get_mut(&root).expect("a root holds");
            held.domain = Some(match &held.domain {
                Some(own) => own.meet(&domain).ok_or_else(|| Conflicted::Holds {
                    members: held.members.clone(),
                    conflict: Conflict::Domain(domain.clone(), None),
                })?,
                None => domain,
            });
        }
        let pairs = self.normalize(root)?;
        self.unify_pairs(pairs)
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
            let domain = match (&kept.domain, &merged.domain) {
                (Some(domain), Some(other)) => Some(
                    domain
                        .meet(other)
                        .ok_or_else(|| apart(Conflict::Domain(other.clone(), None)))?,
                ),
                (domain, other) => domain.clone().or_else(|| other.clone()),
            };
            self.classes.parent[gone.0] = keep;
            let merged = self.classes.held.remove(&gone).expect("a root holds");
            let held = self.classes.held.get_mut(&keep).expect("a root holds");
            held.members.extend(merged.members);
            held.values.extend(merged.values);
            held.structure_constraints
                .extend(merged.structure_constraints);
            held.structure = structure;
            held.domain = domain;
            held.solved |= merged.solved;
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

/// The part of `structure` the structure constraint names, where the structure meets the structure constraint. A
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

/// A structure written by the text or a declaration keeps the equivalence class
/// "with an equation" whatever else it is unified with.
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
