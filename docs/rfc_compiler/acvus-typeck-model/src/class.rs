//! Equivalence classes: their names, union-find, type variables and kinds
//! (l.111–169).
//!
//! A type variable is one type, represented by an equivalence class (l.113).
//! An equivalence class holds type variables, of which the model keeps the
//! LitVar, and at most one kind (l.117). Unifying two merges them, the kind
//! the conjunction of the two, and unifies their parts of one name in turn
//! (l.158). `Unifying` reports each union it makes, with its cause, and the
//! conflict that stops it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use crate::lit::LitVar;
use crate::sig::{DeclaredTypes, SigVar};
use crate::ty::{Constructor, Member, Name, PartKey, Term};

#[cfg(test)]
mod tests;

/// A name an equivalence class keeps whatever the order of solving.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClassName {
    /// A type the text gives, by the name the program gives it.
    Text(Name),
    /// The part `key` of the named class: made anew when its equivalence class
    /// takes a constructor (l.269, l.278), or written with it.
    Part(Rc<ClassName>, PartKey),
    /// A class a use or a lend of this name made anew.
    Node(Rc<Name>, crate::node::NodeAt),
    /// The copy of a class of another binding group's C that the use `at`
    /// made anew, or of a class outside C, whose closed type it writes
    /// (l.215).
    Copy { at: Rc<Name>, of: Rc<ClassName> },
    /// A class of the signature of the cast at `step` of the path that links
    /// the parts of a relation from `from` to `to` (l.278), or of the view
    /// that links the container `from` to its element `to` (l.154).
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

/// A kind known only in part (l.133–140).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Fact {
    /// A reference to `τ`.
    Ref(ClassId),
    /// Has field `f: τ`.
    Field(Name, ClassId),
    /// Has component `i: τ`.
    Component(usize, ClassId),
    /// Indexes `τ`.
    Indexes(ClassId),
    /// Has variant `V` with payload `τ`; `enum_name` is the enum the text
    /// writes it of, `E` of `E::V`, and none for the language's `Some`,
    /// `None`, `Ok` and `Err` (l.316).
    Variant {
        enum_name: Option<Name>,
        variant: Name,
        payload: ClassId,
    },
}

impl Fact {
    pub fn map<F>(self, mut f: F) -> Fact
    where
        F: FnMut(ClassId) -> ClassId,
    {
        match self {
            Fact::Ref(class) => Fact::Ref(f(class)),
            Fact::Field(field, class) => Fact::Field(field, f(class)),
            Fact::Component(i, class) => Fact::Component(i, f(class)),
            Fact::Indexes(class) => Fact::Indexes(f(class)),
            Fact::Variant {
                enum_name,
                variant,
                payload,
            } => Fact::Variant {
                enum_name,
                variant,
                payload: f(payload),
            },
        }
    }

    pub fn class(&self) -> ClassId {
        match self {
            Fact::Ref(class)
            | Fact::Field(_, class)
            | Fact::Component(_, class)
            | Fact::Indexes(class) => *class,
            Fact::Variant { payload, .. } => *payload,
        }
    }

    /// The member this fact names where the body's text writes it, which is
    /// visible to this body (l.276): a field, or a variant of an enum the text
    /// names.
    pub fn visible_member(&self) -> Option<(Constructor, Member)> {
        match self {
            Fact::Field(field, _) => Some((Constructor::Object, Member::Field(field.clone()))),
            Fact::Variant {
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

/// Where the kind an equivalence class holds came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KindSource {
    /// A type the text or a declaration writes, held through equations from
    /// before solving decides it (l.273, "with an equation").
    Written,
    /// The equivalence class took it when it was solved (l.269).
    Solved,
}

/// What one equivalence class holds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Held {
    pub members: BTreeSet<ClassId>,
    pub kind: Option<(Term<ClassId>, KindSource)>,
    pub facts: BTreeSet<Fact>,
    pub lit: Option<LitVar>,
    /// Whether it is a reference, as a use records it before it is decided
    /// (l.246).
    pub reference: Option<bool>,
    /// The lambdas the Values of the function values met in it name, joined
    /// (l.173, l.181).
    pub values: BTreeSet<Name>,
    /// Whether it took its constructor and members, written or joined
    /// (l.269).
    pub solved: bool,
}

/// Why no type holds the conjunction of what two equivalence classes hold.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conflict {
    /// Two kinds of different constructors or members.
    Kinds(Constructor, Constructor),
    /// A LitVar and a constructor that does not admit its values (l.162), or
    /// two LitVars of different families (l.163).
    Lit(LitVar, Option<Constructor>),
    /// A fact that the kind does not meet (l.276).
    Fact(Fact, Constructor),
    /// A record of whether the class is a reference that its kind, or another
    /// record, does not meet (l.246).
    Reference(bool, Option<Constructor>),
}

/// A kind known in part written with its class's name (`design/19-cases.md`,
/// Notation): `ref T`, `field x: T`, `component i: T`, `index T`,
/// `variant V: T`, `variant E::V: T`.
pub struct NamedFact<'a> {
    pub fact: &'a Fact,
    pub arena: &'a Arena,
}

impl fmt::Display for NamedFact<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = |id: &ClassId| self.arena.name(*id);
        match self.fact {
            Fact::Ref(target) => write!(f, "ref {}", name(target)),
            Fact::Field(field, part) => write!(f, "field {field}: {}", name(part)),
            Fact::Component(i, part) => write!(f, "component {i}: {}", name(part)),
            Fact::Indexes(element) => write!(f, "index {}", name(element)),
            Fact::Variant {
                enum_name: Some(enum_name),
                variant,
                payload,
            } => write!(f, "variant {enum_name}::{variant}: {}", name(payload)),
            Fact::Variant {
                enum_name: None,
                variant,
                payload,
            } => write!(f, "variant {variant}: {}", name(payload)),
        }
    }
}

/// What no type holds written with its classes' names, `<a> vs <b>`
/// (`design/19-cases.md`, Notation).
pub struct NamedConflict<'a> {
    pub conflict: &'a Conflict,
    pub arena: &'a Arena,
}

impl fmt::Display for NamedConflict<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reference = |reference: bool| {
            if reference {
                "reference"
            } else {
                "no reference"
            }
        };
        match self.conflict {
            Conflict::Kinds(a, b) => write!(f, "{a:?} vs {b:?}"),
            Conflict::Lit(lit, Some(constructor)) => write!(f, "{lit} vs {constructor:?}"),
            Conflict::Lit(lit, None) => write!(f, "{lit} vs another family"),
            Conflict::Fact(fact, constructor) => {
                let fact = NamedFact {
                    fact,
                    arena: self.arena,
                };
                write!(f, "{constructor:?} vs {fact}")
            }
            Conflict::Reference(held, Some(constructor)) => {
                write!(f, "{} vs {constructor:?}", reference(*held))
            }
            Conflict::Reference(held, None) => {
                write!(f, "{} vs {}", reference(*held), reference(!*held))
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

/// Why two equivalence classes are unified (l.158).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Cause {
    /// They are the two the caller unifies: an equation's two sides.
    Given,
    /// Congruence: they are the parts of one name of the two kinds the
    /// union at `parent`, of the same unifying, conjoined.
    Congruence { parent: usize, part: PartKey },
    /// Congruence: they are the parts of one name of the kind `class` held
    /// and the kind it was given.
    KindGiven { class: ClassId, part: PartKey },
    /// One is the class a kind known in part of `class` names, and the other
    /// the class its kind, or another kind known in part of it, names at that
    /// part (l.133–140, l.154).
    Fact { class: ClassId, fact: Fact },
}

/// Two classes to unify, and why.
struct Pair {
    a: ClassId,
    b: ClassId,
    cause: Cause,
}

/// A union of two equivalence classes (l.158).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Union {
    pub cause: Cause,
    /// The members of the two classes, each as it was before the union, in
    /// the order the cause gives them.
    pub sides: [BTreeSet<ClassId>; 2],
    /// The kind of the class they make: the conjunction of theirs.
    pub kind: Option<Term<ClassId>>,
}

/// What no type holds, met while unifying (l.158–163).
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
/// `cause` would unify, of these members and holding these kinds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Apart {
    pub cause: Cause,
    pub sides: [BTreeSet<ClassId>; 2],
    pub kinds: [Option<Term<ClassId>>; 2],
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
                    kind: None,
                    facts: BTreeSet::new(),
                    lit: None,
                    reference: None,
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

    /// As `Unifying::give_kind`, with the conflict alone.
    pub fn give_kind(
        &mut self,
        id: ClassId,
        term: Term<ClassId>,
        source: KindSource,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        self.unifying(types).give_kind(id, term, source).result()
    }

    /// Marks the equivalence class of `id` as having taken its constructor.
    pub fn mark_solved(&mut self, id: ClassId) {
        let root = self.find(id);
        self.held.get_mut(&root).expect("a root holds").solved = true;
    }

    /// As `Unifying::give_fact`, with the conflict alone.
    pub fn give_fact(
        &mut self,
        id: ClassId,
        fact: Fact,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        self.unifying(types).give_fact(id, fact).result()
    }

    /// As `Unifying::give_lit`, with the conflict alone.
    pub fn give_lit(
        &mut self,
        id: ClassId,
        lit: LitVar,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        self.unifying(types).give_lit(id, lit).result()
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
    /// facts of one part name with each other and with its kind's part, and
    /// the LitVar checked against its kind's constructor.
    fn normalize(&mut self, root: ClassId, types: &DeclaredTypes) -> Result<Vec<Pair>, Conflict> {
        let held = self.held.get_mut(&root).expect("a root holds");
        if let (Some(lit), Some((kind, _))) = (&held.lit, &held.kind) {
            if !matches!(kind, Term::Scalar(scalar) if lit.admits(*scalar)) {
                return Err(Conflict::Lit(lit.clone(), Some(kind.constructor())));
            }
            held.lit = None;
        }
        if let (Some(reference), Some((kind, _))) = (held.reference, &held.kind) {
            let constructor = kind.constructor();
            let is_reference = matches!(constructor, Constructor::Ref(_));
            if constructor != Constructor::Never && is_reference != reference {
                return Err(Conflict::Reference(reference, Some(constructor)));
            }
        }
        let mut pairs = Vec::new();
        let mut by_part: BTreeMap<FactPart, ClassId> = BTreeMap::new();
        for fact in &held.facts {
            let part = FactPart::of(fact);
            let cause = || Cause::Fact {
                class: root,
                fact: fact.clone(),
            };
            match by_part.get(&part) {
                Some(&first) => pairs.push(Pair {
                    a: first,
                    b: fact.class(),
                    cause: cause(),
                }),
                None => {
                    by_part.insert(part, fact.class());
                }
            }
            if let Some((kind, _)) = &held.kind {
                if let (Term::Extern { .. }, Fact::Indexes(_)) = (kind, fact) {
                    // The state links it along the container's view (l.154).
                    continue;
                }
                if let Term::Never = kind {
                    // `!`, which has no value, meets every kind (l.276).
                    continue;
                }
                let resolved = resolve(kind, fact, types)
                    .ok_or_else(|| Conflict::Fact(fact.clone(), kind.constructor()))?;
                pairs.push(Pair {
                    a: resolved,
                    b: fact.class(),
                    cause: cause(),
                });
            }
        }
        Ok(pairs)
    }
}

/// Unifying in the equivalence classes (l.158–163), which reports each union
/// it makes and the conflict that stops it.
pub struct Unifying<'a> {
    classes: &'a mut Classes,
    types: &'a DeclaredTypes,
    unions: Vec<Union>,
}

impl Unifying<'_> {
    /// Gives the equivalence class of `id` the kind `term`.
    pub fn give_kind(mut self, id: ClassId, term: Term<ClassId>, source: KindSource) -> Unified {
        let given = self.kind_given(id, term, source);
        self.finish(given)
    }

    fn kind_given(
        &mut self,
        id: ClassId,
        term: Term<ClassId>,
        source: KindSource,
    ) -> Result<(), Conflicted> {
        let root = self.classes.find(id);
        let held = self.classes.held.get_mut(&root).expect("a root holds");
        let mut pairs = Vec::new();
        held.kind = Some(match held.kind.take() {
            Some((kind, held_source)) => {
                if let Err(conflict) = agree(&kind, &term) {
                    held.kind = Some((kind, held_source));
                    let members = held.members.clone();
                    return Err(Conflicted::Holds { members, conflict });
                }
                pairs.extend(
                    part_pairs(&kind, &term)
                        .into_iter()
                        .map(|(part, a, b)| Pair {
                            a,
                            b,
                            cause: Cause::KindGiven { class: root, part },
                        }),
                );
                (kind, joined_source(held_source, source))
            }
            None => (term, source),
        });
        pairs.extend(self.normalize(root)?);
        self.unify_pairs(pairs)
    }

    pub fn give_fact(mut self, id: ClassId, fact: Fact) -> Unified {
        let root = self.classes.find(id);
        self.classes
            .held
            .get_mut(&root)
            .expect("a root holds")
            .facts
            .insert(fact);
        let given = self
            .normalize(root)
            .and_then(|pairs| self.unify_pairs(pairs));
        self.finish(given)
    }

    pub fn give_lit(mut self, id: ClassId, lit: LitVar) -> Unified {
        let given = self.lit_given(id, lit);
        self.finish(given)
    }

    fn lit_given(&mut self, id: ClassId, lit: LitVar) -> Result<(), Conflicted> {
        let root = self.classes.find(id);
        {
            let held = self.classes.held.get_mut(&root).expect("a root holds");
            held.lit = Some(match &held.lit {
                Some(own) => own.union(&lit).ok_or_else(|| Conflicted::Holds {
                    members: held.members.clone(),
                    conflict: Conflict::Lit(lit.clone(), None),
                })?,
                None => lit,
            });
        }
        let pairs = self.normalize(root)?;
        self.unify_pairs(pairs)
    }

    /// Records whether the equivalence class of `id` is a reference.
    pub fn give_reference(mut self, id: ClassId, reference: bool) -> Unified {
        let given = self.reference_given(id, reference);
        self.finish(given)
    }

    fn reference_given(&mut self, id: ClassId, reference: bool) -> Result<(), Conflicted> {
        let root = self.classes.find(id);
        {
            let held = self.classes.held.get_mut(&root).expect("a root holds");
            if held.reference.is_some_and(|own| own != reference) {
                return Err(Conflicted::Holds {
                    members: held.members.clone(),
                    conflict: Conflict::Reference(reference, None),
                });
            }
            held.reference = Some(reference);
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
                    kinds: [a, b].map(|side| {
                        classes.held[&side]
                            .kind
                            .as_ref()
                            .map(|(kind, _)| kind.clone())
                    }),
                    conflict,
                }))
            };
            let parent = self.unions.len();
            let (kept, merged) = (&classes.held[&keep], &classes.held[&gone]);
            let kind = match (&kept.kind, &merged.kind) {
                (Some((kind, s)), Some((other, t))) => {
                    agree(kind, other).map_err(apart)?;
                    work.extend(
                        part_pairs(kind, other)
                            .into_iter()
                            .map(|(part, a, b)| Pair {
                                a,
                                b,
                                cause: Cause::Congruence { parent, part },
                            }),
                    );
                    Some((kind.clone(), joined_source(*s, *t)))
                }
                (kind, other) => kind.clone().or_else(|| other.clone()),
            };
            let lit = match (&kept.lit, &merged.lit) {
                (Some(lit), Some(other)) => Some(
                    lit.union(other)
                        .ok_or_else(|| apart(Conflict::Lit(other.clone(), None)))?,
                ),
                (lit, other) => lit.clone().or_else(|| other.clone()),
            };
            let reference = match (kept.reference, merged.reference) {
                (Some(own), Some(other)) if own != other => {
                    return Err(apart(Conflict::Reference(other, None)));
                }
                (own, other) => own.or(other),
            };
            self.classes.parent[gone.0] = keep;
            let merged = self.classes.held.remove(&gone).expect("a root holds");
            let held = self.classes.held.get_mut(&keep).expect("a root holds");
            held.members.extend(merged.members);
            held.values.extend(merged.values);
            held.facts.extend(merged.facts);
            held.kind = kind;
            held.lit = lit;
            held.reference = reference;
            held.solved |= merged.solved;
            let kind = held.kind.as_ref().map(|(kind, _)| kind.clone());
            self.unions.push(Union { cause, sides, kind });
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

/// What part a fact names, so that two facts of one part name one class.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum FactPart {
    Target,
    Field(Name),
    Component(usize),
    Element,
    Variant(Name),
}

impl FactPart {
    fn of(fact: &Fact) -> FactPart {
        match fact {
            Fact::Ref(_) => FactPart::Target,
            Fact::Field(field, _) => FactPart::Field(field.clone()),
            Fact::Component(i, _) => FactPart::Component(*i),
            Fact::Indexes(_) => FactPart::Element,
            Fact::Variant { variant, .. } => FactPart::Variant(variant.clone()),
        }
    }
}

/// The part of `kind` the fact names, where the kind meets the fact. A
/// struct's field is the part at the field's place in the declared order
/// (l.314). An extern-defined container's element is not a part of it.
fn resolve(kind: &Term<ClassId>, fact: &Fact, types: &DeclaredTypes) -> Option<ClassId> {
    match (kind, fact) {
        (Term::Ref(_, target), Fact::Ref(_)) => Some(*target),
        (Term::Object(fields), Fact::Field(field, _)) => fields.get(field).copied(),
        (Term::Struct(module, args), Fact::Field(field, _)) => {
            let place = types
                .struct_decl(module)?
                .fields
                .iter()
                .position(|declared| declared == field)?;
            args.get(place).copied()
        }
        (Term::Tuple(components), Fact::Component(i, _)) => components.get(*i).copied(),
        (Term::Array(_, element) | Term::Slice(element), Fact::Indexes(_)) => Some(*element),
        (
            term,
            Fact::Variant {
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

/// A kind written by the text or a declaration keeps the equivalence class
/// "with an equation" whatever else it is unified with.
fn joined_source(a: KindSource, b: KindSource) -> KindSource {
    if a == KindSource::Written || b == KindSource::Written {
        KindSource::Written
    } else {
        KindSource::Solved
    }
}

/// Two kinds that one equivalence class may hold together.
fn agree(a: &Term<ClassId>, b: &Term<ClassId>) -> Result<(), Conflict> {
    if a.constructor() == b.constructor() && a.members() == b.members() {
        Ok(())
    } else {
        Err(Conflict::Kinds(a.constructor(), b.constructor()))
    }
}

/// The parts of one name of two kinds of one constructor and members, each
/// with its name.
fn part_pairs(a: &Term<ClassId>, b: &Term<ClassId>) -> Vec<(PartKey, ClassId, ClassId)> {
    a.parts()
        .into_iter()
        .zip(b.parts())
        .map(|((key, x), (_, y))| (key, *x, *y))
        .collect()
}
