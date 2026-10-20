//! Equivalence classes: their names, union-find, type variables and kinds
//! (l.118–176).
//!
//! A type variable is one type, represented by an equivalence class (l.120).
//! An equivalence class holds type variables, of which the model keeps the
//! LitVar, and at most one kind (l.124). Unifying two merges them, the kind
//! the conjunction of the two, and unifies their parts of one name in turn
//! (l.165).

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
    /// takes a constructor (l.275, l.284), or written with it.
    Part(Rc<ClassName>, PartKey),
    /// A class a use or a lend of this name made anew.
    Node(Rc<Name>, crate::node::NodeAt),
    /// A class of the signature of the cast at `step` of the path that links
    /// the parts of a relation from `from` to `to` (l.291), or of the view
    /// that links the container `from` to its element `to` (l.161).
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
            ClassName::Link { from, to, step, at } => {
                write!(f, "link({from} → {to})[{step}].{at:?}")
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

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// A kind known only in part (l.140–147).
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
    /// `None`, `Ok` and `Err` (l.327).
    Variant {
        enum_name: Option<Name>,
        variant: Name,
        payload: ClassId,
    },
}

impl Fact {
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
    /// visible to this body (l.282): a field, or a variant of an enum the text
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
    /// before solving decides it (l.279, "with an equation").
    Written,
    /// The equivalence class took it when it was solved (l.275).
    Solved,
}

/// What one equivalence class holds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Held {
    pub members: BTreeSet<ClassId>,
    pub kind: Option<(Term<ClassId>, KindSource)>,
    pub facts: BTreeSet<Fact>,
    pub lit: Option<LitVar>,
    /// Whether it took its constructor and members, written or joined
    /// (l.275).
    pub solved: bool,
}

/// Why no type holds the conjunction of what two equivalence classes hold.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conflict {
    /// Two kinds of different constructors or members.
    Kinds(Constructor, Constructor),
    /// A LitVar and a constructor that does not admit its values (l.169), or
    /// two LitVars of different families (l.170).
    Lit(LitVar, Option<Constructor>),
    /// A fact that the kind does not meet (l.282).
    Fact(Fact, Constructor),
}

/// The equivalence classes over one arena.
#[derive(Clone, Default, Debug)]
pub struct Classes {
    parent: Vec<ClassId>,
    held: BTreeMap<ClassId, Held>,
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

    /// Gives the equivalence class of `id` the kind `term`.
    pub fn give_kind(
        &mut self,
        id: ClassId,
        term: Term<ClassId>,
        source: KindSource,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        let root = self.find(id);
        let held = self.held.get_mut(&root).expect("a root holds");
        let mut pairs = Vec::new();
        held.kind = Some(match held.kind.take() {
            Some((kind, held_source)) => {
                if let Err(conflict) = agree(&kind, &term) {
                    held.kind = Some((kind, held_source));
                    return Err(conflict);
                }
                pairs.extend(part_pairs(&kind, &term));
                (kind, joined_source(held_source, source))
            }
            None => (term, source),
        });
        pairs.extend(self.normalize(root, types)?);
        self.unify_pairs(pairs, types)
    }

    /// Marks the equivalence class of `id` as having taken its constructor.
    pub fn mark_solved(&mut self, id: ClassId) {
        let root = self.find(id);
        self.held.get_mut(&root).expect("a root holds").solved = true;
    }

    pub fn give_fact(
        &mut self,
        id: ClassId,
        fact: Fact,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        let root = self.find(id);
        self.held
            .get_mut(&root)
            .expect("a root holds")
            .facts
            .insert(fact);
        let pairs = self.normalize(root, types)?;
        self.unify_pairs(pairs, types)
    }

    pub fn give_lit(
        &mut self,
        id: ClassId,
        lit: LitVar,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        let root = self.find(id);
        {
            let held = self.held.get_mut(&root).expect("a root holds");
            held.lit = Some(match &held.lit {
                Some(own) => own
                    .union(&lit)
                    .ok_or_else(|| Conflict::Lit(lit.clone(), None))?,
                None => lit,
            });
        }
        let pairs = self.normalize(root, types)?;
        self.unify_pairs(pairs, types)
    }

    /// Merges the two equivalence classes, and their parts of one name in turn.
    pub fn unify(&mut self, a: ClassId, b: ClassId, types: &DeclaredTypes) -> Result<(), Conflict> {
        self.unify_pairs(vec![(a, b)], types)
    }

    fn unify_pairs(
        &mut self,
        mut work: Vec<(ClassId, ClassId)>,
        types: &DeclaredTypes,
    ) -> Result<(), Conflict> {
        while let Some((a, b)) = work.pop() {
            let (a, b) = (self.find(a), self.find(b));
            if a == b {
                continue;
            }
            let (keep, gone) = if a < b { (a, b) } else { (b, a) };
            let (kept, merged) = (&self.held[&keep], &self.held[&gone]);
            let kind = match (&kept.kind, &merged.kind) {
                (Some((kind, s)), Some((other, t))) => {
                    agree(kind, other)?;
                    work.extend(part_pairs(kind, other));
                    Some((kind.clone(), joined_source(*s, *t)))
                }
                (kind, other) => kind.clone().or_else(|| other.clone()),
            };
            let lit = match (&kept.lit, &merged.lit) {
                (Some(lit), Some(other)) => Some(
                    lit.union(other)
                        .ok_or_else(|| Conflict::Lit(other.clone(), None))?,
                ),
                (lit, other) => lit.clone().or_else(|| other.clone()),
            };
            self.parent[gone.0] = keep;
            let merged = self.held.remove(&gone).expect("a root holds");
            let held = self.held.get_mut(&keep).expect("a root holds");
            held.members.extend(merged.members);
            held.facts.extend(merged.facts);
            held.kind = kind;
            held.lit = lit;
            held.solved |= merged.solved;
            work.extend(self.normalize(keep, types)?);
        }
        Ok(())
    }

    /// The pairs of classes that what the equivalence class holds makes one:
    /// facts of one part name with each other and with its kind's part, and
    /// the LitVar checked against its kind's constructor.
    fn normalize(
        &mut self,
        root: ClassId,
        types: &DeclaredTypes,
    ) -> Result<Vec<(ClassId, ClassId)>, Conflict> {
        let held = self.held.get_mut(&root).expect("a root holds");
        if let (Some(lit), Some((kind, _))) = (&held.lit, &held.kind) {
            if !matches!(kind, Term::Scalar(scalar) if lit.admits(*scalar)) {
                return Err(Conflict::Lit(lit.clone(), Some(kind.constructor())));
            }
            held.lit = None;
        }
        let mut pairs = Vec::new();
        let mut by_part: BTreeMap<FactPart, ClassId> = BTreeMap::new();
        for fact in &held.facts {
            let part = FactPart::of(fact);
            match by_part.get(&part) {
                Some(&first) => pairs.push((first, fact.class())),
                None => {
                    by_part.insert(part, fact.class());
                }
            }
            if let Some((kind, _)) = &held.kind {
                if let (Term::Extern { .. }, Fact::Indexes(_)) = (kind, fact) {
                    // The state links it along the container's view (l.161).
                    continue;
                }
                let resolved = resolve(kind, fact, types)
                    .ok_or_else(|| Conflict::Fact(fact.clone(), kind.constructor()))?;
                pairs.push((resolved, fact.class()));
            }
        }
        Ok(pairs)
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
/// (l.325). An extern-defined container's element is not a part of it.
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

fn part_pairs(a: &Term<ClassId>, b: &Term<ClassId>) -> Vec<(ClassId, ClassId)> {
    a.parts()
        .into_iter()
        .zip(b.parts())
        .map(|((_, x), (_, y))| (*x, *y))
        .collect()
}
