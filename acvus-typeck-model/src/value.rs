//! An equivalence class's value: the join of the constructors and members its
//! sources give, and the kind it takes with its parts made anew (l.273–282).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Classes, Fact};
use crate::decl::Node;
use crate::program::Visibility;
use crate::read::ReadValue;
use crate::relation::Recorder;
use crate::solve::State;
use crate::ty::{Constructor, HasTerm, Member, Name, PartKey, Term};

/// A constructor and its members: what an equivalence class takes when it is
/// solved (l.275), each part then solved on its own.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Head {
    pub constructor: Constructor,
    pub members: BTreeSet<Member>,
}

pub fn head(term: &Term<ClassId>) -> Head {
    Head {
        constructor: term.constructor(),
        members: term.members(),
    }
}

impl State {
    /// The join of two sources' constructors and members (l.86–90): `!` below
    /// every one, two references at the lesser mutability, and two others
    /// raised by casts to their nearest common ancestor node, their members
    /// united. A reference's target is a part, and the constructor of a
    /// reference is its mutability alone (l.66), so two references join
    /// whatever their targets.
    pub(crate) fn head_join(&self, a: &Head, b: &Head) -> Option<Head> {
        match (&a.constructor, &b.constructor) {
            (Constructor::Never, _) => return Some(b.clone()),
            (_, Constructor::Never) => return Some(a.clone()),
            (Constructor::Ref(m), Constructor::Ref(n)) => {
                return Some(Head {
                    constructor: Constructor::Ref(*m.min(n)),
                    members: BTreeSet::new(),
                });
            }
            (Constructor::Ref(_), _) | (_, Constructor::Ref(_)) => return None,
            _ => {}
        }
        let nca = self.decls.tree().nca(
            &Node::Owned(a.constructor.clone()),
            &Node::Owned(b.constructor.clone()),
        )?;
        let (a, b) = (self.raise_head(a, &nca)?, self.raise_head(b, &nca)?);
        if a.constructor != b.constructor {
            return None;
        }
        let members = match a.constructor {
            Constructor::Object | Constructor::Enum(_) => {
                a.members.union(&b.members).cloned().collect()
            }
            _ if a.members == b.members => a.members,
            _ => return None,
        };
        Some(Head {
            constructor: a.constructor,
            members,
        })
    }

    /// The constructors and members the kinds known in part of the class give:
    /// those of the members visible to this body (l.282).
    pub(crate) fn fact_heads(&self, root: ClassId) -> Vec<Head> {
        let host_bound = self
            .lower_bounds(root)
            .into_iter()
            .any(|lower| self.is_host(lower));
        if self.readings.visibility == Visibility::BySources && host_bound {
            return Vec::new();
        }
        self.classes
            .held(root)
            .facts
            .iter()
            .filter_map(Fact::visible_member)
            .map(|(constructor, member)| Head {
                constructor,
                members: [member].into(),
            })
            .collect()
    }

    /// Whether the equivalence class holds a type the host gives.
    fn is_host(&self, id: ClassId) -> bool {
        self.classes
            .held(id)
            .members
            .iter()
            .any(|member| self.hosts.contains(member))
    }

    /// The constructor and members at `node` that the casts up from `head`'s
    /// node reach: those of the last cast's result.
    fn raise_head(&self, head: &Head, node: &Node) -> Option<Head> {
        let path = self
            .decls
            .tree()
            .path(&Node::Owned(head.constructor.clone()), node)?;
        let Some(last) = path.last() else {
            return Some(head.clone());
        };
        let result = self.decls.decl(last).sig.result().clone();
        let crate::sig::SigTy::Term(term) = result else {
            return None;
        };
        Some(Head {
            constructor: term.constructor(),
            members: term.members(),
        })
    }

    /// The kind of `head`, each part a class made anew and named after
    /// `owner` (l.275). Every kind's constructor agrees with its declaration:
    /// the text's are checked when they are recorded, and a signature's when
    /// it is declared.
    pub(crate) fn skeleton(&mut self, head: &Head, owner: &ClassName) -> Term<ClassId> {
        let owner = Rc::new(owner.clone());
        let decls = Rc::clone(&self.decls);
        let mut part = |key: PartKey| {
            let class = self.arena.intern(ClassName::Part(owner.clone(), key));
            self.classes.grow(&self.arena);
            class
        };
        let named = |members: &BTreeSet<Member>| -> Vec<Name> {
            members
                .iter()
                .map(|member| match member {
                    Member::Field(name) | Member::Variant(name) => name.clone(),
                })
                .collect()
        };
        match &head.constructor {
            Constructor::Ref(mutability) => Term::Ref(*mutability, part(PartKey::Target)),
            Constructor::Tuple(n) => {
                Term::Tuple((0..*n).map(|i| part(PartKey::Component(i))).collect())
            }
            Constructor::Array(n) => Term::Array(*n, part(PartKey::Element)),
            Constructor::Slice => Term::Slice(part(PartKey::Element)),
            Constructor::Option => Term::Option(part(PartKey::Payload(Name::new(crate::ty::SOME)))),
            Constructor::Result => Term::Result {
                ok: part(PartKey::Payload(Name::new(crate::ty::OK))),
                err: part(PartKey::Payload(Name::new(crate::ty::ERR))),
            },
            Constructor::Function(arity) => Term::FunctionValue { arity: *arity },
            Constructor::Object => Term::Object(
                named(&head.members)
                    .into_iter()
                    .map(|field| (field.clone(), part(PartKey::Field(field))))
                    .collect(),
            ),
            Constructor::Enum(name) => Term::Enum(
                name.clone(),
                named(&head.members)
                    .into_iter()
                    .map(|variant| (variant.clone(), part(PartKey::Payload(variant))))
                    .collect(),
            ),
            Constructor::Struct(module) => {
                let fields = decls
                    .types()
                    .struct_decl(module)
                    .expect("a kind's struct is declared")
                    .fields
                    .len();
                Term::Struct(
                    module.clone(),
                    (0..fields).map(|i| part(PartKey::Arg(i))).collect(),
                )
            }
            Constructor::Extern(name) => {
                let decl = decls
                    .types()
                    .extern_decl(name)
                    .expect("a kind's extern-defined type is declared");
                Term::Extern {
                    name: name.clone(),
                    args: (0..decl.arity).map(|i| part(PartKey::Arg(i))).collect(),
                    identity: decl.identity.then(|| part(PartKey::Identity)),
                }
            }
            Constructor::Scalar(scalar) => Term::Scalar(*scalar),
            Constructor::String => Term::String,
            Constructor::Str => Term::Str,
            Constructor::Unit => Term::Unit,
            Constructor::Never => Term::Never,
            Constructor::Identity(name) => Term::Identity(name.clone()),
        }
    }
}

impl State {
    /// The state a node's value read is taken from (l.267): the relations
    /// recorded by neither the nodes `excluded` nor solving, solved where
    /// each class's value is decided by its sources whatever its other
    /// dependencies, no node decided, nothing fallen back, and a class with
    /// no relation left giving nothing. Classes are solved in one order here.
    pub(crate) fn read_state(&self, excluded: &BTreeSet<Name>) -> State {
        let mut read = State {
            decls: Rc::clone(&self.decls),
            arena: self.arena.clone(),
            classes: Classes::default(),
            log: Vec::new(),
            texts: self.texts.clone(),
            decided: self.decided.clone(),
            split: BTreeSet::new(),
            no_value: BTreeSet::new(),
            refusals: BTreeSet::new(),
            nodes: BTreeMap::new(),
            nothing: BTreeSet::new(),
            readings: self.readings,
            hosts: self.hosts.clone(),
            viewed: BTreeSet::new(),
            deferred: BTreeMap::new(),
            broken: BTreeMap::new(),
        };
        read.classes.grow(&read.arena);
        for recorded in &self.log {
            if recorded.recorder.is_derived() {
                continue;
            }
            if let Recorder::Node(name) = &recorded.recorder
                && excluded.contains(name)
            {
                continue;
            }
            read.record(recorded.relation.clone(), recorded.recorder.clone());
        }
        read.settle();
        loop {
            if let Some(&root) = read.eligible().0.first() {
                read.solve(root);
                read.settle();
                continue;
            }
            let more: Vec<ClassId> = read
                .fallback_targets()
                .into_iter()
                .filter(|&root| !read.gives_nothing(root))
                .collect();
            if more.is_empty() {
                return read;
            }
            read.nothing.extend(more);
        }
    }

    /// The value the class has here: a position with no kind is undecided,
    /// and one holding only a LitVar is that LitVar.
    pub(crate) fn read_value(&self, id: ClassId) -> ReadValue {
        self.read_value_in(id, &mut BTreeSet::new())
    }

    fn read_value_in(&self, id: ClassId, visiting: &mut BTreeSet<ClassId>) -> ReadValue {
        let root = self.root(id);
        let held = self.classes.held(root);
        if self.has_no_value(root) || !visiting.insert(root) {
            return ReadValue::Undecided;
        }
        let value = match (&held.kind, &held.lit) {
            (Some((kind, _)), _) => {
                let term = kind
                    .clone()
                    .map(|_, part| self.read_value_in(part, visiting));
                ReadValue::from_term(term).unwrap_or(ReadValue::Undecided)
            }
            (None, Some(lit)) => ReadValue::Lit(lit.clone()),
            (None, None) => ReadValue::Undecided,
        };
        visiting.remove(&root);
        value
    }
}
