//! An equivalence class's value (l.172–176, l.273–282): a cycle of lower
//! bounds alone merged first, a class that reaches a part of its own refused,
//! a LitVar's join unifying it, and the join of the constructors and members
//! the sources give, the kind the class takes with its parts made anew.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::Refusal;
use crate::class::{ClassId, ClassName, Fact, KindSource};
use crate::decl::Node;
use crate::program::Visibility;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Member, Name, PartKey, Term, Ty};

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
    /// Refuses the equivalence class where it reaches a part of its own along
    /// recorded lower bounds and parts: no finite type holds it, so it has no
    /// value (l.275). Returns whether it is refused.
    pub(crate) fn refuse_infinite(&mut self, root: ClassId) -> bool {
        if !self.reaches_own_part(root) {
            return false;
        }
        let name = self.class_name(root);
        let members = self.classes.held(root).members.clone();
        self.no_value.extend(members);
        self.refusals.insert(Refusal::Infinite(name));
        true
    }

    /// Unifies with the equivalence class each LitVar lower bound of it that
    /// waits for nothing: the join of a LitVar (l.176).
    pub(crate) fn join_lit_lowers(&mut self, root: ClassId) {
        let settled = self.settled();
        for lower in self.lower_bounds(root) {
            if self.is_lit(lower) && settled.contains(&self.root(lower)) {
                self.record(Relation::Equation(lower, root), Recorder::Split(root));
            }
        }
    }

    /// The equivalence class takes its constructor and members, and is
    /// solved (l.275): where it holds no kind, the join of the constructors
    /// and members its solved lower bounds and its kinds known in part give
    /// (l.280), its parts made anew. Where they give none it is left
    /// unsolved, and where they have no join it is refused.
    pub(crate) fn join(&mut self, root: ClassId) {
        let root = self.root(root);
        if self.classes.held(root).kind.is_none() {
            let mut sources: Vec<Head> = self
                .lower_bounds(root)
                .into_iter()
                .filter(|&lower| self.is_solved(lower))
                .filter_map(|lower| self.classes.held(lower).kind.as_ref().map(|(k, _)| head(k)))
                .collect();
            sources.extend(self.fact_heads(root));
            let Some(first) = sources.first().cloned() else {
                return;
            };
            let joined = sources
                .iter()
                .skip(1)
                .try_fold(first, |joined, source| self.head_join(&joined, source));
            let Some(joined) = joined else {
                let name = self.class_name(root);
                let members = self.classes.held(root).members.clone();
                self.no_value.extend(members);
                self.refusals.insert(Refusal::NoJoin(name));
                return;
            };
            let owner = self.class_name(root);
            let kind = self.skeleton(&joined, &owner);
            let decls = Rc::clone(&self.decls);
            if let Err(conflict) =
                self.classes
                    .give_kind(root, kind, KindSource::Solved, decls.types())
            {
                self.refuse_conflict(format!("the value of {owner}"), conflict);
            }
        }
        self.classes.mark_solved(root);
    }

    /// Unifies a LitVar lower bound, at `i` in the log, with the solved
    /// equivalence class it bounds, where the LitVar's class waits for nothing
    /// (`settled`): the join of a LitVar (l.176). It is then marked split.
    pub(crate) fn join_lit_at(&mut self, i: usize, settled: &BTreeSet<ClassId>) {
        let Relation::LowerBound { lower, upper } = self.log[i].relation else {
            return;
        };
        if self.is_lit(lower) && self.is_solved(upper) && settled.contains(&self.root(lower)) {
            self.split.insert(i);
            self.record(Relation::Equation(lower, upper), Recorder::Split(upper));
        }
    }

    /// Merges each cycle of lower bounds alone into one equivalence class
    /// (l.275).
    pub(crate) fn merge_cycles(&mut self) {
        let mut edges: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        for recorded in &self.log {
            if let Relation::LowerBound { lower, upper } = recorded.relation {
                let (lower, upper) = (self.root(lower), self.root(upper));
                if lower != upper {
                    edges.entry(upper).or_default().insert(lower);
                }
            }
        }
        let reach = |from: ClassId| {
            let mut seen = BTreeSet::new();
            let mut work = vec![from];
            while let Some(at) = work.pop() {
                for &next in edges.get(&at).into_iter().flatten() {
                    if seen.insert(next) {
                        work.push(next);
                    }
                }
            }
            seen
        };
        let mut merges = Vec::new();
        for &from in edges.keys() {
            for to in reach(from) {
                if to != from && reach(to).contains(&from) && from < to {
                    merges.push((from, to));
                }
            }
        }
        for (a, b) in merges {
            if self.root(a) != self.root(b) {
                self.record(Relation::Equation(a, b), Recorder::Cycle);
            }
        }
    }

    /// Whether the equivalence class reaches a part of its own along recorded
    /// lower bounds and parts (l.275).
    fn reaches_own_part(&self, root: ClassId) -> bool {
        let root = self.root(root);
        let mut seen: BTreeSet<(ClassId, bool)> = BTreeSet::new();
        let mut work = vec![(root, false)];
        while let Some((at, through_part)) = work.pop() {
            let mut next: Vec<(ClassId, bool)> = self
                .lower_bounds(at)
                .into_iter()
                .map(|lower| (self.root(lower), through_part))
                .collect();
            if let Some((kind, _)) = &self.classes.held(at).kind {
                next.extend(
                    kind.parts()
                        .into_iter()
                        .map(|(_, part)| (self.root(*part), true)),
                );
            }
            for (class, through_part) in next {
                if class == root && through_part {
                    return true;
                }
                if seen.insert((class, through_part)) {
                    work.push((class, through_part));
                }
            }
        }
        false
    }

    /// The closed type the equivalence class holds, where it has one.
    pub fn value(&self, id: ClassId) -> Option<Ty> {
        self.value_in(id, &mut BTreeSet::new())
    }

    fn value_in(&self, id: ClassId, visiting: &mut BTreeSet<ClassId>) -> Option<Ty> {
        let root = self.root(id);
        if self.has_no_value(root) || !visiting.insert(root) {
            return None;
        }
        let (kind, _) = self.classes.held(root).kind.clone()?;
        let term = kind
            .try_map(|_, part| self.value_in(part, visiting).ok_or(()))
            .ok()?;
        visiting.remove(&root);
        Ty::new(term).ok()
    }
}
