//! An equivalence class's value (l.167–171, l.269–278): a cycle of lower
//! bounds alone merged first, a class that reaches a part of its own refused,
//! a LitVar's join unifying it, and the join of the constructors and members
//! the sources give, the kind the class takes with its parts made anew.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::Refusal;
use crate::class::{ClassId, ClassName, Fact, KindSource, Unified};
use crate::decl::Node;
use crate::program::Visibility;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Member, Name, PartKey, Term, Ty};

/// A constructor and its members: what an equivalence class takes when it is
/// solved (l.271), each part then solved on its own.
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

/// What an equivalence class took when it was solved (l.271).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Took {
    /// The kind it held already.
    Held,
    /// The join of the constructors and members its sources give, its parts
    /// made anew, with what unification did as the class took it, and the
    /// refusal where the class could not take it.
    Joined {
        head: Head,
        unified: Unified,
        refused: Option<Refusal>,
    },
    /// No source gives a constructor, so it is left unsolved.
    Nothing,
    /// Its sources have no join, so it has no value and is refused.
    NoJoin(Refusal),
}

/// What an equivalence class's value is taken from when it is solved
/// (l.275–278).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sources {
    /// The relations of the log that make the class hold a kind, by their
    /// places: the equations on its members, and the kinds recorded on them
    /// (l.275).
    pub equations: Vec<usize>,
    /// Each solved lower bound, with the constructor and members it gives
    /// (l.276).
    pub lowers: Vec<(ClassId, Head)>,
    /// The constructors and members its kinds known in part give (l.278).
    pub facts: Vec<Head>,
    /// The lower bounds not solved, which give nothing.
    pub left_out: Vec<ClassId>,
}

/// What an equivalence class took when it was solved, and its sources then
/// (l.271–278).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Joined {
    pub sources: Sources,
    pub took: Took,
}

impl State {
    /// The join of two sources' constructors and members (l.86–88): `!` below
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
    /// those of the members visible to this body (l.278).
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
    /// `owner` (l.271). Every kind's constructor agrees with its declaration:
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
    /// value (l.271). Returns the refusal, where it is refused.
    pub(crate) fn refuse_infinite(&mut self, root: ClassId) -> Option<Refusal> {
        if !self.reaches_own_part(root) {
            return None;
        }
        let name = self.class_name(root);
        let members = self.classes.held(root).members.clone();
        self.no_value.extend(members);
        let refusal = Refusal::Infinite(name);
        self.refusals.insert(refusal.clone());
        Some(refusal)
    }

    /// Unifies with the equivalence class each LitVar lower bound of it that
    /// waits for nothing: the join of a LitVar (l.171). Returns the
    /// equations recorded.
    pub(crate) fn join_lit_lowers(&mut self, root: ClassId) -> Recording {
        let mut recorded = Recording::default();
        let settled = self.settled();
        for lower in self.lower_bounds(root) {
            if self.is_lit(lower) && settled.contains(&self.root(lower)) {
                let relation = Relation::Equation(lower, root);
                recorded.push(self.record(relation, Recorder::Split(root)));
            }
        }
        recorded
    }

    /// The sources of the equivalence class's value (l.275–278).
    pub(crate) fn sources(&self, root: ClassId) -> Sources {
        let root = self.root(root);
        let equations = self
            .log
            .iter()
            .enumerate()
            .filter(|(_, recorded)| match &recorded.relation {
                Relation::Equation(a, b) => self.root(*a) == root || self.root(*b) == root,
                Relation::Kind(class, ..) => self.root(*class) == root,
                _ => false,
            })
            .map(|(at, _)| at)
            .collect();
        let mut lowers = Vec::new();
        let mut left_out = Vec::new();
        for lower in self.lower_bounds(root) {
            let given = self
                .classes
                .held(lower)
                .kind
                .as_ref()
                .filter(|_| self.is_solved(lower))
                .map(|(kind, _)| head(kind));
            match given {
                Some(given) => lowers.push((lower, given)),
                None => left_out.push(lower),
            }
        }
        Sources {
            equations,
            lowers,
            facts: self.fact_heads(root),
            left_out,
        }
    }

    /// The equivalence class takes its constructor and members, and is
    /// solved (l.271): where it holds no kind, the join of the constructors
    /// and members its solved lower bounds and its kinds known in part give
    /// (l.276), its parts made anew. Where they give none it is left
    /// unsolved, and where they have no join it is refused.
    pub(crate) fn join(&mut self, root: ClassId) -> Joined {
        let root = self.root(root);
        let sources = self.sources(root);
        let with_sources = |took| Joined {
            sources: sources.clone(),
            took,
        };
        if self.classes.held(root).kind.is_some() {
            self.classes.mark_solved(root);
            return with_sources(Took::Held);
        }
        let given: Vec<Head> = sources
            .lowers
            .iter()
            .map(|(_, given)| given.clone())
            .chain(sources.facts.iter().cloned())
            .collect();
        let Some(first) = given.first().cloned() else {
            return with_sources(Took::Nothing);
        };
        let joined = given
            .iter()
            .skip(1)
            .try_fold(first, |joined, source| self.head_join(&joined, source));
        let Some(joined) = joined else {
            let refusal = Refusal::NoJoin(self.class_name(root));
            return with_sources(self.no_join(root, refusal));
        };
        let owner = self.class_name(root);
        let kind = self.skeleton(&joined, &owner);
        let decls = Rc::clone(&self.decls);
        let unified =
            self.classes
                .unifying(decls.types())
                .give_kind(root, kind, KindSource::Solved);
        let refused = unified
            .result()
            .err()
            .map(|conflict| self.refuse_conflict(format!("the value of {owner}"), conflict));
        self.classes.mark_solved(root);
        with_sources(Took::Joined {
            head: joined,
            unified,
            refused,
        })
    }

    /// The equivalence class, holding no kind, has LitVar lower bounds whose
    /// unification with it, their join (l.171), conflicts: two LitVars of two
    /// families meet (l.165). Its sources have no join, so it has no value
    /// and is refused there (l.292), by the conflict's refusal.
    pub(crate) fn lit_no_join(&mut self, root: ClassId, refusal: Refusal) -> Joined {
        let root = self.root(root);
        Joined {
            sources: self.sources(root),
            took: self.no_join(root, refusal),
        }
    }

    /// The equivalence class has no value, and is never solved again (l.292).
    fn no_join(&mut self, root: ClassId, refusal: Refusal) -> Took {
        let members = self.classes.held(root).members.clone();
        self.no_value.extend(members);
        self.refusals.insert(refusal.clone());
        Took::NoJoin(refusal)
    }

    /// Unifies a LitVar lower bound, at `i` in the log, with the solved
    /// equivalence class it bounds, where the LitVar's class waits for nothing
    /// (`settled`): the join of a LitVar (l.171). It is then marked split.
    /// Returns the equation recorded, where one is.
    pub(crate) fn join_lit_at(
        &mut self,
        i: usize,
        settled: &BTreeSet<ClassId>,
    ) -> Option<Recording> {
        let Relation::LowerBound { lower, upper } = self.log[i].relation else {
            return None;
        };
        if !(self.is_lit(lower) && self.is_solved(upper) && settled.contains(&self.root(lower))) {
            return None;
        }
        self.split.insert(i);
        let mut recorded = Recording::default();
        let relation = Relation::Equation(lower, upper);
        recorded.push(self.record(relation, Recorder::Split(upper)));
        Some(recorded)
    }

    /// Merges each cycle of lower bounds alone into one equivalence class
    /// (l.271). Returns the equations recorded.
    pub(crate) fn merge_cycles(&mut self) -> Recording {
        let mut recorded = Recording::default();
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
                recorded.push(self.record(Relation::Equation(a, b), Recorder::Cycle));
            }
        }
        recorded
    }

    /// Whether the equivalence class reaches a part of its own along recorded
    /// lower bounds and parts (l.271).
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
