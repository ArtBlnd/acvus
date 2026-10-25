//! An equivalence class's value: a cycle of lower
//! bounds and part lower bounds alone merged first, a class that reaches a
//! part of its own refused, the join of the constructors and members the
//! sources give, and the kind the class takes with its parts made anew.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::Refusal;
use crate::class::{ClassId, ClassName, Fact, KindSource, Unified};
use crate::decl::Node;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Constructor, Member, Name, PartKey, Term, Ty};

/// A constructor and its members: what an equivalence class takes when it is
/// solved, each part then solved on its own.
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

/// What an equivalence class took when it was solved.
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

/// What an equivalence class's value is taken from when it is solved.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sources {
    /// The relations of the log that make the class hold a kind, by their
    /// places: the equations on its members, and the kinds recorded on them.
    pub equations: Vec<usize>,
    /// Each solved lower bound, with the constructor and members it gives.
    pub lowers: Vec<(ClassId, Head)>,
    /// Each solved part lower bound, with the constructor and members it
    /// gives, which reach the value by `≤ₚ`.
    pub part_lowers: Vec<(ClassId, Head)>,
    /// The constructors and members its kinds known in part give.
    pub facts: Vec<Head>,
    /// The lower bounds and part lower bounds not solved, which give
    /// nothing.
    pub left_out: Vec<ClassId>,
}

/// What an equivalence class took when it was solved, and its sources then.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Joined {
    pub sources: Sources,
    pub took: Took,
}

impl State {
    /// The join of two sources' constructors and members: `!` below
    /// every one, two references at the lesser mutability, and two others
    /// raised by casts to their nearest common ancestor node, their members
    /// united. A reference's target is a part, and the constructor of a
    /// reference is its mutability alone, so two references join
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

    /// The join raised where the sources carry two different identities, which
    /// no class holds: to the nearest node above that carries none,
    /// each value cast up to it; none where no such node is, or where a part
    /// lower bound carries one, which reaches with no cast.
    fn without_two_identities(&self, sources: &Sources, joined: Head) -> Option<Head> {
        let identities = |given: &[(ClassId, Head)]| -> BTreeSet<Ty> {
            given
                .iter()
                .filter_map(|(class, _)| self.value(*class))
                .filter_map(|value| crate::join::identity(&value).cloned())
                .collect()
        };
        let (lowers, part_lowers) = (
            identities(&sources.lowers),
            identities(&sources.part_lowers),
        );
        if lowers.union(&part_lowers).count() < 2 {
            return Some(joined);
        }
        if !part_lowers.is_empty() {
            return None;
        }
        let node = Node::Owned(joined.constructor.clone());
        let above = crate::join::without_identity(&self.decls, &node)?;
        self.raise_head(&joined, &above)
    }

    /// The constructors and members the kinds known in part of the class give:
    /// where kinds are its only sources, the constructor of each member they
    /// name, visible or not; otherwise those of the members visible to this
    /// body, which alone a kind adds to a type another source gives.
    pub(crate) fn fact_heads(&self, root: ClassId) -> Vec<Head> {
        let lowers = self.lower_bounds(root);
        if lowers.is_empty() {
            return self
                .classes
                .held(root)
                .facts
                .iter()
                .filter_map(Fact::named_member)
                .map(|(constructor, members)| Head {
                    constructor,
                    members,
                })
                .collect();
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

    /// The constructor a range admitting one alone gives the class holding
    /// it.
    pub(crate) fn range_head(&self, root: ClassId) -> Option<Head> {
        let range = self.classes.held(self.root(root)).range.as_ref()?;
        let constructor = range.single()?.clone();
        Some(Head {
            constructor,
            members: BTreeSet::new(),
        })
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
    /// `owner`. Every kind's constructor agrees with its declaration:
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
            Constructor::ErrorOf => Term::ErrorOf(part(PartKey::Wrapped)),
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
    /// recorded lower bounds, part lower bounds and parts: no finite type
    /// holds it, so it has no value. Returns the refusal, where it is
    /// refused.
    pub(crate) fn refuse_infinite(&mut self, root: ClassId) -> Option<Refusal> {
        if !self.reaches_own_part(root) {
            return None;
        }
        let refusal = Refusal::Infinite(self.class_name(root));
        self.refuse_value(root, &refusal);
        Some(refusal)
    }

    /// The sources of the equivalence class's value.
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
        let given = |lower: ClassId| {
            self.classes
                .held(lower)
                .kind
                .as_ref()
                .filter(|_| self.is_solved(lower))
                .map(|(kind, _)| head(kind))
        };
        let bounds =
            |lower: ClassId, upper: ClassId| self.root(upper) == root && self.root(lower) != root;
        let mut lowers = Vec::new();
        let mut part_lowers = Vec::new();
        let mut left_out = Vec::new();
        for recorded in &self.log {
            match recorded.relation {
                Relation::LowerBound { lower, upper } if bounds(lower, upper) => {
                    match given(lower) {
                        Some(given) => lowers.push((lower, given)),
                        None => left_out.push(lower),
                    }
                }
                Relation::PartLowerBound { lower, upper } if bounds(lower, upper) => {
                    match given(lower) {
                        Some(given) => part_lowers.push((lower, given)),
                        None => left_out.push(lower),
                    }
                }
                _ => {}
            }
        }
        Sources {
            equations,
            lowers,
            part_lowers,
            facts: self.fact_heads(root),
            left_out,
        }
    }

    /// The equivalence class takes its constructor and members, and is
    /// solved: where it holds no kind, the join of the constructors
    /// and members its solved lower bounds, part lower bounds and kinds known
    /// in part give, its parts made anew, each part lower bound's
    /// constructor reaching it by `≤ₚ`, with no cast. Where they give
    /// none it is left unsolved, and where they have no join it is refused.
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
            .chain(&sources.part_lowers)
            .map(|(_, given)| given.clone())
            .chain(sources.facts.iter().cloned())
            .chain(self.range_head(root))
            .collect();
        let Some(first) = given.first().cloned() else {
            return with_sources(Took::Nothing);
        };
        let joined = given
            .iter()
            .skip(1)
            .try_fold(first, |joined, source| self.head_join(&joined, source))
            .filter(|joined| {
                sources.part_lowers.iter().all(|(_, given)| {
                    given.constructor == Constructor::Never
                        || given.constructor == joined.constructor
                })
            });
        let joined = joined.and_then(|joined| self.without_two_identities(&sources, joined));
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
        if let Some(conflicted) = &unified.conflict {
            self.conflicted_have_no_value(conflicted);
        }
        self.classes.mark_solved(root);
        with_sources(Took::Joined {
            head: joined,
            unified,
            refused,
        })
    }

    /// The equivalence class's sources have no join.
    fn no_join(&mut self, root: ClassId, refusal: Refusal) -> Took {
        self.refuse_value(root, &refusal);
        Took::NoJoin(refusal)
    }

    /// The equivalence class has no value, and is never solved again.
    fn refuse_value(&mut self, root: ClassId, refusal: &Refusal) {
        let members = self.classes.held(root).members.clone();
        self.no_value.extend(members);
        self.refusals.insert(refusal.clone());
    }

    /// Gives the class a lower bound or a part lower bound at `i` in the log
    /// bounds the range of its lower side, where that holds no kind and a
    /// range a value reaching the class by `≤` stays within
    /// (`Range::upward`), unless the class's range is within it already or
    /// either side has no value. Returns the relation recorded, where one is.
    pub(crate) fn range_up(&mut self, i: usize) -> Option<Recording> {
        let (Relation::LowerBound { lower, upper } | Relation::PartLowerBound { lower, upper }) =
            self.log[i].relation
        else {
            return None;
        };
        if self.has_no_value(lower) || self.has_no_value(upper) {
            return None;
        }
        let held = self.classes.held(self.root(lower));
        if held.kind.is_some() {
            return None;
        }
        let range = held.range.as_ref()?.upward()?;
        let own = &self.classes.held(self.root(upper)).range;
        if own.as_ref().is_some_and(|own| own.within(&range)) {
            return None;
        }
        let mut recorded = Recording::default();
        let relation = Relation::Range(upper, range);
        recorded.push(self.record(relation, Recorder::Split(lower)));
        Some(recorded)
    }

    /// Merges each cycle of lower bounds and part lower bounds alone into one
    /// equivalence class. Returns the equations recorded.
    pub(crate) fn merge_cycles(&mut self) -> Recording {
        let mut recorded = Recording::default();
        let mut edges: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        for recorded in &self.log {
            if let Relation::LowerBound { lower, upper }
            | Relation::PartLowerBound { lower, upper } = recorded.relation
            {
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
    /// lower bounds, part lower bounds and parts.
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
