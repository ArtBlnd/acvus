//! The judgment: what the record decides for an equivalence class or a
//! disjunctive constraint.
//!
//! The module is five layers and one orchestrator. Each layer is a pure
//! function: it takes values read from the record and returns values, and it
//! imports no `State` and no other layer.
//!
//! - [`lower_bounds`]: the join of the heads of the lower bounds that enter a
//!   class (`join`), or `NoJoin`, by the language's own orderings alone.
//! - [`requirements`]: the structure constraints on a class checked against a
//!   head its lower bounds or its occurrences give (`check`), with the members
//!   they name added, or a `Conflict`; and its domains read as one
//!   (`domain_of`). A requirement never gives a class a head.
//! - [`occurrences`]: the one head the decided classes a class passes into
//!   give it (`context`), or `Contexts`.
//! - [`disjunction`]: the alternative a disjunctive constraint takes from
//!   the constructors of the classes it reads (`lend`, `project`, `bind`),
//!   waiting where one is not known.
//! - [`fallback`]: what each fallback candidate takes at the fixpoint
//!   (`fall_back`, rule 4), a literal's default before any `!` (D2).
//!
//! [`view`] reads the record into values (`Bounds`) and knows none of the
//! layers. The orchestrator, `State::decided`, takes the layers in this order:
//!
//! - where something may still enter the class (`lower_pending`), it waits;
//! - where something enters it, it takes
//!   `check(join(lowers, part_lowers)?, requirements)`;
//! - where something may still be passed into (`upper_pending`), it waits;
//! - where it passes into something, it takes
//!   `check(context(occurrences)?, requirements)`;
//! - otherwise nothing decides it (rule 4), and at the fixpoint fallback
//!   reads the record at the class itself (`fallback_candidate`, D2).
//!
//! A class a waiting fact holds of takes no rule, and no reading leaves an
//! undecided disjunctive constraint out to give it one (D6).

pub mod disjunction;
pub mod fallback;
pub mod lower_bounds;
pub mod occurrences;
pub mod requirements;
pub mod view;

use std::collections::BTreeSet;
use std::rc::Rc;

use crate::check::Cause;
use crate::class::{ClassId, ClassName, Conflict, StructureConstraint, StructureSource};
use crate::domain::Domain;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::trace::{Report, Trace};
use crate::ty::{Constructor, Member, Mutability, Name, PartKey, Term, Ty};

use self::fallback::{Candidate, Fallen};
use self::lower_bounds::NoJoin;
use self::occurrences::Contexts;

/// A ready decision step that the solver may take.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Judgment {
    /// An equivalence class takes the head, or the refusal, its record gives.
    Join(ClassId),
    /// A disjunctive constraint takes the single alternative its record gives.
    Decide(Name),
}

/// A head: a class's top-level constructor and its open members (fields or
/// enum variants). A reference's head is `Ref(m)` alone: its target is a
/// part, never part of the head.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Head {
    pub constructor: Constructor,
    pub members: BTreeSet<Member>,
}

impl Head {
    pub fn owned(constructor: Constructor, members: BTreeSet<Member>) -> Self {
        Self {
            constructor,
            members,
        }
    }

    /// The head of a reference of `mutability`: `Ref(mutability)`, with no
    /// members.
    pub fn reference(mutability: Mutability) -> Self {
        Self {
            constructor: Constructor::Ref(mutability),
            members: BTreeSet::new(),
        }
    }

    pub fn constructor(&self) -> Constructor {
        self.constructor.clone()
    }
}

/// What enters an equivalence class: its decided lower bounds and part lower
/// bounds, each with the head it gives. A requirement is no source.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Sources {
    /// Decided lower bounds (`L`).
    pub lowers: Vec<(ClassId, Head)>,
    /// Decided part lower bounds (`≤ₚ`, which take no cast).
    pub part_lowers: Vec<(ClassId, Head)>,
}

impl Sources {
    fn of(bounds: &Bounds) -> Self {
        Self {
            lowers: bounds.lowers.clone(),
            part_lowers: bounds.part_lowers.clone(),
        }
    }

    /// Whether no lower bound enters the class.
    pub fn is_empty(&self) -> bool {
        self.lowers.is_empty() && self.part_lowers.is_empty()
    }
}

/// The heads that `given` gives, in order.
fn heads(given: &[(ClassId, Head)]) -> Vec<Head> {
    given.iter().map(|(_, head)| head.clone()).collect()
}

/// What an equivalence class took at its `Judgment::Join` step.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Took {
    /// The head its lower bounds join at, checked against its requirements.
    Joined { head: Head, recorded: Recording },
    /// The head the classes it passes into give it, where nothing enters it,
    /// checked against its requirements.
    Contextual { head: Head, recorded: Recording },
    /// Refused: its lower bounds have no join.
    NoJoin,
    /// Refused: the classes it passes into give two heads.
    Contexts,
    /// Refused: the head its lower bounds or occurrences give does not satisfy
    /// one of its requirements.
    Conflict,
    /// Refused: it reaches a part of its own along lower bounds and parts.
    Infinite,
}

/// An equivalence class fallback gave something at the fixpoint: a default
/// recorded as its structure, or a refusal.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FellBack {
    pub class: ClassId,
    pub fallen: Fallen,
    pub recorded: Recording,
}

/// What the orchestrator can do for an unsolved equivalence class now: the
/// rule of the judgment that applies to it (rules 1 to 3, or fallback's rule
/// 4), or that a waiting fact holds of it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Ready {
    /// Rules 1 and 2: something enters the class and nothing more may; its
    /// lower bounds and its requirements decide it.
    Join {
        sources: Sources,
        requirements: Vec<StructureConstraint>,
    },
    /// Rule 3: nothing enters the class, nothing may, and nothing more may
    /// be passed into; its occurrences and its requirements decide it.
    Contextual {
        occurrences: Vec<Head>,
        requirements: Vec<StructureConstraint>,
    },
    /// A waiting fact holds: something may still enter the class
    /// (`lower_pending`), or, where nothing enters it, be passed into
    /// (`upper_pending`). No rule runs on it.
    Waiting,
    /// A lower bound was refused and nothing else enters or takes the class.
    Refused,
    /// Rule 4: nothing decides the class: no lower bound gives it a head, it
    /// passes into nothing, and nothing more may come to it. Fallback reads
    /// the record at the class itself (`State::own_held`, D2).
    Rest,
}

/// What the orchestrator decides from a class's `Bounds`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Decided {
    /// Something may still enter the class (`lower_pending`), or, where
    /// nothing enters it, be passed into (`upper_pending`).
    Wait,
    /// The head its lower bounds join at, checked against its requirements.
    Join(Head),
    /// Its lower bounds have no join.
    NoJoin,
    /// The head its occurrences give, checked against its requirements.
    Contextual(Head),
    /// Its occurrences give two heads.
    Contexts,
    /// The head its lower bounds or occurrences give does not satisfy one of
    /// its requirements.
    Conflict(Conflict),
    /// Neither lower bounds nor occurrences give a head.
    Nothing,
}

impl Decided {
    /// The head decided, where `Join` or `Contextual` gives one.
    pub(crate) fn head(self) -> Option<Head> {
        match self {
            Decided::Join(head) | Decided::Contextual(head) => Some(head),
            Decided::Wait
            | Decided::NoJoin
            | Decided::Contexts
            | Decided::Conflict(_)
            | Decided::Nothing => None,
        }
    }
}

/// What the record holds for an equivalence class, read over the component
/// of the undecided classes it waits for (`view`).
#[derive(Clone, Default, Debug)]
pub(crate) struct Bounds {
    /// Lower bounds: the decided lower bounds and the head each gives.
    pub(crate) lowers: Vec<(ClassId, Head)>,
    /// Lower bounds: the decided part lower bounds (`≤ₚ`) and the head each
    /// gives.
    pub(crate) part_lowers: Vec<(ClassId, Head)>,
    /// Requirements: the structure constraints on the class.
    pub(crate) constraints: Vec<StructureConstraint>,
    /// Occurrences: the heads of the decided classes the class passes into.
    pub(crate) uppers: Vec<Head>,
    /// Requirements: the domains on the class and on the undecided classes
    /// below it that nothing enters, a generalized one read against its use's
    /// choice now; what a reading of the class gives where nothing gives it a
    /// head, and nothing fallback reads (D2).
    pub(crate) domains: Vec<Domain>,
    /// `may_enter`: an undecided disjunctive constraint or decomposition, or
    /// a use whose candidates are not yet one, may still add a lower bound.
    pub(crate) lower_pending: bool,
    /// `may_take`: an undecided disjunctive constraint or decomposition may
    /// still add an occurrence.
    pub(crate) upper_pending: bool,
    /// Whether a refused lower bound reaches the class, and nothing else
    /// enters it or is passed into.
    pub(crate) refused: bool,
}

impl State {
    /// The `Bounds` of `root` without the relations the disjunctive
    /// constraints in `left_out` recorded, the view reading the head an
    /// undecided class would take by `decided`.
    pub(crate) fn bounds(&self, root: ClassId, left_out: &BTreeSet<Name>) -> Bounds {
        self.bounds_with(root, left_out, &|bounds: &Bounds| self.decided(bounds))
    }

    /// The orchestrator: what `bounds` decide, taking the layers in order.
    ///
    /// - `lower_pending`: `Wait`.
    /// - Lower bounds or part lower bounds: `check(join(..)?, requirements)`,
    ///   which is `Join`, `NoJoin` or `Conflict`.
    /// - `upper_pending`: `Wait`.
    /// - Occurrences: `check(context(..)?, requirements)`, which is
    ///   `Contextual`, `Contexts` or `Conflict`.
    /// - Otherwise `Nothing`: the domains go to fallback.
    ///
    /// It reads `bounds` and the declarations alone, and records nothing.
    pub(crate) fn decided(&self, bounds: &Bounds) -> Decided {
        if bounds.lower_pending {
            return Decided::Wait;
        }
        let sources = Sources::of(bounds);
        if !sources.is_empty() {
            let lowers = heads(&sources.lowers);
            let part_lowers = heads(&sources.part_lowers);
            return match lower_bounds::join(&lowers, &part_lowers) {
                Err(NoJoin) => Decided::NoJoin,
                Ok(head) => match requirements::check(head, &bounds.constraints) {
                    Ok(head) => Decided::Join(head),
                    Err(conflict) => Decided::Conflict(conflict),
                },
            };
        }
        if bounds.upper_pending {
            return Decided::Wait;
        }
        if bounds.uppers.is_empty() {
            return Decided::Nothing;
        }
        match occurrences::context(&bounds.uppers) {
            Err(Contexts) => Decided::Contexts,
            Ok(head) => match requirements::check(head, &bounds.constraints) {
                Ok(head) => Decided::Contextual(head),
                Err(conflict) => Decided::Conflict(conflict),
            },
        }
    }

    /// What the orchestrator can do for an unsolved class `root` now, reading
    /// the whole record: a class whose refused lower bound alone reaches it
    /// is `Refused`; otherwise the orchestrator's `Wait` (a waiting fact) is
    /// `Waiting` and its `Nothing` is `Rest` (rule 4), and any other decision
    /// is taken by the lower bounds (`Join`, rules 1 and 2) where something
    /// enters the class, and by the occurrences (`Contextual`, rule 3) where
    /// nothing does. It leaves out no undecided disjunctive constraint, and
    /// records nothing.
    pub(crate) fn ready(&self, root: ClassId) -> Ready {
        let root = self.root(root);
        let bounds = self.bounds(root, &BTreeSet::new());
        if bounds.refused {
            return Ready::Refused;
        }
        match self.decided(&bounds) {
            Decided::Wait => Ready::Waiting,
            Decided::Nothing => Ready::Rest,
            Decided::Join(_)
            | Decided::NoJoin
            | Decided::Contextual(_)
            | Decided::Contexts
            | Decided::Conflict(_) => {
                let sources = Sources::of(&bounds);
                let requirements = bounds.constraints;
                if sources.is_empty() {
                    Ready::Contextual {
                        occurrences: bounds.uppers,
                        requirements,
                    }
                } else {
                    Ready::Join {
                        sources,
                        requirements,
                    }
                }
            }
        }
    }

    /// The judgments the record allows now: a `Judgment::Join` for each
    /// unsolved class its lower bounds or occurrences decide (rules 1 to 3),
    /// and a `Judgment::Decide` for each undecided disjunctive constraint the
    /// record gives an alternative. A class a waiting fact holds of is in
    /// none (D6): where nothing is ready, the fixpoint is broken by
    /// fallback's defaults alone (D2). It records nothing.
    pub fn judgments(&self) -> Vec<Judgment> {
        let mut judgments = Vec::new();
        for root in self.unsolved() {
            match self.ready(root) {
                Ready::Join { .. } | Ready::Contextual { .. } => {
                    judgments.push(Judgment::Join(root))
                }
                Ready::Waiting | Ready::Refused | Ready::Rest => {}
            }
        }
        for name in self.undecided() {
            if self.reading(&name, &BTreeSet::new()).decision.is_some() {
                judgments.push(Judgment::Decide(name));
            }
        }
        judgments
    }

    /// Takes one ready `judgment`, one `judgments` gives, and records what it
    /// decides. It panics on a judgment the record does not allow now.
    pub fn judge(&mut self, judgment: &Judgment) -> Trace {
        let mut trace = Trace::default();
        match judgment {
            Judgment::Join(class) => {
                let root = self.root(*class);
                let took = match self.ready(root) {
                    Ready::Join {
                        sources,
                        requirements,
                    } => self.take_join(root, &sources, &requirements),
                    Ready::Contextual {
                        occurrences,
                        requirements,
                    } => self.take_contextual(root, &occurrences, &requirements),
                    Ready::Waiting | Ready::Refused | Ready::Rest => {
                        panic!("the record gives {} no type now", self.class_name(root))
                    }
                };
                trace.push(Report::Joined { class: root, took });
            }
            Judgment::Decide(name) => {
                let decision = self
                    .decision(name, &BTreeSet::new())
                    .expect("the record decides a disjunctive constraint it allows to decide");
                self.set_decision(name, decision.clone());
                let recorded = self.record_decision(name);
                trace.push(Report::Decided {
                    disjunction: name.clone(),
                    decision,
                    recorded,
                });
            }
        }
        trace
    }

    /// At the `entry` group's fixpoint, records what `fall_back` gives the
    /// fallback candidates (`fallback_candidate`, D2): a default as its
    /// structure, its parts made anew or kept (`skeleton`), or a refusal:
    /// `conflict` where its domains admit no type in common or several, and
    /// `no_value` where a requirement is on a class that takes no `!`.
    /// The loop runs again after it (`State::run`), so a literal's default
    /// decides the classes it enters by rules 1 and 2 before any class takes
    /// `!`.
    pub(crate) fn take_defaults(&mut self) -> Vec<FellBack> {
        let candidates: Vec<Candidate> = self
            .unsolved()
            .into_iter()
            .filter_map(|root| self.fallback_candidate(root))
            .collect();
        let mut fell = Vec::new();
        for (root, fallen) in fallback::fall_back(&candidates) {
            let mut recorded = Recording::default();
            match &fallen {
                Fallen::Default(constructor) => {
                    let owner = self.class_name(root);
                    let structure = self.skeleton(root, constructor, &BTreeSet::new(), &owner);
                    let relation = Relation::Structure(root, structure, StructureSource::Solved);
                    recorded.push(self.record(relation, Recorder::Fallback));
                }
                Fallen::NoType(domain) => {
                    let cause = self.conflict_cause(Conflict::Domain(domain.clone(), None));
                    self.classes.refuse(root, cause);
                }
                Fallen::NoValue => self.classes.refuse(root, Cause::NoValue),
            }
            fell.push(FellBack {
                class: root,
                fallen,
                recorded,
            });
        }
        fell
    }

    /// The fallback candidate an unsolved class `root` is at the fixpoint
    /// (D2), read off the record at the class itself (`own_held`):
    ///
    /// - A class with a lower bound, recorded or pending, is none: rules 1
    ///   and 2 decide it once its lower bounds are decided.
    /// - A class with no lower bound that holds a literal's domain as its own
    ///   is one with the meet of its own domains, whether or not an
    ///   occurrence of it may still change: the literal's default replaces
    ///   rule 3 there.
    /// - A class with no lower bound and no domain, the operand of no
    ///   undecided projection (`requirement_pending`), is one with no domain
    ///   where no requirement is on it, whether or not an occurrence of it
    ///   may still be recorded (`taking`): it takes `!`, which promises only
    ///   that no value exists, and no occurrence makes one. Where a
    ///   requirement is on it, it is one only where no occurrence of it may
    ///   still be recorded, and takes `no_value`. An occurrence recorded
    ///   whose class has no type yet is no waiting fact, and keeps it out of
    ///   neither.
    /// - The operand of an undecided projection is none: the projection's
    ///   decision may still record a requirement on it, so it takes no `!`,
    ///   and, staying without a value at the fixpoint, it is `no_value` in
    ///   the outcome. A lend's place is no such class: a lend records a
    ///   requirement on its place only where the place is a reference.
    /// - Any other class, one with a domain that is no literal's among them,
    ///   is none.
    ///
    /// It reads no class's lower bounds' domains, records nothing and takes
    /// no rule of the judgment.
    fn fallback_candidate(&self, root: ClassId) -> Option<Candidate> {
        let own = self.own_held(root);
        if own.entered {
            return None;
        }
        if own.literal {
            return Some(Candidate {
                class: root,
                domain: requirements::domain_of(own.domains),
                required: own.required,
            });
        }
        let waiting = own.requirement_pending || (own.required && own.taking);
        (own.domains.is_empty() && !waiting).then_some(Candidate {
            class: root,
            domain: None,
            required: own.required,
        })
    }

    /// The head `bounds` decide, where the orchestrator gives one.
    pub(crate) fn value_head(&self, bounds: &Bounds) -> Option<Head> {
        self.decided(bounds).head()
    }

    /// Refuses `root` with the conflict between its head and a requirement.
    fn refuse_conflict(&mut self, root: ClassId, conflict: Conflict) -> Took {
        let cause = self.conflict_cause(conflict);
        self.classes.refuse(root, cause);
        Took::Conflict
    }

    /// Rules 1 and 2 taken: the head the lower bounds in `sources` join at
    /// (`lower_bounds::join`), checked against `requirements`, recorded as a
    /// skeleton structure whose parts are classes made anew or kept
    /// (`skeleton`), into which the decomposition of each lower bound passes
    /// its parts. Refuses `root`
    /// `Infinite` where it reaches a part of its own, `NoJoin` where its lower
    /// bounds have no join or carry two distinct identities
    /// (`lower_bounds::identities_agree`, D3), and with a `Conflict` where
    /// the joined head does not satisfy a requirement. It takes no cast,
    /// erases no identity, and gives a reference's target no structure of
    /// its own.
    pub(crate) fn take_join(
        &mut self,
        root: ClassId,
        sources: &Sources,
        requirements: &[StructureConstraint],
    ) -> Took {
        let root = self.root(root);
        let cycle = self.own_part_cycle(root);
        if !cycle.is_empty() {
            for class in cycle {
                self.classes.refuse(class, Cause::Infinite);
            }
            return Took::Infinite;
        }
        let lowers = heads(&sources.lowers);
        let part_lowers = heads(&sources.part_lowers);
        let identities: BTreeSet<Ty> = sources
            .lowers
            .iter()
            .chain(&sources.part_lowers)
            .filter_map(|(class, _)| self.value(*class))
            .filter_map(|value| crate::join::identity(&value).cloned())
            .collect();
        let joined = lower_bounds::join(&lowers, &part_lowers)
            .ok()
            .filter(|_| lower_bounds::identities_agree(&identities));
        let Some(joined) = joined else {
            self.classes.refuse(root, Cause::NoJoin);
            return Took::NoJoin;
        };
        let joined = match requirements::check(joined, requirements) {
            Ok(joined) => joined,
            Err(conflict) => return self.refuse_conflict(root, conflict),
        };
        let owner = self.class_name(root);
        let recorded = self.take_skeleton(root, &joined, &owner, &Recorder::Join(root));
        Took::Joined {
            head: joined,
            recorded,
        }
    }

    /// Takes the one head `occurrences` give, checked against `requirements`,
    /// and records its skeleton structure; refuses `root` `Contexts` where the
    /// occurrences give two heads, and with a `Conflict` where the head does
    /// not satisfy a requirement.
    pub(crate) fn take_contextual(
        &mut self,
        root: ClassId,
        occurrences: &[Head],
        requirements: &[StructureConstraint],
    ) -> Took {
        let root = self.root(root);
        let Ok(head) = occurrences::context(occurrences) else {
            self.classes.refuse(root, Cause::Contexts);
            return Took::Contexts;
        };
        let head = match requirements::check(head, requirements) {
            Ok(head) => head,
            Err(conflict) => return self.refuse_conflict(root, conflict),
        };
        let owner = self.class_name(root);
        let recorded = self.take_skeleton(root, &head, &owner, &Recorder::Join(root));
        Took::Contextual { head, recorded }
    }

    /// Records at `root` the structure `head` gives (rules 2 and 3): its
    /// constructor and members, each part the part a structure a union
    /// dropped from `root` has at its key, or a class made anew and named
    /// after `owner` (`skeleton`), as `Relation::Structure(root, structure,
    /// StructureSource::Solved)` recorded by `recorder`. A reference's target
    /// is one such part, and is given no structure here. Returns what the
    /// relation recorded.
    pub(crate) fn take_skeleton(
        &mut self,
        root: ClassId,
        head: &Head,
        owner: &ClassName,
        recorder: &Recorder,
    ) -> Recording {
        let mut recorded = Recording::default();
        let structure = self.skeleton(root, &head.constructor, &head.members, owner);
        let relation = Relation::Structure(root, structure, StructureSource::Solved);
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }

    /// The structure `root` takes for `constructor` and `members`, as a
    /// `Term<ClassId>`: each part the one a structure a union dropped
    /// from `root` has at its key (`Classes::kept_part`), so that what
    /// decompositions recorded against that part stays on it, and otherwise
    /// a class made anew named after `owner`. It records nothing: the
    /// caller records the structure, which unifies every dropped part of
    /// its key with it.
    pub(crate) fn skeleton(
        &mut self,
        root: ClassId,
        constructor: &Constructor,
        members: &BTreeSet<Member>,
        owner: &ClassName,
    ) -> Term<ClassId> {
        let owner = Rc::new(owner.clone());
        let decls = Rc::clone(&self.decls);
        let mut part = |key: PartKey| match self.classes.kept_part(root, &key) {
            Some(kept) => kept,
            None => self.fresh(ClassName::Part(owner.clone(), key)),
        };
        let named = |members: &BTreeSet<Member>| -> Vec<Name> {
            members
                .iter()
                .map(|member| match member {
                    Member::Field(name) | Member::Variant(name) => name.clone(),
                })
                .collect()
        };
        match constructor {
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
                named(members)
                    .into_iter()
                    .map(|field| (field.clone(), part(PartKey::Field(field))))
                    .collect(),
            ),
            Constructor::Enum(name) => Term::Enum(
                name.clone(),
                named(members)
                    .into_iter()
                    .map(|variant| (variant.clone(), part(PartKey::Payload(variant))))
                    .collect(),
            ),
            Constructor::ErrorOf => Term::ErrorOf(part(PartKey::Wrapped)),
            Constructor::Struct(module) => {
                let fields = decls
                    .types()
                    .struct_decl(module)
                    .expect("a structure's struct is declared")
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
                    .expect("a structure's extern-defined type is declared");
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

    /// Reads the closed type `Ty` of equivalence class `id`, if all its parts
    /// are decided and finite.
    pub fn value(&self, id: ClassId) -> Option<Ty> {
        self.value_in(id, &mut BTreeSet::new())
    }

    fn value_in(&self, id: ClassId, visiting: &mut BTreeSet<ClassId>) -> Option<Ty> {
        let root = self.root(id);
        if self.has_no_value(root) || !visiting.insert(root) {
            return None;
        }
        let (structure, _) = self.classes.held(root).structure.clone()?;
        let term = structure
            .try_map(|_, part| self.value_in(part, visiting).ok_or(()))
            .ok()?;
        visiting.remove(&root);
        Ty::new(term).ok()
    }
}
