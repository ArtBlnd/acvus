//! The judgment: the one place where solving decides, reading the record. A
//! join position takes the type its record gives, and a disjunctive
//! constraint takes the alternative its record gives. I\* asks the same
//! judgment again, with relations left out of the record.
//!
//! A judgment reads every occurrence of its class in the record, in one
//! order: the structure the class holds and its equations, then the lower
//! bounds that have a node, joined at the node, then, where no lower bound
//! gives a node, the decided classes it passes into, which fill it, and
//! otherwise the domain alone, which fallback reads. A bound that is a
//! class not yet decided gives its own bounds in its place, so that a class
//! and the undecided classes it is bound by are decided together; a class
//! waits only for a disjunctive constraint that may still record into it or
//! into one of those. A disjunctive constraint decides from what the record
//! gives now, and the loop runs again.

use std::collections::BTreeSet;

use crate::class::{ClassId, Side, StructureConstraint};
use crate::coerce::{Rank, coerce_read};
use crate::decl::DeclId;
use crate::disjunction::{Decision, Disjunction, LendTarget};
use crate::domain::Domain;
use crate::program::Projection;
use crate::read::ReadValue;
use crate::relation::{Item, Recorder, Relation, Role};
use crate::sig::Signature;
use crate::solve::State;
use crate::trace::{Report, Trace};
use crate::ty::{Constructor, HasTerm, Mutability, Name, Term};
use crate::value::{Head, Sources};

/// What the record allows to be decided.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Judgment {
    /// The join position takes the type its record gives.
    Join(ClassId),
    /// The disjunctive constraint takes the alternative its record gives.
    Decide(Name),
}

/// What the record gives an equivalence class at its judgment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Verdict {
    /// Its lower bounds give nodes: it takes their join.
    Join(Sources),
    /// No lower bound gives a node, and the classes it passes into give one
    /// node: it takes that.
    Fill(Vec<Head>),
    /// Something that bears on it is not yet decided: a disjunctive
    /// constraint that may record into it, or into a class whose bounds it
    /// takes.
    Waiting,
    /// Nothing decides it: the domain the record gives it, which fallback
    /// reads.
    Rest(Option<Domain>),
}

/// What a disjunctive constraint reads and becomes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Reading {
    /// The alternative the record gives it, none where it decides nothing.
    pub(crate) decision: Option<Decision>,
}

/// Which way bounds are gathered: what enters a class, or what it passes
/// into.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Direction {
    Lower,
    Upper,
}

/// The bounds of an equivalence class, as its judgment reads them from the
/// record: a bound that is an equivalence class not yet decided gives its
/// own bounds in its place, so that a class and the undecided classes it is
/// bound by are decided together.
#[derive(Clone, Default, Debug)]
pub(crate) struct Bounds {
    /// Each decided lower bound, with the node it gives.
    pub(crate) lowers: Vec<(ClassId, Head)>,
    /// Each decided part lower bound, with the node it gives, which reaches
    /// by `≤ₚ`.
    pub(crate) part_lowers: Vec<(ClassId, Head)>,
    /// The structure constraints on the class and on the undecided classes
    /// it takes lower bounds from.
    pub(crate) constraints: Vec<StructureConstraint>,
    /// The node each decided class it passes into gives.
    pub(crate) uppers: Vec<Head>,
    /// The domains on the class and on the undecided classes it takes lower
    /// bounds from.
    pub(crate) domains: Vec<Domain>,
    /// Whether an undecided disjunctive constraint may still record a lower
    /// bound or a structure constraint into the class or one it takes lower
    /// bounds from.
    pub(crate) lower_pending: bool,
    /// Whether an undecided disjunctive constraint may still take what the
    /// class, or one it takes upper bounds from, passes into it.
    pub(crate) upper_pending: bool,
}

impl State {
    /// The judgments the record allows now: each unsolved class whose bounds
    /// give it a type, and each undecided disjunctive constraint whose
    /// record gives it an alternative.
    pub fn judgments(&self) -> Vec<Judgment> {
        let mut judgments = Vec::new();
        for root in self.unsolved() {
            match self.verdict(root) {
                Verdict::Join(_) | Verdict::Fill(_) => judgments.push(Judgment::Join(root)),
                Verdict::Rest(Some(domain)) if self.bounds_anew(root, &domain) => {
                    judgments.push(Judgment::Join(root));
                }
                Verdict::Waiting | Verdict::Rest(_) => {}
            }
        }
        for name in self.undecided() {
            if self.reading(&name, &BTreeSet::new()).decision.is_some() {
                judgments.push(Judgment::Decide(name));
            }
        }
        judgments
    }

    /// Takes the judgment, recording what it decides.
    pub fn judge(&mut self, judgment: &Judgment) -> Trace {
        let mut trace = Trace::default();
        match judgment {
            Judgment::Join(class) => {
                let root = self.root(*class);
                let took = match self.verdict(root) {
                    Verdict::Join(sources) => self.take_join(root, &sources),
                    Verdict::Fill(uppers) => self.take_fill(root, &uppers),
                    Verdict::Rest(Some(domain)) if self.bounds_anew(root, &domain) => {
                        self.take_domain(root, domain)
                    }
                    Verdict::Waiting | Verdict::Rest(_) => {
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

    /// What the disjunctive constraint `name` decides reading the record
    /// without the relations the disjunctive constraints `left_out` recorded;
    /// none where that decides nothing.
    pub(crate) fn decision(&self, name: &Name, left_out: &BTreeSet<Name>) -> Option<Decision> {
        self.reading(name, left_out).decision
    }

    /// What the record gives the unsolved class `root` now, by its bounds.
    pub(crate) fn verdict(&self, root: ClassId) -> Verdict {
        let root = self.root(root);
        let bounds = self.bounds(root, &BTreeSet::new());
        if bounds.lower_pending {
            return Verdict::Waiting;
        }
        let sources = Sources {
            structure_constraints: self.constraint_heads(
                &bounds.constraints,
                bounds.lowers.is_empty() && bounds.part_lowers.is_empty(),
            ),
            lowers: bounds.lowers,
            part_lowers: bounds.part_lowers,
        };
        if !sources.is_empty() {
            return Verdict::Join(sources);
        }
        if bounds.upper_pending {
            return Verdict::Waiting;
        }
        if !bounds.uppers.is_empty() {
            return Verdict::Fill(bounds.uppers);
        }
        let domain = bounds
            .domains
            .into_iter()
            .reduce(|met, domain| met.meet(&domain).unwrap_or(Domain::Only(BTreeSet::new())));
        Verdict::Rest(domain)
    }

    /// The bounds of the class, read without the relations the disjunctive
    /// constraints `left_out` recorded.
    pub(crate) fn bounds(&self, root: ClassId, left_out: &BTreeSet<Name>) -> Bounds {
        let mut bounds = Bounds::default();
        let mut visiting = BTreeSet::new();
        self.gather_lowers(root, left_out, false, &mut bounds, &mut visiting);
        self.gather_uppers(root, left_out, &mut bounds, &mut visiting);
        bounds
    }

    /// Gathers into `bounds` what enters the class: the node of each decided
    /// lower bound, the bounds of each undecided one in its place, the
    /// structure constraints and domains on it, and whether an undecided
    /// disjunctive constraint may still record into it.
    fn gather_lowers(
        &self,
        root: ClassId,
        left_out: &BTreeSet<Name>,
        part: bool,
        bounds: &mut Bounds,
        visiting: &mut BTreeSet<(ClassId, Direction)>,
    ) {
        let root = self.root(root);
        if !visiting.insert((root, Direction::Lower)) {
            return;
        }
        let kept = |at: usize| !left_out_by(&self.classes.items()[at].recorder, left_out);
        for occurrence in self.classes.occurrences(root) {
            if !kept(occurrence.at) {
                continue;
            }
            match &self.classes.items()[occurrence.at].item {
                Item::Relation(relation) => match relation {
                    Relation::LowerBound { lower, .. } | Relation::PartLowerBound { lower, .. }
                        if occurrence.side() == Some(Side::Upper) && self.root(*lower) != root =>
                    {
                        let lower = self.root(*lower);
                        // A lower bound with no value gives the class none:
                        // nothing decides it.
                        if self.has_no_value(lower) {
                            bounds.lower_pending = true;
                            continue;
                        }
                        let part = part || matches!(relation, Relation::PartLowerBound { .. });
                        match self.node_of(lower, left_out, visiting) {
                            Some(head) if part => bounds.part_lowers.push((lower, head)),
                            Some(head) => bounds.lowers.push((lower, head)),
                            None if self.is_solved(lower) => {
                                // A reference whose target has no node yet.
                                bounds.lower_pending |= self
                                    .target_bounds(lower, left_out, visiting)
                                    .lower_pending;
                            }
                            // An undecided lower bound gives the type its own
                            // judgment would give it: its bounds' one type,
                            // or its domain.
                            None => {
                                let mut own = Bounds::default();
                                self.gather_lowers(lower, left_out, false, &mut own, visiting);
                                self.gather_uppers(lower, left_out, &mut own, visiting);
                                let nothing_enters = own.lowers.is_empty()
                                    && own.part_lowers.is_empty()
                                    && own.constraints.is_empty();
                                bounds.lower_pending |=
                                    own.lower_pending || (nothing_enters && own.upper_pending);
                                match self.value_head(&own) {
                                    Some(head) if part => bounds.part_lowers.push((lower, head)),
                                    Some(head) => bounds.lowers.push((lower, head)),
                                    None => bounds.domains.extend(own.domains),
                                }
                            }
                        }
                    }
                    Relation::Domain(_, domain) if occurrence.side() == Some(Side::Own) => {
                        bounds.domains.push(domain.clone());
                    }
                    Relation::StructureConstraint(_, constraint)
                        if occurrence.side() == Some(Side::Own) =>
                    {
                        bounds.constraints.push(constraint.clone());
                    }
                    // A part of a class is recorded into by the decomposition
                    // of what enters that class, once both are solved.
                    Relation::Structure(owner, ..)
                        if occurrence.side() == Some(Side::Part)
                            && self.undecomposed_at(*owner, Side::Upper, left_out) =>
                    {
                        bounds.lower_pending = true;
                    }
                    Relation::StructureConstraint(owner, _)
                        if occurrence.side() == Some(Side::Part)
                            && !self.classes.items()[occurrence.at].decomposed
                            && !self.has_no_value(*owner) =>
                    {
                        bounds.lower_pending = true;
                    }
                    _ => {}
                },
                Item::Disjunction { kind, decision, .. } => {
                    if decision.is_none()
                        && matches!(occurrence.side(), Some(Side::Upper | Side::Own | Side::Part))
                        && !matches!(kind, Disjunction::CallValue { callee, .. }
                            if self.root(*callee) == root)
                    {
                        bounds.lower_pending = true;
                    }
                }
                Item::Instance { .. } | Item::Lambda { .. } => {}
            }
        }
    }

    /// Gathers into `bounds` what the class passes into: the node of each
    /// decided class, the bounds of each undecided one in its place, and
    /// whether an undecided disjunctive constraint may still take what the
    /// class passes into it.
    fn gather_uppers(
        &self,
        root: ClassId,
        left_out: &BTreeSet<Name>,
        bounds: &mut Bounds,
        visiting: &mut BTreeSet<(ClassId, Direction)>,
    ) {
        let root = self.root(root);
        if !visiting.insert((root, Direction::Upper)) {
            return;
        }
        let kept = |at: usize| !left_out_by(&self.classes.items()[at].recorder, left_out);
        for occurrence in self.classes.occurrences(root) {
            if !kept(occurrence.at) {
                continue;
            }
            match &self.classes.items()[occurrence.at].item {
                Item::Relation(relation) => match relation {
                    Relation::LowerBound { upper: into, .. }
                    | Relation::PartLowerBound { upper: into, .. }
                    | Relation::CoercionConstraint { to: into, .. }
                        if occurrence.side() == Some(Side::Lower) && self.root(*into) != root =>
                    {
                        let into = self.root(*into);
                        if self.has_no_value(into) {
                            continue;
                        }
                        match self.node_of(into, left_out, visiting) {
                            Some(head) => bounds.uppers.push(head),
                            None if self.is_solved(into) => bounds.upper_pending = true,
                            // An undecided class passed into gives the type
                            // its own judgment would give it.
                            None => {
                                let mut own = Bounds::default();
                                self.gather_lowers(into, left_out, false, &mut own, visiting);
                                self.gather_uppers(into, left_out, &mut own, visiting);
                                bounds.upper_pending |= own.lower_pending || own.upper_pending;
                                if let Some(head) = self.value_head(&own) {
                                    bounds.uppers.push(head);
                                }
                            }
                        }
                    }
                    // A part of a class is recorded into by the decomposition
                    // of what that class passes into, once both are solved.
                    Relation::Structure(owner, ..)
                        if occurrence.side() == Some(Side::Part)
                            && self.undecomposed_at(*owner, Side::Lower, left_out) =>
                    {
                        bounds.upper_pending = true;
                    }
                    Relation::StructureConstraint(owner, _)
                        if occurrence.side() == Some(Side::Part)
                            && !self.classes.items()[occurrence.at].decomposed
                            && !self.has_no_value(*owner) =>
                    {
                        bounds.upper_pending = true;
                    }
                    _ => {}
                },
                Item::Disjunction { kind, decision, .. } => {
                    if decision.is_none()
                        && occurrence.side() == Some(Side::Lower)
                        && !matches!(kind, Disjunction::Bind { position, .. }
                            if self.root(*position) == root)
                    {
                        bounds.upper_pending = true;
                    }
                }
                Item::Instance { .. } | Item::Lambda { .. } => {}
            }
        }
    }

    /// Whether a relation the class `owner` is the `side` of, with a value
    /// on its other side, is not yet decomposed: its decomposition will
    /// record into the parts of `owner`.
    fn undecomposed_at(&self, owner: ClassId, side: Side, left_out: &BTreeSet<Name>) -> bool {
        let owner = self.root(owner);
        self.classes.occurrences(owner).iter().any(|occurrence| {
            let recorded = &self.classes.items()[occurrence.at];
            if recorded.decomposed
                || occurrence.side() != Some(side)
                || left_out_by(&recorded.recorder, left_out)
            {
                return false;
            }
            let other = match recorded.relation() {
                Some(
                    Relation::LowerBound { lower, upper }
                    | Relation::PartLowerBound { lower, upper }
                    | Relation::CoercionConstraint {
                        from: lower,
                        to: upper,
                    }
                    | Relation::Wrap {
                        from: lower,
                        to: upper,
                    },
                ) => match side {
                    Side::Upper => *lower,
                    _ => *upper,
                },
                _ => return false,
            };
            self.root(other) != owner && !self.has_no_value(other)
        })
    }

    /// The node of a decided class: its own, or, for a reference whose
    /// target is not yet decided, its mutability with the node its target's
    /// bounds give.
    fn node_of(
        &self,
        id: ClassId,
        left_out: &BTreeSet<Name>,
        visiting: &mut BTreeSet<(ClassId, Direction)>,
    ) -> Option<Head> {
        let root = self.root(id);
        if let Some(head) = self.head_of(root) {
            return Some(head);
        }
        let Some((Term::Ref(mutability, _), _)) = &self.classes.held(root).structure else {
            return None;
        };
        let target = self.target_bounds(root, left_out, visiting);
        if target.lower_pending {
            return None;
        }
        let joined = self.join_of(&target)?;
        Some(Head {
            node: crate::decl::Node::Ref(*mutability, joined.constructor()),
            members: BTreeSet::new(),
        })
    }

    /// The bounds of the target of the reference the class holds.
    fn target_bounds(
        &self,
        reference: ClassId,
        left_out: &BTreeSet<Name>,
        visiting: &mut BTreeSet<(ClassId, Direction)>,
    ) -> Bounds {
        let mut bounds = Bounds::default();
        if let Some((Term::Ref(_, target), _)) = &self.classes.held(self.root(reference)).structure
        {
            self.gather_lowers(*target, left_out, false, &mut bounds, visiting);
        }
        bounds
    }

    /// The node `bounds` give a class: that its lower bounds and structure
    /// constraints join at, or, where they give none, the one node of what
    /// it passes into; none where they give none.
    fn value_head(&self, bounds: &Bounds) -> Option<Head> {
        if let Some(head) = self.join_of(bounds) {
            return Some(head);
        }
        let first = bounds.uppers.first()?;
        bounds
            .uppers
            .iter()
            .all(|head| head == first)
            .then(|| first.clone())
    }

    /// The node the lower bounds and structure constraints of `bounds` join
    /// at, none where they give none or have no join.
    fn join_of(&self, bounds: &Bounds) -> Option<Head> {
        let constraints = self.constraint_heads(
            &bounds.constraints,
            bounds.lowers.is_empty() && bounds.part_lowers.is_empty(),
        );
        let given: Vec<&Head> = bounds
            .lowers
            .iter()
            .chain(&bounds.part_lowers)
            .map(|(_, head)| head)
            .chain(constraints.iter())
            .collect();
        let first = (*given.first()?).clone();
        given
            .into_iter()
            .skip(1)
            .try_fold(first, |joined, head| self.head_join(&joined, head))
    }

    /// Whether the domain the class's sources give bounds it more than the
    /// domains on it do, so that a judgment is to record it.
    fn bounds_anew(&self, root: ClassId, domain: &Domain) -> bool {
        match self.classes.domain(root) {
            None => true,
            Some(own) => own.meet(domain).is_some_and(|met| met != own),
        }
    }

    /// The roots of the classes the disjunctive constraint `name` reads: a
    /// use, the arguments where its candidates differ and its result; a lend, its place; a
    /// projection other than a Deref, its operand; a binding, its position;
    /// a call of a function value, its callee and the class called.
    pub(crate) fn reads(&self, name: &Name) -> BTreeSet<ClassId> {
        let (kind, _) = self.disjunction(name);
        let classes: Vec<ClassId> = match kind {
            Disjunction::Use {
                callee,
                args,
                result,
                ..
            } => {
                let candidates: Vec<&Signature> = self
                    .candidates(callee, args.len())
                    .into_iter()
                    .map(|decl| &decl.sig)
                    .collect();
                Self::differing(&candidates, args.len())
                    .into_iter()
                    .filter_map(|i| args[i])
                    .chain(*result)
                    .collect()
            }
            Disjunction::Lend { place, .. } => vec![*place],
            Disjunction::Project {
                projection: Projection::Deref,
                ..
            } => Vec::new(),
            Disjunction::Project { operand, .. } => vec![*operand],
            Disjunction::Bind { position, .. } => vec![*position],
            Disjunction::CallValue { callee, .. } => vec![*callee, self.called(*callee)],
        };
        classes.into_iter().map(|class| self.root(class)).collect()
    }

    /// The roots of the classes a decision still open to the undecided
    /// disjunctive constraint `name` records into: those at whose occurrence
    /// the alternative taken records a relation, its result, its exit, a
    /// lend's result and place, a projection's operand and result, and a
    /// binding's binding and position.
    pub(crate) fn records_into(&self, name: &Name) -> BTreeSet<ClassId> {
        let (kind, _) = self.disjunction(name);
        kind.occurrences()
            .into_iter()
            .filter(|(_, role)| {
                matches!(role.side(), Some(Side::Upper | Side::Own | Side::Part))
                    || matches!(role, Role::Bound)
            })
            .filter(|(class, _)| {
                !matches!(kind, Disjunction::CallValue { callee, .. }
                    if self.root(*callee) == self.root(*class))
            })
            .map(|(class, _)| self.root(class))
            .collect()
    }

    /// The class whose function value a call of `callee` calls: the target of
    /// the reference `callee` holds, as a projection is taken of a reference's
    /// target, and `callee` itself otherwise.
    pub(crate) fn called(&self, callee: ClassId) -> ClassId {
        let root = self.root(callee);
        match &self.classes.held(root).structure {
            Some((Term::Ref(_, target), _)) => self.root(*target),
            _ => root,
        }
    }

    /// What the disjunctive constraint `name` reads and becomes, reading the
    /// record without the relations the disjunctive constraints `left_out`
    /// recorded: what the classes at its positions hold, or their bounds
    /// give them now.
    pub(crate) fn reading(&self, name: &Name, left_out: &BTreeSet<Name>) -> Reading {
        let (kind, _) = self.disjunction(name);
        let mut left_out = left_out.clone();
        left_out.insert(name.clone());
        let read = |class: ClassId| self.read(class, &left_out);
        let decision = match kind {
            Disjunction::Use {
                callee,
                args,
                result,
                ..
            } => {
                let candidates = self.candidates(callee, args.len());
                let ids: Vec<DeclId> = candidates.iter().map(|decl| decl.id.clone()).collect();
                let sigs: Vec<&Signature> = candidates.iter().map(|decl| &decl.sig).collect();
                let values: Vec<(usize, ReadValue)> = Self::differing(&sigs, args.len())
                    .into_iter()
                    .filter_map(|i| Some((i, read(args[i]?))))
                    .collect();
                let result = result.map(read);
                narrow(&self.decls, &ids, &sigs, &values, result.as_ref())
            }
            Disjunction::Lend {
                mutability, place, ..
            } => lend(&read(*place), Some(*mutability)),
            Disjunction::Project {
                projection: Projection::Deref,
                ..
            } => Some(Decision::Applied),
            Disjunction::Project { operand, .. } => match reads_as_reference(&read(*operand)) {
                AsReference::Yes(_) => Some(Decision::AppliedToTarget),
                AsReference::No | AsReference::Undecided => Some(Decision::Applied),
            },
            Disjunction::Bind { path, position, .. } => {
                let mut passed: Option<Option<Mutability>> = None;
                let after_deref = path
                    .iter()
                    .rposition(|step| {
                        matches!(
                            self.disjunction(step).0,
                            Disjunction::Project { projection: Projection::Deref, .. }
                        )
                    })
                    .map_or(0, |deref| deref + 1);
                // Each projection on the path decides whether the binding
                // passes through a reference there: as recorded, or as its
                // reading gives it now; where one gives nothing, the record
                // lacks what the binding reads.
                let mut steps_read = true;
                for step in &path[after_deref..] {
                    let (kind, decided) = self.disjunction(step);
                    let decided = decided
                        .cloned()
                        .or_else(|| self.reading(step, &left_out).decision);
                    match decided {
                        None => steps_read = false,
                        Some(Decision::AppliedToTarget) => {
                            let Disjunction::Project { operand, .. } = kind else {
                                unreachable!("a binding's path is of projections");
                            };
                            let mutability = match read(*operand).decided() {
                                Some(Term::Ref(mutability, _)) => Some(*mutability),
                                _ => None,
                            };
                            let so_far = passed.unwrap_or(Some(Mutability::Exclusive));
                            passed = Some(lesser(so_far, mutability));
                        }
                        Some(_) => {}
                    }
                }
                match passed {
                    _ if !steps_read => None,
                    None => Some(Decision::Read),
                    Some(passed) => lend(&read(*position), passed).map(|decision| match decision {
                        Decision::Lend { target, mutability } => {
                            Decision::BindLend { target, mutability }
                        }
                        other => other,
                    }),
                }
            }
            Disjunction::CallValue { callee, .. } => {
                let lambdas = self.values_of(self.called(*callee));
                (!lambdas.is_empty()).then_some(Decision::Calls(lambdas))
            }
        };
        Reading { decision }
    }

    /// The value the class `id` holds, read without the relations the
    /// disjunctive constraints `left_out` recorded: its structure, each part
    /// read in turn, or what its bounds give it now.
    pub(crate) fn read(&self, id: ClassId, left_out: &BTreeSet<Name>) -> ReadValue {
        self.read_in(id, left_out, &mut BTreeSet::new())
    }

    fn read_in(
        &self,
        id: ClassId,
        left_out: &BTreeSet<Name>,
        visiting: &mut BTreeSet<ClassId>,
    ) -> ReadValue {
        let root = self.root(id);
        if self.has_no_value(root) || !visiting.insert(root) {
            return ReadValue::Undecided;
        }
        let kept = |at: usize| !left_out_by(&self.classes.items()[at].recorder, left_out);
        let occurrences = self.classes.occurrences(root);
        let structure = self
            .classes
            .held(root)
            .structure
            .as_ref()
            .filter(|_| {
                occurrences.iter().any(|occurrence| {
                    occurrence.side() == Some(Side::Own)
                        && kept(occurrence.at)
                        && matches!(
                            self.classes.items()[occurrence.at].relation(),
                            Some(Relation::Structure(..))
                        )
                })
            })
            .map(|(structure, _)| structure.clone());
        let value = match structure {
            Some(structure) => {
                let term = structure.map(|_, part| self.read_in(part, left_out, visiting));
                ReadValue::from_term(term).unwrap_or(ReadValue::Undecided)
            }
            None => {
                let bounds = self.bounds(root, left_out);
                match self.value_head(&bounds) {
                    Some(head) => head_value(head),
                    None => bounds
                        .domains
                        .into_iter()
                        .reduce(|met, domain| {
                            met.meet(&domain).unwrap_or(Domain::Only(BTreeSet::new()))
                        })
                        .map_or(ReadValue::Undecided, ReadValue::Domain),
                }
            }
        };
        visiting.remove(&root);
        value
    }
}

/// A head read as a value: its node, each part undecided; a reference's
/// target is its constructor, each part of that undecided too.
fn head_value(head: Head) -> ReadValue {
    fn of(constructor: &Constructor, members: &BTreeSet<crate::ty::Member>) -> ReadValue {
        let named = |members: &BTreeSet<crate::ty::Member>| -> Vec<Name> {
            members
                .iter()
                .map(|member| match member {
                    crate::ty::Member::Field(name) | crate::ty::Member::Variant(name) => {
                        name.clone()
                    }
                })
                .collect()
        };
        let part = || ReadValue::Undecided;
        let term: Term<ReadValue> = match constructor {
            Constructor::Ref(mutability) => Term::Ref(*mutability, part()),
            Constructor::Tuple(n) => Term::Tuple((0..*n).map(|_| part()).collect()),
            Constructor::Array(n) => Term::Array(*n, part()),
            Constructor::Slice => Term::Slice(part()),
            Constructor::Option => Term::Option(part()),
            Constructor::Result => Term::Result {
                ok: part(),
                err: part(),
            },
            Constructor::Function(arity) => Term::FunctionValue { arity: *arity },
            Constructor::Object => Term::Object(
                named(members)
                    .into_iter()
                    .map(|field| (field, part()))
                    .collect(),
            ),
            Constructor::Enum(name) => Term::Enum(
                name.clone(),
                named(members)
                    .into_iter()
                    .map(|variant| (variant, part()))
                    .collect(),
            ),
            Constructor::ErrorOf => Term::ErrorOf(part()),
            // A struct's and an extern-defined type's parts are as many as
            // the declaration says, which a read at the node does not see:
            // the node alone is read.
            Constructor::Struct(_) | Constructor::Extern(_) => {
                return ReadValue::Domain(Domain::Only([constructor.clone()].into()));
            }
            Constructor::Scalar(scalar) => Term::Scalar(*scalar),
            Constructor::String => Term::String,
            Constructor::Str => Term::Str,
            Constructor::Unit => Term::Unit,
            Constructor::Never => Term::Never,
            Constructor::Identity(name) => Term::Identity(name.clone()),
        };
        ReadValue::from_term(term).unwrap_or(ReadValue::Undecided)
    }
    match &head.node {
        crate::decl::Node::Owned(constructor) => of(constructor, &head.members),
        crate::decl::Node::Ref(mutability, target) => {
            let target = of(target, &BTreeSet::new());
            ReadValue::from_term(Term::Ref(*mutability, target)).unwrap_or(ReadValue::Undecided)
        }
    }
}

/// Whether a relation `recorder` recorded is left out of a read that leaves
/// out the relations of the disjunctive constraints `left_out`: one of them
/// recorded it.
pub(crate) fn left_out_by(recorder: &Recorder, left_out: &BTreeSet<Name>) -> bool {
    recorder
        .disjunctions()
        .iter()
        .any(|disjunction| left_out.contains(disjunction))
}

/// What a class reads as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AsReference {
    /// The value decides neither.
    Undecided,
    /// No reference.
    No,
    /// A reference, of its mutability where the value decides it.
    Yes(Option<Mutability>),
}

/// What a class reads as in the value `value`: a term decides it, and a
/// domain where it admits references alone, of the one mutability it admits,
/// or no reference at all.
fn reads_as_reference(value: &ReadValue) -> AsReference {
    match value {
        ReadValue::Term(term) => match &**term {
            Term::Ref(mutability, _) => AsReference::Yes(Some(*mutability)),
            _ => AsReference::No,
        },
        ReadValue::Domain(domain) => {
            let shared = domain.admits(&Constructor::Ref(Mutability::Shared));
            let exclusive = domain.admits(&Constructor::Ref(Mutability::Exclusive));
            if !shared && !exclusive {
                AsReference::No
            } else if domain.admits_no_reference() {
                AsReference::Undecided
            } else {
                AsReference::Yes(match (shared, exclusive) {
                    (true, false) => Some(Mutability::Shared),
                    (false, true) => Some(Mutability::Exclusive),
                    _ => None,
                })
            }
        }
        ReadValue::Undecided => AsReference::Undecided,
    }
}

/// The lend of a place reading as `value`, at `mutability`: a place that
/// reads as nothing, its unit being ready, is no reference; none where the
/// place is a reference whose mutability, and so the lesser, is undecided.
fn lend(value: &ReadValue, mutability: Option<Mutability>) -> Option<Decision> {
    let (target, through) = match reads_as_reference(value) {
        AsReference::Undecided | AsReference::No => (LendTarget::Place, Some(Mutability::Exclusive)),
        AsReference::Yes(through) => (LendTarget::Through, through),
    };
    let mutability = lesser(mutability, through)?;
    Some(Decision::Lend { target, mutability })
}

/// The lesser of two mutabilities, where either may be undecided.
fn lesser(a: Option<Mutability>, b: Option<Mutability>) -> Option<Mutability> {
    match (a, b) {
        (Some(Mutability::Shared), _) | (_, Some(Mutability::Shared)) => Some(Mutability::Shared),
        (Some(Mutability::Exclusive), Some(Mutability::Exclusive)) => Some(Mutability::Exclusive),
        _ => None,
    }
}

/// Narrows a use's candidates by the value read at each argument where they
/// differ, in order, and then by the value read at its result: at each, only
/// the candidates whose passing goes by a coercion of the lowest rank remain,
/// an argument passing into the candidate's parameter and the candidate's
/// result type into the result. A position that decides nothing reaches
/// every candidate, and an argument that reads as `!` decides the use as
/// never carried out. The candidate where exactly one remains.
fn narrow(
    decls: &crate::decl::Decls,
    ids: &[DeclId],
    sigs: &[&Signature],
    values: &[(usize, ReadValue)],
    result: Option<&ReadValue>,
) -> Option<Decision> {
    if values
        .iter()
        .any(|(_, value)| matches!(value.decided(), Some(Term::Never)))
    {
        return Some(Decision::Never);
    }
    let mut remaining: Vec<usize> = (0..sigs.len()).collect();
    let stages = values
        .iter()
        .map(|(i, value)| (Some(*i), value))
        .chain(result.map(|result| (None, result)));
    for (position, value) in stages {
        let ranks: Vec<(usize, Option<Rank>)> = remaining
            .iter()
            .map(|&c| {
                let rank = match position {
                    Some(i) => {
                        let written = ReadValue::of_sig(&sigs[c].params()[i]);
                        coerce_read(decls, value, &written)
                    }
                    None => {
                        let written = ReadValue::of_sig(sigs[c].result());
                        coerce_read(decls, &written, value)
                    }
                };
                (c, rank.ok().and_then(|coercion| coercion.rank()))
            })
            .collect();
        let lowest = ranks.iter().filter_map(|(_, rank)| rank.clone()).min();
        remaining = ranks
            .iter()
            .filter(|(_, rank)| rank.is_some() && *rank == lowest)
            .map(|(c, _)| *c)
            .collect();
    }
    match remaining.as_slice() {
        [one] => Some(Decision::Use(ids[*one].clone())),
        _ => None,
    }
}
