//! The view: what the record holds for an equivalence class, read as values
//! (`Bounds`): its lower bounds, part lower bounds, requirements, domains (a
//! generalized one read against its use's choice now) and occurrences,
//! whether something may still enter it (`lower_pending`) or be passed into
//! (`upper_pending`), and whether a refused lower bound alone reaches it
//! (`refused`), read over the strongly connected component of the undecided
//! classes it waits for; and what the record holds at the class itself, for
//! fallback (`OwnHeld`, D2).
//!
//! The view knows none of the five layers. Where it reads the head an
//! undecided member of a component would take, it calls the decision it is
//! given (`State::bounds_with`).

use std::collections::{BTreeMap, BTreeSet};

use crate::choice::Choice;
use crate::class::{ClassId, Side, StructureConstraint};
use crate::disjunction::generalization::Generalization;
use crate::disjunction::{Disjunction, DisjunctionAt};
use crate::domain::Domain;
use crate::program::Position;
use crate::relation::{Item, Recorder, Relation, Role};
use crate::solve::State;
use crate::ty::{Name, Term};

use super::{Bounds, Decided, Head};

/// Whether a relation `recorder` recorded is left out of a read that excludes
/// the disjunctive constraints in `left_out`.
pub(crate) fn left_out_by(recorder: &Recorder, left_out: &BTreeSet<Name>) -> bool {
    recorder
        .disjunctions()
        .iter()
        .any(|disjunction| left_out.contains(disjunction))
}

/// A bound of a member of a component, as the record gives it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Bound {
    class: ClassId,
    /// Whether the relation is a part lower bound (`≤ₚ`).
    part: bool,
}

/// Whether something enters a member of a component.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Sourced {
    /// Nothing enters it, and nothing more will.
    No,
    /// Nothing enters it, nothing more will, and a lower bound of it was
    /// refused: it passes into nothing, so nothing decides it.
    Refused,
    /// Nothing enters it now, but something may still (`may_enter`).
    Pending,
    /// Something enters it (`L ≠ ∅`).
    Yes,
}

/// What the record holds locally at one equivalence class before component
/// propagation.
#[derive(Clone, Default, Debug)]
struct LocalHeld {
    lowers: Vec<Bound>,
    refused_lower: bool,
    uppers: Vec<ClassId>,
    constraints: Vec<StructureConstraint>,
    domains: Vec<Domain>,
    /// `may_enter`, the waiting fact of rules 1 and 2: whether a lower bound
    /// may still be recorded on the class: by the decomposition of a relation
    /// the class is a part of a side of, or of a wrap into it; by an
    /// undecided disjunctive constraint it is the upper side or a part of;
    /// or by a use whose candidates are not yet one
    /// (`receives_from_candidate`). A lend's place and a projection's
    /// operand are no upper side and no part: what their decision records
    /// on them is a requirement, or a structure on a class made anew.
    lower_pending: bool,
    /// Whether one of `domains` is a literal's: a `Relation::Domain` the
    /// text's origin records (`Recorder::is_origin`, D2).
    literal: bool,
    /// `may_take`: whether an undecided disjunctive constraint or decomposition
    /// may still add an occurrence.
    upper_pending: bool,
    /// `may_require`, a fact of rule 4 alone: whether an undecided projection
    /// has the class as its operand, so that its decision may still record a
    /// requirement on it. No rule of 1 to 3 reads it.
    requirement_pending: bool,
}

/// What the record holds at an equivalence class itself, read without its
/// component, for fallback (D2): whether a lower bound is recorded on it,
/// refused or not, or may still be (`entered`); whether it holds a literal's
/// domain as its own (`literal`); its own domains; whether a requirement is
/// on it (`required`); whether an undecided projection may still record one
/// (`requirement_pending`); and whether an occurrence of it may still be
/// recorded (`taking`). `requirement_pending` and `taking` are facts of the
/// record; an occurrence recorded whose class has no type yet is none.
#[derive(Clone, Default, Debug)]
pub(crate) struct OwnHeld {
    pub(crate) entered: bool,
    pub(crate) literal: bool,
    pub(crate) domains: Vec<Domain>,
    pub(crate) required: bool,
    pub(crate) requirement_pending: bool,
    pub(crate) taking: bool,
}

/// Strongly connected component of mutually waiting undecided classes, read
/// with `decide`, which gives the head an undecided member would take from
/// its `Bounds`.
struct Component<'a, F>
where
    F: Fn(&Bounds) -> Decided,
{
    state: &'a State,
    decide: &'a F,
    members: BTreeSet<ClassId>,
    held: BTreeMap<ClassId, LocalHeld>,
    sourced: BTreeMap<ClassId, Sourced>,
    taken: BTreeMap<ClassId, Bounds>,
}

impl State {
    /// The constructor and members (`Head`) of a decided class's structure,
    /// `None` where `id` holds none. A reference's head is `Ref(m)` alone; its
    /// target is not read.
    pub(crate) fn head_of(&self, id: ClassId) -> Option<Head> {
        let root = self.root(id);
        let (structure, _) = self.classes.held(root).structure.as_ref()?;
        Some(match structure {
            Term::Ref(mutability, _) => Head::reference(*mutability),
            _ => Head::owned(structure.constructor(), structure.members()),
        })
    }

    /// Reads the `Bounds` of `root` without the relations the disjunctive
    /// constraints in `left_out` recorded. Where a lower bound or an
    /// occurrence of `root` is an undecided member of its component, the head
    /// it gives is the one `decide` gives that class's `Bounds`, and none
    /// where `decide` gives no head. It records nothing, and calls no layer
    /// of its own.
    pub(crate) fn bounds_with<F>(
        &self,
        root: ClassId,
        left_out: &BTreeSet<Name>,
        decide: &F,
    ) -> Bounds
    where
        F: Fn(&Bounds) -> Decided,
    {
        let root = self.root(root);
        let members = self.component(root, left_out);
        let held = members
            .iter()
            .map(|&member| (member, self.held_at(member, left_out)))
            .collect();
        let mut component = Component {
            state: self,
            decide,
            members,
            held,
            sourced: BTreeMap::new(),
            taken: BTreeMap::new(),
        };
        component.read(root, &mut BTreeSet::new())
    }

    /// What the record holds at `root` itself (`OwnHeld`), reading every
    /// relation, for fallback (D2). It reads no component, decides nothing
    /// and records nothing.
    pub(crate) fn own_held(&self, root: ClassId) -> OwnHeld {
        let held = self.held_at(root, &BTreeSet::new());
        OwnHeld {
            entered: !held.lowers.is_empty() || held.refused_lower || held.lower_pending,
            literal: held.literal,
            domains: held.domains,
            required: !held.constraints.is_empty(),
            requirement_pending: held.requirement_pending,
            taking: held.upper_pending,
        }
    }

    /// The undecided classes that `root` directly waits for.
    fn waits_for(&self, root: ClassId, left_out: &BTreeSet<Name>) -> BTreeSet<ClassId> {
        let held = self.held_at(root, left_out);
        let mut waiting = BTreeSet::new();
        for bound in &held.lowers {
            if let Some(undecided) = self.undecided_of(bound.class) {
                waiting.insert(undecided);
            }
        }
        for &upper in &held.uppers {
            if let Some(undecided) = self.undecided_of(upper) {
                waiting.insert(undecided);
            }
        }
        waiting
    }

    /// Returns `Some(root)` if `class` is still undecided and not refused.
    fn undecided_of(&self, class: ClassId) -> Option<ClassId> {
        let root = self.root(class);
        (!self.has_no_value(root) && !self.is_solved(root)).then_some(root)
    }

    /// The strongly connected component of `root` in the `waits_for` graph.
    fn component(&self, root: ClassId, left_out: &BTreeSet<Name>) -> BTreeSet<ClassId> {
        let forward = self.reach(root, left_out);
        let mut members = BTreeSet::from([root]);
        for class in forward {
            if self.reach(class, left_out).contains(&root) {
                members.insert(class);
            }
        }
        members
    }

    /// Transitive closure of `waits_for` from `from`.
    fn reach(&self, from: ClassId, left_out: &BTreeSet<Name>) -> BTreeSet<ClassId> {
        let mut seen = BTreeSet::new();
        let mut work = vec![from];
        while let Some(at) = work.pop() {
            for next in self.waits_for(at, left_out) {
                if seen.insert(next) {
                    work.push(next);
                }
            }
        }
        seen
    }

    /// What the record holds directly at the class of `root`, without the
    /// relations the disjunctive constraints and uses in `left_out` recorded
    /// (`LocalHeld`): its lower bounds, occurrences, requirements and
    /// domains, and the facts of what may still be recorded on it. A lower
    /// bound may still be (`lower_pending`, rules 1 and 2) from an undecided
    /// disjunctive constraint the class is the upper side or a part of, and
    /// an occurrence (`upper_pending`, rule 3) from one it is the lower side
    /// of; of a lend's place, a projection's operand and a call's callee
    /// neither is, the decision recording no lower bound and no occurrence
    /// into them. A requirement may still be (`requirement_pending`, rule 4)
    /// from an undecided projection whose operand it is. It reads no other
    /// class's component, decides nothing and records nothing.
    fn held_at(&self, root: ClassId, left_out: &BTreeSet<Name>) -> LocalHeld {
        let root = self.root(root);
        let mut held = LocalHeld::default();
        for occurrence in self.classes.occurrences(root) {
            let recorded = &self.classes.items()[occurrence.at];
            if left_out_by(&recorded.recorder, left_out) {
                continue;
            }
            match &recorded.item {
                Item::Relation(relation) => match relation {
                    Relation::LowerBound { lower, .. } | Relation::PartLowerBound { lower, .. }
                        if occurrence.side() == Some(Side::Upper) && self.root(*lower) != root =>
                    {
                        let lower = self.root(*lower);
                        if self.has_no_value(lower) {
                            held.refused_lower = true;
                            continue;
                        }
                        held.lowers.push(Bound {
                            class: lower,
                            part: matches!(relation, Relation::PartLowerBound { .. }),
                        });
                    }
                    Relation::LowerBound { upper: into, .. }
                    | Relation::PartLowerBound { upper: into, .. }
                    | Relation::CoercionConstraint { to: into, .. }
                        if occurrence.side() == Some(Side::Lower) && self.root(*into) != root =>
                    {
                        let into = self.root(*into);
                        if !self.has_no_value(into) {
                            held.uppers.push(into);
                        }
                    }
                    Relation::Domain(_, domain) if occurrence.side() == Some(Side::Own) => {
                        held.domains.push(domain.clone());
                        held.literal |= recorded.recorder.is_origin();
                    }
                    Relation::GeneralizedDomain(_, domain)
                        if occurrence.side() == Some(Side::Own) =>
                    {
                        held.domains.push(self.classes.resolve(domain));
                    }
                    Relation::StructureConstraint(_, constraint)
                        if occurrence.side() == Some(Side::Own) =>
                    {
                        held.constraints.push(constraint.clone());
                    }
                    Relation::Structure(owner, ..) if occurrence.side() == Some(Side::Part) => {
                        held.lower_pending |= self.undecomposed_at(*owner, Side::Upper, left_out);
                        held.upper_pending |= self.undecomposed_at(*owner, Side::Lower, left_out);
                    }
                    Relation::StructureConstraint(owner, _)
                        if occurrence.side() == Some(Side::Part)
                            && !recorded.decomposed
                            && !self.has_no_value(*owner) =>
                    {
                        held.lower_pending = true;
                        held.upper_pending = true;
                    }
                    Relation::Wrap { .. }
                        if occurrence.side() == Some(Side::Upper) && !recorded.decomposed =>
                    {
                        held.lower_pending = true;
                    }
                    _ => {}
                },
                Item::Disjunction {
                    name,
                    kind,
                    decision,
                } if decision.is_none() && !left_out.contains(name) => {
                    if matches!(occurrence.side(), Some(Side::Upper | Side::Part)) {
                        held.lower_pending = true;
                    }
                    if occurrence.side() == Some(Side::Lower) {
                        held.upper_pending = true;
                    }
                    if matches!(kind, Disjunction::Project { .. })
                        && occurrence.role == Role::Operand
                    {
                        held.requirement_pending = true;
                    }
                }
                Item::Choice { name, of, choice }
                    if !left_out.contains(name)
                        && choice.remaining.len() > 1
                        && self.chosen_of(name, of, choice, left_out).is_none() =>
                {
                    held.lower_pending |= self.receives_from_candidate(choice, &occurrence.role);
                }
                Item::Disjunction { .. }
                | Item::Instance { .. }
                | Item::Lambda { .. }
                | Item::Choice { .. } => {}
            }
        }
        held
    }

    /// The waiting fact of a use whose candidates are not yet one: whether
    /// the class in `role` of the use whose choice is `choice` will receive a
    /// lower bound once one candidate remains. A variable `Generalized(k)`
    /// the generalization's result writes receives the candidate's type there
    /// (`record_candidate`), and the exit of `e?` receives the exit's value
    /// (`record_exit`). The use's result receives nothing it waits for: the
    /// generalization's result type enters it as the use is recorded, and
    /// the `!` the never rule gives it changes no join (D4). A variable only
    /// a parameter writes receives nothing new, its lower bounds being the
    /// arguments, recorded already. It reads the record alone and decides
    /// nothing.
    fn receives_from_candidate(&self, choice: &Choice, role: &Role) -> bool {
        match role {
            Role::Made(DisjunctionAt::Generalized(k)) => {
                let sigs = self.signatures(choice);
                Generalization::of(&sigs.iter().collect::<Vec<_>>()).writes_in_result(*k)
            }
            Role::Position(Position::Exit) => true,
            _ => false,
        }
    }

    /// Whether a relation on `owner` at `side` is not yet decomposed.
    pub(crate) fn undecomposed_at(
        &self,
        owner: ClassId,
        side: Side,
        left_out: &BTreeSet<Name>,
    ) -> bool {
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

    /// Direct lower bounds and part lower bounds of `id`, excluding itself.
    pub(crate) fn lower_bounds(&self, id: ClassId) -> Vec<ClassId> {
        let root = self.root(id);
        self.classes
            .occurrences(root)
            .iter()
            .filter(|occurrence| occurrence.side() == Some(Side::Upper))
            .filter_map(|occurrence| self.classes.items()[occurrence.at].relation())
            .filter_map(|relation| match relation {
                Relation::LowerBound { lower, .. } | Relation::PartLowerBound { lower, .. }
                    if self.root(*lower) != root =>
                {
                    Some(*lower)
                }
                _ => None,
            })
            .collect()
    }

    /// Detects whether `root` reaches a part of its own along lower bounds and
    /// structure parts (an infinite type cycle).
    pub(crate) fn own_part_cycle(&self, root: ClassId) -> Vec<ClassId> {
        let root = self.root(root);
        let mut seen: BTreeSet<(ClassId, bool)> = BTreeSet::new();
        let mut came_from: BTreeMap<(ClassId, bool), (ClassId, bool)> = BTreeMap::new();
        let mut work = vec![(root, false)];
        while let Some((at, through_part)) = work.pop() {
            let mut next: Vec<(ClassId, bool)> = self
                .lower_bounds(at)
                .into_iter()
                .map(|lower| (self.root(lower), through_part))
                .collect();
            if let Some((structure, _)) = &self.classes.held(at).structure {
                next.extend(
                    structure
                        .parts()
                        .into_iter()
                        .map(|(_, part)| (self.root(*part), true)),
                );
            }
            for step in next {
                if step.0 == root && step.1 {
                    let mut path = vec![root];
                    let mut back = (at, through_part);
                    while back.0 != root || back.1 {
                        path.push(back.0);
                        let Some(previous) = came_from.get(&back) else {
                            break;
                        };
                        back = *previous;
                    }
                    return path;
                }
                if seen.insert(step) {
                    came_from.insert(step, (at, through_part));
                    work.push(step);
                }
            }
        }
        Vec::new()
    }
}

impl<F> Component<'_, F>
where
    F: Fn(&Bounds) -> Decided,
{
    fn sourced(&mut self, member: ClassId, visiting: &mut BTreeSet<ClassId>) -> Sourced {
        if let Some(&sourced) = self.sourced.get(&member) {
            return sourced;
        }
        if !visiting.insert(member) {
            return Sourced::Pending;
        }
        let held = self.held[&member].clone();
        let mut sourced = if held.lower_pending {
            Sourced::Pending
        } else if held.refused_lower && held.uppers.is_empty() {
            Sourced::Refused
        } else {
            Sourced::No
        };
        for bound in &held.lowers {
            let of_bound = match self.state.undecided_of(bound.class) {
                None => Sourced::Yes,
                Some(undecided) if self.members.contains(&undecided) => {
                    self.sourced(undecided, visiting)
                }
                Some(_) => Sourced::Pending,
            };
            sourced = sourced.max(of_bound);
        }
        visiting.remove(&member);
        self.sourced.insert(member, sourced);
        sourced
    }

    /// The domains on `member` and on the undecided members below it that
    /// nothing enters: what a class is read as where nothing gives it a head
    /// (`State::read`), a use reading its argument position as the family
    /// of the literal entering it. Fallback reads none of it, a class with a
    /// lower bound taking no default (D2).
    fn domains_below(&mut self, member: ClassId, visiting: &mut BTreeSet<ClassId>) -> Vec<Domain> {
        if !visiting.insert(member) {
            return Vec::new();
        }
        let held = self.held[&member].clone();
        let mut domains = held.domains.clone();
        for bound in &held.lowers {
            if let Some(undecided) = self.state.undecided_of(bound.class)
                && self.members.contains(&undecided)
                && self.sourced(undecided, &mut BTreeSet::new()) < Sourced::Yes
            {
                domains.extend(self.domains_below(undecided, visiting));
            }
        }
        domains
    }

    fn read(&mut self, member: ClassId, visiting: &mut BTreeSet<ClassId>) -> Bounds {
        if let Some(bounds) = self.taken.get(&member) {
            return bounds.clone();
        }
        if !visiting.insert(member) {
            return Bounds {
                lower_pending: true,
                upper_pending: true,
                ..Bounds::default()
            };
        }
        let held = self.held[&member].clone();
        let mut bounds = Bounds {
            constraints: held.constraints.clone(),
            domains: held.domains.clone(),
            lower_pending: held.lower_pending,
            upper_pending: held.upper_pending,
            ..Bounds::default()
        };
        let mut refused_lower = held.refused_lower;
        for bound in &held.lowers {
            let head = match self.state.undecided_of(bound.class) {
                None => self.state.head_of(bound.class),
                Some(undecided) if !self.members.contains(&undecided) => {
                    bounds.lower_pending = true;
                    continue;
                }
                Some(undecided) => match self.sourced(undecided, &mut BTreeSet::new()) {
                    Sourced::No => {
                        let below = self.domains_below(undecided, &mut BTreeSet::new());
                        bounds.domains.extend(below);
                        continue;
                    }
                    Sourced::Refused => {
                        refused_lower = true;
                        continue;
                    }
                    Sourced::Pending => {
                        let below = self.domains_below(undecided, &mut BTreeSet::new());
                        bounds.domains.extend(below);
                        bounds.lower_pending = true;
                        continue;
                    }
                    Sourced::Yes => {
                        let inner = self.read(undecided, visiting);
                        bounds.lower_pending |= inner.lower_pending;
                        self.head_of_bounds(&inner)
                    }
                },
            };
            match head {
                Some(head) if bound.part => bounds.part_lowers.push((bound.class, head)),
                Some(head) => bounds.lowers.push((bound.class, head)),
                None => {}
            }
        }
        let unsourced = bounds.lowers.is_empty() && bounds.part_lowers.is_empty();
        for &into in held.uppers.iter().filter(|_| unsourced) {
            let head = match self.state.undecided_of(into) {
                None => self.state.head_of(into),
                Some(undecided) if !self.members.contains(&undecided) => {
                    bounds.upper_pending = true;
                    continue;
                }
                Some(undecided) => {
                    let inner = self.read(undecided, visiting);
                    bounds.upper_pending |= inner.lower_pending || inner.upper_pending;
                    self.head_of_bounds(&inner)
                }
            };
            if let Some(head) = head {
                bounds.uppers.push(head);
            } else {
                bounds.upper_pending = true;
            }
        }
        bounds.refused = refused_lower
            && bounds.lowers.is_empty()
            && bounds.part_lowers.is_empty()
            && !bounds.lower_pending
            && held.uppers.is_empty();
        visiting.remove(&member);
        self.taken.insert(member, bounds.clone());
        bounds
    }

    /// The head `decide` gives `bounds`, where it gives one.
    fn head_of_bounds(&self, bounds: &Bounds) -> Option<Head> {
        (self.decide)(bounds).head()
    }
}
