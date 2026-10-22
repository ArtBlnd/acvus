//! Checks and acceptance (l.278, l.280, l.290): each value the text makes
//! enter a type checked once everything is solved, the places where a
//! definition gives no result, and what solving decides (I3).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Conflict, Conflicted, NamedConflict};
use crate::coerce::{CoerceError, Coercion, along_le, coerce};
use crate::node::Decision;
use crate::relation::{Named, Relation};
use crate::solve::State;
use crate::ty::{Name, Ty};

/// Where a definition gives no result (l.290).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Refusal {
    /// No type holds what a relation made one equivalence class hold.
    Conflict { at: String, conflict: String },
    /// The types an equivalence class's sources give have no join.
    NoJoin(ClassName),
    /// The equivalence class reaches a part of its own (l.269).
    Infinite(ClassName),
    /// The places the equivalence class nothing enters passes into have two
    /// types (l.282).
    NoFill(ClassName),
    /// The equivalence class has no value once solving ends.
    NoValue(ClassName),
    /// A type the text writes disagrees with its declaration.
    Undeclared(String),
}

/// A relation an entering value is checked by once everything is solved
/// (l.278, l.280), with the coercion that changes it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Check {
    pub relation: String,
    pub coercion: Result<Coercion, CoerceError>,
}

/// What solving decides (I3).
#[derive(Clone, Debug)]
pub struct Outcome {
    pub values: BTreeMap<ClassName, Option<Ty>>,
    pub decisions: BTreeMap<Name, Option<Decision>>,
    /// Which of the text's classes are one, reported but not compared: two
    /// orders may make two literals one class or leave them apart, of one
    /// type either way (`same_typing`).
    pub partition: BTreeSet<BTreeSet<ClassName>>,
    pub checks: Vec<Check>,
    pub refusals: BTreeSet<Refusal>,
    pub accepted: bool,
}

impl Outcome {
    /// Whether the two outcomes are one typing, which is what does not depend
    /// on the order of solving (types §6, I2): each class's value, each node's
    /// decision, each check, each refusal and whether the program is accepted.
    /// Which classes are one is no part of it.
    pub fn same_typing(&self, other: &Outcome) -> bool {
        self.values == other.values
            && self.decisions == other.decisions
            && self.checks == other.checks
            && self.refusals == other.refusals
            && self.accepted == other.accepted
    }
}

impl State {
    /// Refuses at `at` what no type holds, and returns the refusal.
    /// The classes a conflict names hold a conjunction no type meets, so they
    /// have no value (l.158, l.276), and are never solved or filled again.
    pub(crate) fn conflicted_have_no_value(&mut self, conflicted: &Conflicted) {
        let members: Vec<ClassId> = match conflicted {
            Conflicted::Apart(apart) => apart.sides.iter().flatten().copied().collect(),
            Conflicted::Holds { members, .. } => members.iter().copied().collect(),
        };
        self.no_value.extend(members);
    }

    pub(crate) fn refuse_conflict(&mut self, at: String, conflict: Conflict) -> Refusal {
        let conflict = NamedConflict {
            conflict: &conflict,
            arena: &self.arena,
        };
        let refusal = Refusal::Conflict {
            at,
            conflict: conflict.to_string(),
        };
        self.refusals.insert(refusal.clone());
        refusal
    }

    /// Checks each value the text makes enter a type (l.280), in the order of
    /// their relations' names, which no order of solving changes (I3).
    fn checks(&self) -> Vec<Check> {
        let mut checks = Vec::new();
        for recorded in &self.log {
            if !recorded.recorder.is_entering() {
                continue;
            }
            let (from, to, lower) = match recorded.relation {
                Relation::LowerBound { lower, upper } => (lower, upper, true),
                Relation::Boundary { from, to } => (from, to, false),
                Relation::PartLowerBound { .. }
                | Relation::Equation(..)
                | Relation::Fact(..)
                | Relation::Kind(..)
                | Relation::Lit(..)
                | Relation::Reference(..)
                | Relation::Values(..)
                | Relation::Wrap { .. } => continue,
            };
            let (Some(from_value), Some(to_value)) = (self.value(from), self.value(to)) else {
                continue;
            };
            let coercion = if lower {
                along_le(&self.decls, &from_value, &to_value)
            } else {
                coerce(&self.decls, &from_value, &to_value)
            };
            let relation = Named {
                relation: &recorded.relation,
                arena: &self.arena,
            }
            .to_string();
            checks.push(Check { relation, coercion });
        }
        checks.sort_by(|a, b| {
            (&a.relation, format!("{:?}", a.coercion))
                .cmp(&(&b.relation, format!("{:?}", b.coercion)))
        });
        checks
    }

    pub fn outcome(&self) -> Outcome {
        let values = self
            .texts
            .iter()
            .map(|&id| (self.arena.name(id).clone(), self.value(id)))
            .collect();
        let mut partition: BTreeMap<ClassId, BTreeSet<ClassName>> = BTreeMap::new();
        for &id in &self.texts {
            partition
                .entry(self.root(id))
                .or_default()
                .insert(self.arena.name(id).clone());
        }
        let checks = self.checks();
        let mut refusals = self.refusals.clone();
        for root in self.classes.roots() {
            if self.value(root).is_none() && !self.has_no_value(root) {
                refusals.insert(Refusal::NoValue(self.class_name(root)));
            }
        }
        let decisions: BTreeMap<Name, Option<Decision>> = self
            .nodes
            .iter()
            .map(|(name, node)| (name.clone(), node.decision.clone()))
            .collect();
        let accepted = refusals.is_empty()
            && checks.iter().all(|check| check.coercion.is_ok())
            && decisions.values().all(Option::is_some);
        Outcome {
            values,
            decisions,
            partition: partition.into_values().collect(),
            checks,
            refusals,
            accepted,
        }
    }

    /// The outcome of the entry's state, with what each other binding group
    /// decided at its fixpoint: the values of its text's classes, its decided
    /// nodes, its checks of values entering classes it solved, and its
    /// refusals. Its classes and nodes left in C need no value and no decision
    /// there (l.290).
    pub fn outcome_with(&self, groups: &BTreeMap<Name, Rc<State>>) -> Outcome {
        let mut outcome = self.outcome();
        for group in groups.values() {
            outcome.values.extend(
                group
                    .texts
                    .iter()
                    .map(|&id| (group.arena.name(id).clone(), group.value(id))),
            );
            let decided: Vec<(Name, Decision)> = group
                .nodes
                .iter()
                .filter_map(|(name, node)| Some((name.clone(), node.decision.clone()?)))
                .collect();
            // A node is one group's: a use of several candidates has one, at
            // the positions its choice decides, which one constraint holds
            // (l.186), and a use of one has none.
            for (name, decision) in decided {
                let previous = outcome.decisions.insert(name.clone(), Some(decision));
                assert!(previous.is_none(), "two groups have the node {name}");
            }
            let checks = group.checks();
            let checked = checks.iter().all(|check| check.coercion.is_ok());
            outcome.checks.extend(checks);
            outcome.refusals.extend(group.refusals.iter().cloned());
            outcome.accepted &= checked && group.refusals.is_empty();
        }
        outcome
    }
}
