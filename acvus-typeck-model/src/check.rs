//! Checks and acceptance (l.291, l.293, l.301): each value the text makes
//! enter a type checked once everything is solved, the places where a
//! definition gives no result, and what solving decides (I3).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Conflict, KindSource};
use crate::coerce::{CoerceError, Coercion, along_le, coerce};
use crate::node::Decision;
use crate::relation::{Named, Relation};
use crate::solve::State;
use crate::ty::{Name, Ty};

/// Where a definition gives no result (l.301).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Refusal {
    /// No type holds what a relation made one equivalence class hold.
    Conflict { at: String, conflict: String },
    /// The types an equivalence class's sources give have no join.
    NoJoin(ClassName),
    /// The equivalence class reaches a part of its own (l.275).
    Infinite(ClassName),
    /// The equivalence class has no value once solving ends.
    NoValue(ClassName),
    /// A type the text writes disagrees with its declaration.
    Undeclared(String),
}

/// A relation an entering value is checked by once everything is solved
/// (l.291, l.293), with the coercion that changes it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Check {
    pub relation: String,
    pub coercion: Result<Coercion, CoerceError>,
}

/// What solving decides (I3).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Outcome {
    pub values: BTreeMap<ClassName, Option<Ty>>,
    pub decisions: BTreeMap<Name, Option<Decision>>,
    pub partition: BTreeSet<BTreeSet<ClassName>>,
    pub checks: Vec<Check>,
    pub refusals: BTreeSet<Refusal>,
    pub accepted: bool,
}

impl State {
    /// Refuses at `at` what no type holds, and returns the refusal.
    pub(crate) fn refuse_conflict(&mut self, at: String, conflict: Conflict) -> Refusal {
        let refusal = Refusal::Conflict {
            at,
            conflict: format!("{conflict:?}"),
        };
        self.refusals.insert(refusal.clone());
        refusal
    }

    /// Checks each value the text makes enter a type (l.293), in the order of
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
                Relation::Equation(..)
                | Relation::Fact(..)
                | Relation::Kind(..)
                | Relation::Lit(..)
                | Relation::Reference(..)
                | Relation::Values(..) => continue,
            };
            let (Some(from_value), Some(to_value)) = (self.value(from), self.value(to)) else {
                continue;
            };
            let written = matches!(self.classes.held(to).kind, Some((_, KindSource::Written)));
            let coercion = if lower {
                match along_le(&self.decls, &from_value, &to_value) {
                    Ok(coercion) => Ok(coercion),
                    Err(_) if written => coerce(&self.decls, &from_value, &to_value),
                    Err(error) => Err(error),
                }
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
            && decisions
                .values()
                .all(|decision| matches!(decision, Some(decision) if decision.is_reached()));
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
    /// there (l.301).
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
            let reached = decided.iter().all(|(_, decision)| decision.is_reached());
            outcome.decisions.extend(
                decided
                    .into_iter()
                    .map(|(name, decision)| (name, Some(decision))),
            );
            let checks = group.checks();
            let checked = checks.iter().all(|check| check.coercion.is_ok());
            outcome.checks.extend(checks);
            outcome.refusals.extend(group.refusals.iter().cloned());
            outcome.accepted &= reached && checked && group.refusals.is_empty();
        }
        outcome
    }
}
