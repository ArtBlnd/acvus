//! Unification: the equivalence classes hold what a relation
//! records — an equation unifies two into one, whose structure is the conjunction
//! of theirs and whose parts of one name are unified in turn, and a
//! structure, a structure constraint, a domain, or a lambda its Values name is held
//! by the class it is given to — and a cycle of lower bounds and part lower
//! bounds alone is merged into one class. Where no type holds what a relation
//! records, the classes it names have no value, for that cause.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::Cause;
use crate::class::{ClassId, Conflict, Conflicted, NamedConflict, Unified};
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;

/// What the equivalence classes did as they took a relation: the
/// unions made, and the cause where no type holds what it records or a type
/// the text writes disagrees with its declaration, for which the classes it
/// names have no value.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Unification {
    pub unified: Unified,
    pub refused: Option<Cause>,
}

impl State {
    /// The equivalence classes take what `relation` records, as it is recorded.
    /// Where no type holds it, or a type the text writes disagrees with its
    /// declaration, the classes it names have no value.
    pub(crate) fn hold(&mut self, relation: &Relation) -> Unification {
        let decls = Rc::clone(&self.decls);
        let types = decls.types();
        let unified = match relation {
            Relation::Equation(a, b) => self.classes.unifying(types).unify(*a, *b),
            Relation::StructureConstraint(class, _) => {
                self.classes.unifying(types).constrained(*class)
            }
            Relation::Structure(class, term, source) => match types.check(term) {
                Ok(()) => {
                    self.classes
                        .unifying(types)
                        .give_structure(*class, term.clone(), *source)
                }
                Err(error) => {
                    let cause = Cause::Undeclared(format!("{error:?}"));
                    self.classes.refuse(*class, cause.clone());
                    return Unification {
                        unified: Unified::default(),
                        refused: Some(cause),
                    };
                }
            },
            Relation::Domain(class, domain) => {
                self.classes.unifying(types).domain_given(*class, domain)
            }
            Relation::Values(..)
            | Relation::Standing(..)
            | Relation::LowerBound { .. }
            | Relation::PartLowerBound { .. }
            | Relation::CoercionConstraint { .. }
            | Relation::Wrap { .. } => Unified::default(),
        };
        let Some(conflict) = unified.conflict.as_ref().map(|c| c.conflict().clone()) else {
            return Unification {
                unified,
                refused: None,
            };
        };
        let cause = self.conflict_cause(conflict);
        if let Some(conflicted) = &unified.conflict {
            self.conflicted_have_no_value(conflicted, &cause);
        }
        Unification {
            unified,
            refused: Some(cause),
        }
    }

    /// Merges each cycle of lower bounds and part lower bounds alone into one
    /// equivalence class. Returns the equations recorded.
    pub(crate) fn merge_cycles(&mut self) -> Recording {
        let mut recorded = Recording::default();
        let mut edges: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        for recorded in self.classes.items() {
            if let Some(
                Relation::LowerBound { lower, upper } | Relation::PartLowerBound { lower, upper },
            ) = recorded.relation()
            {
                let (lower, upper) = (*lower, *upper);
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

    /// The classes a conflict names hold a conjunction no type satisfies, so
    /// they have no value.
    pub(crate) fn conflicted_have_no_value(&mut self, conflicted: &Conflicted, cause: &Cause) {
        let members: Vec<ClassId> = match conflicted {
            Conflicted::Apart(apart) => apart.sides.iter().flatten().copied().collect(),
            Conflicted::Holds { members, .. } => members.iter().copied().collect(),
        };
        for member in members {
            self.classes.refuse(member, cause.clone());
        }
    }

    /// The cause of a class's having no value: what no type holds.
    pub(crate) fn conflict_cause(&self, conflict: Conflict) -> Cause {
        let conflict = NamedConflict {
            conflict: &conflict,
            arena: &self.arena,
        };
        Cause::Conflict(conflict.to_string())
    }
}
