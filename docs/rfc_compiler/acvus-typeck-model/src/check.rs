//! Checks and acceptance: each value the text makes
//! enter a type checked once everything is solved, the places where a
//! definition gives no result, and what solving decides.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use crate::class::{ClassId, ClassName};
use crate::coerce::{CoerceError, Coercion, along_le, coerce};
use crate::decl::DeclId;
use crate::disjunction::Decision;
use crate::relation::{Item, Named, Relation};
use crate::sig::{SigTy, SigVar};
use crate::solve::State;
use crate::ty::{Name, Term, Ty};

/// Why an equivalence class has no value.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Cause {
    /// No type holds the conjunction of what its relations give it.
    Conflict(String),
    /// What enters it has no join.
    NoJoin,
    /// It reaches a part of its own.
    Infinite,
    /// A type the text writes for it disagrees with its declaration.
    Undeclared(String),
    /// Nothing decided it once solving ended.
    NoValue,
}

impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cause::Conflict(conflict) => write!(f, "conflict: {conflict}"),
            Cause::NoJoin => f.write_str("no_join"),
            Cause::Infinite => f.write_str("infinite"),
            Cause::Undeclared(what) => write!(f, "undeclared: {what}"),
            Cause::NoValue => f.write_str("no_value"),
        }
    }
}

/// Where a definition gives no result: an equivalence class, with every
/// type variable it holds, the items of the record on it, which cannot hold
/// together, and why. A refusal is read from the record once solving ends,
/// so it depends on no order of solving.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Refusal {
    pub members: BTreeSet<ClassName>,
    /// The items on the class, each as the record names it.
    pub items: BTreeSet<String>,
    pub cause: Cause,
}

/// A relation an entering value is checked by once everything is solved,
/// with the coercion that changes it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Check {
    pub relation: String,
    pub coercion: Result<Coercion, CoerceError>,
    /// Where the coercion is a declared conversion, the lambdas a function
    /// value it carries brings that the Values where it reaches do not name;
    /// none elsewhere.
    pub unnamed: BTreeSet<Name>,
}

impl Check {
    /// Whether the value reaches: by a coercion, and, by a conversion, with
    /// every lambda it carries named where it reaches.
    pub fn holds(&self) -> bool {
        self.coercion.is_ok() && self.unnamed.is_empty()
    }
}

/// What solving decides.
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
    /// on the order of solving: each class's value, each disjunctive constraint's
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
    /// Every refusal the record gives: one for each equivalence class with no
    /// value, with its members, the items on it and the cause.
    pub fn refusals(&self) -> BTreeSet<Refusal> {
        self.classes
            .roots()
            .filter_map(|root| {
                let cause = self.classes.held(root).no_value.clone()?;
                Some(self.refusal(root, cause))
            })
            .collect()
    }

    /// The refusal of the class of `id` for `cause`.
    pub(crate) fn refusal(&self, id: ClassId, cause: Cause) -> Refusal {
        let root = self.root(id);
        let held = self.classes.held(root);
        let members = held
            .members
            .iter()
            .map(|&member| self.arena.name(member).clone())
            .collect();
        let items = held
            .occurrences
            .iter()
            .map(|occurrence| match &self.classes.items()[occurrence.at].item {
                Item::Relation(relation) => Named {
                    relation,
                    arena: &self.arena,
                }
                .to_string(),
                Item::Disjunction { name, .. } => format!("disjunctive constraint {name}"),
                Item::Instance { at, .. } => format!("instance {at}"),
                Item::Lambda { name, .. } => format!("lambda {name}"),
            })
            .collect();
        Refusal {
            members,
            items,
            cause,
        }
    }

    /// Checks each value the text makes enter a type, in the order of
    /// their relations' names, which no order of solving changes.
    fn checks(&self) -> Vec<Check> {
        let mut checks = Vec::new();
        for recorded in self.classes.items() {
            if !recorded.recorder.is_entering() {
                continue;
            }
            let (from, to, lower) = match recorded.relation() {
                Some(Relation::LowerBound { lower, upper }) => (*lower, *upper, true),
                Some(Relation::CoercionConstraint { from, to }) => (*from, *to, false),
                _ => continue,
            };
            let (Some(from_value), Some(to_value)) = (self.value(from), self.value(to)) else {
                continue;
            };
            let coercion = if lower {
                along_le(&self.decls, &from_value, &to_value)
            } else {
                coerce(&self.decls, &from_value, &to_value)
            };
            let unnamed = match &coercion {
                Ok(Coercion::Conversion(id)) => self.unnamed(id, from, to),
                _ => BTreeSet::new(),
            };
            let relation = Named {
                relation: recorded.relation().expect("a lower bound or a coercion constraint"),
                arena: &self.arena,
            }
            .to_string();
            checks.push(Check {
                relation,
                coercion,
                unnamed,
            });
        }
        checks.sort_by(|a, b| {
            (&a.relation, format!("{:?}", a.coercion))
                .cmp(&(&b.relation, format!("{:?}", b.coercion)))
        });
        checks
    }

    /// The lambdas the function values that the declared conversion `id`
    /// carries from `from` bring where they reach in `to`, which the Values
    /// there do not name. The decomposition links nothing through a
    /// conversion, so nothing records them there: each function value
    /// at a position the conversion's parameter writes a variable reaches the
    /// one at the same place of each position its result writes it, and its
    /// lambdas are to be among that one's.
    fn unnamed(&self, id: &DeclId, from: ClassId, to: ClassId) -> BTreeSet<Name> {
        let sig = &self.decls.decl(id).sig;
        let [param] = sig.params() else {
            unreachable!("a conversion takes one parameter");
        };
        let mut params = BTreeMap::new();
        self.classes_at_vars(param, from, &mut params);
        let mut results = BTreeMap::new();
        self.classes_at_vars(sig.result(), to, &mut results);
        let mut unnamed = BTreeSet::new();
        for (var, carried) in &params {
            let Some(reached) = results.get(var) else {
                continue;
            };
            for &at in carried {
                for &into in reached {
                    self.unnamed_at(at, into, &mut unnamed);
                }
            }
        }
        unnamed
    }

    /// The class at each position `ty` writes a variable, `ty` being written at
    /// the class `at`.
    fn classes_at_vars(&self, ty: &SigTy, at: ClassId, vars: &mut BTreeMap<SigVar, Vec<ClassId>>) {
        match ty {
            SigTy::Var(var) => vars.entry(*var).or_default().push(self.root(at)),
            SigTy::Term(term) => {
                let structure = self.held_structure(at);
                for (key, part) in term.parts() {
                    let held = structure
                        .part(&key)
                        .expect("a conversion writes the shape of the class it reaches");
                    self.classes_at_vars(part, *held, vars);
                }
            }
        }
    }

    /// Each lambda a function value in `from` brings that the function value
    /// at its place in `to` does not name, the two of one type.
    fn unnamed_at(&self, from: ClassId, to: ClassId, unnamed: &mut BTreeSet<Name>) {
        let structure = self.held_structure(from);
        if let Term::FunctionValue { .. } = structure {
            let named = self.values_of(to);
            unnamed.extend(
                self.values_of(from)
                    .into_iter()
                    .filter(|lambda| !named.contains(lambda)),
            );
            return;
        }
        let reached = self.held_structure(to);
        for (key, part) in structure.parts() {
            let into = reached
                .part(&key)
                .expect("a conversion binds one type at a variable");
            self.unnamed_at(*part, *into, unnamed);
        }
    }

    /// The structure of a class with a value.
    fn held_structure(&self, id: ClassId) -> Term<ClassId> {
        let Some((structure, _)) = self.classes.held(self.root(id)).structure.clone() else {
            unreachable!("a class with a value holds its structure");
        };
        structure
    }

    pub fn outcome(&self) -> Outcome {
        let texts = self.texts();
        let values = texts
            .iter()
            .map(|&id| (self.arena.name(id).clone(), self.value(id)))
            .collect();
        let mut partition: BTreeMap<ClassId, BTreeSet<ClassName>> = BTreeMap::new();
        for &id in &texts {
            partition
                .entry(self.root(id))
                .or_default()
                .insert(self.arena.name(id).clone());
        }
        let checks = self.checks();
        let mut refusals = self.refusals();
        for root in self.classes.roots() {
            if self.value(root).is_none() && !self.has_no_value(root) {
                refusals.insert(self.refusal(root, Cause::NoValue));
            }
        }
        let decisions: BTreeMap<Name, Option<Decision>> = self.decisions();
        let accepted = refusals.is_empty()
            && checks.iter().all(Check::holds)
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
    /// disjunctive constraints, its checks of values entering classes it solved, and its
    /// refusals. Its classes and disjunctive constraints left in C need no value and no decision
    /// there.
    pub fn outcome_with(&self, groups: &BTreeMap<Name, Rc<State>>) -> Outcome {
        let mut outcome = self.outcome();
        for group in groups.values() {
            outcome.values.extend(
                group
                    .texts()
                    .into_iter()
                    .map(|id| (group.arena.name(id).clone(), group.value(id))),
            );
            let decided: Vec<(Name, Decision)> = group
                .decisions()
                .into_iter()
                .filter_map(|(name, decision)| Some((name, decision?)))
                .collect();
            // A disjunctive constraint is one: a use of several candidates is
            // decided in the one group that records the positions where
            // they differ, and a use of one candidate, decided as it is
            // recorded in each group that records a position of it, is
            // decided alike in each.
            for (name, decision) in decided {
                let previous = outcome.decisions.insert(name.clone(), Some(decision.clone()));
                assert!(
                    previous.is_none_or(|previous| previous == Some(decision)),
                    "two groups decide the disjunctive constraint {name} apart"
                );
            }
            let checks = group.checks();
            let checked = checks.iter().all(Check::holds);
            outcome.checks.extend(checks);
            let refusals = group.refusals();
            outcome.accepted &= checked && refusals.is_empty();
            outcome.refusals.extend(refusals);
        }
        outcome
    }
}
