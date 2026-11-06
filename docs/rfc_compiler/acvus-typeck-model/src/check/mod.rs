//! Checks and acceptance: each value the text makes
//! enter a type checked once everything is solved, the places where a
//! definition gives no result, and what solving decides.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

pub mod istar;
pub mod trace;

use crate::choice::Chosen;
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
    /// Nothing enters it, and what it passes into is of two types.
    Contexts,
    /// It reaches a part of its own.
    Infinite,
    /// A type the text writes for it disagrees with its declaration.
    Undeclared(String),
    /// Nothing decided it once solving ended.
    NoValue,
    /// It is a variable of a use's generalization, and the structures the
    /// use's variables took reach no candidate.
    NoCandidate,
    /// A use's choice that admits several candidates gives its domain, which
    /// has no default; or, of a use, several candidates remain to its choice
    /// once solving ended.
    Ambiguous,
}

impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cause::Conflict(conflict) => write!(f, "conflict: {conflict}"),
            Cause::NoJoin => f.write_str("no_join"),
            Cause::Contexts => f.write_str("contexts"),
            Cause::Infinite => f.write_str("infinite"),
            Cause::Undeclared(what) => write!(f, "undeclared: {what}"),
            Cause::NoValue => f.write_str("no_value"),
            Cause::NoCandidate => f.write_str("no_candidate"),
            Cause::Ambiguous => f.write_str("ambiguous"),
        }
    }
}

/// Where a definition gives no result: an equivalence class, with every
/// type variable it holds, the items of the record on it, which cannot hold
/// together, and why; or a use left `ambiguous`, with the members of the
/// variables of its generalization, its item and those variables. A refusal
/// is read from the record once solving ends, so it depends on no order of
/// solving.
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

/// What a disjunctive constraint is once solving ends.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum DecisionAtEnd {
    /// It took this alternative.
    Decided(Decision),
    /// It took no alternative, and every class it reads has a value.
    Undecided,
    /// It took no alternative, and a class it reads has no value
    /// (`State::refused_with`, D5): it has nothing to read, and is refused
    /// with that class, whose refusal names it among its items.
    RefusedWith,
}

impl DecisionAtEnd {
    /// The alternative taken, where one is.
    pub fn decision(&self) -> Option<&Decision> {
        match self {
            DecisionAtEnd::Decided(decision) => Some(decision),
            DecisionAtEnd::Undecided | DecisionAtEnd::RefusedWith => None,
        }
    }
}

/// What solving decides.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub values: BTreeMap<ClassName, Option<Ty>>,
    pub decisions: BTreeMap<Name, DecisionAtEnd>,
    /// What each use became, read off its choice (`State::chosen`).
    pub choices: BTreeMap<Name, Option<Chosen>>,
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
    /// decision, what each use became, each check, each refusal and whether
    /// the program is accepted. Which classes are one is no part of it.
    pub fn same_typing(&self, other: &Outcome) -> bool {
        self.values == other.values
            && self.decisions == other.decisions
            && self.choices == other.choices
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
            .filter(|occurrence| !self.classes.items()[occurrence.at].dropped)
            .map(|occurrence| match &self.classes.items()[occurrence.at].item {
                Item::Relation(relation) => Named {
                    relation,
                    arena: &self.arena,
                }
                .to_string(),
                Item::Disjunction { name, .. } => format!("disjunctive constraint {name}"),
                Item::Instance { at, .. } => format!("instance {at}"),
                Item::Lambda { name, .. } => format!("lambda {name}"),
                Item::Choice { name, .. } => format!("choice {name}"),
            })
            .collect();
        Refusal {
            members,
            items,
            cause,
        }
    }

    /// The read-off candidate at the end: each use to which several
    /// candidates remain once solving ends, and which the never rule did not
    /// make `never`, refused `ambiguous`, whether or not the variables of its
    /// generalization have values. Its members are those of each variable
    /// `Generalized(k)`, and its items the use's item, `choice n`, and each
    /// such variable. It chooses no candidate.
    fn ambiguous_uses(&self) -> Vec<Refusal> {
        self.classes
            .choices()
            .filter(|(_, name, of, choice)| {
                choice.remaining.len() > 1
                    && self.chosen_of(name, of, choice, &BTreeSet::new()).is_none()
            })
            .map(|(_, name, _, _)| {
                let positions = self.generalized(name);
                let members = positions
                    .iter()
                    .flat_map(|&class| self.classes.held(class).members.iter())
                    .map(|&member| self.arena.name(member).clone())
                    .collect();
                let items = std::iter::once(format!("choice {name}"))
                    .chain(
                        positions
                            .iter()
                            .map(|&class| self.class_name(class).to_string()),
                    )
                    .collect();
                Refusal {
                    members,
                    items,
                    cause: Cause::Ambiguous,
                }
            })
            .collect()
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
                        .expect("a conversion writes the structure of the class it reaches");
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
        refusals.extend(self.ambiguous_uses());
        for root in self.classes.roots() {
            if self.value(root).is_none() && !self.has_no_value(root) {
                refusals.insert(self.refusal(root, Cause::NoValue));
            }
        }
        let decisions: BTreeMap<Name, DecisionAtEnd> = self
            .decisions()
            .into_iter()
            .map(|(name, decision)| {
                let at_end = match decision {
                    Some(decision) => DecisionAtEnd::Decided(decision),
                    None if self.refused_with(&name) => DecisionAtEnd::RefusedWith,
                    None => DecisionAtEnd::Undecided,
                };
                (name, at_end)
            })
            .collect();
        let choices = self.choices();
        let accepted = refusals.is_empty()
            && checks.iter().all(Check::holds)
            && decisions
                .values()
                .all(|decision| decision.decision().is_some())
            && choices.values().all(Option::is_some);
        Outcome {
            values,
            decisions,
            choices,
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
            for (name, decision) in decided {
                let previous = outcome
                    .decisions
                    .insert(name.clone(), DecisionAtEnd::Decided(decision.clone()));
                assert!(
                    previous.is_none_or(|previous| previous == DecisionAtEnd::Decided(decision)),
                    "two groups decide the disjunctive constraint {name} apart"
                );
            }
            // A use is one: a use of several candidates is recorded whole
            // in one group, and a use of one candidate, recorded in each
            // group that records a position of it, became that candidate in
            // each.
            let chosen: Vec<(Name, Chosen)> = group
                .choices()
                .into_iter()
                .filter_map(|(name, chosen)| Some((name, chosen?)))
                .collect();
            for (name, became) in chosen {
                let previous = outcome.choices.insert(name.clone(), Some(became.clone()));
                assert!(
                    previous.is_none_or(|previous| previous == Some(became)),
                    "two groups choose for the use {name} apart"
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
