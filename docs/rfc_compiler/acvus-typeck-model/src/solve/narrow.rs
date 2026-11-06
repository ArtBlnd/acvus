//! Narrowing: the candidates that remain to a use's choice, read off the
//! types the variables of its candidates' generalization will take, run as
//! part of settling.
//!
//! A candidate is never chosen by a rule: it remains while the type each
//! variable `Generalized(k)` will take reaches the candidate's type there,
//! and the candidates that remain only shrink. Where one remains, its
//! signature is recorded; where none does, the variables have no value.
//! Where an argument is `!`, the use is never carried out (the never rule).

use std::collections::BTreeSet;

use crate::check::Cause;
use crate::coerce::{coerce_read, le_read};
use crate::disjunction::generalization::Generalization;
use crate::read::ReadValue;
use crate::record::{Entry, Recording};
use crate::relation::Relation;
use crate::solve::State;
use crate::trace::{Report, Trace};
use crate::ty::Name;

impl State {
    /// Narrows the choice of each use to which several candidates remain,
    /// in the record's order (the read-off candidate):
    ///
    /// - The never rule: where an argument reads as `!` (`never_argument`,
    ///   read as `reached` reads a position), the use is never carried out.
    ///   Every candidate remains, and its result is `!` (`record_never`).
    /// - Otherwise the candidates the types its variables will take reach
    ///   remain (`reached`).
    ///
    /// Where the candidates that remain shrink, each variable's domain is
    /// read again against them, refusing a variable whose structure it no
    /// longer admits; where one remains, its signature is recorded
    /// (`record_candidate`); where none does, each variable has no value,
    /// `no_candidate`. Returns a report for each choice it changed. It
    /// prefers no candidate to another: no rank is read.
    pub(crate) fn narrow(&mut self) -> Trace {
        let mut trace = Trace::default();
        let several: Vec<Name> = self
            .classes
            .choices()
            .filter(|(_, _, _, choice)| choice.remaining.len() > 1)
            .map(|(_, name, _, _)| name.clone())
            .collect();
        for name in several {
            let (of, choice) = self.choice(&name);
            let never = self.never_argument(&name, of, &BTreeSet::new());
            let before = choice.remaining.clone();
            if never {
                let recorded = self.record_never(&name);
                if !recorded.is_empty() {
                    trace.push(self.narrowed(&name, recorded));
                }
                continue;
            }
            let remaining = self.reached(&name);
            if remaining == before {
                continue;
            }
            let at = self.choice_at(&name);
            self.classes.narrow(at, remaining.clone());
            let mut recorded = self.domains_read_again(&name);
            match remaining.len() {
                0 => {
                    for class in self.generalized(&name) {
                        self.classes.refuse(class, Cause::NoCandidate);
                    }
                }
                1 => recorded.append(self.record_candidate(&name)),
                _ => {}
            }
            trace.push(self.narrowed(&name, recorded));
        }
        trace
    }

    /// The report of a choice narrowed: the candidates that remain now, what
    /// the use became, and what that recorded.
    fn narrowed(&self, name: &Name, recorded: Recording) -> Report {
        let (_, choice) = self.choice(name);
        Report::Narrowed {
            choice: name.clone(),
            remaining: choice
                .remaining
                .iter()
                .map(|&i| choice.candidates[i].clone())
                .collect(),
            chosen: self.chosen(name),
            recorded,
        }
    }

    /// The candidates that remain to the choice of the use `name` and that
    /// the type each variable `Generalized(k)` of their generalization will
    /// take reaches (the read-off candidate): at a variable a parameter
    /// writes, that type `⇝` the candidate's type there; at one the result
    /// writes, the candidate's type there `≤` that type; both where both
    /// write it.
    ///
    /// The type a variable will take is its reading without what the use
    /// itself records (`read`, leaving out `name`), so that a variable the
    /// result writes, which waits for the candidate's type, is read by its
    /// other relations. A variable that reads undecided keeps every
    /// candidate; one that reads `!` is checked as any other reading, `! ⇝`
    /// every type at a parameter and no candidate's type `≤ !` at the result;
    /// a variable of the candidate's own reaches it. The coercion takes no
    /// conversion, and every candidate it reaches remains, whatever its rank.
    fn reached(&self, name: &Name) -> BTreeSet<usize> {
        let (_, choice) = self.choice(name);
        let sigs = self.signatures(choice);
        let part = Generalization::of(&sigs.iter().collect::<Vec<_>>());
        let left_out = BTreeSet::from([name.clone()]);
        let mut remaining = choice.remaining.clone();
        for (k, class) in self.generalized(name).into_iter().enumerate() {
            if self.has_no_value(class) {
                continue;
            }
            let value = self.read(class, &left_out);
            if matches!(value, ReadValue::Undecided) {
                continue;
            }
            let (at_param, at_result) = (part.writes_in_params(k), part.writes_in_result(k));
            remaining.retain(|&i| {
                let written = ReadValue::of_sig(&part.vars[k][i]);
                (!at_param || coerce_read(&self.decls, &value, &written).is_ok())
                    && (!at_result || le_read(&self.decls, &written, &value).is_ok())
            });
        }
        remaining
    }

    /// Each domain the choice of the use `name` gives a variable, read again
    /// against the candidates that remain now, as the classes took it when
    /// it was recorded: a variable whose structure it no longer admits has no
    /// value, a conflict. Returns the relations whose reading changed a
    /// class.
    fn domains_read_again(&mut self, name: &Name) -> Recording {
        let domains: Vec<(usize, Relation)> = self
            .classes
            .items()
            .iter()
            .enumerate()
            .filter_map(|(at, recorded)| match recorded.relation() {
                Some(relation @ Relation::GeneralizedDomain(_, domain))
                    if domain.choice == *name =>
                {
                    Some((at, relation.clone()))
                }
                _ => None,
            })
            .collect();
        let mut recorded = Recording::default();
        for (at, relation) in domains {
            let unification = self.hold(&relation);
            if unification.refused.is_some() || !unification.unified.unions.is_empty() {
                recorded.push(Entry { at, unification });
            }
        }
        recorded
    }
}
