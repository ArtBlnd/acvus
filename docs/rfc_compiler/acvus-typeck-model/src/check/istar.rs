//! I\*: each decision against the one the judgment gives its disjunctive constraint reading
//! what is recorded once solving ends, without what the decision and the decisions that
//! depend on it recorded; and each use that became a candidate or `never` against the
//! candidates the values its positions hold once solving ends reach.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::choice::{Choice, Chosen, Use};
use crate::class::ClassId;
use crate::coerce::{coerce_read, le_read};
use crate::disjunction::{Decision, Disjunction};
use crate::explore::Final;
use crate::instance::{Instance, Made};
use crate::program::{ENTRY, Program};
use crate::read::ReadValue;
use crate::relation::{Recorder, Relation, copy_name};
use crate::solve::State;
use crate::ty::Name;

/// Which way a decision fails I\*, by the outcome it is part of.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Kind {
    /// The program is refused.
    Completeness,
    /// The program is accepted, with a typing other than the one I\* gives.
    Soundness,
}

/// A decision that is not the one its disjunctive constraint makes reading what is recorded at
/// the end, or a use whose choice the values at the end do not bear out.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Violation {
    pub kind: Kind,
    /// The binding group of the state the disjunctive constraint is decided again in, or the
    /// use is checked in.
    pub group: Name,
    /// The disjunctive constraint's or the use's name.
    pub disjunction: Name,
    pub found: Found,
}

/// What I\* found of a disjunctive constraint or a use.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Found {
    /// The decision; what it becomes decided again, none where that decides nothing; and the
    /// disjunctive constraints whose relations the decision again leaves out.
    Decision {
        decided: Decision,
        again: Option<Decision>,
        left_out: BTreeSet<Name>,
    },
    /// What the use became; the candidates that remain to its choice; and those the values
    /// its positions hold at the end reach, some of which do not remain, or several of which
    /// remain.
    Choice {
        chosen: Chosen,
        remaining: BTreeSet<usize>,
        reached: BTreeSet<usize>,
    },
}

/// Every decided disjunctive constraint of a final state, decided again in each context its
/// decision holds for.
pub fn check(program: &Program, end: &Final<'_>) -> Vec<Violation> {
    let accepted = end.entry.outcome_with(end.groups).accepted;
    let mut violations = Vec::new();
    let states = end.groups.values().map(|state| &**state).chain([end.entry]);
    for state in states {
        violations.extend(check_view(&view(state, end.groups), accepted, |_| true));
    }
    for (group, parent) in &program.groups {
        let Some(parent) = parent else {
            continue;
        };
        let enclosing = if parent.as_str() == ENTRY {
            end.entry
        } else {
            &*end.groups[parent]
        };
        let lambda = view(&end.groups[group], end.groups);
        let at = Name::new(&format!("{group}#check"));
        let mut here = enclosing.clone();
        let mut copy = Instance::new(&at, Recorder::Check, &lambda);
        copy.record_rest(&mut here);
        let made = here
            .instance(&at)
            .expect("an instance is kept where it is recorded");
        add_decided(&mut here, &made, &lambda);
        violations.extend(check_view(&here, accepted, |name| {
            name.as_str().starts_with(&format!("{at}/"))
        }));
    }
    violations
}

/// The state with, for each instance it made, a disjunctive constraint for each disjunctive constraint the
/// instantiated group decided, under the name of its copy, so that the
/// relations copied from it are that disjunctive constraint's (`Recorder::disjunctions`),
/// and a use for each use that became a candidate or `never` there, so that it is checked
/// in the instance too.
fn view(state: &State, groups: &BTreeMap<Name, Rc<State>>) -> State {
    let mut view = state.clone();
    for made in state.instances() {
        let from = self::view(&groups[&made.group], groups);
        add_decided(&mut view, &made, &from);
    }
    view
}

fn add_decided(here: &mut State, made: &Made, from: &State) {
    let mut copy = Instance::resume(here, made, from);
    // An instance of parts of C has the decided disjunctive constraints in those parts.
    let components = made.parts.as_ref().map(|_| from.components());
    for (_, name, kind, decision) in from.classes.disjunctions() {
        let Some(decision) = decision else {
            continue;
        };
        if let (Some(parts), Some(components)) = (&made.parts, &components)
            && !components
                .of_disjunction(from, kind)
                .is_some_and(|part| parts.contains(&part))
        {
            continue;
        }
        let kind = copy.kind(here, kind);
        let copied = copy_name(&made.at, name);
        let entry = here.add_disjunction(&copied, kind, Recorder::Check);
        here.classes.decide(entry.at, Some(decision.clone()));
    }
    // And a use for each use that became a candidate or `never` there.
    for (at, name, of, choice) in from.classes.choices() {
        if from.chosen_of(name, of, choice, &BTreeSet::new()).is_none() {
            continue;
        }
        if let (Some(parts), Some(components)) = (&made.parts, &components)
            && !from
                .classes
                .roles_of(at)
                .into_iter()
                .find_map(|(class, _)| components.of(from, class))
                .is_some_and(|part| parts.contains(&part))
        {
            continue;
        }
        let of = of.clone().map(|class| copy.class(here, class));
        let copied = copy_name(&made.at, name);
        // A use's choice relates no class as it is recorded.
        let _ = here.add_choice(&copied, of, choice.clone(), Recorder::Check);
    }
}

/// Decides again each decided disjunctive constraint `subject` admits, and checks each use
/// `subject` admits that became a candidate or `never` (`check_choice`), in a final state
/// whose program is `accepted` or not. A disjunctive constraint a class of which it reads has
/// no value is refused with that class (`State::refused_with`, D5): it has nothing to read,
/// and is not decided again.
fn check_view<F>(view: &State, accepted: bool, subject: F) -> Vec<Violation>
where
    F: Fn(&Name) -> bool,
{
    let graph = EndGraph::of(view);
    let fails_as = if accepted {
        Kind::Soundness
    } else {
        Kind::Completeness
    };
    let mut violations = Vec::new();
    for (_, name, kind, decision) in view.classes.disjunctions() {
        let Some(decided) = decision.cloned() else {
            continue;
        };
        if !subject(name) || view.refused_with(name) {
            continue;
        }
        let left_out = graph.exclusion(name);
        let again = match kind {
            Disjunction::Bind { path, .. } if path.iter().any(|step| left_out.contains(step)) => {
                let mut cut = view.clone();
                for step in path.iter().filter(|step| left_out.contains(*step)) {
                    let at = cut.disjunction_at(step);
                    cut.classes.decide(at, None);
                }
                cut.decision(name, &left_out)
            }
            _ => view.decision(name, &left_out),
        };
        if again.as_ref() == Some(&decided) {
            continue;
        }
        violations.push(Violation {
            kind: fails_as,
            group: view.group.clone(),
            disjunction: name.clone(),
            found: Found::Decision {
                decided,
                again,
                left_out,
            },
        });
    }
    for (_, name, of, choice) in view.classes.choices() {
        if !subject(name) {
            continue;
        }
        if let Some(found) = check_choice(view, name, of, choice) {
            violations.push(Violation {
                kind: fails_as,
                group: view.group.clone(),
                disjunction: name.clone(),
                found,
            });
        }
    }
    violations
}

/// The independent check of a use that became a candidate or `never`: every candidate the
/// values its positions hold at the end reach (`reached`) remains, and one candidate
/// remains, or, an argument being `!`, the never rule keeps every one. None where it holds, or
/// the use became nothing.
fn check_choice(view: &State, name: &Name, of: &Use, choice: &Choice) -> Option<Found> {
    let chosen = view.chosen_of(name, of, choice, &BTreeSet::new())?;
    let reached = reached(view, of, choice);
    let one = choice.remaining.len() == 1 || chosen == Chosen::Never;
    (!reached.is_subset(&choice.remaining) || !one).then(|| Found::Choice {
        chosen,
        remaining: choice.remaining.clone(),
        reached,
    })
}

/// The candidates of the use's choice the values its positions hold at the end reach: each
/// argument's value `⇝` the candidate's parameter, and the candidate's result `≤` the
/// result's value, by the coercion table, taking no conversion and no rank. A position with
/// no value, and a variable of the candidate's own, decide nothing.
fn reached(view: &State, of: &Use, choice: &Choice) -> BTreeSet<usize> {
    let read = |class: ClassId| {
        view.value(class)
            .map_or(ReadValue::Undecided, |value| ReadValue::of_ty(&value))
    };
    view.signatures(choice)
        .iter()
        .enumerate()
        .filter(|(_, sig)| {
            let arguments = of.args.iter().zip(sig.params()).all(|(arg, param)| {
                arg.is_none_or(|arg| {
                    coerce_read(&view.decls, &read(arg), &ReadValue::of_sig(param)).is_ok()
                })
            });
            let result = of.result.is_none_or(|result| {
                le_read(&view.decls, &ReadValue::of_sig(sig.result()), &read(result)).is_ok()
            });
            arguments && result
        })
        .map(|(i, _)| i)
        .collect()
}

/// The dependency at the end: for each
/// disjunctive constraint, the disjunctive constraints it depends on.
struct EndGraph {
    on: BTreeMap<Name, BTreeSet<Name>>,
}

impl EndGraph {
    fn of(view: &State) -> EndGraph {
        let reach = Reach::of(view);
        let names: Vec<Name> = view
            .classes
            .disjunctions()
            .map(|(_, name, _, _)| name.clone())
            .collect();
        let records: BTreeMap<Name, BTreeSet<ClassId>> = names
            .iter()
            .map(|name| (name.clone(), records(view, name)))
            .collect();
        let on = names
            .iter()
            .map(|name| (name.clone(), depends_on(view, &reach, &records, name)))
            .collect();
        EndGraph { on }
    }

    /// `n` and every disjunctive constraint that transitively depends on it.
    fn exclusion(&self, n: &Name) -> BTreeSet<Name> {
        let mut left_out = BTreeSet::from([n.clone()]);
        let mut work = vec![n.clone()];
        while let Some(at) = work.pop() {
            for (disjunction, on) in &self.on {
                if on.contains(&at) && left_out.insert(disjunction.clone()) {
                    work.push(disjunction.clone());
                }
            }
        }
        left_out
    }
}

/// The disjunctive constraints `name` depends on at the end: those that record into a class
/// reached from the classes it reads, and a binding's path.
fn depends_on(
    view: &State,
    reach: &Reach,
    records: &BTreeMap<Name, BTreeSet<ClassId>>,
    name: &Name,
) -> BTreeSet<Name> {
    let (kind, _) = view.disjunction(name);
    let reached = reach.from(view, view.reads(name));
    let mut on: BTreeSet<Name> = records
        .iter()
        .filter(|(other, records)| *other != name && !records.is_disjoint(&reached))
        .map(|(other, _)| other.clone())
        .collect();
    if let Disjunction::Bind { path, .. } = kind {
        for step in path {
            assert!(
                view.classes.disjunction_at(step).is_some(),
                "a binding's path is of the view's disjunctive constraints"
            );
            on.insert(step.clone());
        }
    }
    on
}

/// The roots of the classes the disjunctive constraint records into at the end: a decided
/// one's, what the relations it recorded record into; one not decided, what a decision
/// still open to it would record into.
fn records(view: &State, name: &Name) -> BTreeSet<ClassId> {
    let classes: BTreeSet<ClassId> = match view.disjunction(name).1 {
        Some(_) => view
            .classes
            .items()
            .iter()
            .filter(|recorded| recorded.recorder.disjunctions().contains(name))
            .filter_map(|recorded| recorded.relation())
            .flat_map(into)
            .collect(),
        None => view.records_into(name),
    };
    classes.into_iter().map(|class| view.root(class)).collect()
}

/// The classes a relation records into: the upper side of a lower bound or a
/// part lower bound, the type passed into at a coercion constraint, both sides of an
/// equation, and the class a structure, a structure constraint, a domain or a lambda is given to.
fn into(relation: &Relation) -> Vec<ClassId> {
    match relation {
        Relation::LowerBound { upper, .. } | Relation::PartLowerBound { upper, .. } => {
            vec![*upper]
        }
        Relation::CoercionConstraint { to, .. } | Relation::Wrap { to, .. } => vec![*to],
        Relation::Equation(a, b) => vec![*a, *b],
        Relation::StructureConstraint(class, _)
        | Relation::Structure(class, ..)
        | Relation::Domain(class, _)
        | Relation::GeneralizedDomain(class, _)
        | Relation::Values(class, _)
        | Relation::Standing(class, _) => vec![*class],
    }
}

/// Reach at the end: from a class to each
/// class it has as a lower bound or a part lower bound, along every one
/// recorded, a decomposition's among them. Every class has taken its constructor by
/// the end.
struct Reach {
    edges: BTreeMap<ClassId, BTreeSet<ClassId>>,
}

impl Reach {
    fn of(view: &State) -> Reach {
        let mut edges: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        for recorded in view.classes.items() {
            if let Some(
                Relation::LowerBound { lower, upper } | Relation::PartLowerBound { lower, upper },
            ) = recorded.relation()
            {
                let (lower, upper) = (*lower, *upper);
                let (lower, upper) = (view.root(lower), view.root(upper));
                if lower != upper {
                    edges.entry(upper).or_default().insert(lower);
                }
            }
        }
        Reach { edges }
    }

    fn from<I>(&self, view: &State, classes: I) -> BTreeSet<ClassId>
    where
        I: IntoIterator<Item = ClassId>,
    {
        let mut reached: BTreeSet<ClassId> =
            classes.into_iter().map(|class| view.root(class)).collect();
        let mut work: Vec<ClassId> = reached.iter().copied().collect();
        while let Some(at) = work.pop() {
            for &next in self.edges.get(&at).into_iter().flatten() {
                if reached.insert(next) {
                    work.push(next);
                }
            }
        }
        reached
    }
}
