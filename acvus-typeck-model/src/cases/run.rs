//! A case's program solved as its orders say, each final state handed to the
//! checks of notes 14 and 15 and to the case's changed certificates
//! (`design/19-cases.md`, The runner).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::certificate::Certificate;
use crate::check::Outcome;
use crate::class::ClassName;
use crate::explore::{Final, explore_with, sample_with};
use crate::node::Decision;
use crate::program::{ENTRY, Program};
use crate::solve::{State, Step};
use crate::ty::{Name, Ty};

use super::parse::{Orders, StepText};
use super::render::{self, Rendered};

/// The budget of states for `orders: all` (`design/13-exploration.md`).
pub const BUDGET: usize = 100_000;

/// A certificate changed as a solver that erred would leave it.
pub enum Corruption {
    Value {
        group: String,
        class: ClassName,
        ty: Ty,
    },
    Decision {
        group: String,
        node: String,
        decision: Decision,
    },
}

/// What the checker found of one corruption, over every final state.
#[derive(Default)]
pub struct Corrupted {
    /// The violations' `what`.
    pub fails: BTreeSet<String>,
    /// How many final states' corrupted certificate holds.
    pub held: usize,
}

/// A final state's outcome, its I\* violations, and the violations of its
/// certificate unchanged.
pub struct Checked {
    pub outcome: Outcome,
    pub istar: BTreeSet<String>,
    pub unsound: Vec<String>,
}

/// One step a written order took, or the start of a group's state, with its
/// phases' reports named in the state it left, and the refusals it made.
pub struct Taken {
    pub group: String,
    pub step: String,
    pub reports: Vec<Rendered>,
    pub refused: Vec<String>,
}

pub enum Explored {
    /// Every order: the states visited, and whether every order was explored
    /// within the budget.
    All {
        states: usize,
        complete: bool,
    },
    Sampled(u64),
    Written(Vec<Taken>),
}

/// A run: its outcomes, distinct, in the order first reached, and its final
/// states.
pub struct Run {
    pub explored: Explored,
    pub outcomes: Vec<Outcome>,
    pub finals: Vec<Checked>,
    pub corrupted: Vec<Corrupted>,
}

/// The checks each final state is handed to.
struct Visitor<'a> {
    program: &'a Program,
    corruptions: &'a [Corruption],
    finals: Vec<Checked>,
    corrupted: Vec<Corrupted>,
}

impl Visitor<'_> {
    fn visit(&mut self, end: &Final<'_>) {
        let unsound = crate::certificate::check(self.program, &Certificate::of(end))
            .iter()
            .map(|violation| format!("{violation:?}"))
            .collect();
        let istar = crate::istar::check(self.program, end)
            .into_iter()
            .map(|violation| render::istar(violation.kind, &violation.node))
            .collect();
        for (corruption, corrupted) in self.corruptions.iter().zip(&mut self.corrupted) {
            let mut certificate = Certificate::of(end);
            match corruption {
                Corruption::Value { group, class, ty } => {
                    certificate.set_value(group, class, ty.clone());
                }
                Corruption::Decision {
                    group,
                    node,
                    decision,
                } => certificate.set_decision(group, node, decision.clone()),
            }
            let violations = crate::certificate::check(self.program, &certificate);
            if violations.is_empty() {
                corrupted.held += 1;
            }
            corrupted
                .fails
                .extend(violations.into_iter().map(|violation| violation.what));
        }
        self.finals.push(Checked {
            outcome: end.entry.outcome_with(end.groups),
            istar,
            unsound,
        });
    }
}

pub fn run(program: &Program, orders: &Orders, corruptions: &[Corruption]) -> Result<Run, String> {
    let mut visitor = Visitor {
        program,
        corruptions,
        finals: Vec::new(),
        corrupted: corruptions.iter().map(|_| Corrupted::default()).collect(),
    };
    let (explored, outcomes) = match orders {
        Orders::All => {
            let exploration = explore_with(program, BUDGET, |end| visitor.visit(end));
            let explored = Explored::All {
                states: exploration.states,
                complete: exploration.complete,
            };
            (explored, exploration.outcomes)
        }
        Orders::Sampled(orders) => {
            let outcomes = sample_with(program, *orders, |end| visitor.visit(end));
            (Explored::Sampled(*orders), outcomes)
        }
        Orders::Written(steps) => {
            let (taken, outcome) = written(program, steps, &mut |end| visitor.visit(end))?;
            (Explored::Written(taken), vec![outcome])
        }
    };
    Ok(Run {
        explored,
        outcomes,
        finals: visitor.finals,
        corrupted: visitor.corrupted,
    })
}

/// A step a case writes, as a step the state may take now.
fn step_of(state: &State, text: &StepText) -> Result<Step, String> {
    let steps = state.steps();
    let wanted = match text {
        StepText::Solve(class) => {
            let roots: BTreeSet<_> = state
                .arena
                .classes()
                .filter(|(_, name)| name.to_string() == *class)
                .map(|(id, _)| state.root(id))
                .collect();
            let root = match roots.into_iter().collect::<Vec<_>>().as_slice() {
                [] => return Err(format!("solve {class}: no class of that name")),
                [root] => *root,
                _ => {
                    return Err(format!(
                        "solve {class}: classes of two roots have that name"
                    ));
                }
            };
            Step::Solve(root)
        }
        StepText::Decide(node) => Step::Decide(Name::new(node)),
        StepText::Fallback => return Err("fallback is no step".to_string()),
    };
    if steps.contains(&wanted) {
        Ok(wanted)
    } else {
        let listed: Vec<String> = steps.iter().map(|step| render::step(state, step)).collect();
        Err(format!(
            "{} is not a step now; the steps are [{}]",
            render::step(state, &wanted),
            listed.join(", ")
        ))
    }
}

/// The written steps of one group, each with its phases' reports.
fn take(
    state: &mut State,
    group: &str,
    steps: &[StepText],
    taken: &mut Vec<Taken>,
) -> Result<(), String> {
    for text in steps {
        let before = state.refusals.clone();
        let (step, trace) = match text {
            StepText::Fallback => {
                if group != ENTRY {
                    return Err(format!(
                        "fallback in {group}, inside which there is none (l.297)"
                    ));
                }
                if !state.steps().is_empty() {
                    return Err("fallback before the fixpoint (l.299)".to_string());
                }
                ("fallback".to_string(), state.fall_back())
            }
            _ => {
                let step = step_of(state, text)?;
                (render::step(state, &step), state.step(&step))
            }
        };
        let reports = trace
            .reports
            .iter()
            .map(|report| render::report(state, report))
            .collect();
        let refused = state
            .refusals
            .difference(&before)
            .map(render::refusal)
            .collect();
        taken.push(Taken {
            group: group.to_string(),
            step,
            reports,
            refused,
        });
    }
    Ok(())
}

/// A group's state as solving starts it, with its start taken as a step.
fn start(
    program: &Program,
    group: &Name,
    solved: &BTreeMap<Name, Rc<State>>,
    taken: &mut Vec<Taken>,
) -> State {
    let started = State::start(program, group, solved);
    let reports = started
        .trace
        .reports
        .iter()
        .map(|report| render::report(&started.state, report))
        .collect();
    taken.push(Taken {
        group: group.to_string(),
        step: "start".to_string(),
        reports,
        refused: started.state.refusals.iter().map(render::refusal).collect(),
    });
    started.state
}

/// One order: each group's written steps, then its first step until none is
/// left, the entry falling back wherever none is (l.299).
fn written<F>(
    program: &Program,
    steps: &BTreeMap<String, Vec<StepText>>,
    visit: &mut F,
) -> Result<(Vec<Taken>, Outcome), String>
where
    F: FnMut(&Final<'_>),
{
    let groups = program.group_order();
    for group in steps.keys() {
        if group != ENTRY && !groups.iter().any(|name| name.as_str() == group) {
            return Err(format!("the order of {group}, which is no group"));
        }
    }
    let mut taken = Vec::new();
    let mut solved: BTreeMap<Name, Rc<State>> = BTreeMap::new();
    for group in groups {
        let mut state = start(program, &group, &solved, &mut taken);
        if let Some(steps) = steps.get(group.as_str()) {
            take(&mut state, group.as_str(), steps, &mut taken)?;
        }
        while let Some(step) = state.steps().first().cloned() {
            state.step(&step);
        }
        solved.insert(group, Rc::new(state));
    }
    let mut state = start(program, &Name::new(ENTRY), &solved, &mut taken);
    if let Some(steps) = steps.get(ENTRY) {
        take(&mut state, ENTRY, steps, &mut taken)?;
    }
    loop {
        if let Some(step) = state.steps().first().cloned() {
            state.step(&step);
            continue;
        }
        if !state.fall_back().fell_back() {
            break;
        }
    }
    visit(&Final {
        entry: &state,
        groups: &solved,
    });
    Ok((taken, state.outcome_with(&solved)))
}
