//! Every order of solving a program, up to a budget of states, each final
//! state handed on with the order that reached it.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::Outcome;
use crate::judgment::Judgment;
use crate::program::{ENTRY, Program};
use crate::solve::State;
use crate::ty::Name;

/// The outcomes every order of solving reaches.
#[derive(Clone, Debug)]
pub struct Exploration {
    /// The outcomes of distinct typings (`Outcome::same_typing`), each as
    /// first reached.
    pub outcomes: Vec<Outcome>,
    /// The states visited, each once.
    pub states: usize,
    /// Whether every order was explored within the budget.
    pub complete: bool,
}

/// An order of solving: the judgments that took each binding group other
/// than the entry to its fixpoint, in the order the groups are solved,
/// then the entry's judgments to each of its fixpoints, a fallback that
/// fell a class back taken between one fixpoint and the next judgments.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Order {
    pub groups: Vec<(Name, Vec<Judgment>)>,
    pub entry: Vec<Vec<Judgment>>,
}

impl Order {
    /// The order of no judgment, before any group is solved.
    pub fn new() -> Order {
        Order {
            groups: Vec::new(),
            entry: vec![Vec::new()],
        }
    }

    /// Takes `judgment` next in the entry.
    pub fn push_judgment(&mut self, judgment: Judgment) {
        self.entry
            .last_mut()
            .expect("the entry's judgments have a fixpoint to reach")
            .push(judgment);
    }

    /// Takes a fallback next in the entry, which fell a class back.
    pub fn push_fallback(&mut self) {
        self.entry.push(Vec::new());
    }
}

impl Default for Order {
    fn default() -> Order {
        Order::new()
    }
}

/// A final state of an order of solving: the entry's, once nothing is left to
/// fall back, with each group solved before it at its fixpoint,
/// and the order that reached it.
pub struct Final<'a> {
    pub entry: &'a State,
    pub groups: &'a BTreeMap<Name, Rc<State>>,
    pub order: &'a Order,
}

/// Solves each binding group other than the entry to its fixpoint in every
/// order, before the groups that make instances of it, then the entry
/// in every order, falling back wherever no judgment is left. A state
/// reached twice is explored once, along the order that reached it first.
pub fn explore(program: &Program, budget: usize) -> Exploration {
    explore_with(program, budget, |_| {})
}

/// As `explore`, handing each distinct final state to `visit` before it is
/// reduced to an outcome.
pub fn explore_with<F>(program: &Program, budget: usize, mut visit: F) -> Exploration
where
    F: FnMut(&Final<'_>),
{
    let mut choices: Vec<(BTreeMap<Name, Rc<State>>, Order)> =
        vec![(BTreeMap::new(), Order::new())];
    let mut states = 0;
    for group in program.group_order() {
        let mut next = Vec::new();
        for (solved, order) in &choices {
            let start = State::of_group(program, &group, solved);
            let Some(fixpoints) = fixpoints(start, budget, &mut states) else {
                return Exploration {
                    outcomes: Vec::new(),
                    states,
                    complete: false,
                };
            };
            for (fixpoint, judgments) in fixpoints {
                let mut solved = solved.clone();
                solved.insert(group.clone(), Rc::new(fixpoint));
                let mut order = order.clone();
                order.groups.push((group.clone(), judgments));
                next.push((solved, order));
            }
        }
        choices = next;
    }
    let mut outcomes: Vec<Outcome> = Vec::new();
    for (solved, order) in &choices {
        let start = State::of_group(program, &Name::new(ENTRY), solved);
        let mut seen = BTreeSet::new();
        let mut work = vec![(start, order.clone())];
        while let Some((state, order)) = work.pop() {
            if !seen.insert(state.fingerprint()) {
                continue;
            }
            states += 1;
            if states > budget {
                return Exploration {
                    outcomes,
                    states,
                    complete: false,
                };
            }
            let judgments = state.judgments();
            if judgments.is_empty() {
                let mut next = state.clone();
                if next.fall_back().fell_back() {
                    let mut order = order.clone();
                    order.push_fallback();
                    work.push((next, order));
                } else {
                    visit(&Final {
                        entry: &state,
                        groups: solved,
                        order: &order,
                    });
                    let outcome = state.outcome_with(solved);
                    if !outcomes.iter().any(|reached| reached.same_typing(&outcome)) {
                        outcomes.push(outcome);
                    }
                }
                continue;
            }
            for judgment in judgments {
                let mut next = state.clone();
                next.step(&judgment);
                let mut order = order.clone();
                order.push_judgment(judgment);
                work.push((next, order));
            }
        }
    }
    Exploration {
        outcomes,
        states,
        complete: true,
    }
}

/// The distinct states at which a binding group's solving reaches its
/// fixpoint, inside which there is no fallback, each with the judgments
/// that first reached it, or none where the budget runs out.
fn fixpoints(
    start: State,
    budget: usize,
    states: &mut usize,
) -> Option<Vec<(State, Vec<Judgment>)>> {
    let mut seen = BTreeSet::new();
    let mut reached = BTreeSet::new();
    let mut fixpoints = Vec::new();
    let mut work = vec![(start, Vec::new())];
    while let Some((state, taken)) = work.pop() {
        if !seen.insert(state.fingerprint()) {
            continue;
        }
        *states += 1;
        if *states > budget {
            return None;
        }
        let judgments = state.judgments();
        if judgments.is_empty() {
            if reached.insert(state.fingerprint()) {
                fixpoints.push((state, taken));
            }
            continue;
        }
        for judgment in judgments {
            let mut next = state.clone();
            next.step(&judgment);
            let mut taken = taken.clone();
            taken.push(judgment);
            work.push((next, taken));
        }
    }
    Some(fixpoints)
}

/// The outcomes of distinct typings `orders` orders of solving reach (as
/// `Exploration::outcomes`), each taking at
/// every state the judgment its own sequence picks: the first judgment always, the
/// last always, then sequences a fixed generator gives.
pub fn sample(program: &Program, orders: u64) -> Vec<Outcome> {
    sample_with(program, orders, |_| {})
}

/// As `sample`, handing each order's final state, with the order, to
/// `visit`.
pub fn sample_with<F>(program: &Program, orders: u64, mut visit: F) -> Vec<Outcome>
where
    F: FnMut(&Final<'_>),
{
    let mut outcomes: Vec<Outcome> = Vec::new();
    for order in 0..orders {
        let mut picks = Picks::new(order);
        let mut taken = Order::new();
        let mut solved: BTreeMap<Name, Rc<State>> = BTreeMap::new();
        for group in program.group_order() {
            let mut state = State::of_group(program, &group, &solved);
            let mut judgments_taken = Vec::new();
            loop {
                let judgments = state.judgments();
                if judgments.is_empty() {
                    break;
                }
                let judgment = judgments[picks.next(judgments.len())].clone();
                state.step(&judgment);
                judgments_taken.push(judgment);
            }
            taken.groups.push((group.clone(), judgments_taken));
            solved.insert(group, Rc::new(state));
        }
        let mut state = State::of_group(program, &Name::new(ENTRY), &solved);
        loop {
            let judgments = state.judgments();
            if !judgments.is_empty() {
                let judgment = judgments[picks.next(judgments.len())].clone();
                state.step(&judgment);
                taken.push_judgment(judgment);
                continue;
            }
            if !state.fall_back().fell_back() {
                break;
            }
            taken.push_fallback();
        }
        visit(&Final {
            entry: &state,
            groups: &solved,
            order: &taken,
        });
        let outcome = state.outcome_with(&solved);
        if !outcomes.iter().any(|reached| reached.same_typing(&outcome)) {
            outcomes.push(outcome);
        }
    }
    outcomes
}

/// The judgments one sampled order picks.
struct Picks {
    order: u64,
    state: u64,
}

impl Picks {
    fn new(order: u64) -> Picks {
        Picks {
            order,
            state: order.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    fn next(&mut self, judgments: usize) -> usize {
        match self.order {
            0 => 0,
            1 => judgments - 1,
            _ => {
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state ^= self.state << 17;
                (self.state % judgments as u64) as usize
            }
        }
    }
}

impl State {
    /// A text of the state that names classes as the outcome does, so that
    /// two states reached apart compare by it.
    pub fn fingerprint(&self) -> String {
        let mut roots: Vec<String> = self
            .classes
            .roots()
            .map(|root| {
                let held = self.classes.held(root);
                let structure = held.structure.as_ref().map(|(structure, source)| {
                    let structure = structure.clone().map(|_, part| self.class_name(part));
                    format!("{structure:?}/{source:?}")
                });
                format!(
                    "{}:{structure:?}:{:?}:{}",
                    self.class_name(root),
                    self.classes.domain(root),
                    self.has_no_value(root),
                )
            })
            .collect();
        roots.sort();
        let decisions: Vec<_> = self.decisions().into_values().collect();
        let decomposed: Vec<usize> = self
            .classes
            .items()
            .iter()
            .enumerate()
            .filter(|(_, recorded)| recorded.decomposed)
            .map(|(at, _)| at)
            .collect();
        format!(
            "{roots:?}|{}|{decomposed:?}|{:?}|{decisions:?}",
            self.classes.items().len(),
            self.refusals(),
        )
    }
}
