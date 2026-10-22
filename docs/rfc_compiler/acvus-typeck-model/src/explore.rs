//! Every order of solving a program, up to a budget of states (I2, I3), each
//! final state handed on with the order that reached it.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::Outcome;
use crate::program::{ENTRY, Program};
use crate::solve::{State, Step};
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

/// An order of solving (I2): the steps that took each binding group other
/// than the entry to its fixpoint, in the order the groups are solved
/// (l.213), then the entry's steps to each of its fixpoints, a fallback that
/// fell a class back taken between one fixpoint and the next steps (l.288).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Order {
    pub groups: Vec<(Name, Vec<Step>)>,
    pub entry: Vec<Vec<Step>>,
}

impl Order {
    /// The order of no step, before any group is solved.
    pub fn new() -> Order {
        Order {
            groups: Vec::new(),
            entry: vec![Vec::new()],
        }
    }

    /// Takes `step` next in the entry.
    pub fn push_step(&mut self, step: Step) {
        self.entry
            .last_mut()
            .expect("the entry's steps have a fixpoint to reach")
            .push(step);
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
/// fall back (l.288), with each group solved before it at its fixpoint
/// (l.213), and the order that reached it.
pub struct Final<'a> {
    pub entry: &'a State,
    pub groups: &'a BTreeMap<Name, Rc<State>>,
    pub order: &'a Order,
}

/// Solves each binding group other than the entry to its fixpoint in every
/// order, before the groups that make instances of it (l.213), then the entry
/// in every order, falling back wherever no step is left (l.288). A state
/// reached twice is explored once, along the order that reached it first.
pub fn explore(program: &Program, budget: usize) -> Exploration {
    explore_with(program, budget, |_| {})
}

/// As `explore`, handing each distinct final state to `visit` before it is
/// reduced to an outcome (`design/13-exploration.md`, Checks).
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
            for (fixpoint, steps) in fixpoints {
                let mut solved = solved.clone();
                solved.insert(group.clone(), Rc::new(fixpoint));
                let mut order = order.clone();
                order.groups.push((group.clone(), steps));
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
            let steps = state.steps();
            if steps.is_empty() {
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
            for step in steps {
                let mut next = state.clone();
                next.step(&step);
                let mut order = order.clone();
                order.push_step(step);
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
/// fixpoint, inside which there is no fallback (l.286), each with the steps
/// that first reached it, or none where the budget runs out.
fn fixpoints(start: State, budget: usize, states: &mut usize) -> Option<Vec<(State, Vec<Step>)>> {
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
        let steps = state.steps();
        if steps.is_empty() {
            if reached.insert(state.fingerprint()) {
                fixpoints.push((state, taken));
            }
            continue;
        }
        for step in steps {
            let mut next = state.clone();
            next.step(&step);
            let mut taken = taken.clone();
            taken.push(step);
            work.push((next, taken));
        }
    }
    Some(fixpoints)
}

/// The outcomes of distinct typings `orders` orders of solving reach (as
/// `Exploration::outcomes`), each taking at
/// every state the step its own sequence picks: the first step always, the
/// last always, then sequences a fixed generator gives. It checks I2 on a
/// sample, for programs with too many states to explore.
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
            let mut steps_taken = Vec::new();
            loop {
                let steps = state.steps();
                if steps.is_empty() {
                    break;
                }
                let step = steps[picks.next(steps.len())].clone();
                state.step(&step);
                steps_taken.push(step);
            }
            taken.groups.push((group.clone(), steps_taken));
            solved.insert(group, Rc::new(state));
        }
        let mut state = State::of_group(program, &Name::new(ENTRY), &solved);
        loop {
            let steps = state.steps();
            if !steps.is_empty() {
                let step = steps[picks.next(steps.len())].clone();
                state.step(&step);
                taken.push_step(step);
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

/// The steps one sampled order picks.
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

    fn next(&mut self, steps: usize) -> usize {
        match self.order {
            0 => 0,
            1 => steps - 1,
            _ => {
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state ^= self.state << 17;
                (self.state % steps as u64) as usize
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
                let kind = held.kind.as_ref().map(|(kind, source)| {
                    let kind = kind.clone().map(|_, part| self.class_name(part));
                    format!("{kind:?}/{source:?}")
                });
                format!(
                    "{}:{kind:?}:{:?}:{}:{}",
                    self.class_name(root),
                    held.lit,
                    held.solved,
                    self.has_no_value(root),
                )
            })
            .collect();
        roots.sort();
        let decisions: Vec<_> = self.nodes.values().map(|node| &node.decision).collect();
        format!(
            "{roots:?}|{}|{:?}|{:?}|{decisions:?}|{:?}",
            self.log.len(),
            self.split,
            self.refusals,
            self.deferred,
        )
    }
}
