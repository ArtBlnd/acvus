//! Every order of solving a program, up to a budget of states (I2, I3).

use std::collections::BTreeSet;

use crate::program::Program;
use crate::solve::{Outcome, State};

/// The outcomes every order of solving reaches.
#[derive(Clone, Debug)]
pub struct Exploration {
    /// The distinct outcomes, in the order first reached.
    pub outcomes: Vec<Outcome>,
    /// The states visited, each once.
    pub states: usize,
    /// Whether every order was explored within the budget.
    pub complete: bool,
}

/// Takes every step each state allows, a state reached twice explored once,
/// and falls back wherever no step is left.
pub fn explore(program: &Program, budget: usize) -> Exploration {
    let mut seen = BTreeSet::new();
    let mut outcomes: Vec<Outcome> = Vec::new();
    let mut work = vec![State::new(program)];
    let mut states = 0;
    while let Some(state) = work.pop() {
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
            if next.fall_back() {
                work.push(next);
            } else {
                let outcome = state.outcome();
                if !outcomes.contains(&outcome) {
                    outcomes.push(outcome);
                }
            }
            continue;
        }
        for step in steps {
            let mut next = state.clone();
            next.step(&step);
            work.push(next);
        }
    }
    Exploration {
        outcomes,
        states,
        complete: true,
    }
}
