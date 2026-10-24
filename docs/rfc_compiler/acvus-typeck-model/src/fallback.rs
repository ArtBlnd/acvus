//! Fallback: once every binding group is at its fixpoint, each
//! equivalence class left unsolved with no unbroken dependency takes what
//! its range falls back to, its family's default or `!`, or `!` where it holds
//! none. The value it takes is recorded as a decision, which every read after
//! it takes as given.

use crate::class::{ClassId, KindSource};
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::Term;

/// An equivalence class fallen back: the default it took, and the relation
/// that records it, with what unification did as it took it and the refusal
/// where it could not take it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FellBack {
    pub class: ClassId,
    pub default: Term<ClassId>,
    pub recorded: Recording,
}

impl State {
    /// The equivalence classes left to fall back to a default take it, each
    /// recorded by fallback: those that are not solved and have no unbroken
    /// dependency.
    pub(crate) fn take_defaults(&mut self) -> Vec<FellBack> {
        let mut fell = Vec::new();
        for root in self.fallback_targets() {
            let default = match &self.classes.held(root).range {
                Some(range) => range.fallback(),
                None => Term::Never,
            };
            let relation = Relation::Kind(root, default.clone(), KindSource::Solved);
            let mut recorded = Recording::default();
            recorded.push(self.record(relation, Recorder::Fallback));
            fell.push(FellBack {
                class: root,
                default,
                recorded,
            });
        }
        fell
    }

    /// The equivalence classes to fall back once nothing is eligible: those
    /// unblocked.
    pub(crate) fn fallback_targets(&self) -> Vec<ClassId> {
        let graph = self.graph();
        let met = self.met_in(&graph);
        self.unblocked(&graph, &met)
    }
}
