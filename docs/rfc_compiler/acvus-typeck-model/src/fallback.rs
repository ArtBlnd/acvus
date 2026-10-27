//! Fallback: at the entry's fixpoint, each class the judgment leaves with
//! nothing to decide it (`Verdict::Rest`) takes a default. The classes whose
//! record gives them a domain take its default first, a literal's family's
//! default; only where no class takes one do the classes with no domain take
//! `!`. The structure taken is recorded, and every judgment after it takes
//! it as given.

use crate::class::{ClassId, StructureSource};
use crate::judgment::Verdict;
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
    /// The equivalence classes left to fall back take a default, each
    /// recorded by fallback: those with a domain its default, and, where
    /// none has one, those without `!`.
    pub(crate) fn take_defaults(&mut self) -> Vec<FellBack> {
        let rest: Vec<(ClassId, Option<Term<ClassId>>)> = self
            .unsolved()
            .into_iter()
            .filter_map(|root| match self.verdict(root) {
                Verdict::Rest(domain) => Some((root, domain.map(|domain| domain.fallback()))),
                Verdict::Join(_) | Verdict::Fill(_) | Verdict::Waiting => None,
            })
            .collect();
        let with_domain: Vec<(ClassId, Term<ClassId>)> = rest
            .iter()
            .filter_map(|(root, default)| Some((*root, default.clone()?)))
            .collect();
        let taking: Vec<(ClassId, Term<ClassId>)> = if with_domain.is_empty() {
            rest.into_iter()
                .map(|(root, _)| (root, Term::Never))
                .collect()
        } else {
            with_domain
        };
        let mut fell = Vec::new();
        for (root, default) in taking {
            let relation = Relation::Structure(root, default.clone(), StructureSource::Solved);
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
}
