//! Fallback (l.297–299): once every binding group is at its fixpoint, each
//! equivalence class left unsolved with no unbroken dependency takes its
//! LitVar's family's default, or `!`.

use std::rc::Rc;

use crate::check::Refusal;
use crate::class::{ClassId, KindSource};
use crate::solve::State;
use crate::ty::Term;

/// An equivalence class fallen back (l.299): the default it took, with the
/// refusal where it could not take it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FellBack {
    pub class: ClassId,
    pub default: Term<ClassId>,
    pub refused: Option<Refusal>,
}

impl State {
    /// The equivalence classes left to fall back to a default take it
    /// (l.299): those that are not solved and have no unbroken dependency.
    pub(crate) fn take_defaults(&mut self) -> Vec<FellBack> {
        let targets = self.fallback_targets();
        let decls = Rc::clone(&self.decls);
        let types = decls.types();
        let mut fell = Vec::new();
        for root in targets {
            let term = match &self.classes.held(root).lit {
                Some(lit) => Term::Scalar(lit.default_scalar()),
                None => Term::Never,
            };
            let refused =
                match self
                    .classes
                    .give_kind(root, term.clone(), KindSource::Solved, types)
                {
                    Ok(()) => None,
                    Err(conflict) => {
                        let name = self.class_name(root);
                        Some(self.refuse_conflict(format!("the default of {name}"), conflict))
                    }
                };
            self.classes.mark_solved(root);
            fell.push(FellBack {
                class: root,
                default: term,
                refused,
            });
        }
        fell
    }

    /// The equivalence classes to fall back once nothing is eligible: those
    /// unblocked (l.299).
    pub(crate) fn fallback_targets(&self) -> Vec<ClassId> {
        let graph = self.graph();
        let settled = self.settled_in(&graph);
        self.unblocked(&graph, &settled)
    }
}
