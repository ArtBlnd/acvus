//! Fallback (l.297–299): once every binding group is at its fixpoint, each
//! equivalence class left unsolved with no unbroken dependency takes its
//! LitVar's family's default, or `!`.

use std::rc::Rc;

use crate::class::{ClassId, KindSource};
use crate::solve::State;
use crate::ty::Term;

impl State {
    /// The equivalence classes left to fall back to a default take it
    /// (l.299): those that are not solved and have no unbroken dependency.
    /// Returns whether any did.
    pub(crate) fn take_defaults(&mut self) -> bool {
        let targets = self.fallback_targets();
        let decls = Rc::clone(&self.decls);
        let types = decls.types();
        for &root in &targets {
            let term = match &self.classes.held(root).lit {
                Some(lit) => Term::Scalar(lit.default_scalar()),
                None => Term::Never,
            };
            if let Err(conflict) = self
                .classes
                .give_kind(root, term, KindSource::Solved, types)
            {
                let name = self.class_name(root);
                self.refuse_conflict(format!("the default of {name}"), conflict);
            }
            self.classes.mark_solved(root);
        }
        !targets.is_empty()
    }

    /// The equivalence classes to fall back once nothing is eligible: those
    /// unblocked (l.299).
    pub(crate) fn fallback_targets(&self) -> Vec<ClassId> {
        let graph = self.graph();
        let settled = self.settled_in(&graph);
        self.unblocked(&graph, &settled)
    }
}
