//! Fallback (l.286–288): once every binding group is at its fixpoint, each
//! equivalence class left unsolved with no unbroken dependency takes its
//! LitVar's family's default, or `!`.

use std::rc::Rc;

use crate::check::Refusal;
use crate::class::{ClassId, KindSource, Unified};
use crate::solve::State;
use crate::ty::Term;

/// An equivalence class fallen back (l.288): the default it took, with what
/// unification did as it took it, and the refusal where it could not take it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FellBack {
    pub class: ClassId,
    pub default: Term<ClassId>,
    pub unified: Unified,
    pub refused: Option<Refusal>,
}

impl State {
    /// The equivalence classes left to fall back to a default take it
    /// (l.288): those that are not solved and have no unbroken dependency.
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
            let unified =
                self.classes
                    .unifying(types)
                    .give_kind(root, term.clone(), KindSource::Solved);
            let refused = unified.result().err().map(|conflict| {
                let name = self.class_name(root);
                self.refuse_conflict(format!("the default of {name}"), conflict)
            });
            self.classes.mark_solved(root);
            fell.push(FellBack {
                class: root,
                default: term,
                unified,
                refused,
            });
        }
        fell
    }

    /// The equivalence classes to fall back once nothing is eligible: those
    /// unblocked (l.288).
    pub(crate) fn fallback_targets(&self) -> Vec<ClassId> {
        let graph = self.graph();
        let met = self.met_in(&graph);
        self.unblocked(&graph, &met)
    }
}
