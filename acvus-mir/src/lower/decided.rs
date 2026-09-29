//! What typeck decided, as one body being lowered reads it (functions §2):
//! the run's own decisions, or, for a lambda body lowered at an instance of
//! a generalized binding, that instance's, then the instances it was made
//! within, outward, then the run's. Every form reads its site's type and
//! decisions through this view; none reads `Typeck`'s site tables directly.

use acvus_ast::AstId;

use crate::ty::Ty;
use crate::typeck::{Adjustment, AsCast, Chosen, Coercion, InstanceDecided, InstanceId, Iteration, StructureTree, TryExit, Typeck};

/// A view of `Typeck` at one instance, or at none.
#[derive(Clone, Copy)]
pub(super) struct Decided<'a> {
    pub typeck: &'a Typeck,
    pub instance: Option<InstanceId>,
}

impl<'a> Decided<'a> {
    /// The run's own view.
    pub fn run(typeck: &'a Typeck) -> Decided<'a> {
        Decided { typeck, instance: None }
    }

    /// `get` of the view's instance, then of each instance it was made
    /// within, outward, then `run` of the run's tables.
    fn read<T>(&self, get: impl Fn(&'a InstanceDecided) -> Option<T>, run: impl FnOnce(&'a Typeck) -> Option<T>) -> Option<T> {
        let mut at = self.instance;
        while let Some(instance) = at {
            let decided = self
                .typeck
                .instances
                .get(&instance)
                .unwrap_or_else(|| panic!("typeck records every instance lower is at, and not {instance:?}"));
            if let Some(found) = get(decided) {
                return Some(found);
            }
            at = decided.within;
        }
        run(self.typeck)
    }

    /// The type at `site`.
    ///
    /// # Panics
    /// No table the view reads types `site`: every site of an admitted
    /// program is typed at every instance it is lowered at.
    pub fn ty(&self, site: AstId) -> Ty {
        self.read(|at| at.types.get(&site).copied(), |run| run.types.get(&site).copied())
            .unwrap_or_else(|| panic!("typeck types every site of an admitted program, and not {site:?}"))
    }

    pub fn reached(&self, site: AstId) -> Option<&'a Chosen> {
        self.read(|at| at.reached.get(&site), |run| run.reached.get(&site))
    }

    pub fn callee(&self, site: AstId) -> Option<Ty> {
        self.read(|at| at.callees.get(&site).copied(), |run| run.callees.get(&site).copied())
    }

    pub fn coercion(&self, site: AstId) -> Option<&'a Coercion> {
        self.read(|at| at.coercions.get(&site), |run| run.coercions.get(&site))
    }

    pub fn structure_tree(&self, site: AstId) -> Option<&'a StructureTree> {
        self.read(|at| at.structure_trees.get(&site), |run| run.structure_trees.get(&site))
    }

    pub fn adjustment(&self, site: AstId) -> Option<Adjustment> {
        self.read(|at| at.adjustments.get(&site).copied(), |run| run.adjustments.get(&site).copied())
    }

    pub fn iteration(&self, site: AstId) -> Option<Iteration> {
        self.read(|at| at.iteration.get(&site).copied(), |run| run.iteration.get(&site).copied())
    }

    pub fn try_exit(&self, site: AstId) -> Option<TryExit<Ty>> {
        self.read(|at| at.tries.get(&site).copied(), |run| run.tries.get(&site).copied())
    }

    pub fn cast(&self, site: AstId) -> Option<AsCast> {
        self.read(|at| at.casts.get(&site).copied(), |run| run.casts.get(&site).copied())
    }

    /// The same view at `instance`, a lambda body's instance within this
    /// one.
    pub fn at(&self, instance: InstanceId) -> Decided<'a> {
        Decided { typeck: self.typeck, instance: Some(instance) }
    }
}
