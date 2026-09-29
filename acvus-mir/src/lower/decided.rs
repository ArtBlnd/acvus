//! What typeck decided, as one body being lowered reads it (functions §2):
//! the run's own decisions, or, for a lambda body lowered at an instance of
//! a generalized binding, that instance's, then the instances it was made
//! within, outward, then the run's. Every form reads its site's type and
//! decisions through this view; none reads `Typeck`'s site tables directly.

use acvus_ast::AstId;

use crate::ty::Ty;
use crate::typeck::{Adjustment, AsCast, Chosen, Coercion, InstanceId, Iteration, StructureTree, TryExit, Typeck};

/// A view of `Typeck` at one instance, or at none.
#[derive(Clone, Copy)]
pub(super) struct Decided<'a> {
    pub typeck: &'a Typeck,
    pub instance: Option<InstanceId>,
}

impl<'a> Decided<'a> {
    /// The type at `site`.
    ///
    /// # Panics
    /// No table the view reads types `site`: every site of an admitted
    /// program is typed at every instance it is lowered at.
    pub fn ty(&self, site: AstId) -> Ty {
        todo!("the instance's table, then each within, then the run's")
    }

    pub fn reached(&self, site: AstId) -> Option<&'a Chosen> {
        todo!("as `ty`")
    }

    pub fn callee(&self, site: AstId) -> Option<Ty> {
        todo!("as `ty`")
    }

    pub fn coercion(&self, site: AstId) -> Option<&'a Coercion> {
        todo!("as `ty`")
    }

    pub fn structure_tree(&self, site: AstId) -> Option<&'a StructureTree> {
        todo!("as `ty`")
    }

    pub fn adjustment(&self, site: AstId) -> Option<Adjustment> {
        todo!("as `ty`")
    }

    pub fn iteration(&self, site: AstId) -> Option<Iteration> {
        todo!("as `ty`")
    }

    pub fn try_exit(&self, site: AstId) -> Option<TryExit<Ty>> {
        todo!("as `ty`")
    }

    pub fn cast(&self, site: AstId) -> Option<AsCast> {
        todo!("as `ty`")
    }

    /// The same view at `instance`, a lambda body's instance within this
    /// one.
    pub fn at(&self, instance: InstanceId) -> Decided<'a> {
        Decided { typeck: self.typeck, instance: Some(instance) }
    }
}
