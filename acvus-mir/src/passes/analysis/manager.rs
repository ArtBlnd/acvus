//! The analysis manager (passes-structure §2): an analysis is computed on
//! first request and kept until an edit changes an aspect it reads. No pass
//! declares what it preserves; the Rewriter's record of what changed
//! decides (ir::cfg::Changes).

use std::any::{Any, TypeId};
use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::decl::Declarations;
use crate::ir::cfg::{Aspect, BodyId, Changes, Module};

/// An analysis the manager keeps. `READS` lists every aspect of a body its
/// facts depend on; its doc states its cost and its fixpoint's bound.
pub trait Analysis: 'static {
    type Facts: 'static;

    const READS: &'static [Aspect];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts;
}

pub struct AnalysisManager<'d> {
    decls: &'d Declarations,
    kept: FxHashMap<(TypeId, BodyId), (Rc<dyn Any>, &'static [Aspect])>,
}

impl<'d> AnalysisManager<'d> {
    pub fn new(decls: &'d Declarations) -> Self {
        Self { decls, kept: FxHashMap::default() }
    }

    pub fn decls(&self) -> &'d Declarations {
        self.decls
    }

    /// The facts of `A` for `body`, computed on first request.
    pub fn get<A>(&mut self, module: &Module, body: BodyId) -> Rc<A::Facts>
    where
        A: Analysis,
    {
        todo!("passes::analysis::AnalysisManager::get")
    }

    /// Drops every kept analysis of `body` that reads an aspect `changes`
    /// touched.
    pub fn invalidate(&mut self, body: BodyId, changes: &Changes) {
        todo!("passes::analysis::AnalysisManager::invalidate")
    }
}
