//! The editor's facts over the analysis manager (passes-analyses.md): one
//! adapter serves every scope. `refresh` invalidates the manager for what the
//! last edit changed and recomputes what the adapter serves; every query
//! reads only facts kept since the last refresh, never the module.

use std::rc::Rc;

use crate::ir::cfg::edit::{DominanceFacts, Facts, InstRef, LoanFacts, OrderFacts, Position};
use crate::ir::cfg::{BodyId, Changes, Module, Site};
use crate::ir::ValueId;
use crate::passes::analysis::{AnalysisManager, Dominators, Escapes, Loans, PostDominators};

pub struct EditFacts<'a, 'd> {
    am: &'a mut AnalysisManager<'d>,
    dominators: Option<Rc<Dominators>>,
    post_dominators: Option<Rc<PostDominators>>,
    loans: Option<Rc<Loans>>,
    escapes: Option<Rc<Escapes>>,
}

impl<'a, 'd> EditFacts<'a, 'd> {
    /// Facts for editing `body`, computed now.
    pub fn new(am: &'a mut AnalysisManager<'d>, module: &Module, body: BodyId) -> Self {
        todo!("passes::EditFacts::new")
    }
}

impl Facts for EditFacts<'_, '_> {
    fn refresh(&mut self, module: &Module, body: BodyId, changes: &Changes) {
        todo!("passes::EditFacts::refresh")
    }
}

impl DominanceFacts for EditFacts<'_, '_> {
    fn dominates(&self, value: ValueId, at: Position) -> bool {
        todo!("passes::EditFacts::dominates")
    }
    fn position_dominates(&self, at: Position, site: Site) -> bool {
        todo!("passes::EditFacts::position_dominates")
    }
    fn control_equivalent(&self, a: Position, b: Position) -> bool {
        todo!("passes::EditFacts::control_equivalent")
    }
}

impl LoanFacts for EditFacts<'_, '_> {
    fn conflicts(&self, inst: InstRef, write: InstRef) -> bool {
        todo!("passes::EditFacts::conflicts")
    }
    fn escapes(&self, slot: ValueId) -> Option<InstRef> {
        todo!("passes::EditFacts::escapes")
    }
}

impl OrderFacts for EditFacts<'_, '_> {
    /// Both are calls whose callee's declared effect commutes (effects §1),
    /// precomputed at refresh from the callee types.
    fn may_reorder(&self, a: InstRef, b: InstRef) -> bool {
        todo!("passes::EditFacts::may_reorder")
    }
}
