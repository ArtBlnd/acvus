//! The one entry (passes-pipeline.md): the required steps, then P1-P7 at
//! `Level::Full`, each step's required form asserted in debug and test
//! builds, and the scalar fixpoint P3* bounded by the measure instructions +
//! blocks, which every round that changes something strictly decreases.

use crate::ir::cfg::Module;
use crate::passes::analysis::AnalysisManager;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// The required steps only.
    None,
    Full,
}

pub fn optimize(module: &mut Module, am: &mut AnalysisManager<'_>, level: Level) {
    todo!("passes::pipeline::optimize")
}
