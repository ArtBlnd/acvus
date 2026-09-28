//! Natural loops: a back edge's target dominates its source; each loop's blocks, its nesting, and its preheader where one exists. Reads `Dominators`. Cost: O(blocks + edges).
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager};

pub struct LoopTree {}

impl Analysis for LoopTree {
    type Facts = LoopTree;

    const READS: &'static [Aspect] = &[Aspect::CfgShape];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> LoopTree {
        todo!("passes::analysis::LoopTree::compute")
    }
}
