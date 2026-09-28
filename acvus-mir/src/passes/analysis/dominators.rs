//! Immediate dominators by Cooper, Harvey and Kennedy (2001), and dominance between positions (same block: by index). Cost: O(blocks + edges) per iteration, O(depth of the dominator tree) iterations.
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager};

pub struct Dominators {}

impl Analysis for Dominators {
    type Facts = Dominators;

    const READS: &'static [Aspect] = &[Aspect::CfgShape];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Dominators {
        todo!("passes::analysis::Dominators::compute")
    }
}
