//! Post-dominators: Cooper-Harvey-Kennedy on the reverse CFG, with one virtual exit that every `Return` and `Diverge` block reaches. Cost as `Dominators`.
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager};

pub struct PostDominators {}

impl Analysis for PostDominators {
    type Facts = PostDominators;

    const READS: &'static [Aspect] = &[Aspect::CfgShape];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> PostDominators {
        todo!("passes::analysis::PostDominators::compute")
    }
}
