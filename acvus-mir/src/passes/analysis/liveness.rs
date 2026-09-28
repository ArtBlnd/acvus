//! The values live at each block's start and end: the dense engine, backward, over value sets. Cost: O(edges × values); height: the body's values.
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager};

pub struct Liveness {}

impl Analysis for Liveness {
    type Facts = Liveness;

    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Liveness {
        todo!("passes::analysis::Liveness::compute")
    }
}
