//! List scheduling within a block over def-use, tokens included (effects §3: token dominance is the order); runs last. `Editor<Motion>`. Cost: O(n log n) per block.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Schedule;

impl BodyTransform for Schedule {
    const NAME: &'static str = "schedule";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::schedule")
    }
}
