//! Global value numbering over the dominator tree: a pure computation, a pure call or a merge (effects §3: associative, commutative, idempotent) equal to a dominating one takes its value. Scope `Value`. F1 → F1. Cost: O(instructions) expected. The dominating one already ran on every path, and a trapping one already trapped where this one would.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Gvn;

impl BodyTransform for Gvn {
    const NAME: &'static str = "gvn";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::gvn")
    }
}
