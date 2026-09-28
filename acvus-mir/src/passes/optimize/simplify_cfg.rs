//! CFG simplification: an empty block that only jumps is bypassed, a block joins its sole predecessor whose sole successor it is, unreachable blocks are pruned; a `Diamond`, `Switch` or loop label is never merged away (D45, D57). Scope `Cfg`. Cost: O(blocks) per round. Only edges and empty blocks change.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct SimplifyCfg;

impl BodyTransform for SimplifyCfg {
    const NAME: &'static str = "simplify_cfg";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::simplify_cfg")
    }
}
