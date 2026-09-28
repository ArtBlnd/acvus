//! SSA construction: every slot `Escapes` does not name is promoted (Braun et al. 2013, via `Editor<Storage>::promote`). F0 → F1. Cost: O(blocks + edges + uses) per slot. A promoted slot is reached through no reference, so its loads and stores are its values.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Mem2Reg;

impl BodyTransform for Mem2Reg {
    const NAME: &'static str = "mem2reg";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::mem2reg")
    }
}
