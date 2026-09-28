//! A reference to the whole of what a reference names is that reference (types §4). Scope `Value`. F1 → F1. Cost: O(instructions). The same loan.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Reborrow;

impl BodyTransform for Reborrow {
    const NAME: &'static str = "reborrow";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::reborrow")
    }
}
