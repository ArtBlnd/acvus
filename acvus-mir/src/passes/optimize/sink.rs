//! Sinking: a value moves to the nearest common dominator of its uses where that is later (`Editor<Motion>`). F1 → F1. Cost: O(instructions × depth). As licm.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Sink;

impl BodyTransform for Sink {
    const NAME: &'static str = "sink";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::sink")
    }
}
