//! Bounds-check elimination: a check the interval domain proves never fires goes, on `NeverFires` (`discharge`, G1). The interval domain reads declared Relations (extern fn §3). F1 → F1. Cost: O(uses × widened height). A check that never fires does not change whether a run traps.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Bce;

impl BodyTransform for Bce {
    const NAME: &'static str = "bce";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::bce")
    }
}
