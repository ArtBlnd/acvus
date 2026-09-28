//! Sparse conditional constant propagation (Wegman-Zadeck) over the constant domain; a value proven constant becomes that constant, an arm no run takes is dropped (`take_arm`). Scope `Sccp`. F1 → F1. Cost: O(uses × 3).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Sccp;

impl BodyTransform for Sccp {
    const NAME: &'static str = "sccp";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::sccp")
    }
}
