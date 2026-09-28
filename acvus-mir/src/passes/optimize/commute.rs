//! Adjacent calls declared commutative trade places to shorten chains (effects §1, `swap`, G4). Scope `Effect`. Cost: O(chain length).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Commute;

impl BodyTransform for Commute {
    const NAME: &'static str = "commute";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::commute")
    }
}
