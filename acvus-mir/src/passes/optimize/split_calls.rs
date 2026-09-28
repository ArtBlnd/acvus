//! Every call whose declaration allows it is split into its start and its taking (effects §6), then the start moves up and the taking down under motion's rules; each start is taken once on every path (SplitTakenOnce). Cost: O(calls × motion).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct SplitCalls;

impl BodyTransform for SplitCalls {
    const NAME: &'static str = "split_calls";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::split_calls")
    }
}
