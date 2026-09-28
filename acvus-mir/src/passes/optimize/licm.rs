//! Loop-invariant code motion: invariant instructions move to the preheader, inner loops first (`Editor<Motion>`); pure non-trapping values freely, others only where control-equivalent (D54, D56). Requires F2. Cost: O(loop blocks × instructions).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Licm;

impl BodyTransform for Licm {
    const NAME: &'static str = "licm";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::licm")
    }
}
