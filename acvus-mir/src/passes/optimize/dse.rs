//! Dead store elimination: a store no read reaches goes, on `NeverRead` (`Editor<Discharge>::discharge`, G1) from `Loans` and `Liveness`. F1 → F1. Cost: O(stores × uses). A store no read reaches is unobservable.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Dse;

impl BodyTransform for Dse {
    const NAME: &'static str = "dse";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::dse")
    }
}
