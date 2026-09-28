//! Scalar replacement: a slot of a structural type that no reference to the whole escapes becomes one slot per field (`split_slot`, S1); extern-defined types never (compilation §3). Then `mem2reg` promotes the fields. Cost: O(fields × uses). Fields of a structural value are independent places (types §1).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Sroa;

impl BodyTransform for Sroa {
    const NAME: &'static str = "sroa";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::sroa")
    }
}
