//! Dead code elimination: an unused result whose instruction neither traps, writes storage, touches a context nor is a non-pure call goes (D54), over a worklist. Scope `Remove`. Any form, kept. Cost: O(instructions).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Dce;

impl BodyTransform for Dce {
    const NAME: &'static str = "dce";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("passes::optimize::dce")
    }
}
