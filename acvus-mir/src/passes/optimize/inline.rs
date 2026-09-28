//! Inlining: a direct call of a body of this module is replaced by that body
//! (`Editor<Inline>::inline`, S2), visiting the call graph's strongly
//! connected components callees first; a body is never inlined into itself,
//! and each call site at most once, so the pass is bounded by the module's
//! call sites. The callee's token chain continues the caller's. A call means
//! its body.

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::ModuleTransform;

pub struct Inline;

impl ModuleTransform for Inline {
    const NAME: &'static str = "inline";

    fn run(module: &mut Module, am: &mut AnalysisManager<'_>) -> Vec<(BodyId, Changes)> {
        todo!("passes::optimize::inline")
    }
}
