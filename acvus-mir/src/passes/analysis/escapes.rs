//! For each slot, the first use through which a reference to it leaves the uses a promotion rewrites: passed to a call, stored, returned, captured, or joined into a block parameter. Reads `Loans`. Cost: O(uses).
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager};

pub struct Escapes {}

impl Analysis for Escapes {
    type Facts = Escapes;

    const READS: &'static [Aspect] = &[Aspect::Instructions, Aspect::Storage];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Escapes {
        todo!("passes::analysis::Escapes::compute")
    }
}
