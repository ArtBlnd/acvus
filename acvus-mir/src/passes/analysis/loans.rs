//! Loans (ownership §5): for each value and each position of its type, the storages it may name, shared or exclusive; which storage each instruction reads and writes. The sparse engine over def-use; a call moves loans only as its function type's flows state. Storage: a slot, or the storage a reference parameter or capture lends (one opaque storage each). Cost: O(uses × storages).
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager};

pub struct Loans {}

impl Analysis for Loans {
    type Facts = Loans;

    const READS: &'static [Aspect] = &[Aspect::Instructions, Aspect::Storage, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Loans {
        todo!("passes::analysis::Loans::compute")
    }
}
