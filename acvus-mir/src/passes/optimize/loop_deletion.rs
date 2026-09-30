//! A loop that does nothing observable goes (LoopDeletion; compilation §1).
//!
//! A natural loop is deleted where every instruction in it has no effect, may
//! not trap and finishes, every loop inside it finishes (`analysis::Raise`),
//! and every value it gives its exit, or defines and has used after it, is
//! invariant in it or computable at the exit by the scalar evolution. Those
//! values are computed in the preheader or after the loop instead, under
//! undefined behaviour §2's rule for a value the run as written did not
//! compute, and the preheader jumps to the exit.
//!
//! Form: F2, SSA, in and out.
//!
//! Cost: O(loops) deletions, each O(the loop's blocks and exit values) edits.
//!
//! Meaning: the loop computes nothing the run observes but the values replaced
//! by equal ones, and it may not trap and finishes, so no run's effects, trap
//! or ending changes (compilation §1). No type changes; extern-defined types
//! are untouched (compilation §3).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct LoopDeletion;

impl BodyTransform for LoopDeletion {
    const NAME: &'static str = "loop_deletion";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("delete each loop the conditions admit, inner loops first")
    }
}
