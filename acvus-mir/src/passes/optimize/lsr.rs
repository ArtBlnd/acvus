//! Loop strength reduction (Allen, Cocke and Kennedy; compilation §4).
//!
//! A value computed in a loop whose evolution is an affine recurrence
//! `{S, +, D}` of the loop, computed there by a multiplication, becomes a new
//! join (`add_join`): its entry argument `S` computed in the preheader, and on
//! each back edge the parameter plus `D`, `D` invariant in the loop. The
//! multiplication's uses read the parameter, and the multiplication goes. The
//! step is taken only where it costs less in the backend's measure than the
//! multiplication it replaces.
//!
//! The value the new join computes on the back edge after the last iteration
//! is one the run as written did not compute, so the join is made only where
//! each value it computes is proven within its type by the interval domain or
//! computed wider (undefined behaviour §2).
//!
//! Form: F2, SSA, in and out.
//!
//! Cost: O(loops · the loop's instructions) candidates, each O(the edits it
//! composes), over the scalar evolution the manager keeps.
//!
//! Meaning: on the k-th iteration the new parameter is `S + k·D`, the value
//! the multiplication computed, exactly; every use reads the value it read
//! before, and nothing traps, ends or orders differently (compilation §1). No
//! type changes; extern-defined types are untouched (compilation §3).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct Lsr;

impl BodyTransform for Lsr {
    const NAME: &'static str = "lsr";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("reduce each affine multiplication the conditions admit")
    }
}
