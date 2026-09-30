//! Induction-variable simplification over a `For` (IndVarSimplify; compilation
//! §4: a loop's join holds as little as it can).
//!
//! In a loop the scalar evolution reads, the canonical induction variable is
//! the `For`'s counter. The pass does two rewrites, each an edit of the loop
//! composed with the body's value edits:
//! 1. A use in the body of a header parameter whose evolution is an affine
//!    recurrence `{S, +, D}` of the loop, other than its own step, reads
//!    `S + k·D` made by `expand` from the counter's position `k`. Where no use
//!    is left but the step, the parameter leaves the join (`remove_join`); a
//!    use the body does not dominate keeps it.
//! 2. A value the loop gives its exit, or a value defined in the loop and used
//!    after it, whose evolution at the exit is computable from an exact trip
//!    count (`S + n·D`), is made by `expand` after the loop instead
//!    (`remove_exit`, or a replacement of the uses outside the loop).
//!
//! A value computed in either rewrite that the run as written did not compute
//! steers the run only where it is proven within its type by the interval
//! domain or computed wider (undefined behaviour §2).
//!
//! Form: F2 (loops canonical), SSA, in and out.
//!
//! Cost: O(loops · the loop's header and exit parameters) rewrites, each
//! O(the edits it composes), over the scalar evolution the manager keeps.
//!
//! Meaning: a recurrence's value at the k-th iteration is `S + k·D` exactly
//! (integer arithmetic means the integers', undefined behaviour §2), and at
//! the exit `S + n·D`; every rewritten use reads the value it read before, and
//! nothing traps, ends or orders differently (compilation §1). No type
//! changes; extern-defined types are untouched (compilation §3).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct IvCanon;

impl BodyTransform for IvCanon {
    const NAME: &'static str = "iv_canon";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("rewrite each For's recurrences and computable exit values, inner loops first")
    }
}
