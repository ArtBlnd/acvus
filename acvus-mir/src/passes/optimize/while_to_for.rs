//! A `While` whose trip count is known becomes a `For` over a range
//! (compilation §4: a transformation reshapes a loop so that its join holds as
//! little as it can; a value handed from one iteration to the next that the
//! iteration's position decides is computed from the position instead).
//!
//! A `While` is promoted where, for its header's test:
//! 1. the scalar evolution names its trip count `n`, exact or at most (a loop
//!    left early keeps its other exits inside the `For`);
//! 2. the test, whose evaluations the promotion removes, has no effect and
//!    may not trap, and finishes (`analysis::Raise`);
//! 3. `n` is computed in the preheader so that it steers the run soundly: it
//!    is proven within its type by the interval domain, or computed wider
//!    (undefined behaviour §2: the run as written never computed it).
//!
//! The promoted loop is `For` over `0..n` with its counter `k`; each header
//! parameter that is a recurrence `{S, +, D}` of the loop becomes `S + k·D`
//! computed in the body and leaves the join; every other carried value stays
//! a header parameter. A stream head (`while let Some(x) = next(&mut it)`) is
//! not this pass's: its head comes from declared sources, named by the loop
//! analysis to come.
//!
//! Form: F1, SSA, loops canonical, in and out.
//!
//! Cost: O(loops) promotions, each O(the loop's header parameters and its
//! blocks) edits, over the scalar evolution and the endings the manager keeps.
//!
//! Meaning: the promoted loop runs its body `n` times with each header
//! parameter at the value the `While` gave it on that iteration, since a
//! recurrence's value at the k-th iteration is `S + k·D` exactly (integer
//! arithmetic means the integers', undefined behaviour §2); the removed tests
//! had no effect, could not trap and finished, so no run's effects, trap or
//! ending changes (compilation §1). No type changes; extern-defined types are
//! untouched (compilation §3).

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct WhileToFor;

impl BodyTransform for WhileToFor {
    const NAME: &'static str = "while_to_for";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        todo!("promote each While the conditions admit, inner loops first")
    }
}
