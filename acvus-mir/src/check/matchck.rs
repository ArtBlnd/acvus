//! matchck (patterns §3): no run reaches the block where no arm of a
//! `match` answers. Lower sends every such place, of a `match` with no
//! catch-all arm, to one block ending in `Terminator::NoArmMatched`: an arm
//! tested after another, and a `Switch` whose keys leave a value, by its
//! `default`. The check reads `MatchFacts` and refuses `NotExhaustive`, at
//! the terminator's span, where an edge into such a block can run.
//!
//! It is sound whatever the facts know, since a fact only over-approximates;
//! what it proves grows with the domains `MatchFacts` runs.

use super::Refusal;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("each NoArmMatched block of the body with an executable incoming edge under MatchFacts")
}
