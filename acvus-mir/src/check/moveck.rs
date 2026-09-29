//! moveck (ownership §2, §6): a value that does not copy is not used, whole
//! or in any part that moved, after a move out of it; a part moved out ends
//! with its holder, and the value is not used whole again. Over `Presence`.

use super::Refusal;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(body: BodyId, module: &Module, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("moveck over the analyses its module doc names")
}
