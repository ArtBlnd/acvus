//! moveck (ownership §2, §6): a value that does not copy is not used after
//! a use moved it, whole or in the part moved; a value a part of which moved
//! out is not used whole again. It reads `Initialization`: at each `Take` or
//! `Ref` whose target is a slot or a by-value parameter's storage, a unit its
//! path overlaps that some path moved out refuses the read `UsedAfterMove`
//! at its span, naming the move. A read one of whose units is only never
//! written is `initck`'s.

use super::Refusal;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("each Take or Ref by name of the body, by the rule above")
}
