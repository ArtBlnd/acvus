//! initck (ownership §4): a place is read by name only where every path to
//! the read has written what it reads: a field where that field is written,
//! the value whole where every field of its type is. It reads
//! `Initialization`: at each `Take` or `Ref` whose target is a slot or a
//! by-value parameter's storage, each unit its path overlaps must be written
//! and nothing else (`Written::is_written`). Where one is not and none of
//! them was moved out, the read is refused `NotWritten` at its span; a unit
//! moved out is `moveck`'s. What a read through a reference reaches is not
//! located by name, and is not this check's.

use super::Refusal;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("each Take or Ref by name of the body, by the rule above")
}
