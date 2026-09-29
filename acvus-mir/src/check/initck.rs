//! initck (ownership §4): a place is read, and a value used whole, only
//! where every path to it has written what the read needs: a field read
//! where that field is written, a whole use where every field of its type
//! is. A path on which a value lacks what a type grown in its slot holds is
//! refused where the value is used whole (types §1). Over `Presence`.

use super::Refusal;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(body: BodyId, module: &Module, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("initck over the analyses its module doc names")
}
