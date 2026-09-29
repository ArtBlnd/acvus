//! matchck (patterns §3): the block lower marks where no arm of a `match`
//! answers (`Terminator::NoArmMatched`) is reached by no run. Within the one
//! body, the variants a scrutinee may hold are known where every value
//! reaching it was made in the body (`TagValues`); a value from outside it,
//! a parameter, a capture, a call's result, a context or an input, holds
//! any variant its type has. An `Option`, a `Result` and an enum a
//! declaration writes have their variants by their type (types §2, patterns
//! §2), so every one of them is seen; a program's enum from outside the body
//! is not, and its `match` needs `_`.

use super::Refusal;
use crate::ir::cfg::{Body, BodyId};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(body: BodyId, module_body: &Body, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("matchck over the analyses its module doc names")
}
