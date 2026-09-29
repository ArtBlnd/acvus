//! borrowck (ownership §5): while an exclusive loan lives its storage is
//! reached only through it; while a shared one lives its storage is not
//! written, moved or lent exclusively. What a path of references reaches is
//! held by the first reference on it, and is exclusive only where every
//! reference on it is. A reference to a transparent structured value is the
//! structure of references to its parts (`Intrinsic::Gather`, `as_ref`,
//! `as_mut`, a coercion through a reference). Over `Loans` and `Liveness`.

use super::Refusal;
use crate::ir::cfg::{Body, BodyId};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(body: BodyId, module_body: &Body, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("borrowck over the analyses its module doc names")
}
