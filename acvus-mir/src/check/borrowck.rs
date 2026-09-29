//! borrowck (ownership §5): while a loan lives, its place is reached only
//! as the loan allows. It reads `Loans` for what each holder holds,
//! `Liveness` and `ContentLiveness` for which holders are live.
//!
//! At every access of a place, by name (`RefTarget::Var`, `Param`) or
//! through a reference `r` (`RefTarget::Through`), a read, a write, a move or
//! a lend, every live loan whose place overlaps the accessed one is held
//! against it:
//! - a write, a move and a `&mut` lend meet every such loan;
//! - a read and a `&` lend meet an exclusive one;
//! - an access through `r` does not meet the loans `r` itself holds, which
//!   it is made through.
//! A loan the access meets refuses it `LoanConflict`, at the access's span,
//! naming where the loan was made.

use super::Refusal;
use crate::ir::cfg::{BodyId, Module};
use crate::passes::analysis::AnalysisManager;

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    todo!("each access of the body against the live loans overlapping it, by the rule above")
}
