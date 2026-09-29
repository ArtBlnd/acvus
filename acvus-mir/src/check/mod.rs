//! check (D86): the checker's judgments over the lowered IR. With typeck
//! they are the checker (refusals): a program is refused here or not at
//! all, and nothing after admits or refuses again. Each judgment reads the
//! analyses the manager keeps and edits nothing.
//!
//! - `initck`: a place is read, and a value used whole, only where every
//!   path has written what the read needs (ownership §4), over `Presence`.
//! - `moveck`: a value that does not copy is not used after a move out of
//!   it, whole or in part (ownership §2, §6), over `Presence`.
//! - `borrowck`: while a loan lives, its storage is reached only as its
//!   mutability allows, a path of references held by its first and exclusive
//!   only where every reference on it is (ownership §5), over `Loans` and
//!   `Liveness`.
//! - `matchck`: the block lower marks where no arm of a `match` answers is
//!   reached by no run (patterns §3), over `TagValues`, within the one body.
//!
//! They run after `required::remove_dead_context_ops` and before
//! `required::insert_drops`, which places drops where the moves they admit
//! leave values.

mod borrowck;
mod initck;
mod matchck;
mod moveck;

use acvus_ast::Span;

use crate::ir::cfg::Module;
use crate::passes::analysis::AnalysisManager;

/// A refusal the checker makes over the IR, in the program's terms
/// (refusals): what was written, where, and the demand that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub kind: RefusalKind,
    pub span: Span,
}

/// Grouped by the judgment that refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefusalKind {
    /// A read of a place, or a whole use, some path to which did not write
    /// what it needs (ownership §4).
    NotWritten,
    /// A use of a value, or of a part of it, after it moved (ownership §2):
    /// `moved` where the move was.
    UsedAfterMove { moved: Span },
    /// A storage reached while a loan on it forbids it (ownership §5): the
    /// span is the reach, and `loan` where the loan was made.
    LoanConflict { loan: Span },
    /// A `match` some value reaches with no arm answering (patterns §3):
    /// where its scrutinee's variants are not all seen, it needs `_`.
    NotExhaustive,
}

/// Every judgment over every body of `module`, in the order the module doc
/// lists; every refusal each makes.
pub fn check(module: &Module, am: &mut AnalysisManager<'_>) -> Vec<Refusal> {
    let mut refusals = Vec::new();
    for body in module.body_ids() {
        // Placeholder: `initck::check(module, body, am, &mut refusals)`, filled by initck's task.
        // Placeholder: `moveck::check(module, body, am, &mut refusals)`, filled by moveck's task.
        borrowck::check(module, body, am, &mut refusals);
        // Placeholder: `matchck::check(module, body, am, &mut refusals)`, filled by matchck's task.
    }
    refusals
}
