//! check (D86): the checker's judgments over the lowered IR. With typeck
//! they are the checker (refusals): a program is refused here or not at
//! all, and nothing after admits or refuses again. Each judgment reads the
//! analyses the manager keeps and edits nothing.
//!
//! - `initck`: a place is read, and a value used whole, only where every
//!   path has written what the read needs (ownership §4), over
//!   `Initialization`.
//! - `moveck`: a value that does not copy is not used after a move out of
//!   it, whole or in part (ownership §2, §6), over `Initialization`.
//! - `borrowck`: while a loan lives, its storage is reached only as its
//!   mode allows, a reborrow a loan of its own within the loan it is made
//!   through, and nothing that outlives the body holds a loan on its own
//!   storage (ownership §5), over `Loans` and `Liveness`.
//! - `matchck`: the block lower marks where no arm of a `match` answers is
//!   reached by no run (patterns §3), over `MatchFacts`, within the one body.
//!
//! They run after `required::clone_strings` and before
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

/// A loan conflict's whole trace, each loan named by where it was made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictTrace {
    /// The live loan the access met: where it was made, then each loan it was made through, innermost first.
    pub earlier: Vec<Span>,
    /// The conflicting side's loan chain, where the access made or used a loan (a lend or reborrow), innermost first; empty for an access by name.
    pub later: Vec<Span>,
    /// Every other access that met the same pair while both lived.
    pub uses: Vec<Span>,
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
    /// Two loans that exclude each other while both live, or a place reached
    /// by name while a loan forbids it (ownership §5), one refusal each
    /// (refusals): the span is the access that conflicted, and `trace` all
    /// of the conflict, what a message shows of it left to the message.
    LoanConflict { trace: ConflictTrace },
    /// A loan on the body's own storage that its result or a storage
    /// outside it holds when it returns (ownership §5): `loan` where the
    /// loan was made.
    LoanOutlivesBody { loan: Span },
    /// A `match` some value reaches with no arm answering (patterns §3):
    /// where its scrutinee's variants are not all seen, it needs `_`.
    NotExhaustive,
    /// A return with storage outside the body, a return or a call where a
    /// context is seen beyond the program, or a call where the body may
    /// suspend, with either, a part of which some path took out and did not
    /// write back (ownership §4, context §1): `taken` where it was taken.
    NotRefilled { taken: Span },
    /// A move of what a shared reference lends (ownership §5).
    MoveThroughShared,
    /// A move through a reference that lends more than one place on the
    /// paths to it (ownership §4).
    MoveThroughSeveral,
    /// A store whose old content the body cannot end (ownership §6): a
    /// store through a reference that may reach several places, where what
    /// it replaces may be absent at one of them on some path, since which
    /// content it replaces is not known; and a store into one place a part
    /// of which is present on some paths only, where no root names that
    /// place on an edge it is last known present on
    /// (`lower::required::rooted_place`). `moved` where it was moved out.
    StoreOverContentMaybeMoved { moved: Span },
}

impl RefusalKind {
    /// The refusal in notation (no prose): its kind and the spans it names,
    /// `@start..end`.
    pub fn written(&self) -> String {
        let at = |span: &Span| format!("@{}..{}", span.start, span.end);
        let all = |spans: &[Span]| format!("[{}]", spans.iter().map(at).collect::<Vec<_>>().join(", "));
        match self {
            RefusalKind::NotWritten => "NotWritten".to_string(),
            RefusalKind::UsedAfterMove { moved } => format!("UsedAfterMove moved {}", at(moved)),
            RefusalKind::LoanConflict { trace } => {
                format!("LoanConflict earlier {} later {} uses {}", all(&trace.earlier), all(&trace.later), all(&trace.uses))
            }
            RefusalKind::LoanOutlivesBody { loan } => format!("LoanOutlivesBody loan {}", at(loan)),
            RefusalKind::NotExhaustive => "NotExhaustive".to_string(),
            RefusalKind::NotRefilled { taken } => format!("NotRefilled taken {}", at(taken)),
            RefusalKind::MoveThroughShared => "MoveThroughShared".to_string(),
            RefusalKind::MoveThroughSeveral => "MoveThroughSeveral".to_string(),
            RefusalKind::StoreOverContentMaybeMoved { moved } => format!("StoreOverContentMaybeMoved moved {}", at(moved)),
        }
    }
}

/// Every judgment over every body of `module`, in the order the module doc
/// lists; every refusal each makes.
pub fn check(module: &Module, am: &mut AnalysisManager<'_>) -> Vec<Refusal> {
    let mut refusals = Vec::new();
    for body in module.body_ids() {
        initck::check(module, body, am, &mut refusals);
        moveck::check(module, body, am, &mut refusals);
        borrowck::check(module, body, am, &mut refusals);
        matchck::check(module, body, am, &mut refusals);
    }
    refusals
}
