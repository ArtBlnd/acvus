//! What each phase reports it did: the trace a step, a
//! settling and a fallback return, in the order the phases ran, which the
//! cases' runner reads and the solver's callers drop. A report names classes
//! by `ClassId`, whose names are read in the state the trace ends in.

use crate::check::Refusal;
use crate::class::ClassId;
use crate::fallback::FellBack;
use crate::node::Decided;
use crate::record::Recording;
use crate::split::Split;
use crate::unify::Linked;
use crate::value::Joined;

/// What one phase did, a relation named by its place in the log.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Report {
    /// Recording: relations of the text recorded.
    Recorded(Recording),
    /// Unification: a kind "indexes `τ`" linked along a view.
    Linked(Linked),
    /// The join: cycles of lower bounds alone merged.
    Merged(Recording),
    /// The join: the equivalence class reaches a part of its own, and is
    /// refused.
    Infinite { class: ClassId, refusal: Refusal },
    /// The join: what the equivalence class took, and from what.
    Joined { class: ClassId, joined: Joined },
    /// The join: the range the lower bound or part lower bound at `bound`
    /// gives the class it bounds.
    Ranged { bound: usize, recorded: Recording },
    /// A split: what became of the relation at `at`.
    Split { at: usize, split: Split },
    /// The nodes: the common part of all the nodes not yet decided may
    /// become.
    Common(Recording),
    /// The nodes: a node decided.
    Decided(Decided),
    /// Fallback.
    FellBack(FellBack),
}

/// The reports of a step, a settling or a fallback, in order.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Trace {
    pub reports: Vec<Report>,
}

impl Trace {
    pub fn push(&mut self, report: Report) {
        self.reports.push(report);
    }

    pub fn append(&mut self, mut other: Trace) {
        self.reports.append(&mut other.reports);
    }

    /// Whether an equivalence class fell back.
    pub fn fell_back(&self) -> bool {
        self.reports
            .iter()
            .any(|report| matches!(report, Report::FellBack(_)))
    }
}
