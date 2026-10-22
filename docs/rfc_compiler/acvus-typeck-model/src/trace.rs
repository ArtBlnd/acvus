//! What each phase reports it did (`design/19-cases.md`): the trace a step, a
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
use crate::value::{Filled, Joined};

/// What one phase did, a relation named by its place in the log.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Report {
    /// Recording: relations of the text recorded (l.217–231).
    Recorded(Recording),
    /// Unification: a kind "indexes `τ`" linked along a view (l.154).
    Linked(Linked),
    /// The join: cycles of lower bounds alone merged (l.269).
    Merged(Recording),
    /// The join: the equivalence class reaches a part of its own, and is
    /// refused (l.269).
    Infinite { class: ClassId, refusal: Refusal },
    /// The join: the equivalence class's LitVar lower bounds unified with it
    /// (l.169).
    LitLowers { class: ClassId, recorded: Recording },
    /// A split: `?N ⇝ U` unifying the equivalence class with each type it
    /// passes into (l.278).
    LitBoundaries { class: ClassId, recorded: Recording },
    /// The join: what the equivalence class took, and from what (l.269).
    Joined { class: ClassId, joined: Joined },
    /// The fill: what the equivalence class nothing enters took (l.282).
    Filled(Filled),
    /// The join: the LitVar lower bound at `bound` unified with the solved
    /// class it bounds (l.169).
    LitJoined { bound: usize, recorded: Recording },
    /// A split: what became of the relation at `at` (l.278).
    Split { at: usize, split: Split },
    /// The nodes: the common part of all the nodes not yet decided may
    /// become (l.246).
    Common(Recording),
    /// The nodes: a node decided (l.261).
    Decided(Decided),
    /// Fallback (l.288).
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

    /// Whether an equivalence class fell back (l.288).
    pub fn fell_back(&self) -> bool {
        self.reports
            .iter()
            .any(|report| matches!(report, Report::FellBack(_)))
    }
}
