//! What each phase reports it did: the trace a judgment, a
//! settling and a fallback return, in the order the phases ran, which the
//! cases' runner reads and the solver's callers drop. A report names classes
//! by `ClassId`, whose names are read in the state the trace ends in.

use crate::class::ClassId;
use crate::decomposition::Decomposition;
use crate::disjunction::Decision;
use crate::fallback::FellBack;
use crate::record::Recording;
use crate::ty::Name;
use crate::value::Took;

/// What one phase did, a relation named by its place in the record.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Report {
    /// Recording: relations of the text recorded.
    Recorded(Recording),
    /// Unification: cycles of lower bounds and part lower bounds alone merged.
    Merged(Recording),
    /// A decomposition: what became of the relation at `at`.
    Decomposed {
        at: usize,
        decomposition: Decomposition,
    },
    /// A judgment: what the equivalence class took, its join or its fill,
    /// or the refusal where it has no value.
    Joined { class: ClassId, took: Took },
    /// A judgment: the alternative a disjunctive constraint is decided as, and
    /// what that records.
    Decided {
        disjunction: Name,
        decision: Decision,
        recorded: Recording,
    },
    /// Fallback.
    FellBack(FellBack),
}

/// The reports of a judgment, a settling or a fallback, in order.
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
