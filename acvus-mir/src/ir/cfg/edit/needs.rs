//! The facts an editor's legality checks read. `passes` implements them over
//! its analysis manager; `refresh` recomputes what the editor's own edits made
//! stale, so no check reads a stale fact.

use super::{InstRef, Position};
use crate::ir::cfg::{BodyId, Changes, Module, Site};
use crate::ir::ValueId;

pub trait Facts {
    fn refresh(&mut self, module: &Module, body: BodyId, changes: &Changes);
}

/// For scopes whose checks read nothing beyond the core.
pub struct NoFacts;

impl Facts for NoFacts {
    fn refresh(&mut self, _: &Module, _: BodyId, _: &Changes) {}
}

pub trait DominanceFacts: Facts {
    /// Whether `value`'s definition dominates `at`.
    fn dominates(&self, value: ValueId, at: Position) -> bool;
    /// Whether an instruction standing at `at` would be defined wherever
    /// `site` reads it: `at` comes first on every path to `site`, `site`
    /// itself included when `at` stands just before it.
    fn position_dominates(&self, at: Position, site: Site) -> bool;
    /// Whether `a` and `b` run on exactly the same runs (compilation §1):
    /// one dominates the other and the other post-dominates the one, and no
    /// path from the earlier to the later enters a block of a loop that does
    /// not hold both, since such a loop may never end.
    fn control_equivalent(&self, a: Position, b: Position) -> bool;
}

pub trait LoanFacts: Facts {
    /// Whether `write` writes storage that `inst`'s operands or result name
    /// (ownership §5).
    fn conflicts(&self, inst: InstRef, write: InstRef) -> bool;
    /// The first use of `slot` a promotion cannot rewrite: anything but a
    /// whole-slot `Take` or `Assign`. `None` is exactly what `promote`
    /// requires.
    fn escapes(&self, slot: ValueId) -> Option<Site>;
}

pub trait OrderFacts: Facts {
    /// Whether the declarations let `a` and `b` trade places (effects §2).
    fn may_reorder(&self, a: InstRef, b: InstRef) -> bool;
}

/// Why an instruction the core would keep is unobservable here: the fact an
/// analysis proves, which lets it go (D55, general capability G1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unobserved {
    /// A check never fires on any run that reaches it.
    NeverFires,
    /// A store or a load whose value no read reaches.
    NeverRead,
    /// A commit of a context the body never writes (context §3).
    NeverWritten,
}

pub trait ContentFacts: Facts {
    /// Whether the content of `place`, a slot or a by-value parameter's
    /// own storage, may be read from `at` on, before a whole store replaces
    /// it or the body ends.
    fn content_read_after(&self, place: ValueId, at: Position) -> bool;
}

pub trait DischargeFacts: Facts {
    /// The fact that makes `inst` unobservable, if an analysis proves one.
    fn discharged(&self, inst: InstRef) -> Option<Unobserved>;
}
