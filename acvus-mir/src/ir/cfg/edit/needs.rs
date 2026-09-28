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
    /// Whether `a` and `b` run on exactly the same runs: one dominates the
    /// other and the other post-dominates the one.
    fn control_equivalent(&self, a: Position, b: Position) -> bool;
}

pub trait LoanFacts: Facts {
    /// Whether `write` writes storage that `inst`'s operands or result name
    /// (ownership §5).
    fn conflicts(&self, inst: InstRef, write: InstRef) -> bool;
    /// Whether a reference to `slot` escapes the uses a promotion rewrites.
    fn escapes(&self, slot: ValueId) -> Option<InstRef>;
}

pub trait OrderFacts: Facts {
    /// Whether the declarations let `a` and `b` trade places (effects §2).
    fn may_reorder(&self, a: InstRef, b: InstRef) -> bool;
}
