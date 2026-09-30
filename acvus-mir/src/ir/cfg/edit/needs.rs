//! The facts an editor's legality checks read. `passes` implements them over
//! its analysis manager. Every query is lent the body it asks about as the
//! editor holds it now, and `refresh` only drops what an edit made stale, so
//! a fact is recomputed on the first query after its invalidation and kept
//! until the next one (D47).

use super::{InstRef, Position};
use crate::ir::cfg::{Body, BodyId, Changes, Module, Site};
use crate::ir::{PathSeg, RefTarget, ValueId};

/// The body a query asks about, in the module that holds it, as it stands
/// when the query is made.
#[derive(Clone, Copy)]
pub struct Asked<'m> {
    module: &'m Module,
    body: BodyId,
}

impl<'m> Asked<'m> {
    /// # Panics
    /// `body` is not a body of `module`.
    pub fn new(module: &'m Module, body: BodyId) -> Asked<'m> {
        module.assert_body(body);
        Asked { module, body }
    }

    pub fn module(self) -> &'m Module {
        self.module
    }

    pub fn body_id(self) -> BodyId {
        self.body
    }

    pub fn body(self) -> &'m Body {
        self.module.body(self.body)
    }
}

pub trait Facts {
    /// Drops every fact of `body` an aspect of `changes` made stale.
    fn refresh(&mut self, body: BodyId, changes: &Changes);
}

/// For scopes whose checks read nothing beyond the core.
pub struct NoFacts;

impl Facts for NoFacts {
    fn refresh(&mut self, _: BodyId, _: &Changes) {}
}

pub trait DominanceFacts: Facts {
    /// Whether `value`'s definition dominates `at`.
    fn dominates(&mut self, asked: Asked<'_>, value: ValueId, at: Position) -> bool;
    /// Whether an instruction standing at `at` would be defined wherever
    /// `site` reads it: `at` comes first on every path to `site`, `site`
    /// itself included when `at` stands just before it.
    fn position_dominates(&mut self, asked: Asked<'_>, at: Position, site: Site) -> bool;
    /// Whether `a` and `b` run on exactly the same runs (compilation §1):
    /// one dominates the other and the other post-dominates the one, and no
    /// path from the earlier to the later enters a block of a loop that does
    /// not hold both, since such a loop may never end.
    fn control_equivalent(&mut self, asked: Asked<'_>, a: Position, b: Position) -> bool;
}

pub trait LoanFacts: Facts {
    /// Whether `write` writes storage that `inst`'s operands or result name
    /// (ownership §5).
    fn conflicts(&mut self, asked: Asked<'_>, inst: InstRef, write: InstRef) -> bool;
    /// Whether `inst` names any storage, by its operands, its result, or
    /// what it reads and writes. Where it names none, `conflicts` is false
    /// with it on either side, for every other instruction.
    fn names_storage(&mut self, asked: Asked<'_>, inst: InstRef) -> bool;
    /// The first use of `slot` a promotion cannot rewrite: anything but a
    /// whole-slot `Take` or `Assign`. `None` is exactly what `promote`
    /// requires.
    fn escapes(&mut self, asked: Asked<'_>, slot: ValueId) -> Option<Site>;
}

pub trait OrderFacts: Facts {
    /// Whether the declarations let `a` and `b` trade places (effects §2).
    fn may_reorder(&mut self, asked: Asked<'_>, a: InstRef, b: InstRef) -> bool;
}

/// Why an instruction the core would keep is unobservable here: the fact an
/// analysis proves, which lets it go (D55, general capability G1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unobserved {
    /// A check never fires on any run that reaches it.
    NeverFires,
    /// A store whose value no read reaches.
    NeverRead,
}

pub trait ContentFacts: Facts {
    /// Whether the content of `place`, a place by name, may be read from
    /// `at` on, before a whole store replaces it or the body ends.
    fn content_read_after(&mut self, asked: Asked<'_>, place: RefTarget, at: Position) -> bool;
    /// Whether any part of `place`'s content overlapping `path` may be read
    /// from `at` on, before a store replaces that part or the body ends.
    fn part_read_after(&mut self, asked: Asked<'_>, place: RefTarget, path: &[PathSeg], at: Position) -> bool;
}

pub trait DischargeFacts: Facts {
    /// The fact that makes `inst` unobservable, if an analysis proves one.
    fn discharged(&mut self, asked: Asked<'_>, inst: InstRef) -> Option<Unobserved>;
}
