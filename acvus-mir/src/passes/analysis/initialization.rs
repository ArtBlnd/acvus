//! Initialization (ownership §2, §4, §6): whether each part of each place
//! the body holds is written, never written, or moved out, at every point:
//! the dense engine, forward, before promotion, when every binding is a
//! slot. `check::initck` and `check::moveck` read it.
//!
//! - A place is a slot or a by-value parameter's own storage, whatever its
//!   type: a binding is built field by field whether its fields copy or
//!   not (ownership §4).
//! - A place's units are its type cut at every path the body names in it
//!   by name, a store's, a read's or a move's, by `ir::parts::units`, so they
//!   cover the whole type.
//! - Entry: a slot's units are never written; a by-value parameter's are
//!   written.
//! - An `Assign` into a path by name writes every unit that path covers.
//! - A `Take` by name whose value does not copy moves every unit its path
//!   covers, at its span (ownership §2, §6). A `Take` of a value that copies
//!   moves nothing.
//! - Everything reached through a reference is not located by name, and is
//!   not this analysis's.
//! - A join unions what each path may leave.
//!
//! Cost: a unit's fact only grows, and it holds two flags and the spans of
//! the body's moves, so each unit rises at most O(moves + 2) times; the
//! engine runs O(units · (moves + 2) · (blocks + edges)) block transfers,
//! each O(its instructions · the units they cover).

use std::collections::BTreeSet;

use acvus_ast::Span;
use rustc_hash::FxHashMap;

use crate::ir::ValueId;
use crate::ir::cfg::{BodyId, InstId, Module};
use crate::ir::parts::Path;
use crate::passes::analysis::{Analysis, AnalysisManager};
use crate::passes::domain::{JoinSemiLattice, Widen};

/// What the paths reaching a point may leave in one unit: written, never
/// written, or moved out at these spans. A point no path reaches holds none
/// of them; a unit is written there only where it is written and nothing
/// else.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Written {
    pub written: bool,
    pub never: bool,
    pub moved: BTreeSet<Span>,
}

impl Written {
    /// Whether the unit holds its content on every path that reaches.
    pub fn is_written(&self) -> bool {
        self.written && !self.never && self.moved.is_empty()
    }
}

impl JoinSemiLattice for Written {
    fn bottom() -> Self {
        Written::default()
    }

    fn join(&mut self, other: &Self) -> bool {
        let before = self.clone();
        self.written |= other.written;
        self.never |= other.never;
        self.moved.extend(other.moved.iter().copied());
        *self != before
    }

    fn le(&self, other: &Self) -> bool {
        (!self.written || other.written) && (!self.never || other.never) && self.moved.is_subset(&other.moved)
    }
}

/// Finite: two flags and the body's move spans. The widening is the join.
impl Widen for Written {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// The facts: each place's units, and each unit's `Written` just before
/// every instruction.
pub struct Initialization {
    units: FxHashMap<ValueId, Vec<Path>>,
    before: FxHashMap<InstId, FxHashMap<(ValueId, usize), Written>>,
}

impl Initialization {
    /// The units of `place`, in one order; `None` for a value that is not a
    /// place of the body.
    pub fn units(&self, place: ValueId) -> Option<&[Path]> {
        self.units.get(&place).map(Vec::as_slice)
    }

    /// What each unit of `place` overlapping `path` holds just before
    /// `inst`, with the unit.
    pub fn before(&self, inst: InstId, place: ValueId, path: &[crate::ir::PathSeg]) -> Vec<(&Path, &Written)> {
        todo!("the units of `place` overlapping `path`, each with its fact before `inst`")
    }
}

impl Analysis for Initialization {
    type Facts = Initialization;

    const READS: &'static [crate::ir::cfg::Aspect] = &[
        crate::ir::cfg::Aspect::Instructions,
        crate::ir::cfg::Aspect::CfgShape,
        crate::ir::cfg::Aspect::Types,
    ];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts {
        todo!("the units by the module doc, then the dense engine forward with the module doc's transfer, recorded before each instruction")
    }
}
