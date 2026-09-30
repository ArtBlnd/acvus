//! The scalar evolution: how each integer value of a body evolves over the
//! iterations of the loops around it, and how many iterations a loop runs.
//!
//! A value is an element of the value ring (`domain::ring`) over the body's
//! atoms, or a first-degree recurrence `{start, +, step}_L` over one loop `L`:
//! its value at `L`'s k-th iteration is `start + k·step`, where `step` is
//! invariant in `L` and `start` is invariant in `L` too (it may itself be a
//! recurrence of a loop around `L`). No recurrence is of a higher degree in its
//! own loop: a value whose step changes with its loop's own iteration is
//! unknown (the user's decision: each loop first degree, coefficients in the
//! ring; no full ring of recurrences).
//!
//! Arithmetic is exact over the integers: overflow is undefined (operators
//! §intro), so a recurrence needs no wrap flag and no pass reasons about one.
//! Sign and order are not the evolution's; a reader asks the interval domain.
//!
//! The evolution is a query over the SSA, kept by the analysis manager; it
//! writes nothing into the IR (compilation §4: the backend decides what to do
//! with a loop).
//!
//! Cost: one pass over the body's values in dominance order, each value's
//! evolution built from its operands' by the ring's bounded operations, so
//! O(values · MAX_TERMS²); no fixpoint, since a loop's header parameters are
//! read off their entry and back-edge arguments once.

use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::ir::ValueId;
use crate::ir::cfg::{Aspect, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager, LoopId};
use crate::passes::domain::{Polynomial, Symbol};

/// How a value evolves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evolution {
    /// The same element of the ring on every iteration of every loop around
    /// the value's definition that it is read in.
    Invariant(Polynomial<Symbol>),
    /// `{start, +, step}` over `over`: `start + k·step` at the loop's k-th
    /// iteration, both invariant in `over`.
    Recurrence(Rc<Recurrence>),
    /// No evolution this analysis names.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recurrence {
    pub over: LoopId,
    pub start: Evolution,
    pub step: Evolution,
}

/// How many times a loop's body runs, where the analysis names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripCount {
    /// Exactly this many, on every run that enters the loop, as an
    /// evolution invariant in the loop (it may vary with a loop around it).
    Exact(Evolution),
    /// At most this many: a loop the body may leave early, or a pull loop
    /// whose count its maker's declared relation bounds (extern fn §3,
    /// relation; source, count).
    AtMost(Evolution),
}

/// The evolution facts of one body.
pub struct Evolutions {
    of: FxHashMap<ValueId, Evolution>,
    trips: FxHashMap<LoopId, TripCount>,
}

impl Evolutions {
    /// The evolution of `value`; `Unknown` for a value the analysis does not
    /// name.
    pub fn of(&self, value: ValueId) -> &Evolution {
        todo!()
    }

    /// Whether `value` is invariant in `over`: its evolution holds no
    /// recurrence of `over` or of a loop inside it.
    pub fn is_invariant(&self, value: ValueId, over: LoopId) -> bool {
        todo!()
    }

    /// The trip count of `over`, where the analysis names one. A `For` over a
    /// range runs `max(hi - at, 0)` times, over a slice its length, over an
    /// array its length; a `While` whose only exit compares a recurrence of it
    /// with a value invariant in it runs the count that comparison gives; a
    /// pull loop runs at most what its iterator's maker's relation states.
    pub fn trip_count(&self, over: LoopId) -> Option<&TripCount> {
        todo!()
    }
}

/// The analysis the manager keeps. It reads the body's values and control
/// flow, the loop tree, and the declarations' relations and sources.
pub struct ScalarEvolution;

impl Analysis for ScalarEvolution {
    type Facts = Evolutions;
    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Evolutions {
        todo!("read the loop tree through the manager, then the values in dominance order")
    }
}
