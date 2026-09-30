//! What a run of an instruction, a loop or a body may do besides giving its
//! value: trap, or not finish. Whether a run traps and whether it ends are
//! part of its meaning (compilation §1, effects §4), so a pass that removes,
//! moves or runs ahead an instruction asks here first.
//!
//! What traps is the language's (operators): integer division or remainder
//! by zero, an index outside its container, and `unwrap` of `None` or
//! `Err`. Nothing else traps; integer overflow is undefined, not a trap
//! (undefined behaviour §2). An operation that may trap by its kind does not
//! where the value analyses prove its operand clear of the case: a divisor
//! whose interval excludes zero, an index the bounds proof places inside its
//! slice, an `unwrap` whose variant the match facts know.
//!
//! What may not finish: a `While` the scalar evolution names no trip count
//! for, and a call whose ending is not known to be one. A `For` always
//! finishes. A call's ending composes (extern fn §4): an extern's declared
//! `Ends` states its own code, and the work the call runs ends and traps as
//! that work does, a closure argument by the bodies its function type's
//! values name, and work a value it is given holds by nothing known, so as
//! may-trap and may-not-finish. A call of a body of the module ends as that
//! body does, decided callees first over the call graph; a callee in the
//! caller's own strongly connected component may trap and may not finish.
//!
//! The analysis is a query the manager keeps per body; it writes nothing
//! into the IR. It asks the manager for each callee's endings outside its own
//! component, so the manager never meets a body it is computing.
//!
//! Cost: O(instructions + loops) per body over the facts it reads (the
//! value analyses, the match facts, the scalar evolution and the callees'
//! endings), plus the call graph's O(bodies + instructions) once per module.

use rustc_hash::FxHashMap;

use crate::ir::cfg::{Aspect, BodyId, InstId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager, LoopId};

/// How a run of a body ends, as a caller sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ending {
    /// Some run may end in a trap.
    pub may_trap: bool,
    /// Every run that does not trap ends.
    pub finishes: bool,
}

/// The endings of one body's instructions and loops, and of the body.
pub struct Endings {
    of_inst: FxHashMap<InstId, Ending>,
    of_loop: FxHashMap<LoopId, bool>,
    body: Ending,
}

impl Endings {
    /// Whether a run of `inst` may trap.
    pub fn may_trap(&self, inst: InstId) -> bool {
        todo!()
    }

    /// Whether every run of `inst` that does not trap ends.
    pub fn finishes(&self, inst: InstId) -> bool {
        todo!()
    }

    /// Whether every run of `over` that does not trap ends: a `For` always,
    /// a `While` where the scalar evolution names its trip count and every
    /// instruction of the loop finishes.
    pub fn loop_finishes(&self, over: LoopId) -> bool {
        todo!()
    }

    /// The body's own ending, which its callers read.
    pub fn body(&self) -> Ending {
        self.body
    }
}

/// The analysis the manager keeps.
pub struct Raise;

impl Analysis for Raise {
    type Facts = Endings;
    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Endings {
        todo!("callees outside the body's component first, through the manager")
    }
}
