//! The sparse engine: a value domain over SSA def-use. Cost: each value's
//! fact rises at most the domain's height, and each rise revisits its uses
//! (O(uses × height)); a domain of infinite height widens.

use rustc_hash::FxHashMap;

use crate::ir::cfg::{BlockId, Body, Inst, Terminator};
use crate::ir::ValueId;
use crate::passes::domain::ValueDomain;

/// Which successors of a terminator may run, read from the facts of its
/// operands (conditional propagation, Wegman-Zadeck): only edges that may
/// run carry facts into their targets.
pub enum Executable {
    All,
    Only(Vec<BlockId>),
}

/// A sparse analysis: which value domain it runs, and how an instruction
/// that is not a value operation maps to it (calls read declared facts only).
pub trait SparseAnalysis {
    type Domain: ValueDomain;

    fn domain(&self) -> &Self::Domain;

    /// The fact of an instruction's result from its operands' facts, in the
    /// order the instruction lists its operands.
    fn transfer(
        &self,
        inst: &Inst,
        operands: &[&<Self::Domain as ValueDomain>::Value],
    ) -> <Self::Domain as ValueDomain>::Value;

    /// Which successors may run, from the facts of the terminator's operands.
    fn executable(
        &self,
        terminator: &Terminator,
        operands: &[&<Self::Domain as ValueDomain>::Value],
    ) -> Executable;
}

pub struct SparseFacts<D>
where
    D: ValueDomain,
{
    values: FxHashMap<ValueId, D::Value>,
}

impl<D> SparseFacts<D>
where
    D: ValueDomain,
{
    /// # Panics
    /// `value` is not a value of the body the facts were computed for.
    pub fn of(&self, value: ValueId) -> &D::Value {
        self.values.get(&value).expect("sparse facts cover every value of their body")
    }
}

pub fn solve_sparse<A>(analysis: &A, body: &Body) -> SparseFacts<A::Domain>
where
    A: SparseAnalysis,
{
    todo!("passes::analysis::solve_sparse")
}
