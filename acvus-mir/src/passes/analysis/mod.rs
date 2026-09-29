//! Analyses (passes-structure §1-2): where values flow. The two engines, and
//! the manager that computes, keeps and invalidates facts.

mod content_liveness;
mod initialization;
mod match_facts;
mod dense;
mod dominators;
mod escapes;
mod liveness;
mod loans;
mod loop_tree;
mod manager;
mod post_dominators;
mod sparse;
mod values;

#[cfg(test)]
mod tests;

pub use content_liveness::{ContentLiveness, Presence};
pub use initialization::{Initialization, Written};
pub use match_facts::MatchFacts;
pub use dense::{DenseAnalysis, DenseFacts, Direction, solve_dense};
pub use dominators::{Dominators, Point};
pub use escapes::Escapes;
pub use liveness::Liveness;
pub use loans::{Access, ElementAt, Loan, LoanSet, Loans, Made, Mode, Place, Reach, Storage, TouchedPlace};
pub use loop_tree::{LoopId, LoopTree};
pub use post_dominators::PostDominators;
pub use manager::{Analysis, AnalysisManager};
pub use sparse::{Executable, SparseAnalysis, SparseFacts, solve_sparse};
pub use values::{AffineValues, ConstantValues, IntervalValues};
pub(in crate::passes) use values::ValueBounds;
pub(in crate::passes) use values::switch_key;
