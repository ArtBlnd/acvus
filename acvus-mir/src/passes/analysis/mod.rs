//! Analyses (passes-structure §1-2): where values flow. The two engines, and
//! the manager that computes, keeps and invalidates facts.

mod dense;
mod dominators;
mod escapes;
mod liveness;
mod loans;
mod loop_tree;
mod manager;
mod post_dominators;
mod sparse;

#[cfg(test)]
mod tests;

pub use dense::{DenseAnalysis, DenseFacts, Direction, solve_dense};
pub use dominators::{Dominators, Point};
pub use escapes::Escapes;
pub use liveness::Liveness;
pub use loans::Loans;
pub use loop_tree::{LoopId, LoopTree};
pub use post_dominators::PostDominators;
pub use manager::{Analysis, AnalysisManager};
pub use sparse::{Executable, SparseAnalysis, SparseFacts, solve_sparse};
