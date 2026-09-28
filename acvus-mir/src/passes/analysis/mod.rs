//! Analyses (passes-structure §1-2): where values flow. The two engines, and
//! the manager that computes, keeps and invalidates facts.

mod dense;
mod manager;
mod sparse;

pub use dense::{DenseAnalysis, DenseFacts, Direction, solve_dense};
pub use manager::{Analysis, AnalysisManager};
pub use sparse::{SparseAnalysis, SparseFacts, solve_sparse};
