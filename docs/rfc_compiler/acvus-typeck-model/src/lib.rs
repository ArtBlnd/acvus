//! An executable definition of how the transparency-constraint type system
//! solves a program.
//!
//! Organized by responsibility into:
//! - [`types`]: Closed types, signatures, declarations, literal domains, join (`≤`), and coercions (`⇝`).
//! - [`program`]: Input program AST and constraints.
//! - [`record`]: Equivalence classes, recorded relations, disjunctive constraints, and the uses of declarations with their choices of a candidate.
//! - [`solve`]: Binding-group solver loop, unification, structural decomposition, and polymorphic group summaries (`Cut`, `Certificate`, `Instance`).
//! - [`judgment`]: Unified priority-ordered decision pipeline (`lower_bounds` → `requirements` → `occurrences` → `disjunction` → `fallback`).
//! - [`check`]: Post-solving relation verification, `I*` soundness/completeness checking, and trace reporting.

#[cfg(test)]
mod cases;
pub mod check;
pub mod judgment;
pub mod program;
pub mod record;
pub mod solve;
pub mod solving;
pub mod types;

// Re-export submodules at the crate root so paths stay flat and ergonomic.
pub use check::{istar, trace};
#[cfg(test)]
pub(crate) use program::fixture;
pub use record::{choice, class, disjunction, relation};
pub use solve::{certificate, cut, decomposition, explore, instance, unify};
pub use types::{coerce, decl, domain, join, lit, read, sig, ty};
