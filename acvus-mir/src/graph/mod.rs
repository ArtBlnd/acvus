//! Graph-level type resolution engine.
//!
//! Receives a `CompilationGraph` (flat declaration of compilation units, extern
//! declarations, scopes, and bindings) and resolves all types through DAG
//! topological sort + SCC unification.
//!
//! This module does NOT know about orchestration concepts (nodes, strategies,
//! bind/init). It works with generic compilation units and their dependencies.

mod types;
mod resolve;
#[cfg(test)]
mod tests;

pub use types::*;
pub use resolve::*;
