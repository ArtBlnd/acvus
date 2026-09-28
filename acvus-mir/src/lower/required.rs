//! The required steps after lowering (D40, D55): meaning, not optimization.
//! Each edits through its own scope.
//! - `clone_strings`: a `String` used before its last use is cloned first
//!   (ownership §3), through `insert` and `replace_use` (G3).
//! - `remove_dead_context_ops`: a load whose value is never read and a
//!   commit of a context the body never writes do not exist (context §3),
//!   through `discharge` (G1).
//! - `insert_drops`: where a value ends (ownership §6), through
//!   `insert_drop` (S3). It runs after `check`, which admits the moves.

use crate::ir::cfg::Module;
use crate::passes::analysis::AnalysisManager;

pub fn clone_strings(module: &mut Module, am: &mut AnalysisManager<'_>) {
    todo!("lower::required::clone_strings")
}

pub fn remove_dead_context_ops(module: &mut Module, am: &mut AnalysisManager<'_>) {
    todo!("lower::required::remove_dead_context_ops")
}

pub fn insert_drops(module: &mut Module, am: &mut AnalysisManager<'_>) {
    todo!("lower::required::insert_drops")
}
