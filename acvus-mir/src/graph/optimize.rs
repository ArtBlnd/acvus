//! Phase 5: Optimize
//!
//! Runs the full optimization pipeline on lowered MIR modules.
//!
//! Pass 1 (cross-module): SSA → Inline
//! Pass 2 (per-module):   SpawnSplit → Reorder → SSA → RegColor → Validate

use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::inliner;
use crate::graph::QualifiedRef;
use crate::ir::MirModule;
use crate::optimize::{self, FnTypes, ValidateResult};
use crate::pass::PassContext;
use crate::ty::Ty;
use crate::validate::ValidationError;

/// Result of the optimization pipeline.
pub struct OptimizeResult {
    /// Optimized modules, keyed by function QualifiedRef.
    pub modules: FxHashMap<QualifiedRef, MirModule>,
    /// Validation errors per function (empty = valid).
    pub errors: Vec<(QualifiedRef, Vec<ValidationError>)>,
}

/// Run the full optimization pipeline.
///
/// `modules`: lowered MIR modules from Phase 3 (lower).
/// `fn_types`: QualifiedRef → Ty mapping for all functions.
/// `recursive_fns`: set of functions involved in recursion (SCC).
pub fn optimize(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    // ── Pass 1: SSA (per-module) → Inline (cross-module) ────────────

    let mut ssa_modules = modules;
    for module in ssa_modules.values_mut() {
        optimize::ssa_pass::run(&mut module.main, fn_types);
        for closure in module.closures.values_mut() {
            optimize::ssa_pass::run(closure, fn_types);
        }
    }

    let inlined = inliner::inline(&ssa_modules, recursive_fns);

    // ── Pass 2: Optimize + Validate (per-module via PassManager) ────

    let manager = optimize::pass2_manager();
    let mut result_modules = FxHashMap::default();
    let mut all_errors = Vec::new();

    for (qref, module) in inlined.modules {
        let mut ctx = PassContext::new();
        ctx.insert(FnTypes(fn_types.clone()));

        let (optimized, ctx) = manager.run_with_context(module, ctx);
        let validation = ctx.get::<ValidateResult>();

        if !validation.0.is_empty() {
            all_errors.push((qref, validation.0.clone()));
        }

        result_modules.insert(qref, optimized);
    }

    OptimizeResult {
        modules: result_modules,
        errors: all_errors,
    }
}
