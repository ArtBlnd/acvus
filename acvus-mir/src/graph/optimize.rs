//! Phase 5: Optimize
//!
//! Runs the full optimization pipeline on lowered MIR modules.
//!
//! Pass 1 (cross-module): SSA → Inline
//! Pass 2 (per-module):   SpawnSplit → CodeMotion → Reorder → SSA → RegColor → Validate
//!
//! Each body is promoted to CfgBody once, all passes run, then demoted once.
//! Validate runs on MirBody (after demote) to catch demotion bugs.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{self, CfgBody};
use crate::graph::QualifiedRef;
use crate::graph::inliner;
use crate::ir::MirModule;
use crate::optimize;

use crate::ty::Ty;
use crate::validate::{self, ValidationError};

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
/// `fn_metadata`: QualifiedRef → Ty mapping for all functions.
/// `recursive_fns`: set of functions involved in recursion (SCC).
pub fn optimize(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_metadata: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    optimize_inner(modules, fn_metadata, context_types, recursive_fns, false)
}

/// Optimize with untyped scalar register coloring (for kovac).
pub fn optimize_untyped(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_metadata: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    optimize_inner(modules, fn_metadata, context_types, recursive_fns, true)
}

fn optimize_inner(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_metadata: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
    untyped_scalars: bool,
) -> OptimizeResult {
    // ── Pass 1: SROA → SSA (per-module) → Inline (cross-module) ─────

    let mut ssa_modules = modules;
    for module in ssa_modules.values_mut() {
        run_pass1_body(&mut module.main, fn_metadata, context_types);
        for closure in module.closures.values_mut() {
            run_pass1_body(closure, fn_metadata, context_types);
        }
    }

    let inlined = inliner::inline(&ssa_modules, recursive_fns);

    // ── Pass 2: Optimize + Validate (per-module, direct calls) ──────
    //
    // promote once → all passes on CfgBody → demote once → validate on MirBody.

    let mut result_modules = FxHashMap::default();
    let mut all_errors = Vec::new();

    for (qref, mut module) in inlined.modules {
        run_pass2_body(&mut module.main, fn_metadata, context_types, untyped_scalars);
        for closure in module.closures.values_mut() {
            run_pass2_body(closure, fn_metadata, context_types, untyped_scalars);
        }

        // Validate on MirBody — after demote, catches demotion bugs.
        let errors = validate::validate(&module, fn_metadata, &FxHashMap::default());
        if !errors.is_empty() {
            all_errors.push((qref, errors));
        }

        result_modules.insert(qref, module);
    }

    OptimizeResult {
        modules: result_modules,
        errors: all_errors,
    }
}

/// Pass 1: SROA → SSA → DSE → DCE on a single body.
fn run_pass1_body(
    body: &mut crate::ir::MirBody,
    fn_metadata: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
) {
    optimize::sroa::run_body(body, context_types);
    let mut cfg = cfg::promote(std::mem::take(body));
    optimize::ssa_pass::run(&mut cfg, fn_metadata);
    optimize::dse::run(&mut cfg, fn_metadata);
    optimize::dce::run(&mut cfg, fn_metadata);
    *body = cfg::demote(cfg);
}

/// Pass 2: Full optimization pipeline on a single body.
/// SROA on MirBody, then promote once → all passes on CfgBody → demote once.
fn run_pass2_body(
    body: &mut crate::ir::MirBody,
    fn_metadata: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    untyped_scalars: bool,
) {
    optimize::sroa::run_body(body, context_types);
    let mut cfg = cfg::promote(std::mem::take(body));
    run_pass2(&mut cfg, fn_metadata, untyped_scalars);
    *body = cfg::demote(cfg);
}

/// Pass 2 pipeline on CfgBody: SpawnSplit → SSA → DSE → CodeMotion → Reorder → RegColor.
fn run_pass2(cfg: &mut CfgBody, fn_metadata: &FxHashMap<QualifiedRef, Ty>, untyped_scalars: bool) {
    optimize::spawn_split::run(cfg, fn_metadata);
    optimize::ssa_pass::run(cfg, fn_metadata);
    optimize::dse::run(cfg, fn_metadata);
    optimize::dce::run(cfg, fn_metadata);
    optimize::code_motion::run(cfg, fn_metadata);
    optimize::reorder::run(cfg, fn_metadata);
    debug_validate(cfg);
    optimize::drop_insertion::insert_drops(cfg, &cfg.val_types.clone());
    if untyped_scalars {
        optimize::reg_color::color_body_untyped(cfg);
    } else {
        optimize::reg_color::color_body(cfg);
    }
}

/// Validate CfgBody after optimization: check use-def integrity, SSA dominance, and type coverage.
/// Collects all violations and panics if any are found.
#[cfg(debug_assertions)]
fn debug_validate(cfg: &CfgBody) {
    use rustc_hash::{FxHashMap, FxHashSet};
    use crate::ir::ValueId;
    use crate::analysis::domtree::DomTree;
    use crate::cfg::BlockIdx;

    let mut errors: Vec<String> = Vec::new();

    // ── Build def set and def locations ──
    let mut defs: FxHashSet<ValueId> = FxHashSet::default();
    let mut def_loc: FxHashMap<ValueId, (usize, usize)> = FxHashMap::default();

    // Function params/captures: defined "before" block 0.
    for (_, v) in cfg.params.iter().chain(cfg.captures.iter()) {
        defs.insert(*v);
        def_loc.insert(*v, (0, usize::MAX));
    }

    for (bi, block) in cfg.blocks.iter().enumerate() {
        for &p in &block.params {
            defs.insert(p);
            def_loc.insert(p, (bi, usize::MAX));
        }
        for (ii, inst) in block.insts.iter().enumerate() {
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                defs.insert(d);
                def_loc.insert(d, (bi, ii));
            }
        }
        if let crate::cfg::Terminator::ListStep { dst, index_dst, .. } = &block.terminator {
            defs.insert(*dst);
            defs.insert(*index_dst);
            def_loc.insert(*dst, (bi, usize::MAX - 1));
            def_loc.insert(*index_dst, (bi, usize::MAX - 1));
        }
    }

    let domtree = DomTree::build(cfg);

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // ── Check instructions ──
        for (ii, inst) in block.insts.iter().enumerate() {
            // Type coverage: every def and use must have a type.
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                if !cfg.val_types.contains_key(&d) {
                    errors.push(format!("B{bi}:{ii} DEF missing type: {d:?} in {:?}", inst.kind));
                }
            }
            for u in crate::analysis::inst_info::uses(&inst.kind) {
                if !cfg.val_types.contains_key(&u) {
                    errors.push(format!("B{bi}:{ii} USE missing type: {u:?} in {:?}", inst.kind));
                }
                // Use-def: every use must have a def.
                let Some(&(def_bi, def_ii)) = def_loc.get(&u) else {
                    errors.push(format!("B{bi}:{ii} use without def: {u:?} in {:?}", inst.kind));
                    continue;
                };
                // SSA dominance.
                if def_bi == bi {
                    if def_ii != usize::MAX && def_ii >= ii {
                        errors.push(format!(
                            "B{bi}:{ii} ORDER VIOLATION (same block): \
                             use {u:?} at inst {ii}, def at inst {def_ii} in {:?}",
                            inst.kind
                        ));
                    }
                } else if !domtree.dominates(BlockIdx(def_bi), BlockIdx(bi)) {
                    errors.push(format!(
                        "B{bi}:{ii} DOMINANCE VIOLATION: \
                         use {u:?} in B{bi}, def in B{def_bi} (not dominator) in {:?}",
                        inst.kind
                    ));
                }
            }
        }

        // ── Check terminator uses ──
        let term_uses = match &block.terminator {
            crate::cfg::Terminator::Return(v) => vec![*v],
            crate::cfg::Terminator::Jump { args, .. } => args.clone(),
            crate::cfg::Terminator::JumpIf { cond, then_args, else_args, .. } => {
                let mut v = vec![*cond];
                v.extend(then_args);
                v.extend(else_args);
                v
            }
            crate::cfg::Terminator::ListStep { list, index_src, done_args, .. } => {
                let mut v = vec![*list, *index_src];
                v.extend(done_args);
                v
            }
            crate::cfg::Terminator::Fallthrough => vec![],
        };
        for u in &term_uses {
            if !defs.contains(u) {
                errors.push(format!("B{bi} TERM use without def: {u:?} in {:?}", block.terminator));
            }
        }
    }

    if !errors.is_empty() {
        let msg = errors.join("\n  ");
        panic!("CfgBody validation failed ({} errors):\n  {msg}", errors.len());
    }
}

#[cfg(not(debug_assertions))]
fn debug_validate(_cfg: &CfgBody) {}
