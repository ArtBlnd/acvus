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
/// `fn_types`: QualifiedRef → Ty mapping for all functions.
/// `recursive_fns`: set of functions involved in recursion (SCC).
pub fn optimize(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    optimize_inner(modules, fn_types, context_types, recursive_fns, false)
}

/// Optimize with untyped scalar register coloring (for kovac).
pub fn optimize_untyped(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    optimize_inner(modules, fn_types, context_types, recursive_fns, true)
}

fn optimize_inner(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
    untyped_scalars: bool,
) -> OptimizeResult {
    // ── Pass 1: SROA → SSA (per-module) → Inline (cross-module) ─────

    let mut ssa_modules = modules;
    for module in ssa_modules.values_mut() {
        run_pass1_body(&mut module.main, fn_types, context_types);
        for closure in module.closures.values_mut() {
            run_pass1_body(closure, fn_types, context_types);
        }
    }

    // Debug: check if any Ref(Param/Var) survive Pass 1.
    #[cfg(debug_assertions)]
    for (qref, module) in ssa_modules.iter() {
        for inst in &module.main.insts {
            if let crate::ir::InstKind::Ref { dst, target, .. } = &inst.kind {
                match target {
                    crate::ir::RefTarget::Var(slot) => {
                        eprintln!("[POST-PASS1] {:?} has Ref(Var({slot:?})) dst={dst:?}", qref);
                    }
                    crate::ir::RefTarget::Param(slot) => {
                        eprintln!("[POST-PASS1] {:?} has Ref(Param({slot:?})) dst={dst:?}", qref);
                    }
                    _ => {}
                }
            }
        }
    }
    let inlined = inliner::inline(&ssa_modules, recursive_fns);

    // ── Pass 2: Optimize + Validate (per-module, direct calls) ──────
    //
    // promote once → all passes on CfgBody → demote once → validate on MirBody.

    let mut result_modules = FxHashMap::default();
    let mut all_errors = Vec::new();

    for (qref, mut module) in inlined.modules {
        // Verify inline output: every use must have a def, every def must have a type.
        #[cfg(debug_assertions)]
        {
            use rustc_hash::FxHashSet;
            use crate::ir::ValueId;
            let mut defs: FxHashSet<ValueId> = FxHashSet::default();
            // Params and captures are always defined.
            defs.extend(module.main.param_regs().iter());
            defs.extend(module.main.capture_regs().iter());
            for inst in &module.main.insts {
                for d in crate::analysis::inst_info::defs(&inst.kind) {
                    defs.insert(d);
                    if !module.main.val_types.contains_key(&d) {
                        eprintln!("POST-INLINE def without type: {d:?} in {:?}", inst.kind);
                    }
                }
                // BlockLabel params are defs too.
                if let crate::ir::InstKind::BlockLabel { params, .. } = &inst.kind {
                    for &p in params {
                        defs.insert(p);
                    }
                }
            }
            for inst in &module.main.insts {
                for u in crate::analysis::inst_info::uses(&inst.kind) {
                    if !defs.contains(&u) {
                        eprintln!("POST-INLINE use without def: {u:?} in {:?}", inst.kind);
                    }
                }
            }
        }
        run_pass2_body(&mut module.main, fn_types, context_types, untyped_scalars);
        for closure in module.closures.values_mut() {
            run_pass2_body(closure, fn_types, context_types, untyped_scalars);
        }

        // Debug: check all ValueId types in final MirBody.
        #[cfg(debug_assertions)]
        for (ii, inst) in module.main.insts.iter().enumerate() {
            for u in crate::analysis::inst_info::uses(&inst.kind) {
                if !module.main.val_types.contains_key(&u) {
                    eprintln!("[POST-PASS2] inst {ii} USE missing type: {u:?} in {:?}", inst.kind);
                }
            }
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                if !module.main.val_types.contains_key(&d) {
                    eprintln!("[POST-PASS2] inst {ii} DEF missing type: {d:?} in {:?}", inst.kind);
                }
            }
        }

        // Validate on MirBody — after demote, catches demotion bugs.
        let errors = validate::validate(&module, fn_types, &FxHashMap::default());
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
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
) {
    optimize::sroa::run_body(body, context_types);
    let mut cfg = cfg::promote(std::mem::take(body));
    optimize::ssa_pass::run(&mut cfg, fn_types);
    optimize::dse::run(&mut cfg, fn_types);
    optimize::dce::run(&mut cfg, fn_types);
    *body = cfg::demote(cfg);
}

/// Pass 2: Full optimization pipeline on a single body.
/// SROA on MirBody, then promote once → all passes on CfgBody → demote once.
fn run_pass2_body(
    body: &mut crate::ir::MirBody,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    untyped_scalars: bool,
) {
    optimize::sroa::run_body(body, context_types);
    let mut cfg = cfg::promote(std::mem::take(body));
    run_pass2(&mut cfg, fn_types, untyped_scalars);
    *body = cfg::demote(cfg);
}

/// Pass 2 pipeline on CfgBody: SpawnSplit → SSA → DSE → CodeMotion → Reorder → RegColor.
fn run_pass2(cfg: &mut CfgBody, fn_types: &FxHashMap<QualifiedRef, Ty>, untyped_scalars: bool) {
    debug_check_defs(cfg, "before pass2");
    optimize::spawn_split::run(cfg, fn_types);
    debug_check_defs(cfg, "after spawn_split");
    optimize::ssa_pass::run(cfg, fn_types);
    debug_check_defs(cfg, "after ssa");
    optimize::dse::run(cfg, fn_types);
    debug_check_defs(cfg, "after dse");
    debug_check_types(cfg, "after dse");
    optimize::dce::run(cfg, fn_types);
    debug_check_defs(cfg, "after dce");
    debug_check_types(cfg, "after dce");
    debug_check_order(cfg, "before code_motion");

    // Run hoist and sink separately to isolate the problem.
    #[cfg(debug_assertions)]
    {
        eprintln!("=== B0 before hoist ===");
        for (ii, inst) in cfg.blocks[0].insts.iter().enumerate() {
            eprintln!("  {ii}: defs={:?} uses={:?} {:?}",
                crate::analysis::inst_info::defs(&inst.kind),
                crate::analysis::inst_info::uses(&inst.kind),
                inst.kind);
        }
    }
    optimize::code_motion::run(cfg, fn_types);
    #[cfg(debug_assertions)]
    {
        eprintln!("=== B0 after code_motion ===");
        for (ii, inst) in cfg.blocks[0].insts.iter().enumerate() {
            eprintln!("  {ii}: defs={:?} uses={:?} {:?}",
                crate::analysis::inst_info::defs(&inst.kind),
                crate::analysis::inst_info::uses(&inst.kind),
                inst.kind);
        }
    }
    debug_check_order(cfg, "after code_motion");
    debug_check_types(cfg, "after code_motion");
    debug_check_types(cfg, "after code_motion");
    optimize::reorder::run(cfg, fn_types);
    debug_check_types(cfg, "after reorder");
    // Before reg_color: dump Ref instructions and their types for debugging.
    #[cfg(debug_assertions)]
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for inst in &block.insts {
            if let crate::ir::InstKind::Ref { dst, target, .. } = &inst.kind {
                let has_type = cfg.val_types.contains_key(dst);
                eprintln!("[pre-regcolor] B{bi} Ref {dst:?} → {target:?} typed={has_type}");
            }
        }
    }
    #[cfg(debug_assertions)]
    {
        let ref_count: usize = cfg.blocks.iter().flat_map(|b| &b.insts)
            .filter(|i| matches!(i.kind, crate::ir::InstKind::Ref { .. })).count();
        let load_count: usize = cfg.blocks.iter().flat_map(|b| &b.insts)
            .filter(|i| matches!(i.kind, crate::ir::InstKind::Load { .. })).count();
        let total: usize = cfg.blocks.iter().map(|b| b.insts.len()).sum();
        eprintln!("[pre-regcolor] refs={ref_count} loads={load_count} total={total}");
        // Dump entry block Refs and Loads with their ValueIds.
        for inst in &cfg.blocks[0].insts {
            match &inst.kind {
                crate::ir::InstKind::Ref { dst, target, .. } => {
                    eprintln!("  B0 Ref {dst:?} → {target:?}");
                }
                crate::ir::InstKind::Load { dst, src, .. } => {
                    eprintln!("  B0 Load {dst:?} ← {src:?}");
                }
                _ => {}
            }
        }
    }
    if untyped_scalars {
        optimize::reg_color::color_body_untyped(cfg);
    } else {
        optimize::reg_color::color_body(cfg);
    }
    debug_check_types(cfg, "after reg_color");
}

#[cfg(debug_assertions)]
fn debug_check_defs(cfg: &CfgBody, phase: &str) {
    use rustc_hash::FxHashSet;
    use crate::ir::ValueId;
    let mut defs: FxHashSet<ValueId> = FxHashSet::default();
    defs.extend(cfg.params.iter().map(|(_, v)| v));
    defs.extend(cfg.captures.iter().map(|(_, v)| v));
    for block in &cfg.blocks {
        for &p in &block.params {
            defs.insert(p);
        }
        for inst in &block.insts {
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                defs.insert(d);
            }
        }
        if let crate::cfg::Terminator::ListStep { dst, index_dst, .. } = &block.terminator {
            defs.insert(*dst);
            defs.insert(*index_dst);
        }
    }
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            for u in crate::analysis::inst_info::uses(&inst.kind) {
                if !defs.contains(&u) {
                    eprintln!("[{phase}] B{bi}:{ii} use without def: {u:?} in {:?}", inst.kind);
                }
            }
        }
        // Check terminator uses too.
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
                eprintln!("[{phase}] B{bi} TERM use without def: {u:?} in {:?}", block.terminator);
            }
        }
    }
}

#[cfg(not(debug_assertions))]
fn debug_check_defs(_cfg: &CfgBody, _phase: &str) {}

/// Check SSA dominance property: every use must be dominated by its def.
/// - Same block: def instruction must come before use instruction.
/// - Different blocks: def's block must dominate use's block.
#[cfg(debug_assertions)]
fn debug_check_order(cfg: &CfgBody, phase: &str) {
    use rustc_hash::FxHashMap;
    use crate::ir::ValueId;
    use crate::analysis::domtree::DomTree;
    use crate::cfg::BlockIdx;

    let domtree = DomTree::build(cfg);

    // Build def location: ValueId → (block_idx, inst_idx_within_block).
    // inst_idx = usize::MAX means "block param" (defined at block entry).
    // inst_idx = usize::MAX - 1 means "terminator def" (ListStep).
    let mut def_loc: FxHashMap<ValueId, (usize, usize)> = FxHashMap::default();

    // Function params/captures: defined "before" block 0.
    for (_, v) in cfg.params.iter().chain(cfg.captures.iter()) {
        def_loc.insert(*v, (0, usize::MAX));
    }

    for (bi, block) in cfg.blocks.iter().enumerate() {
        for &p in &block.params {
            def_loc.insert(p, (bi, usize::MAX)); // block param = before all insts
        }
        for (ii, inst) in block.insts.iter().enumerate() {
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                def_loc.insert(d, (bi, ii));
            }
        }
        if let crate::cfg::Terminator::ListStep { dst, index_dst, .. } = &block.terminator {
            def_loc.insert(*dst, (bi, usize::MAX - 1));
            def_loc.insert(*index_dst, (bi, usize::MAX - 1));
        }
    }

    // Check every use.
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            for u in crate::analysis::inst_info::uses(&inst.kind) {
                let Some(&(def_bi, def_ii)) = def_loc.get(&u) else {
                    eprintln!("[{phase}] B{bi}:{ii} USE OF UNDEFINED: {u:?} in {:?}", inst.kind);
                    continue;
                };

                if def_bi == bi {
                    // Same block: def must come before use.
                    if def_ii != usize::MAX && def_ii >= ii {
                        eprintln!(
                            "[{phase}] B{bi}:{ii} ORDER VIOLATION (same block): \
                             use {u:?} at inst {ii}, def at inst {def_ii} in {:?}",
                            inst.kind
                        );
                    }
                } else {
                    // Different block: def's block must dominate use's block.
                    if !domtree.dominates(BlockIdx(def_bi), BlockIdx(bi)) {
                        eprintln!(
                            "[{phase}] B{bi}:{ii} DOMINANCE VIOLATION: \
                             use {u:?} in B{bi}, def in B{def_bi} (not dominator) in {:?}",
                            inst.kind
                        );
                    }
                }
            }
        }
    }
}

#[cfg(not(debug_assertions))]
fn debug_check_order(_cfg: &CfgBody, _phase: &str) {}

#[cfg(debug_assertions)]
fn debug_check_types(cfg: &CfgBody, phase: &str) {
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for inst in &block.insts {
            for def in crate::analysis::inst_info::defs(&inst.kind) {
                if !cfg.val_types.contains_key(&def) {
                    eprintln!("[{phase}] B{bi} DEF missing type: {def:?} in {:?}", inst.kind);
                }
            }
            for u in crate::analysis::inst_info::uses(&inst.kind) {
                if !cfg.val_types.contains_key(&u) {
                    eprintln!("[{phase}] B{bi} USE missing type: {u:?} in {:?}", inst.kind);
                }
            }
        }
    }
}
#[cfg(not(debug_assertions))]
fn debug_check_types(_cfg: &CfgBody, _phase: &str) {}
