//! Phase 5: Optimize
//!
//! Runs the full optimization pipeline on lowered MIR modules.
//!
//! Pass 1 (cross-module): SSA -> Inline
//! Pass 2 (per-module):   SpawnSplit -> CodeMotion -> Lsr -> Reorder -> SSA -> RegColor -> Validate

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
pub fn optimize(
    modules: FxHashMap<QualifiedRef, MirModule>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    optimize_inner(modules, context_types, recursive_fns)
}

fn optimize_inner(
    modules: FxHashMap<QualifiedRef, MirModule>,
    context_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    // -- Pass 0: moves, borrows and exhaustiveness as the source wrote
    // them (RFC-0029, RFC-0051) --

    let mut all_errors = Vec::new();
    for (qref, module) in &modules {
        let mut errors = validate::move_check::check_moves(module);
        errors.extend(validate::borrow_check::check_borrows(module));
        // A `match` is exhaustive (RFC-0051). Like the move check, it reads
        // the shape the source wrote: `switch_expand` in pass 1 replaces
        // the dispatch with the chain the machine runs today.
        errors.extend(validate::exhaustive::check_exhaustive(module));
        if !errors.is_empty() {
            all_errors.push((*qref, errors));
        }
    }

    // -- Pass 1: SSA (per-module) -> Inline (cross-module) -----

    let mut ssa_modules = modules;
    for module in ssa_modules.values_mut() {
        run_pass1_body(&mut module.main);
        for closure in module.closures.values_mut() {
            run_pass1_body(closure);
        }
    }

    let inlined = inliner::inline(&ssa_modules, recursive_fns);

    // -- Pass 2: Optimize + Validate (per-module, direct calls) ------

    let mut result_modules = FxHashMap::default();

    for (qref, mut module) in inlined.modules {
        run_pass2_body(&mut module.main);
        for closure in module.closures.values_mut() {
            run_pass2_body(closure);
        }

        let errors = validate::validate(&module);
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

/// Pass 1: SSA -> DSE -> DCE on a single body.
fn run_pass1_body(body: &mut crate::ir::MirBody) {
    let mut cfg = cfg::promote(std::mem::take(body));
    // RFC-0051, first half: the machine has no `switch` yet, so the
    // dispatch the lowering wrote becomes the chain before any other pass
    // runs. The second half deletes this line.
    optimize::switch_expand::run(&mut cfg);
    optimize::ssa_pass::run(&mut cfg);
    optimize::string_copy::run(&mut cfg);
    optimize::dse::run(&mut cfg);
    optimize::dce::run(&mut cfg);
    *body = cfg::demote(cfg);
}

/// Pass 2: Full optimization pipeline on a single body.
fn run_pass2_body(body: &mut crate::ir::MirBody) {
    let mut cfg = cfg::promote(std::mem::take(body));
    run_pass2(&mut cfg);
    *body = cfg::demote(cfg);
}

/// Pass 2 pipeline on CfgBody.
fn run_pass2(cfg: &mut CfgBody) {
    optimize::commute::run(cfg);
    optimize::spawn_split::run(cfg);
    // RFC-0053: an aggregate no use lets out of the body never exists.
    // Before `ssa_pass`, whose builder places the phis its parts need;
    // before `dce`, which sweeps the constructor left with no reader.
    optimize::sroa::run(cfg);
    optimize::ssa_pass::run(cfg);
    optimize::string_copy::run(cfg);
    // RFC-0055: after `ssa_pass`, which brings a constant and its reader
    // into one body; before `dse`/`dce`, which sweep the operands a fold
    // left with no reader.
    optimize::fold::run(cfg);
    optimize::dse::run(cfg);
    optimize::dce::run(cfg);
    optimize::code_motion::run(cfg);
    // RFC-0056: after the hoist, which puts a loop's invariants above the
    // header and leaves the preheader a block of its own; before the
    // reorder, which schedules within a block.
    optimize::lsr::run(cfg);
    optimize::reorder::run(cfg);
    debug_validate(cfg);
    optimize::drop_insertion::insert_drops(cfg, &cfg.val_types.clone());
}

/// Validate CfgBody after optimization: check use-def integrity, SSA dominance, and type coverage.
/// Collects all violations and panics if any are found.
#[cfg(debug_assertions)]
fn debug_validate(cfg: &CfgBody) {
    use crate::analysis::domtree::DomTree;
    use crate::cfg::BlockIdx;
    use crate::ir::ValueId;
    use rustc_hash::{FxHashMap, FxHashSet};

    let mut errors: Vec<String> = Vec::new();

    // -- Build def set and def locations --
    let mut defs: FxHashSet<ValueId> = FxHashSet::default();
    let mut def_loc: FxHashMap<ValueId, (usize, usize)> = FxHashMap::default();

    // Function params/captures: defined "before" block 0.
    for v in cfg.entry_defs() {
        defs.insert(v);
        def_loc.insert(v, (0, usize::MAX));
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
    }

    let domtree = DomTree::build(cfg);

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // -- Check instructions --
        for (ii, inst) in block.insts.iter().enumerate() {
            // Type coverage: every def and use must have a type.
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                if !cfg.val_types.contains_key(&d) {
                    errors.push(format!(
                        "B{bi}:{ii} DEF missing type: {d:?} in {:?}",
                        inst.kind
                    ));
                }
            }
            for u in crate::analysis::inst_info::uses(&inst.kind) {
                if !cfg.val_types.contains_key(&u) {
                    errors.push(format!(
                        "B{bi}:{ii} USE missing type: {u:?} in {:?}",
                        inst.kind
                    ));
                }
                // Use-def: every use must have a def.
                let Some(&(def_bi, def_ii)) = def_loc.get(&u) else {
                    errors.push(format!(
                        "B{bi}:{ii} use without def: {u:?} in {:?}",
                        inst.kind
                    ));
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

        // -- Check terminator uses --
        let term_uses = match &block.terminator {
            crate::cfg::Terminator::Return { value, order } => {
                std::iter::once(*value).chain(*order).collect()
            }
            crate::cfg::Terminator::Jump { args, .. } => args.clone(),
            crate::cfg::Terminator::JumpIf {
                cond,
                then_args,
                else_args,
                ..
            } => {
                let mut v = vec![*cond];
                v.extend(then_args);
                v.extend(else_args);
                v
            }
            crate::cfg::Terminator::Switch { tag, arms, default } => {
                let mut v = vec![*tag];
                for (_, _, args) in arms {
                    v.extend(args);
                }
                if let Some((_, args)) = default {
                    v.extend(args);
                }
                v
            }
            crate::cfg::Terminator::Fallthrough | crate::cfg::Terminator::Diverge => vec![],
        };
        for u in &term_uses {
            if !defs.contains(u) {
                errors.push(format!(
                    "B{bi} TERM use without def: {u:?} in {:?}",
                    block.terminator
                ));
            }
        }
    }

    if !errors.is_empty() {
        let msg = errors.join("\n  ");
        let mut dump = String::new();
        for (bi, block) in cfg.blocks.iter().enumerate() {
            dump.push_str(&format!("B{bi} params={:?}\n", block.params));
            for (ii, inst) in block.insts.iter().enumerate() {
                dump.push_str(&format!("  {ii}: {:?}\n", inst.kind));
            }
            dump.push_str(&format!("  -> {:?}\n", block.terminator));
        }
        panic!(
            "CfgBody validation failed ({} errors):\n  {msg}\n{dump}",
            errors.len()
        );
    }
}

#[cfg(not(debug_assertions))]
fn debug_validate(_cfg: &CfgBody) {}
