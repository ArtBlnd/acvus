//! Dead Code Elimination (DCE) — mark-sweep on CfgBody.
//!
//! Removes instructions that don't contribute to observable behavior.
//! Observable = Return value, context Store, IO (Eval), effectful FunctionCall.
//!
//! Algorithm:
//! 1. **Root**: instructions with side effects are unconditionally live.
//! 2. **Backward walk**: trace operands of live instructions → mark their
//!    definitions as live → trace their operands → fixpoint.
//! 3. **Sweep**: remove non-live instructions.
//!
//! Runs post-SSA, post-DSE. Catches: dead inline residue, unused Ref/Load,
//! dead computation chains, unused Spawn handles.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{CfgBody, Terminator};
use crate::graph::{FnMetadata, QualifiedRef};
use crate::ir::{Callee, InstKind, ValueId};
use crate::ty::{Effect, Ty};
use crate::analysis::inst_info;

// ── Def location ────────────────────────────────────────────────────

/// Where a ValueId is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DefLoc {
    /// Defined by an instruction at (block, inst_index).
    Inst(usize, usize),
    /// Defined as a block parameter at (block, param_index).
    BlockParam(usize, usize),
    /// Defined by a terminator (ListStep dst/index_dst).
    Terminator(usize),
    /// Function parameter or capture — always live.
    EntryParam,
}

/// Build ValueId → DefLoc mapping.
fn build_def_map(cfg: &CfgBody) -> FxHashMap<ValueId, DefLoc> {
    let mut map = FxHashMap::default();

    // Entry params and captures are always live.
    for &(_, v) in cfg.params.iter().chain(cfg.captures.iter()) {
        map.insert(v, DefLoc::EntryParam);
    }

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // Block params.
        for (pi, &param) in block.params.iter().enumerate() {
            map.insert(param, DefLoc::BlockParam(bi, pi));
        }

        // Instructions.
        for (ii, inst) in block.insts.iter().enumerate() {
            for d in inst_info::defs(&inst.kind) {
                map.insert(d, DefLoc::Inst(bi, ii));
            }
        }

        // Terminator defs.
        if let Terminator::ListStep { dst, index_dst, .. } = &block.terminator {
            map.insert(*dst, DefLoc::Terminator(bi));
            map.insert(*index_dst, DefLoc::Terminator(bi));
        }
    }

    map
}

// ── Root identification ─────────────────────────────────────────────

/// Is this instruction a root (has side effects, unconditionally live)?
///
/// An instruction with ANY effect (read, write, IO, self_modifying) must not
/// be removed. Only provably pure instructions can be dead.
fn is_root(kind: &InstKind, fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>) -> bool {
    match kind {
        // Context store — externally observable.
        InstKind::Store { .. } => true,

        // Eval — IO execution point.
        InstKind::Eval { .. } => true,

        // FunctionCall: root unless provably pure.
        InstKind::FunctionCall { callee, .. } => match callee {
            Callee::Indirect(_) => true, // Unknown effect → root.
            Callee::Direct(qref) => !is_pure_fn(fn_metadata, qref),
        },

        // Spawn: pure (deferred execution). The actual effect happens at Eval.
        // Dead if handle is unused (no Eval consumes it).
        InstKind::Spawn { .. } => false,

        // Everything else: pure computation, dead if result unused.
        _ => false,
    }
}

/// Check if a function is provably pure (no effects at all).
fn is_pure_fn(fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>, qref: &QualifiedRef) -> bool {
    match fn_metadata.get(qref).map(|m| &m.ty) {
        Some(Ty::Fn {
            effect: Effect::Resolved(eff),
            ..
        }) => eff.is_pure(),
        _ => false, // Unknown → not pure (conservative).
    }
}

// ── Mark phase ──────────────────────────────────────────────────────

/// Collect all uses from a terminator.
fn terminator_uses(term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::Return(val) => vec![*val],
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend_from_slice(then_args);
            v.extend_from_slice(else_args);
            v
        }
        Terminator::ListStep {
            list,
            index_src,
            done_args,
            ..
        } => {
            let mut v = vec![*list, *index_src];
            v.extend_from_slice(done_args);
            v
        }
        Terminator::Fallthrough => vec![],
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Run DCE on a CfgBody. Removes all instructions that don't contribute
/// to observable behavior (Return, Store, Eval, effectful calls).
pub fn run(cfg: &mut CfgBody, fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>) {
    let def_map = build_def_map(cfg);

    // Live instruction set: (block_idx, inst_idx).
    let mut live_insts: FxHashSet<(usize, usize)> = FxHashSet::default();
    // Live terminators (always live, but track for block param tracing).
    let mut live_terminators: FxHashSet<usize> = FxHashSet::default();
    // Worklist of ValueIds to trace.
    let mut worklist: Vec<ValueId> = Vec::new();

    // Phase 1: seed roots.
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            if is_root(&inst.kind, fn_metadata) {
                live_insts.insert((bi, ii));
                worklist.extend(inst_info::uses(&inst.kind));
            }
        }

        // Terminators are always live — their uses are roots.
        live_terminators.insert(bi);
        worklist.extend(terminator_uses(&block.terminator));
    }

    // Phase 2: backward walk.
    let mut live_values: FxHashSet<ValueId> = FxHashSet::default();

    while let Some(val) = worklist.pop() {
        if !live_values.insert(val) {
            continue; // Already processed.
        }

        let Some(&def_loc) = def_map.get(&val) else {
            continue; // External value (not defined in this body).
        };

        match def_loc {
            DefLoc::Inst(bi, ii) => {
                if live_insts.insert((bi, ii)) {
                    // Newly live — trace its operands.
                    worklist.extend(inst_info::uses(&cfg.blocks[bi].insts[ii].kind));
                }
            }
            DefLoc::BlockParam(bi, pi) => {
                // Block param is live → trace corresponding jump args from predecessors.
                let block_label = cfg.blocks[bi].label;
                for pred_block in cfg.blocks.iter() {
                    let pred_args: Option<&[ValueId]> = match &pred_block.terminator {
                        Terminator::Jump { label, args } if *label == block_label => {
                            Some(args)
                        }
                        Terminator::JumpIf {
                            then_label,
                            then_args,
                            else_label,
                            else_args,
                            ..
                        } => {
                            if *then_label == block_label {
                                Some(then_args)
                            } else if *else_label == block_label {
                                Some(else_args)
                            } else {
                                None
                            }
                        }
                        Terminator::ListStep { done, done_args, .. }
                            if *done == block_label =>
                        {
                            Some(done_args)
                        }
                        _ => None,
                    };
                    if let Some(args) = pred_args {
                        if let Some(&arg) = args.get(pi) {
                            worklist.push(arg);
                        }
                    }
                }
            }
            DefLoc::Terminator(_) => {
                // ListStep dst/index_dst — terminator is always live,
                // its uses are already seeded.
            }
            DefLoc::EntryParam => {
                // Function param/capture — always live, nothing to trace.
            }
        }
    }

    // Phase 3: sweep — remove dead instructions.
    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let mut ii = 0;
        block.insts.retain(|_| {
            let keep = live_insts.contains(&(bi, ii));
            ii += 1;
            keep
        });
    }
}
