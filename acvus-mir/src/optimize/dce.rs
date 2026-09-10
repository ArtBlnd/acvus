//! Dead Code Elimination (DCE) - mark-sweep on CfgBody.
//!
//! Removes instructions that don't contribute to observable behavior.
//! Observable = Return value, context Store, IO (Eval), effectful FunctionCall.
//!
//! Algorithm:
//! 1. **Root**: instructions with side effects are unconditionally live.
//! 2. **Backward walk**: trace operands of live instructions -> mark their
//!    definitions as live -> trace their operands -> fixpoint.
//! 3. **Sweep**: remove non-live instructions.
//!
//! Runs post-SSA, post-DSE. Catches: dead inline residue, unused Ref/Load,
//! dead computation chains, unused Spawn handles.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::inst_info;
use crate::cfg::{CfgBody, Terminator};
use crate::ir::{InstKind, Label, ValueId};

// -- Def location ----------------------------------------------------

/// Where a ValueId is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DefLoc {
    /// Defined by an instruction at (block, inst_index).
    Inst(usize, usize),
    /// Defined as a block parameter at (block, param_index).
    BlockParam(usize, usize),
    /// Function parameter or capture - always live.
    EntryParam,
}

/// Build ValueId -> DefLoc mapping.
fn build_def_map(cfg: &CfgBody) -> FxHashMap<ValueId, DefLoc> {
    let mut map = FxHashMap::default();

    // Entry params and captures are always live.
    for v in cfg.entry_defs() {
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
    }

    map
}

// -- Root identification ---------------------------------------------

/// Is this instruction a root (has side effects, unconditionally live)?
///
/// An instruction with ANY effect (read, write, IO) must not
/// be removed. Only provably pure instructions can be dead.
fn is_root(kind: &InstKind) -> bool {
    match kind {
        // Context store - externally observable.
        InstKind::Store { .. } => true,

        // Eval - IO execution point.
        InstKind::Eval { .. } => true,

        // A Pure call that writes no context has no effect (RFC-0007,
        // RFC-0017): dead if its result is unused. A call whose effect is
        // unknown stays.
        InstKind::FunctionCall { callee_ty, .. } => !callee_ty
            .effect()
            .is_some_and(|e| e.is_pure() && e.writes.is_empty()),

        // Spawn: pure (deferred execution). The actual effect happens at Eval.
        // Dead if handle is unused (no Eval consumes it).
        InstKind::Spawn { .. } => false,

        // Everything else: pure computation, dead if result unused.
        _ => false,
    }
}

// -- Mark phase ------------------------------------------------------

/// The values a terminator needs regardless of any block param.
fn terminator_roots(term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::Return { value, order } => std::iter::once(*value).chain(*order).collect(),
        Terminator::JumpIf { cond, .. } => vec![*cond],
        Terminator::Jump { .. } | Terminator::Fallthrough => vec![],
    }
}

/// Collect all uses from a terminator.
fn terminator_uses(term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::Return { value, order } => std::iter::once(*value).chain(*order).collect(),
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
        Terminator::Fallthrough => vec![],
    }
}

// -- Public API ------------------------------------------------------

/// Run DCE on a CfgBody. Removes all instructions that don't contribute
/// to observable behavior (Return, Store, Eval, effectful calls).
pub fn run(cfg: &mut CfgBody) {
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
            if is_root(&inst.kind) {
                live_insts.insert((bi, ii));
                worklist.extend(inst_info::uses(&inst.kind));
            }
        }

        // Terminators are always live. A returned value and a branch
        // condition are roots; a jump argument is live only when the block
        // param it feeds is, and BlockParam tracing pulls it in then.
        live_terminators.insert(bi);
        worklist.extend(terminator_roots(&block.terminator));
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
                    // Newly live - trace its operands.
                    worklist.extend(inst_info::uses(&cfg.blocks[bi].insts[ii].kind));
                }
            }
            DefLoc::BlockParam(bi, pi) => {
                // Block param is live -> trace corresponding jump args from predecessors.
                let block_label = cfg.blocks[bi].label;
                for pred_block in cfg.blocks.iter() {
                    let pred_args: Option<&[ValueId]> = match &pred_block.terminator {
                        Terminator::Jump { label, args } if *label == block_label => Some(args),
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
                        _ => None,
                    };
                    if let Some(args) = pred_args {
                        if let Some(&arg) = args.get(pi) {
                            worklist.push(arg);
                        }
                    }
                }
            }
            DefLoc::EntryParam => {
                // Function param/capture - always live, nothing to trace.
            }
        }
    }

    // Phase 3: sweep - remove dead instructions.
    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let mut ii = 0;
        block.insts.retain(|_| {
            let keep = live_insts.contains(&(bi, ii));
            ii += 1;
            keep
        });
    }

    // Phase 4: sweep - remove dead block params and the jump args that
    // fed them. A dead param is a phi nothing reads; left in place it
    // would carry a second copy of a move-only value.
    let dead_params: Vec<(Label, Vec<usize>)> = cfg
        .blocks
        .iter()
        .map(|block| {
            let dead = block
                .params
                .iter()
                .enumerate()
                .filter(|(_, p)| !live_values.contains(p))
                .map(|(pi, _)| pi)
                .collect();
            (block.label, dead)
        })
        .filter(|(_, dead): &(Label, Vec<usize>)| !dead.is_empty())
        .collect();
    if dead_params.is_empty() {
        return;
    }
    let dead_of = |label: Label| -> Option<&Vec<usize>> {
        dead_params
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, d)| d)
    };
    let prune = |args: &mut Vec<ValueId>, dead: &[usize]| {
        let mut pi = 0;
        args.retain(|_| {
            let keep = !dead.contains(&pi);
            pi += 1;
            keep
        });
    };
    for block in &mut cfg.blocks {
        match &mut block.terminator {
            Terminator::Jump { label, args } => {
                if let Some(dead) = dead_of(*label) {
                    prune(args, dead);
                }
            }
            Terminator::JumpIf {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                if let Some(dead) = dead_of(*then_label) {
                    prune(then_args, dead);
                }
                if let Some(dead) = dead_of(*else_label) {
                    prune(else_args, dead);
                }
            }
            Terminator::Return { .. } | Terminator::Fallthrough => {}
        }
        if let Some(dead) = dead_of(block.label) {
            prune(&mut block.params, dead);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::graph::QualifiedRef;
    use crate::ir::{Callee, DebugInfo, Inst, MirBody};
    use crate::ty::Effect;
    use crate::ty::Ty;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    /// `dst = name()` with the given effect; its result is never used.
    fn unused_call(i: &Interner, name: &str, effect: Effect, dst: usize) -> InstKind {
        InstKind::FunctionCall {
            dst: v(dst),
            callee: Callee::Direct(QualifiedRef::root(i.intern(name))),
            callee_ty: Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: effect.into(),
            },
            args: vec![],
            order: None,
            lent: Vec::new(),
        }
    }

    fn body(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for _ in 0..val_count {
            val_types.insert(factory.next(), Ty::Int);
        }
        cfg::promote(MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types,
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn calls(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.kind, InstKind::FunctionCall { .. }))
            .count()
    }

    #[test]
    fn an_unused_pure_call_is_dead() {
        let i = Interner::new();
        let mut cfg = body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                unused_call(&i, "len", Effect::PURE, 1),
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            2,
        );
        run(&mut cfg);
        assert_eq!(
            calls(&cfg),
            0,
            "a Pure call with an unused result is removed"
        );
    }

    #[test]
    fn an_unused_effectful_call_stays() {
        let i = Interner::new();
        for effect in [Effect::IDEMPOTENT.commutative(), Effect::OPAQUE] {
            let mut cfg = body(
                vec![
                    InstKind::Const {
                        dst: v(0),
                        value: acvus_ast::Literal::Int(1),
                    },
                    unused_call(&i, "put", effect, 1),
                    InstKind::Return {
                        value: v(0),
                        order: None,
                    },
                ],
                2,
            );
            run(&mut cfg);
            assert_eq!(
                calls(&cfg),
                1,
                "an effectful call stays whether or not it commutes"
            );
        }
    }
}
