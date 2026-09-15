//! Dead Store Elimination (DSE) for context stores.
//!
//! Runs post-SSA. Removes context Store instructions that are guaranteed
//! to be overwritten on ALL paths before being read.
//!
//! A context store is **live** if any subsequent path may read the context
//! before another store overwrites it. A store is **dead** if on every path
//! from the store, the context is written again before being read.
//!
//! A read is a whole `Take` of the context, a call, or a Return (contexts
//! are externally observable after return); a write is a whole `Assign`.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeSet;

use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{InstKind, RefTarget, ValueId};
use crate::optimize::context_ops::{context_read, context_written};

// -- Per-block context gen/kill sets ---------------------------------

/// For each block, which contexts are read (gen) and which are written (kill)
/// before being read within the block.
///
/// We walk instructions **backwards** within each block to build:
/// - `reads`: contexts that are read before any write in this block (gen set)
/// - `kills`: contexts that are written before any read in this block (kill set)
struct BlockContextInfo {
    /// Contexts read in this block before being written (backward: gen set).
    reads: BTreeSet<QualifiedRef>,
    /// Contexts written in this block before being read (backward: kill set).
    kills: BTreeSet<QualifiedRef>,
    /// Whether this block contains a Return terminator.
    has_return: bool,
}

fn analyze_block(
    block: &crate::cfg::Block,
    written_contexts: &BTreeSet<QualifiedRef>,
) -> BlockContextInfo {
    let mut reads = BTreeSet::new();
    let mut kills = BTreeSet::new();

    let has_return = matches!(block.terminator, Terminator::Return { .. });

    // If block has Return, ALL written contexts are "read" (externally observable).
    if has_return {
        reads = written_contexts.clone();
    }

    // Walk instructions backwards.
    for inst in block.insts.iter().rev() {
        if let Some(qref) = context_read(&inst.kind) {
            kills.remove(&qref);
            reads.insert(qref);
        } else if let Some(qref) = context_written(&inst.kind) {
            reads.remove(&qref);
            kills.insert(qref);
        } else if matches!(
            &inst.kind,
            InstKind::FunctionCall { .. } | InstKind::Spawn { .. } | InstKind::Eval { .. }
        ) {
            kills.clear();
            reads.extend(written_contexts.iter().copied());
        }
    }

    BlockContextInfo {
        reads,
        kills,
        has_return,
    }
}

// -- Backward context liveness ---------------------------------------

/// Compute per-block context liveness: which contexts are "live" at the
/// entry/exit of each block. A context is live if it may be read before
/// being written on some path from this point.
///
/// Standard backward dataflow:
///   live_out[B] =  union  live_in[S] for all successors S of B
///   live_in[B]  = reads[B]  union  (live_out[B] - kills[B])
fn compute_context_liveness(
    cfg: &CfgBody,
    block_infos: &[BlockContextInfo],
) -> Vec<BTreeSet<QualifiedRef>> {
    let n = cfg.blocks.len();
    let mut live_in: Vec<BTreeSet<QualifiedRef>> = vec![BTreeSet::new(); n];
    let mut live_out: Vec<BTreeSet<QualifiedRef>> = vec![BTreeSet::new(); n];

    // Iterative fixpoint.
    let mut changed = true;
    while changed {
        changed = false;
        // Process blocks in reverse order (backward analysis).
        for bi in (0..n).rev() {
            // live_out = union of successors' live_in
            let mut new_out = BTreeSet::new();
            for succ in cfg.successors(BlockIdx(bi)) {
                for qref in &live_in[succ.0] {
                    new_out.insert(*qref);
                }
            }

            // live_in = reads  union  (live_out - kills)
            let info = &block_infos[bi];
            let mut new_in = info.reads.clone();
            for qref in &new_out {
                if !info.kills.contains(qref) {
                    new_in.insert(*qref);
                }
            }

            if new_in != live_in[bi] || new_out != live_out[bi] {
                live_in[bi] = new_in;
                live_out[bi] = new_out;
                changed = true;
            }
        }
    }

    live_out
}

// -- DSE pass --------------------------------------------------------

/// Run Dead Store Elimination on a CfgBody.
///
/// Removes context `Assign` instructions that are dead - the assigned value
/// is guaranteed to be overwritten before being read.
pub fn run(cfg: &mut CfgBody) {
    let written_contexts: BTreeSet<QualifiedRef> = cfg
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|inst| context_written(&inst.kind))
        .collect();
    if written_contexts.is_empty() {
        return;
    }

    // Build per-block info.
    let block_infos: Vec<BlockContextInfo> = cfg
        .blocks
        .iter()
        .map(|block| analyze_block(block, &written_contexts))
        .collect();

    // Compute backward liveness.
    let live_out = compute_context_liveness(cfg, &block_infos);

    // Walk each block forward, tracking local liveness, and mark dead stores.
    let mut dead_insts: FxHashSet<(usize, usize)> = FxHashSet::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // Start with live_out for this block, then walk backwards to find
        // which stores are dead. Actually, we need to walk forward and
        // track which contexts are live *after* each instruction.
        //
        // Walk backwards from block end: start with live_out[bi], process
        // each instruction in reverse order.
        let mut live = live_out[bi].clone();

        // If this block has Return, all written contexts are live at the terminator.
        if matches!(block.terminator, Terminator::Return { .. }) {
            live = written_contexts.clone();
        }

        for (ii, inst) in block.insts.iter().enumerate().rev() {
            if let Some(qref) = context_read(&inst.kind) {
                live.insert(qref);
            } else if let Some(qref) = context_written(&inst.kind) {
                if !live.contains(&qref) {
                    dead_insts.insert((bi, ii));
                }
                live.remove(&qref);
            } else if matches!(
                &inst.kind,
                InstKind::FunctionCall { .. } | InstKind::Spawn { .. } | InstKind::Eval { .. }
            ) {
                live.extend(written_contexts.iter().copied());
            }
        }
    }

    if dead_insts.is_empty() {
        return;
    }

    // Remove dead instructions.
    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let mut ii = 0;
        block.insts.retain(|_| {
            let keep = !dead_insts.contains(&(bi, ii));
            ii += 1;
            keep
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::ir::{DebugInfo, Inst, MirBody};
    use crate::ty::Ty;
    use acvus_ast::Span;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }
    fn span() -> Span {
        Span { start: 0, end: 0 }
    }
    fn inst(kind: InstKind) -> Inst {
        Inst { span: span(), kind }
    }

    fn make_body(insts: Vec<InstKind>, val_types: FxHashMap<ValueId, Ty>) -> MirBody {
        let max_val = val_types.keys().map(|v| v.to_raw()).max().unwrap_or(0);
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..=max_val {
            factory.next();
        }
        MirBody {
            insts: insts.into_iter().map(inst).collect(),
            val_types,
            params: vec![],
            captures: vec![],
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
            order_param: None,
        }
    }

    fn count_assigns(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.kind, InstKind::Assign { .. }))
            .count()
    }

    fn assign(ctx: QualifiedRef, value: ValueId) -> InstKind {
        InstKind::Assign {
            target: RefTarget::Context(ctx),
            path: vec![],
            value,
        }
    }

    fn take(ctx: QualifiedRef, dst: ValueId) -> InstKind {
        InstKind::Take {
            dst,
            target: RefTarget::Context(ctx),
            path: vec![],
        }
    }

    /// `assign @x = v0; assign @x = v1; return v1`: the first assign is dead.
    #[test]
    fn consecutive_assigns_first_dead() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Int);
        val_types.insert(v(1), Ty::Int);
        let body = make_body(
            vec![
                assign(ctx, v(0)),
                assign(ctx, v(1)),
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            val_types,
        );
        let mut cfg = cfg::promote(body);
        assert_eq!(count_assigns(&cfg), 2);
        run(&mut cfg);
        assert_eq!(count_assigns(&cfg), 1, "first dead assign should be removed");
    }

    /// `assign @x = v0; v1 = take @x; return v1`: the assign is read.
    #[test]
    fn assign_then_take_is_live() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Int);
        val_types.insert(v(1), Ty::Int);
        let body = make_body(
            vec![
                assign(ctx, v(0)),
                take(ctx, v(1)),
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            val_types,
        );
        let mut cfg = cfg::promote(body);
        run(&mut cfg);
        assert_eq!(count_assigns(&cfg), 1, "assign before take must not be removed");
    }

    #[test]
    fn assign_then_call_then_assign_keeps_first() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let f = QualifiedRef::root(i.intern("f"));
        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Int);
        val_types.insert(v(1), Ty::Int);
        let body = make_body(
            vec![
                assign(ctx, v(0)),
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: crate::ir::Callee::Direct(f),
                    callee_ty: Ty::error(),
                    args: vec![],
                    order: None,
                },
                assign(ctx, v(1)),
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            val_types,
        );
        let mut cfg = cfg::promote(body);
        run(&mut cfg);
        assert_eq!(
            count_assigns(&cfg),
            2,
            "the call may read @x, so the first assign stays"
        );
    }

    /// An assign before return is externally observable.
    #[test]
    fn assign_before_return_is_live() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Int);
        val_types.insert(v(1), Ty::Int);
        let body = make_body(
            vec![
                assign(ctx, v(0)),
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            val_types,
        );
        let mut cfg = cfg::promote(body);
        run(&mut cfg);
        assert_eq!(count_assigns(&cfg), 1, "assign before return is externally observable");
    }

    /// No context stores -> DSE is a no-op.
    #[test]
    fn no_context_stores_noop() {
        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Int);

        let body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(42),
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            val_types,
        );

        let mut cfg = cfg::promote(body);
        let inst_count_before: usize = cfg.blocks.iter().map(|b| b.insts.len()).sum();

        run(&mut cfg);

        let inst_count_after: usize = cfg.blocks.iter().map(|b| b.insts.len()).sum();
        assert_eq!(inst_count_before, inst_count_after);
    }
}
