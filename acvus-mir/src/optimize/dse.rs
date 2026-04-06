//! Dead Store Elimination (DSE) for context stores.
//!
//! Runs post-SSA. Removes context Store instructions that are guaranteed
//! to be overwritten on ALL paths before being read.
//!
//! A context store is **live** if any subsequent path may read the context
//! before another store overwrites it. A store is **dead** if on every path
//! from the store, the context is written again before being read.
//!
//! "Read" includes: Load from context Ref, FunctionCall with context_uses,
//! Spawn with context_uses, Eval with context_defs (implies prior read),
//! and Return (contexts are externally observable after return).

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeSet;

use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{Callee, InstKind, RefTarget, ValueId};

// ── ref_to_ctx: ValueId → QualifiedRef mapping ─────────────────────

/// Build a map from Ref dst ValueId → QualifiedRef for all identity context Refs.
fn build_ref_to_ctx(cfg: &CfgBody) -> FxHashMap<ValueId, QualifiedRef> {
    let mut map = FxHashMap::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Ref {
                dst,
                target: RefTarget::Context(qref),
                path,
            } = &inst.kind
                && path.is_empty()
            {
                map.insert(*dst, *qref);
            }
        }
    }
    map
}

// ── Per-block context gen/kill sets ─────────────────────────────────

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
    ref_to_ctx: &FxHashMap<ValueId, QualifiedRef>,
    written_contexts: &BTreeSet<QualifiedRef>,
) -> BlockContextInfo {
    let mut reads = BTreeSet::new();
    let mut kills = BTreeSet::new();

    let has_return = matches!(block.terminator, Terminator::Return(_));

    // If block has Return, ALL written contexts are "read" (externally observable).
    if has_return {
        reads = written_contexts.clone();
    }

    // Walk instructions backwards.
    for inst in block.insts.iter().rev() {
        match &inst.kind {
            // Load from context Ref → read.
            InstKind::Load {
                src, volatile: false, ..
            } => {
                if let Some(&qref) = ref_to_ctx.get(src) {
                    // This is a read. Remove from kills (if written later was tracked),
                    // add to reads.
                    kills.remove(&qref);
                    reads.insert(qref);
                }
            }

            // Store to context Ref → write (kill).
            InstKind::Store {
                dst,
                volatile: false,
                ..
            } => {
                if let Some(&qref) = ref_to_ctx.get(dst) {
                    // This is a write. Remove from reads (if read later was tracked),
                    // add to kills.
                    reads.remove(&qref);
                    kills.insert(qref);
                }
            }

            // FunctionCall with context_uses → reads.
            InstKind::FunctionCall {
                context_uses,
                context_defs,
                callee: Callee::Direct(_),
                callee_ty,
                ..
            } => {
                // context_defs are writes (kill).
                for (qref, _) in context_defs {
                    reads.remove(qref);
                    kills.insert(*qref);
                }
                // context_uses are reads (gen).
                for (qref, _) in context_uses {
                    kills.remove(qref);
                    reads.insert(*qref);
                }
                // If context_uses/context_defs are empty, check effect info.
                if context_uses.is_empty() && context_defs.is_empty() {
                    if let Some((fn_reads, fn_writes)) =
                        crate::optimize::ssa_pass::extract_effect_refs(callee_ty)
                    {
                        for qref in fn_writes.iter().rev() {
                            reads.remove(qref);
                            kills.insert(*qref);
                        }
                        for qref in fn_reads.iter().rev() {
                            kills.remove(qref);
                            reads.insert(*qref);
                        }
                    }
                }
            }

            InstKind::FunctionCall {
                context_uses,
                context_defs,
                ..
            } => {
                // Non-direct callee: only handle explicit context_uses/context_defs.
                for (qref, _) in context_defs {
                    reads.remove(qref);
                    kills.insert(*qref);
                }
                for (qref, _) in context_uses {
                    kills.remove(qref);
                    reads.insert(*qref);
                }
            }

            // Spawn with context_uses → reads.
            InstKind::Spawn { context_uses, .. } => {
                for (qref, _) in context_uses {
                    kills.remove(qref);
                    reads.insert(*qref);
                }
            }

            // Eval with context_defs → writes (the callee wrote, we merge).
            InstKind::Eval { context_defs, .. } => {
                for (qref, _) in context_defs {
                    reads.remove(qref);
                    kills.insert(*qref);
                }
            }

            _ => {}
        }
    }

    BlockContextInfo {
        reads,
        kills,
        has_return,
    }
}

// ── Backward context liveness ───────────────────────────────────────

/// Compute per-block context liveness: which contexts are "live" at the
/// entry/exit of each block. A context is live if it may be read before
/// being written on some path from this point.
///
/// Standard backward dataflow:
///   live_out[B] = ∪ live_in[S] for all successors S of B
///   live_in[B]  = reads[B] ∪ (live_out[B] − kills[B])
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

            // live_in = reads ∪ (live_out − kills)
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

// ── DSE pass ────────────────────────────────────────────────────────

/// Run Dead Store Elimination on a CfgBody.
///
/// Removes context Store instructions (and their preceding Ref) that are
/// dead — the stored value is guaranteed to be overwritten before being read.
pub fn run(cfg: &mut CfgBody) {
    let ref_to_ctx = build_ref_to_ctx(cfg);
    if ref_to_ctx.is_empty() {
        return;
    }

    // Collect all written contexts.
    let mut written_contexts = BTreeSet::new();
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Store {
                dst,
                volatile: false,
                ..
            } = &inst.kind
            {
                if let Some(&qref) = ref_to_ctx.get(dst) {
                    written_contexts.insert(qref);
                }
            }
        }
    }

    // Build per-block info.
    let block_infos: Vec<BlockContextInfo> = cfg
        .blocks
        .iter()
        .map(|block| analyze_block(block, &ref_to_ctx, &written_contexts))
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
        if matches!(block.terminator, Terminator::Return(_)) {
            live = written_contexts.clone();
        }

        for (ii, inst) in block.insts.iter().enumerate().rev() {
            match &inst.kind {
                InstKind::Load {
                    src, volatile: false, ..
                } => {
                    if let Some(&qref) = ref_to_ctx.get(src) {
                        live.insert(qref);
                    }
                }

                InstKind::Store {
                    dst,
                    volatile: false,
                    ..
                } => {
                    if let Some(&qref) = ref_to_ctx.get(dst) {
                        if !live.contains(&qref) {
                            // Dead store — context will be overwritten before read.
                            dead_insts.insert((bi, ii));
                            // Also mark the Ref instruction if it immediately precedes.
                            if ii > 0 {
                                if let InstKind::Ref {
                                    dst: ref_dst,
                                    target: RefTarget::Context(_),
                                    ..
                                } = &block.insts[ii - 1].kind
                                {
                                    if ref_dst == dst {
                                        dead_insts.insert((bi, ii - 1));
                                    }
                                }
                            }
                        }
                        // After processing this store (backward), remove from live.
                        // (The store kills liveness of previous stores to same context.)
                        live.remove(&qref);
                    }
                }

                InstKind::FunctionCall {
                    context_uses,
                    context_defs,
                    callee: Callee::Direct(_),
                    callee_ty,
                    ..
                } => {
                    for (qref, _) in context_defs {
                        live.remove(qref);
                    }
                    for (qref, _) in context_uses {
                        live.insert(*qref);
                    }
                    if context_uses.is_empty() && context_defs.is_empty() {
                        if let Some((fn_reads, fn_writes)) =
                            crate::optimize::ssa_pass::extract_effect_refs(callee_ty)
                        {
                            for qref in &fn_writes {
                                live.remove(qref);
                            }
                            for qref in &fn_reads {
                                live.insert(*qref);
                            }
                        }
                    }
                }

                InstKind::FunctionCall {
                    context_uses,
                    context_defs,
                    ..
                } => {
                    for (qref, _) in context_defs {
                        live.remove(qref);
                    }
                    for (qref, _) in context_uses {
                        live.insert(*qref);
                    }
                }

                InstKind::Spawn { context_uses, .. } => {
                    for (qref, _) in context_uses {
                        live.insert(*qref);
                    }
                }

                InstKind::Eval { context_defs, .. } => {
                    for (qref, _) in context_defs {
                        live.remove(qref);
                    }
                }

                _ => {}
            }
        }
    }

    if dead_insts.is_empty() {
        return;
    }

    // Mark orphaned Refs: context Ref whose dst has no live use in ANY block.
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            if let InstKind::Ref {
                dst,
                target: RefTarget::Context(_),
                ..
            } = &inst.kind
            {
                if !ref_has_live_use(&cfg.blocks, *dst, &dead_insts) {
                    dead_insts.insert((bi, ii));
                }
            }
        }
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

/// Check if a Ref's dst ValueId is used by any non-dead Load or Store in any block.
fn ref_has_live_use(
    blocks: &[crate::cfg::Block],
    ref_dst: ValueId,
    dead_insts: &FxHashSet<(usize, usize)>,
) -> bool {
    for (bi, block) in blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            if dead_insts.contains(&(bi, ii)) {
                continue;
            }
            match &inst.kind {
                InstKind::Load { src, .. } if *src == ref_dst => return true,
                InstKind::Store { dst, .. } if *dst == ref_dst => return true,
                _ => {}
            }
        }
    }
    false
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
        let max_val = val_types
            .keys()
            .map(|v| v.to_raw())
            .max()
            .unwrap_or(0);
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
        }
    }

    fn count_stores(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.kind, InstKind::Store { .. }))
            .count()
    }

    fn count_refs(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.kind, InstKind::Ref { .. }))
            .count()
    }

    /// Consecutive stores to the same context: first is dead.
    /// Ref @x → Store @x = v1 → Ref @x → Store @x = v2 → Return v2
    /// After DSE: first Ref+Store removed.
    #[test]
    fn consecutive_stores_first_dead() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Ref(Box::new(Ty::Int), false));
        val_types.insert(v(1), Ty::Int);
        val_types.insert(v(2), Ty::Ref(Box::new(Ty::Int), false));
        val_types.insert(v(3), Ty::Int);

        let body = make_body(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(0),
                    value: v(1),
                    volatile: false,
                },
                InstKind::Ref {
                    dst: v(2),
                    target: RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(2),
                    value: v(3),
                    volatile: false,
                },
                InstKind::Return(v(3)),
            ],
            val_types,
        );

        let mut cfg = cfg::promote(body);
        assert_eq!(count_stores(&cfg), 2);
        assert_eq!(count_refs(&cfg), 2);

        run(&mut cfg);

        assert_eq!(count_stores(&cfg), 1, "first dead store should be removed");
        assert_eq!(count_refs(&cfg), 1, "orphaned ref should be removed");
    }

    /// Store followed by Load: store is live (needed by load).
    /// Ref @x → Store @x = v1 → Ref @x → Load @x → Return loaded
    /// After DSE: nothing removed.
    #[test]
    fn store_then_load_is_live() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Ref(Box::new(Ty::Int), false));
        val_types.insert(v(1), Ty::Int);
        val_types.insert(v(2), Ty::Ref(Box::new(Ty::Int), false));
        val_types.insert(v(3), Ty::Int);

        let body = make_body(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(0),
                    value: v(1),
                    volatile: false,
                },
                InstKind::Ref {
                    dst: v(2),
                    target: RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Load {
                    dst: v(3),
                    src: v(2),
                    volatile: false,
                },
                InstKind::Return(v(3)),
            ],
            val_types,
        );

        let mut cfg = cfg::promote(body);
        let stores_before = count_stores(&cfg);

        run(&mut cfg);

        assert_eq!(
            count_stores(&cfg),
            stores_before,
            "store before load must not be removed"
        );
    }

    /// Volatile store is never removed.
    #[test]
    fn volatile_store_preserved() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Ref(Box::new(Ty::Int), true)); // volatile
        val_types.insert(v(1), Ty::Int);
        val_types.insert(v(2), Ty::Ref(Box::new(Ty::Int), true));
        val_types.insert(v(3), Ty::Int);

        let body = make_body(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(0),
                    value: v(1),
                    volatile: true,
                },
                InstKind::Ref {
                    dst: v(2),
                    target: RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(2),
                    value: v(3),
                    volatile: true,
                },
                InstKind::Return(v(3)),
            ],
            val_types,
        );

        let mut cfg = cfg::promote(body);
        let stores_before = count_stores(&cfg);

        run(&mut cfg);

        assert_eq!(
            count_stores(&cfg),
            stores_before,
            "volatile stores must never be removed"
        );
    }

    /// Store to context before return is live (externally observable).
    /// Ref @x → Store @x = v1 → Return v2
    /// Store is live because return exposes context state.
    #[test]
    fn store_before_return_is_live() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Ref(Box::new(Ty::Int), false));
        val_types.insert(v(1), Ty::Int);
        val_types.insert(v(2), Ty::Int);

        let body = make_body(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(0),
                    value: v(1),
                    volatile: false,
                },
                InstKind::Return(v(2)),
            ],
            val_types,
        );

        let mut cfg = cfg::promote(body);
        let stores_before = count_stores(&cfg);

        run(&mut cfg);

        assert_eq!(
            count_stores(&cfg),
            stores_before,
            "store before return is externally observable"
        );
    }

    /// Cross-block: nested loop with conditional context writes.
    /// SSA creates phi write-back stores at loop headers.
    /// DSE should eliminate dead write-backs that are always overwritten.
    #[test]
    fn cross_block_nested_loop_conditional() {
        let i = Interner::new();
        let module = crate::test::compile_script(
            &i,
            r#"
                row in @matrix {
                    y in row {
                        true = y > 0 { @pos = @pos + y; };
                        true = y < 0 { @neg = @neg + y; };
                    };
                };
                @pos
            "#,
            &[
                ("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int))))),
                ("pos", Ty::Int),
                ("neg", Ty::Int),
            ],
        )
        .expect("compile failed");

        // After test pipeline (SROA+SSA), run SSA again to simulate Pass 2.
        let mut cfg = cfg::promote(module.main);
        crate::optimize::ssa_pass::run(&mut cfg);

        let stores_before = count_stores(&cfg);

        run(&mut cfg);

        let stores_after = count_stores(&cfg);
        // DSE should eliminate at least some dead phi write-backs.
        assert!(
            stores_after < stores_before,
            "DSE should eliminate dead stores (before={stores_before}, after={stores_after})"
        );
    }

    /// No context stores → DSE is a no-op.
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
                InstKind::Return(v(0)),
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

