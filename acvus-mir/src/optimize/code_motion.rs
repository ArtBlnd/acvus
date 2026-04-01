//! Cross-block code motion: hoist pure instructions above branch points.
//!
//! After SpawnSplit, IO calls are Spawn (async start) + Eval (blocking wait).
//! This pass hoists Spawn and other pure instructions out of dominated blocks
//! into their dominator ancestors, maximizing the distance between Spawn and Eval.
//!
//! # Algorithm
//!
//! Each iteration:
//! 1. Build dominator tree + token liveness.
//! 2. For each hoistable instruction, walk UP the dominator chain to find the
//!    **highest ancestor** where all operands are available and no token conflict.
//!    This eliminates the need for multi-iteration fixpoint on deep merge chains.
//! 3. `def_block` is updated after each hoist decision, so later instructions
//!    in the same block see their dependencies' new locations — operand chains
//!    are resolved in a single pass.
//! 4. Repeat until no more instructions can be hoisted (fixpoint for cross-block
//!    chains, but typically converges in 1 iteration).
//!
//! # Hoistability (allowlist)
//!
//! Only provably pure instructions are hoisted. New/unknown instruction kinds
//! default to "not hoistable" (soundness by construction).
//!
//! Pure: arithmetic, value construction, field access, test predicates,
//! Spawn, LoadFunction, pure FunctionCall (no IO/tokens/context effects).
//!
//! NOT hoisted: Eval, context ops, variable ops, indirect calls.

use rustc_hash::FxHashMap;

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::token_liveness;
use crate::cfg::{BlockIdx, CfgBody};
use crate::graph::QualifiedRef;
use crate::ir::*;
use crate::ty::{Effect, EffectTarget, TokenId, Ty};

// ── Entry point ────────────────────────────────────────────────────

pub fn run(cfg: &mut CfgBody, fn_types: &FxHashMap<QualifiedRef, Ty>) {
    // Phase 1: Hoist — move Spawn and pure instructions UP.
    loop {
        if !hoist_pass(cfg, fn_types) {
            break;
        }
    }

    // Phase 2: Sink — move Eval and blocking instructions DOWN.
    sink_pass(cfg);
}

// ── Single hoist pass ──────────────────────────────────────────────

/// One iteration: find hoistable instructions, move them to the highest
/// valid dominator. Returns true if anything moved.
fn hoist_pass(cfg: &mut CfgBody, fn_types: &FxHashMap<QualifiedRef, Ty>) -> bool {
    if cfg.blocks.len() < 2 {
        return false;
    }

    let domtree = DomTree::build(cfg);
    let tok_liveness = token_liveness::analyze(cfg, fn_types);
    let mut def_block = build_def_block(cfg);

    // ── Collect hoists ─────────────────────────────────────────────
    //
    // For each instruction, find the highest dominator ancestor where
    // all operands are available and no token conflicts exist.
    // def_block is updated after each decision so that operand chains
    // within a block are resolved in one pass.

    let mut hoists: Vec<(usize, usize, usize)> = Vec::new(); // (src_block, inst_idx, tgt_block)

    for (bi, block) in cfg.blocks.iter().enumerate() {
        if domtree.idom(BlockIdx(bi)).is_none() {
            continue;
        }

        for (i, inst) in block.insts.iter().enumerate() {
            let kind = &inst.kind;

            if !is_hoistable(kind, fn_types) || inst_info::defs(kind).is_empty() {
                continue;
            }

            let uses = inst_info::uses(kind);
            let tokens = token_ids_of(kind, fn_types, &cfg.val_types);

            if let Some(target) = find_highest_target(
                BlockIdx(bi),
                &uses,
                &tokens,
                &domtree,
                &def_block,
                &tok_liveness,
                cfg,
            ) {
                hoists.push((bi, i, target.0));
                for d in inst_info::defs(kind) {
                    def_block.insert(d, target);
                }
            }
        }
    }

    if hoists.is_empty() {
        return false;
    }

    // ── Apply hoists ───────────────────────────────────────────────

    let mut to_move: FxHashMap<usize, Vec<Inst>> = FxHashMap::default();
    for &(src_bi, inst_i, tgt_bi) in &hoists {
        to_move
            .entry(tgt_bi)
            .or_default()
            .push(cfg.blocks[src_bi].insts[inst_i].clone());
    }

    // Nop originals.
    for &(bi, i, _) in &hoists {
        cfg.blocks[bi].insts[i].kind = InstKind::Nop;
    }

    // Append to target blocks (before terminator, which is separate).
    for (tgt_bi, insts) in to_move {
        cfg.blocks[tgt_bi].insts.extend(insts);
    }

    true
}

// ── Def-block map ──────────────────────────────────────────────────

/// Build ValueId → BlockIdx mapping: where each value is defined.
fn build_def_block(cfg: &CfgBody) -> FxHashMap<ValueId, BlockIdx> {
    let mut def_block = FxHashMap::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let idx = BlockIdx(bi);
        for &p in &block.params {
            def_block.insert(p, idx);
        }
        for inst in &block.insts {
            for d in inst_info::defs(&inst.kind) {
                def_block.insert(d, idx);
            }
        }
        if let crate::cfg::Terminator::ListStep { dst, index_dst, .. } = &block.terminator {
            def_block.insert(*dst, idx);
            def_block.insert(*index_dst, idx);
        }
    }

    for &(_, v) in cfg.params.iter().chain(cfg.captures.iter()) {
        def_block.entry(v).or_insert(BlockIdx(0));
    }

    def_block
}

// ── Target finding ─────────────────────────────────────────────────

/// Walk up the dominator chain from `block_idx` to find the highest ancestor
/// where all operands are available (before the ancestor's terminator) and
/// no token conflicts exist.
fn find_highest_target(
    block_idx: BlockIdx,
    uses: &[ValueId],
    tokens: &smallvec::SmallVec<[TokenId; 2]>,
    domtree: &DomTree,
    def_block: &FxHashMap<ValueId, BlockIdx>,
    tok_liveness: &token_liveness::TokenLivenessResult,
    cfg: &CfgBody,
) -> Option<BlockIdx> {
    let mut best: Option<BlockIdx> = None;
    let mut candidate = domtree.idom(block_idx)?;

    loop {
        // All operands must be available at candidate's body (before terminator).
        let all_available = uses.iter().all(|u| match def_block.get(u) {
            Some(&def_bi) if def_bi == candidate => {
                !is_terminator_def(&cfg.blocks[candidate.0].terminator, *u)
            }
            Some(&def_bi) => domtree.dominates(def_bi, candidate),
            None => true,
        });
        if !all_available {
            break;
        }

        // No token conflict at this level.
        if tokens
            .iter()
            .any(|tid| tok_liveness.is_live_out(candidate, *tid))
        {
            break;
        }

        best = Some(candidate);

        match domtree.idom(candidate) {
            Some(parent) => candidate = parent,
            None => break,
        }
    }

    best
}

// ── Hoistability (allowlist) ───────────────────────────────────────

/// Can this instruction be safely hoisted to a dominator block?
///
/// Allowlist: only provably pure instructions. Unknown kinds default to
/// not hoistable (soundness by construction).
fn is_hoistable(kind: &InstKind, fn_types: &FxHashMap<QualifiedRef, Ty>) -> bool {
    match kind {
        // Arithmetic / logic.
        InstKind::BinOp { .. } | InstKind::UnaryOp { .. } | InstKind::Cast { .. } => true,

        // Value construction.
        InstKind::Const { .. }
        | InstKind::MakeDeque { .. }
        | InstKind::MakeObject { .. }
        | InstKind::MakeTuple { .. }
        | InstKind::MakeRange { .. }
        | InstKind::MakeVariant { .. }
        | InstKind::MakeClosure { .. } => true,

        // Projection path (no-op, pure).
        InstKind::Ref { .. } => true,

        // Field / element access (scalar, pure).
        InstKind::FieldGet { .. }
        | InstKind::FieldSet { .. }
        | InstKind::ObjectGet { .. }
        | InstKind::ListIndex { .. }
        | InstKind::ListGet { .. }
        | InstKind::ListSlice { .. }
        | InstKind::TupleIndex { .. }
        | InstKind::UnwrapVariant { .. } => true,

        // Test predicates.
        InstKind::TestLiteral { .. }
        | InstKind::TestRange { .. }
        | InstKind::TestVariant { .. }
        | InstKind::TestListLen { .. }
        | InstKind::TestObjectKey { .. } => true,

        // Spawn (async start, no side-effect until Eval).
        // Only Direct: Indirect callee tokens can't be tracked.
        InstKind::Spawn {
            callee: Callee::Direct(..),
            ..
        } => true,

        // Function reference.
        InstKind::LoadFunction { .. } => true,

        // Pure direct FunctionCall.
        InstKind::FunctionCall {
            callee: Callee::Direct(qref),
            ..
        } => is_pure_call(fn_types, qref),

        // Everything else: NOT hoistable.
        _ => false,
    }
}

fn is_pure_call(fn_types: &FxHashMap<QualifiedRef, Ty>, qref: &QualifiedRef) -> bool {
    let Some(Ty::Fn {
        effect: Effect::Resolved(eff),
        ..
    }) = fn_types.get(qref)
    else {
        return false;
    };
    eff.is_pure()
}

// ── Terminator helpers ─────────────────────────────────────────────

fn is_terminator_def(term: &crate::cfg::Terminator, val: ValueId) -> bool {
    match term {
        crate::cfg::Terminator::ListStep { dst, index_dst, .. } => val == *dst || val == *index_dst,
        _ => false,
    }
}

fn token_ids_of(
    kind: &InstKind,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    _val_types: &FxHashMap<ValueId, Ty>,
) -> smallvec::SmallVec<[TokenId; 2]> {
    let effect_set = match kind {
        InstKind::FunctionCall {
            callee: Callee::Direct(qref),
            ..
        }
        | InstKind::Spawn {
            callee: Callee::Direct(qref),
            ..
        } => fn_types.get(qref).and_then(|ty| match ty {
            Ty::Fn {
                effect: Effect::Resolved(eff),
                ..
            } => Some(eff),
            _ => None,
        }),
        _ => None,
    };

    let Some(eff) = effect_set else {
        return smallvec::SmallVec::new();
    };

    eff.reads
        .iter()
        .chain(eff.writes.iter())
        .filter_map(|t| match t {
            EffectTarget::Token(tid) => Some(*tid),
            _ => None,
        })
        .collect()
}

// ── Sink pass ─────────────────────────────────────────────────────
//
// Moves Eval (and non-volatile Load) as late as possible — just before
// their result is first needed. This maximizes the distance between
// Spawn (hoisted up) and Eval (sunk down).
//
// Algorithm:
// 1. Build a use map: for each ValueId, where is it used?
// 2. For each sinkable instruction, find the latest safe position.
// 3. Move instructions within their block (reorder down) or to a
//    single-successor block if no in-block uses exist.

/// Position of a use: (block_idx, instruction_index_within_block or TERMINATOR).
fn terminator_uses_vec(term: &crate::cfg::Terminator) -> Vec<ValueId> {
    use crate::cfg::Terminator;
    match term {
        Terminator::Return(val) => vec![*val],
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf { cond, then_args, else_args, .. } => {
            let mut v = vec![*cond];
            v.extend_from_slice(then_args);
            v.extend_from_slice(else_args);
            v
        }
        Terminator::ListStep { list, index_src, done_args, .. } => {
            let mut v = vec![*list, *index_src];
            v.extend_from_slice(done_args);
            v
        }
        Terminator::Fallthrough => vec![],
    }
}

// ── Sink infrastructure ─────────────────────────────────────────────

/// Build ref_to_ctx: ValueId (Ref dst) → QualifiedRef (context).
fn build_ref_to_ctx(cfg: &CfgBody) -> FxHashMap<ValueId, QualifiedRef> {
    let mut map = FxHashMap::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Ref {
                dst,
                target: crate::ir::RefTarget::Context(qref),
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

/// Which context does this Load/Store access? None if not a context op.
fn context_of_load(kind: &InstKind, ref_to_ctx: &FxHashMap<ValueId, QualifiedRef>) -> Option<QualifiedRef> {
    match kind {
        InstKind::Load { src, volatile: false, .. } => ref_to_ctx.get(src).copied(),
        _ => None,
    }
}

fn context_of_store(kind: &InstKind, ref_to_ctx: &FxHashMap<ValueId, QualifiedRef>) -> Option<QualifiedRef> {
    match kind {
        InstKind::Store { dst, volatile: false, .. } => ref_to_ctx.get(dst).copied(),
        _ => None,
    }
}

/// Contexts written by Eval's context_defs.
fn eval_write_contexts(kind: &InstKind) -> Vec<QualifiedRef> {
    match kind {
        InstKind::Eval { context_defs, .. } => context_defs.iter().map(|(qref, _)| *qref).collect(),
        _ => vec![],
    }
}

/// Run the sink pass — move Eval, non-volatile Load, and non-volatile Store
/// as late as possible within their block.
///
/// Processes ONE sinkable instruction per iteration, then re-scans.
/// This avoids index invalidation from multiple moves.
/// Repeats until no more sinking is possible (fixpoint).
fn sink_pass(cfg: &mut CfgBody) {
    let max_iters = cfg.blocks.iter().map(|b| b.insts.len()).sum::<usize>() * 2;
    let mut iters = 0;
    loop {
        if !sink_one(cfg) {
            break;
        }
        iters += 1;
        if iters > max_iters {
            #[cfg(debug_assertions)]
            eprintln!("[sink_pass] hit max iterations ({max_iters}), stopping");
            break;
        }
    }
}

/// Try to sink ONE instruction. Returns true if something moved.
fn sink_one(cfg: &mut CfgBody) -> bool {
    let ref_to_ctx = build_ref_to_ctx(cfg);

    for bi in 0..cfg.blocks.len() {
        for ii in 0..cfg.blocks[bi].insts.len() {
            let kind = &cfg.blocks[bi].insts[ii].kind;

            let sink_info = match kind {
                InstKind::Eval { .. } => Some(SinkKind::Eval),
                InstKind::Load { volatile: false, .. } => {
                    context_of_load(kind, &ref_to_ctx).map(SinkKind::Load)
                }
                InstKind::Store { volatile: false, .. } => {
                    context_of_store(kind, &ref_to_ctx).map(SinkKind::Store)
                }
                _ => None,
            };
            let Some(sink_kind) = sink_info else {
                continue;
            };

            let defs: Vec<ValueId> = inst_info::defs(kind).to_vec();
            let eval_contexts = eval_write_contexts(kind);

            // Scan forward for barrier.
            let mut barrier = cfg.blocks[bi].insts.len();

            for jj in (ii + 1)..cfg.blocks[bi].insts.len() {
                let other = &cfg.blocks[bi].insts[jj].kind;

                let other_uses = inst_info::uses(other);
                if defs.iter().any(|d| other_uses.contains(d)) {
                    barrier = jj;
                    break;
                }

                if let SinkKind::Load(ctx) = &sink_kind {
                    if let Some(store_ctx) = context_of_store(other, &ref_to_ctx) {
                        if store_ctx == *ctx {
                            barrier = jj;
                            break;
                        }
                    }
                }

                if matches!(sink_kind, SinkKind::Eval) {
                    if let Some(load_ctx) = context_of_load(other, &ref_to_ctx) {
                        if eval_contexts.contains(&load_ctx) {
                            barrier = jj;
                            break;
                        }
                    }
                }

                if let SinkKind::Store(ctx) = &sink_kind {
                    if let Some(load_ctx) = context_of_load(other, &ref_to_ctx) {
                        if load_ctx == *ctx {
                            barrier = jj;
                            break;
                        }
                    }
                    if let Some(store_ctx) = context_of_store(other, &ref_to_ctx) {
                        if store_ctx == *ctx {
                            barrier = jj;
                            break;
                        }
                    }
                }
            }

            let term_uses = terminator_uses_vec(&cfg.blocks[bi].terminator);
            if defs.iter().any(|d| term_uses.contains(d)) {
                barrier = barrier.min(cfg.blocks[bi].insts.len());
            }

            let target = if barrier > 0 { barrier - 1 } else { ii };
            if target > ii {
                let inst = cfg.blocks[bi].insts.remove(ii);
                // After remove(ii), the instruction that was at original position
                // `target` is now at position `target - 1`. We want to place our
                // instruction BEFORE it (at original position `target`), which is
                // now position `target - 1` in the modified array. But we want
                // to be AT position target in original coordinates = insert at
                // target - 1 in modified coordinates. However, the "before barrier"
                // semantics means we want to be at original position target,
                // which after remove is at index target - 1.
                // But we also need to account for that target = barrier - 1,
                // so we're really placing at barrier - 1 in original, which is
                // barrier - 2 in modified. That's target - 1.
                //
                // Wait, let me think step by step:
                // Original array: [0..ii..target..barrier..len]
                // We want: [0..target'..barrier..len] where target' has our inst
                //   just before barrier (at position target = barrier-1)
                // After remove(ii): array is len-1, positions ii..len-1 shifted
                // Original position target is now at index target-1
                // Insert at target-1 places inst at that position
                // Result: [...inst(target-1)..barrier_inst(target)...]
                // In original coordinates: inst at target, barrier at target+1
                // But barrier was at barrier=target+1 originally? No, barrier
                // was the first use, at original position barrier. target=barrier-1.
                // So inst goes to original barrier-1, barrier stays at original barrier.
                // That's correct!
                cfg.blocks[bi].insts.insert(target, inst);
                return true;
            }
        }
    }
    false
}

#[derive(Debug)]
enum SinkKind {
    Eval,
    Load(QualifiedRef),
    Store(QualifiedRef),
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{self, CfgBody};
    use crate::ty::{Effect, EffectSet, EffectTarget};
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use std::collections::BTreeSet;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..val_count {
            factory.next();
        }
        cfg::promote(MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn io_fn_type(i: &Interner, name: &str) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        (
            qref,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    io: true,
                    ..Default::default()
                }),
            },
        )
    }

    fn token_fn_type(i: &Interner, name: &str, tid: TokenId) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        let mut reads = BTreeSet::new();
        reads.insert(EffectTarget::Token(tid));
        (
            qref,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    reads,
                    writes: BTreeSet::new(),
                    io: true,
                    self_modifying: false,
                }),
            },
        )
    }

    fn demoted(c: CfgBody) -> MirBody {
        cfg::demote(c)
    }

    fn kinds(body: &MirBody) -> Vec<&InstKind> {
        body.insts.iter().map(|i| &i.kind).collect()
    }

    #[test]
    fn spawn_hoisted_above_branch() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Spawn {
                    dst: v(1),
                    callee: Callee::Direct(qref),
                    args: vec![v(0)],
                    context_uses: vec![],
                },
                InstKind::Return(v(1)),
            ],
            10,
        );

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        let jumpif_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            spawn_idx < jumpif_idx,
            "spawn should be hoisted before branch"
        );
    }

    #[test]
    fn block_param_dependency_prevents_hoist() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![v(0)],
                    else_label: Label(1),
                    else_args: vec![v(0)],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v(1)],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(1)],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(2)],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(2)],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Spawn {
                    dst: v(4),
                    callee: Callee::Direct(qref),
                    args: vec![v(3)],
                    context_uses: vec![],
                },
                InstKind::Return(v(4)),
            ],
            10,
        );

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        let merge_idx = k
            .iter()
            .position(|k| {
                matches!(
                    k,
                    InstKind::BlockLabel {
                        label: Label(2),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(spawn_idx > merge_idx, "spawn should stay in merge block");
    }

    #[test]
    fn eval_not_hoisted() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Const {
                    dst: v(5),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(5),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            10,
        );

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let eval_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Eval { .. }))
            .unwrap();
        let merge_idx = k
            .iter()
            .position(|k| {
                matches!(
                    k,
                    InstKind::BlockLabel {
                        label: Label(2),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(eval_idx > merge_idx, "eval must stay in merge block");
    }

    #[test]
    fn token_conflict_prevents_hoist() {
        let i = Interner::new();
        let tid = TokenId::alloc();
        let (qref1, ty1) = token_fn_type(&i, "io1", tid);
        let (qref2, ty2) = token_fn_type(&i, "io2", tid);
        let fn_types = FxHashMap::from_iter([(qref1, ty1), (qref2, ty2)]);

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref1),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Const {
                    dst: v(5),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(5),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Spawn {
                    dst: v(1),
                    callee: Callee::Direct(qref2),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Return(v(1)),
            ],
            10,
        );

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn2_idx = k
            .iter()
            .rposition(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        let merge_idx = k
            .iter()
            .position(|k| {
                matches!(
                    k,
                    InstKind::BlockLabel {
                        label: Label(2),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(
            spawn2_idx > merge_idx,
            "token-conflicting spawn should stay in merge block"
        );
    }

    #[test]
    fn multi_level_hoist() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        // B0 → diamond → B3 → diamond → B6
        // Spawn in B6 should hoist directly to B0 in one pass.
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(3),
                    then_args: vec![],
                    else_label: Label(4),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(3),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(5),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(4),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(5),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(5),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Spawn {
                    dst: v(1),
                    callee: Callee::Direct(qref),
                    args: vec![v(0)],
                    context_uses: vec![],
                },
                InstKind::Return(v(1)),
            ],
            10,
        );

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        let first_jumpif = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            spawn_idx < first_jumpif,
            "spawn should be hoisted to B0 via highest-target"
        );
    }

    #[test]
    fn pure_instruction_hoisted() {
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return(v(1)),
            ],
            10,
        );

        run(&mut cfg, &FxHashMap::default());
        let body = demoted(cfg);
        let k = kinds(&body);
        let binop_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { .. }))
            .unwrap();
        let jumpif_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            binop_idx < jumpif_idx,
            "pure BinOp should be hoisted before branch"
        );
    }

    // ── Sink tests ──────────────────────────────────────────────────

    /// Eval is sunk past pure computation to just before its result is used.
    ///
    /// Before: Spawn, Eval, BinOp(uses eval result), Return
    /// After:  Spawn, BinOp...(doesn't use eval), Eval, BinOp(uses eval result), Return
    #[test]
    fn eval_sunk_past_independent_computation() {
        let i = Interner::new();
        let (fetch_id, fetch_ty) = io_fn_type(&i, "fetch");
        let fn_types = FxHashMap::from_iter([(fetch_id, fetch_ty)]);

        // v0 = Spawn fetch
        // v1 = Eval v0              ← should sink
        // v2 = BinOp(v3, v3)       ← independent of eval result
        // v4 = BinOp(v1, v2)       ← uses eval result
        // Return v4
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fetch_id),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(3),
                    right: v(3),
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return(v(4)),
            ],
            10,
        );

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);

        let eval_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Eval { .. }))
            .unwrap();
        let use_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(1)))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(3)))
            .unwrap();

        assert!(
            independent_idx < eval_idx,
            "independent computation should be before eval (was {independent_idx}, eval at {eval_idx})"
        );
        assert!(
            eval_idx < use_idx,
            "eval should be before its use (eval at {eval_idx}, use at {use_idx})"
        );
    }

    /// Eval is NOT sunk past a Load of a context it writes.
    ///
    /// Eval writes @ctx, Load reads @ctx → order must be preserved.
    #[test]
    fn eval_not_sunk_past_context_load() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("ctx"));
        let (fetch_id, fetch_ty) = io_fn_type(&i, "fetch");
        let fn_types = FxHashMap::from_iter([(fetch_id, fetch_ty)]);

        // v0 = Spawn fetch
        // v1 = Eval v0, context_defs=[(ctx, v5)]
        // v2 = Ref @ctx
        // v3 = Load v2          ← reads ctx that Eval writes
        // v4 = BinOp(v3, v3)   ← uses load result
        // Return v4
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fetch_id),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![(ctx, v(5))],
                },
                InstKind::Ref {
                    dst: v(2),
                    target: crate::ir::RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Load {
                    dst: v(3),
                    src: v(2),
                    volatile: false,
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: acvus_ast::BinOp::Add,
                    left: v(3),
                    right: v(3),
                },
                InstKind::Return(v(4)),
            ],
            10,
        );

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);

        let eval_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Eval { .. }))
            .unwrap();
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Load { .. }))
            .unwrap();

        assert!(
            eval_idx < load_idx,
            "eval must stay before load of written context (eval {eval_idx}, load {load_idx})"
        );
    }

    /// Load is sunk past independent computation but NOT past a Store to the same context.
    #[test]
    fn load_sunk_but_not_past_store() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        // v0 = Ref @x
        // v1 = Load v0          ← should sink past v2 but not past v4 (store to @x)
        // v2 = BinOp(v5, v5)   ← independent
        // v3 = Ref @x
        // v4 = Store v3 = v5   ← writes @x — barrier for load
        // v6 = BinOp(v1, v2)   ← uses load result
        // Return v6
        let mut cfg = make_cfg(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: crate::ir::RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Load {
                    dst: v(1),
                    src: v(0),
                    volatile: false,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Ref {
                    dst: v(3),
                    target: crate::ir::RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(3),
                    value: v(5),
                    volatile: false,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return(v(6)),
            ],
            10,
        );

        run(&mut cfg, &FxHashMap::default());
        let body = demoted(cfg);
        let k = kinds(&body);

        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Load { .. }))
            .unwrap();
        let store_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Store { .. }))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(5)))
            .unwrap();

        assert!(
            load_idx > independent_idx,
            "load should sink past independent computation"
        );
        assert!(
            load_idx < store_idx,
            "load must NOT sink past store to same context (load {load_idx}, store {store_idx})"
        );
    }

    /// Store is sunk past independent computation but NOT past Load of same context.
    #[test]
    fn store_sunk_but_not_past_load_of_same_context() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        // v0 = Ref @x
        // v1 = Store v0 = v5       ← should sink past v2 but not past v4 (load @x)
        // v2 = BinOp(v5, v5)       ← independent
        // v3 = Ref @x
        // v4 = Load v3             ← reads @x — barrier for store
        // v6 = BinOp(v4, v2)
        // Return v6
        let mut cfg = make_cfg(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: crate::ir::RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(0),
                    value: v(5),
                    volatile: false,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Ref {
                    dst: v(3),
                    target: crate::ir::RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Load {
                    dst: v(4),
                    src: v(3),
                    volatile: false,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(4),
                    right: v(2),
                },
                InstKind::Return(v(6)),
            ],
            10,
        );

        run(&mut cfg, &FxHashMap::default());
        let body = demoted(cfg);
        let k = kinds(&body);

        let store_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Store { .. }))
            .unwrap();
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Load { .. }))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(5)))
            .unwrap();

        assert!(
            store_idx > independent_idx,
            "store should sink past independent computation"
        );
        assert!(
            store_idx < load_idx,
            "store must NOT sink past load of same context (store {store_idx}, load {load_idx})"
        );
    }

    /// Volatile Load is NOT sunk.
    #[test]
    fn volatile_load_not_sunk() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        let mut cfg = make_cfg(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: crate::ir::RefTarget::Context(ctx),
                    path: vec![],
                },
                InstKind::Load {
                    dst: v(1),
                    src: v(0),
                    volatile: true,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(3),
                    right: v(3),
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return(v(4)),
            ],
            10,
        );

        let body_before = demoted(cfg.clone());
        run(&mut cfg, &FxHashMap::default());
        let body_after = demoted(cfg);

        let load_idx_before = kinds(&body_before)
            .iter()
            .position(|k| matches!(k, InstKind::Load { volatile: true, .. }))
            .unwrap();
        let load_idx_after = kinds(&body_after)
            .iter()
            .position(|k| matches!(k, InstKind::Load { volatile: true, .. }))
            .unwrap();

        assert_eq!(
            load_idx_before, load_idx_after,
            "volatile load must not be moved"
        );
    }
}
