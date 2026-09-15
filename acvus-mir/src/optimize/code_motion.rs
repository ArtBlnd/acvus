//! Cross-block code motion: hoist pure instructions above branch points.
//!
//! After SpawnSplit, IO calls are Spawn (the call is issued) + Eval (the
//! result is awaited). This pass hoists pure instructions out of dominated
//! blocks into their dominator ancestors and sinks Evals toward their uses.
//! A Spawn is never hoisted across a branch: the work starts at the Spawn,
//! and issuing it on a path that would not have reached it speculates an
//! effect (RFC-0007).
//!
//! # Algorithm
//!
//! Each iteration:
//! 1. Build the dominator tree.
//! 2. For each hoistable instruction, walk UP the dominator chain to find the
//!    **highest ancestor** where all operands are available.
//!    This eliminates the need for multi-iteration fixpoint on deep merge chains.
//! 3. `def_block` is updated after each hoist decision, so later instructions
//!    in the same block see their dependencies' new locations - operand chains
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
//! LoadFunction.
//!
//! NOT hoisted: Spawn, Eval, calls, context ops, variable ops.

use rustc_hash::FxHashMap;

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::cfg::{BlockIdx, CfgBody};
use crate::ir::*;
use crate::optimize::context_ops::{context_read, context_written};

// -- Entry point ----------------------------------------------------

pub fn run(cfg: &mut CfgBody) {
    // Phase 1: Hoist - move Spawn and pure instructions UP.
    loop {
        if !hoist_pass(cfg) {
            break;
        }
    }

    // Phase 2: Sink - move Eval and blocking instructions DOWN.
    sink_pass(cfg);
}

// -- Single hoist pass ----------------------------------------------

/// One iteration: find hoistable instructions, move them to the highest
/// valid dominator. Returns true if anything moved.
fn hoist_pass(cfg: &mut CfgBody) -> bool {
    if cfg.blocks.len() < 2 {
        return false;
    }

    let domtree = DomTree::build(cfg);
    let mut def_block = build_def_block(cfg);

    // -- Collect hoists ---------------------------------------------
    //
    // For each instruction, find the highest dominator ancestor where
    // all operands are available.
    // def_block is updated after each decision so that operand chains
    // within a block are resolved in one pass.

    let mut hoists: Vec<(usize, usize, usize)> = Vec::new(); // (src_block, inst_idx, tgt_block)

    for (bi, block) in cfg.blocks.iter().enumerate() {
        if domtree.idom(BlockIdx(bi)).is_none() {
            continue;
        }

        for (i, inst) in block.insts.iter().enumerate() {
            let kind = &inst.kind;

            if !is_hoistable(kind) || inst_info::defs(kind).is_empty() {
                continue;
            }

            let uses = inst_info::uses(kind);

            if let Some(target) =
                find_highest_target(BlockIdx(bi), &uses, &domtree, &def_block)
            {
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

    // -- Apply hoists -----------------------------------------------

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

// -- Def-block map --------------------------------------------------

/// Build ValueId -> BlockIdx mapping: where each value is defined.
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
    }

    for v in cfg.entry_defs() {
        def_block.entry(v).or_insert(BlockIdx(0));
    }

    def_block
}

// -- Target finding -------------------------------------------------

/// Walk up the dominator chain from `block_idx` to find the highest ancestor
/// where all operands are available (before the ancestor's terminator).
fn find_highest_target(
    block_idx: BlockIdx,
    uses: &[ValueId],
    domtree: &DomTree,
    def_block: &FxHashMap<ValueId, BlockIdx>,
) -> Option<BlockIdx> {
    let mut best: Option<BlockIdx> = None;
    let mut candidate = domtree.idom(block_idx)?;

    loop {
        // All operands must be available at candidate's body (before terminator).
        let all_available = uses.iter().all(|u| match def_block.get(u) {
            Some(&def_bi) if def_bi == candidate => true,
            Some(&def_bi) => domtree.dominates(def_bi, candidate),
            None => true,
        });
        if !all_available {
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

// -- Hoistability (allowlist) ---------------------------------------

/// Can this instruction be safely hoisted to a dominator block?
///
/// Allowlist: only provably pure instructions. Unknown kinds default to
/// not hoistable (soundness by construction).
fn is_hoistable(kind: &InstKind) -> bool {
    match kind {
        // Arithmetic / logic.
        InstKind::BinOp { .. } | InstKind::UnaryOp { .. } => true,

        // A word constant; a heap value is built where it is used.
        InstKind::Const { value, .. } => {
            !matches!(value, acvus_ast::Literal::String(_) | acvus_ast::Literal::List(_))
        }
        InstKind::MakeArray { .. }
        | InstKind::MakeObject { .. }
        | InstKind::MakeTuple { .. }
        | InstKind::MakeVariant { .. }
        | InstKind::MakeClosure { .. } => false,

        // A place under a variable is an address (no-op, pure); one through
        // a reference is a memory op and stays in order with the other ops
        // through it.
        InstKind::Ref { .. } => false,

        // Field / element access (scalar, pure). UnwrapVariant assumes the
        // tag its test established and ArrayGet assumes an index in range,
        // so neither may run on a path that has not checked; they stay.
        InstKind::FieldGet { .. }
        | InstKind::FieldSet { .. }
        | InstKind::ObjectGet { .. }
        | InstKind::ArrayIndex { .. }
        | InstKind::TupleIndex { .. } => true,

        // Test predicates.
        InstKind::TestLiteral { .. }
        | InstKind::TestVariant { .. }
        | InstKind::TestObjectKey { .. } => true,

        // Function reference.
        InstKind::LoadFunction { .. } => true,

        _ => false,
    }
}

// -- Sink pass -----------------------------------------------------
//
// Moves Eval as late as possible - just before
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

// -- Sink infrastructure ---------------------------------------------

/// Run the sink pass - move each Eval as late as possible within its
/// block. A page op (`Fetch`, `Commit`) is never moved.
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

/// Try to sink ONE Eval. Returns true if something moved.
fn sink_one(cfg: &mut CfgBody) -> bool {
    let loans = Loans::build(cfg);
    for bi in 0..cfg.blocks.len() {
        for ii in 0..cfg.blocks[bi].insts.len() {
            let kind = &cfg.blocks[bi].insts[ii].kind;
            let InstKind::Eval { .. } = kind else {
                continue;
            };
            let effect = loans.storage_effect(kind);

            let defs: Vec<ValueId> = inst_info::defs(kind).to_vec();

            // Scan forward for barrier.
            let mut barrier = cfg.blocks[bi].insts.len();

            for jj in (ii + 1)..cfg.blocks[bi].insts.len() {
                let other = &cfg.blocks[bi].insts[jj].kind;

                let other_uses = inst_info::uses(other);
                if defs.iter().any(|d| other_uses.contains(d)) {
                    barrier = jj;
                    break;
                }

                if is_call(other) || is_page_op(other) || effect.conflicts(&loans.storage_effect(other)) {
                    barrier = jj;
                    break;
                }
            }

            let term_uses = terminator_uses_vec(&cfg.blocks[bi].terminator);
            if defs.iter().any(|d| term_uses.contains(d)) {
                barrier = barrier.min(cfg.blocks[bi].insts.len());
            }

            let target = if barrier > 0 { barrier - 1 } else { ii };
            if target > ii {
                let inst = cfg.blocks[bi].insts.remove(ii);
                cfg.blocks[bi].insts.insert(target, inst);
                return true;
            }
        }
    }
    false
}

fn is_page_op(kind: &InstKind) -> bool {
    context_read(kind).is_some() || context_written(kind).is_some()
}

fn is_call(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::FunctionCall { .. } | InstKind::Spawn { .. } | InstKind::Eval { .. }
    )
}

// -- Tests ----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{self, CfgBody};
    use crate::graph::QualifiedRef;
    use crate::ty::Ty;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

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
            order_param: None,
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
                effect: crate::ty::Effect::OPAQUE.into(),
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
    fn spawn_stays_below_branch() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");

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
                    callee_ty: ty,
                    args: vec![v(0)],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
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
            spawn_idx > jumpif_idx,
            "spawn stays in its block (spawn {spawn_idx}, branch {jumpif_idx})"
        );
    }

    #[test]
    fn block_param_dependency_prevents_hoist() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");

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
                    callee_ty: ty,
                    args: vec![v(3)],
                    order: None,
                },
                InstKind::Return {
                    value: v(4),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
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
    fn a_reference_and_a_heap_constant_are_never_hoisted() {
        assert!(!is_hoistable(&InstKind::Ref {
            dst: v(1),
            target: RefTarget::Var(v(0)),
            path: vec![],
            mutability: crate::ty::Mutability::Shared,
        }));
        assert!(!is_hoistable(&InstKind::Const {
            dst: v(1),
            value: acvus_ast::Literal::String("a".into()),
        }));
        assert!(is_hoistable(&InstKind::Const {
            dst: v(1),
            value: acvus_ast::Literal::Int(1),
        }));
        assert!(!is_hoistable(&InstKind::MakeArray {
            dst: v(1),
            elements: vec![],
        }));
    }

    #[test]
    fn eval_not_hoisted() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref),
                    callee_ty: ty,
                    args: vec![],
                    order: None,
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
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
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
    fn spawn_is_not_hoisted_across_a_branch() {
        let i = Interner::new();
        let (qref1, ty1) = io_fn_type(&i, "io1");
        let (qref2, ty2) = io_fn_type(&i, "io2");

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref1),
                    callee_ty: ty1,
                    args: vec![],
                    order: None,
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
                    callee_ty: ty2,
                    args: vec![],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
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
            "a Spawn stays in its block: hoisting it would issue the call on a path that skips it"
        );
    }

    #[test]
    fn multi_level_hoist() {
        // B0 -> diamond -> B3 -> diamond -> B6
        // The pure op in B6 hoists directly to B0 in one pass.
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
                InstKind::UnaryOp {
                    dst: v(1),
                    op: acvus_ast::UnaryOp::Not,
                    operand: v(0),
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let op_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::UnaryOp { .. }))
            .unwrap();
        let first_jumpif = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            op_idx < first_jumpif,
            "pure op should be hoisted to B0 via highest-target"
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
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
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

    // -- Sink tests --------------------------------------------------

    /// Eval is sunk past pure computation to just before its result is used.
    ///
    /// Before: Spawn, Eval, BinOp(uses eval result), Return
    /// After:  Spawn, BinOp...(doesn't use eval), Eval, BinOp(uses eval result), Return
    #[test]
    fn eval_sunk_past_independent_computation() {
        let i = Interner::new();
        let (fetch_id, fetch_ty) = io_fn_type(&i, "fetch");

        // v0 = Spawn fetch
        // v1 = Eval v0              <- should sink
        // v2 = BinOp(v3, v3)       <- independent of eval result
        // v4 = BinOp(v1, v2)       <- uses eval result
        // Return v4
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fetch_id),
                    callee_ty: fetch_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
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
                InstKind::Return {
                    value: v(4),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
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

    #[test]
    fn fetch_not_sunk_past_call() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let f = QualifiedRef::root(i.intern("f"));
        let mut cfg = make_cfg(
            vec![
                InstKind::Fetch {
                    dst: v(1),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::FunctionCall {
                    dst: v(3),
                    callee: Callee::Direct(f),
                    callee_ty: Ty::error(),
                    args: vec![],
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        let call_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::FunctionCall { .. }))
            .unwrap();
        assert!(
            load_idx < call_idx,
            "fetch must stay before the call (fetch {load_idx}, call {call_idx})"
        );
    }

    #[test]
    fn eval_not_sunk_past_fetch() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref),
                    callee_ty: ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Fetch {
                    dst: v(4),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(4),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let eval_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Eval { .. }))
            .unwrap();
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        assert!(
            eval_idx < load_idx,
            "eval must stay before the fetch (eval {eval_idx}, fetch {load_idx})"
        );
    }

    /// A fetch is not moved, even past independent computation.
    #[test]
    fn fetch_stays_in_place() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        // v1 = fetch @x
        // v2 = BinOp(v5, v5)   <- independent
        // commit @x = v5
        // v6 = BinOp(v1, v2)   <- uses the fetched value
        // Return v6
        let mut cfg = make_cfg(
            vec![
                InstKind::Fetch {
                    dst: v(1),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Commit {
                    context: ctx,
                    value: v(5),
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);

        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        let store_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Commit { .. }))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(5)))
            .unwrap();

        assert!(
            load_idx < independent_idx,
            "fetch is not moved (fetch {load_idx}, independent {independent_idx})"
        );
        assert!(
            load_idx < store_idx,
            "fetch stays before the commit of the same context (fetch {load_idx}, commit {store_idx})"
        );
    }

    /// A commit is not moved, even past independent computation.
    #[test]
    fn commit_stays_in_place() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        // commit @x = v5
        // v2 = BinOp(v5, v5)       <- independent
        // v4 = fetch @x
        // v6 = BinOp(v4, v2)
        // Return v6
        let mut cfg = make_cfg(
            vec![
                InstKind::Commit {
                    context: ctx,
                    value: v(5),
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Fetch {
                    dst: v(4),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(4),
                    right: v(2),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);

        let store_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Commit { .. }))
            .unwrap();
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(5)))
            .unwrap();

        assert!(
            store_idx < independent_idx,
            "commit is not moved (commit {store_idx}, independent {independent_idx})"
        );
        assert!(
            store_idx < load_idx,
            "commit stays before the fetch of the same context (commit {store_idx}, fetch {load_idx})"
        );
    }
}
