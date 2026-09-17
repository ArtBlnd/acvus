//! SSA Pass (mem2reg for local variables)
//!
//! Promotes whole `Take`/`Assign` of locals and parameters to SSA form. A
//! storage that is referenced or accessed by a field path stays in memory.
//! A context is a local variable of the body: the lowering fetches it into
//! its slot at entry (`Fetch`, then `Assign`) and commits it at each exit
//! (`Take`, then `Commit`), so the pass sees only the slot. `Fetch` defines
//! its `dst` and `Commit` uses its `value`; neither is promoted, removed,
//! duplicated, or moved.
//!
//! ## Pipeline
//!
//! 1. **Collect SSA info** (`collect_ssa_info`): every whole `Take` of a
//!    local or parameter is a read, every whole `Assign` a write; a storage
//!    with a `Ref` or a field-path access is not promoted.
//!
//! 2. **SSA builder** (`run_ssa_builder`): PHI insertion at merge points
//!    (Braun et al.). Produces `var_subst` (read → SSA value), the PHI
//!    insertions, and the entry definitions: a local starts `Undef`, a
//!    parameter or capture starts from its own register.
//!
//! 3. **Patch instructions** (`patch_instructions`): add the PHI results as
//!    block params and the incoming values as jump args.
//!
//! 4. **Apply var substitutions** (`apply_var_subst`): rewrite every use of
//!    a promoted read with its SSA value and remove the promoted `Take`s and
//!    `Assign`s.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};

use super::ssa::{ENTRY_BLOCK, SSABuilder, SsaVar};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Callee, Inst, InstKind, Label, RefTarget, ValueId};
use crate::ty::Ty;

/// Run the SSA pass on a CfgBody.
pub fn run(cfg: &mut CfgBody) {
    if cfg.blocks.is_empty() {
        return;
    }
    let ssa_info = collect_ssa_info(cfg);

    let has_work = !ssa_info.written_vars.is_empty()
        || !ssa_info.read_vars.is_empty()
        || !ssa_info.entry_param_defs.is_empty();
    if !has_work {
        return;
    }

    let preds = cfg.predecessors();
    let all_successors: Vec<SmallVec<[BlockIdx; 2]>> = (0..cfg.blocks.len())
        .map(|i| cfg.successors(BlockIdx(i)))
        .collect();
    let SsaBuild {
        phi_insertions,
        var_subst,
        entry_defs,
    } = run_ssa_builder(
        &cfg.blocks,
        &all_successors,
        &preds,
        &ssa_info,
        &mut cfg.val_factory,
        &mut cfg.val_types,
    );
    if !phi_insertions.is_empty() {
        patch_instructions(cfg, &phi_insertions);
    }
    materialize_entry_defs(cfg, entry_defs);

    if !var_subst.is_empty() {
        apply_var_subst(cfg, &var_subst, &ssa_info);
    }
}

/// Apply value substitutions to an instruction's operands.
pub(crate) fn apply_subst(kind: &mut InstKind, subst: &FxHashMap<ValueId, ValueId>) {
    map_uses(kind, &mut |v: &mut ValueId| {
        if let Some(&new) = subst.get(v) {
            *v = new;
        }
    });
}

/// Apply `s` to every use of `kind`, in operand order.
pub(crate) fn map_uses(kind: &mut InstKind, s: &mut impl FnMut(&mut ValueId)) {
    match kind {
        InstKind::Const { .. }
        | InstKind::Fetch { .. }
        | InstKind::Nop
        | InstKind::Diverge
        | InstKind::Poison { .. }
        | InstKind::Undef { .. } => {}
        // A place through a reference uses the reference.
        InstKind::Ref { target, .. } | InstKind::Take { target, .. } => {
            if let RefTarget::Through(r) = target {
                s(r);
            }
        }
        InstKind::Assign { target, value, .. } => {
            if let RefTarget::Through(r) = target {
                s(r);
            }
            s(value);
        }
        InstKind::Commit { value, .. } => s(value),
        InstKind::BinOp { left, right, .. } => {
            s(left);
            s(right);
        }
        InstKind::UnaryOp { operand, .. } => s(operand),
        InstKind::FieldGet { object, .. } => s(object),
        InstKind::FieldSet { object, value, .. } => {
            s(object);
            s(value);
        }
        InstKind::LoadFunction { .. } => {}
        InstKind::FunctionCall {
            callee,
            args,
            order,
            ..
        } => {
            if let Callee::Indirect(v) = callee {
                s(v);
            }
            args.iter_mut().for_each(|v| s(v));
            if let Some(edge) = order {
                s(&mut edge.before);
            }
        }
        InstKind::Spawn {
            callee,
            args,
            order,
            ..
        } => {
            if let Callee::Indirect(v) = callee {
                s(v);
            }
            args.iter_mut().for_each(|v| s(v));
            if let Some(o) = order {
                s(o);
            }
        }
        InstKind::Eval { src, .. } => {
            s(src);
        }
        InstKind::Merge { orders, .. } => orders.iter_mut().for_each(|v| s(v)),
        InstKind::MakeArray { elements, .. } => elements.iter_mut().for_each(|v| s(v)),
        InstKind::StringConcat { parts, .. } => parts.iter_mut().for_each(|v| s(v)),
        InstKind::StringEq { a, b, .. } => {
            s(a);
            s(b);
        }
        InstKind::StringClone { src, .. } => s(src),
        InstKind::MakeObject { fields, .. } => fields.iter_mut().for_each(|(_, v)| s(v)),
        InstKind::MakeTuple { elements, .. } => elements.iter_mut().for_each(|v| s(v)),
        InstKind::TupleIndex { tuple, .. } => s(tuple),
        InstKind::TestLiteral { src, .. } => s(src),
        InstKind::TestObjectKey { src, .. } => s(src),
        InstKind::ArrayIndex { array: list, .. } => s(list),
        InstKind::ArrayGet {
            array: list, index, ..
        } => {
            s(list);
            s(index);
        }
        InstKind::ObjectGet { object, .. } => s(object),
        InstKind::MakeClosure { captures, .. } => captures.iter_mut().for_each(|v| s(v)),
        InstKind::MakeVariant { payload, .. } => {
            if let Some(p) = payload {
                s(p);
            }
        }
        InstKind::TestVariant { src, .. } => s(src),
        InstKind::UnwrapVariant { src, .. } => s(src),
        // BlockLabel, Jump, JumpIf, Return are terminators in CfgBody, not instructions.
        // But they may still exist as InstKind variants for demoted code paths.
        InstKind::BlockLabel { params, .. } => params.iter_mut().for_each(|v| s(v)),
        InstKind::Jump { args, .. } => args.iter_mut().for_each(|v| s(v)),
        InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            s(cond);
            then_args.iter_mut().for_each(|v| s(v));
            else_args.iter_mut().for_each(|v| s(v));
        }
        InstKind::Return { value, order } => {
            s(value);
            if let Some(o) = order {
                s(o);
            }
        }
        InstKind::Drop { src } => s(src),
    }
}

/// Apply value substitutions to a block terminator's operands.
pub(crate) fn apply_subst_terminator(term: &mut Terminator, subst: &FxHashMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&new) = subst.get(v) {
            *v = new;
        }
    };
    match term {
        Terminator::Jump { args, .. } => args.iter_mut().for_each(&s),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            s(cond);
            then_args.iter_mut().for_each(&s);
            else_args.iter_mut().for_each(&s);
        }
        Terminator::Return { value, order } => {
            s(value);
            if let Some(o) = order {
                s(o);
            }
        }
        Terminator::Fallthrough | Terminator::Diverge => {}
    }
}

// -- Step 1: SSA info collection --------------------------------------

/// A single SSA-relevant operation, recorded in instruction order.
#[derive(Debug, Clone)]
enum SsaOp {
    VarStore { slot: ValueId, value: ValueId },
    VarLoad { dst: ValueId, slot: ValueId },
    ParamLoad { dst: ValueId, slot: ValueId },
}

/// Per-block operations in instruction order.
#[derive(Debug, Default)]
struct BlockOps {
    ops: Vec<SsaOp>,
}

/// Aggregated SSA info for local variables.
struct SsaInfo {
    written_vars: BTreeSet<ValueId>,
    /// All vars that are read (VarLoad) - for ensuring entry defines.
    read_vars: BTreeSet<ValueId>,
    /// var slot -> type (from VarStore/VarLoad).
    var_types: FxHashMap<ValueId, Ty>,
    /// param/capture slot -> initial loaded value (from entry region ParamLoad).
    entry_param_defs: BTreeMap<ValueId, ValueId>,
    /// Per-block ops, instruction-ordered.
    block_ops: FxHashMap<BlockIdx, BlockOps>,
}

fn collect_ssa_info(cfg: &CfgBody) -> SsaInfo {
    // A storage that is referenced, or read or written by field, stays in
    // memory; only a storage read and written whole is promoted.
    let mut non_promotable_vars: BTreeSet<ValueId> = BTreeSet::new();

    let mut block_ops: FxHashMap<BlockIdx, BlockOps> = FxHashMap::default();
    let mut written_vars: BTreeSet<ValueId> = BTreeSet::default();
    let mut read_vars: BTreeSet<ValueId> = BTreeSet::default();
    let mut var_types: FxHashMap<ValueId, Ty> = FxHashMap::default();

    // LLVM-style: param_regs ARE the SSA definitions for params.
    // For params/captures: the param_reg IS both the storage slot and the initial value.
    let mut entry_param_defs: BTreeMap<ValueId, ValueId> = BTreeMap::default();
    for (_name, reg) in cfg.params.iter().chain(cfg.captures.iter()) {
        entry_param_defs.insert(*reg, *reg);
        if let Some(ty) = cfg.val_types.get(reg) {
            var_types.insert(*reg, ty.clone());
        }
    }

    // First pass: identify non-promotable storage.
    for block in &cfg.blocks {
        for inst in &block.insts {
            let pinned = match &inst.kind {
                InstKind::Ref { target, .. } => Some(target),
                InstKind::Take { target, path, .. } | InstKind::Assign { target, path, .. }
                    if !path.is_empty() =>
                {
                    Some(target)
                }
                _ => None,
            };
            if let Some(RefTarget::Var(slot) | RefTarget::Param(slot)) = pinned {
                non_promotable_vars.insert(*slot);
            }
        }
    }

    // Second pass: collect SSA ops for whole reads and writes of promotable storage.
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let ops = block_ops.entry(BlockIdx(bi)).or_default();

        for inst in &block.insts {
            match &inst.kind {
                InstKind::Take {
                    dst,
                    target: RefTarget::Var(slot),
                    path,
                } if path.is_empty() && !non_promotable_vars.contains(slot) => {
                    ops.ops.push(SsaOp::VarLoad {
                        dst: *dst,
                        slot: *slot,
                    });
                    read_vars.insert(*slot);
                    if let Some(ty) = cfg.val_types.get(dst) {
                        var_types.entry(*slot).or_insert_with(|| ty.clone());
                    }
                }
                InstKind::Take {
                    dst,
                    target: RefTarget::Param(slot),
                    path,
                } if path.is_empty() && !non_promotable_vars.contains(slot) => {
                    read_vars.insert(*slot);
                    if let Some(ty) = cfg.val_types.get(dst) {
                        var_types.entry(*slot).or_insert_with(|| ty.clone());
                    }
                    ops.ops.push(SsaOp::ParamLoad {
                        dst: *dst,
                        slot: *slot,
                    });
                }
                InstKind::Assign {
                    target: RefTarget::Var(slot),
                    path,
                    value,
                } if path.is_empty() && !non_promotable_vars.contains(slot) => {
                    ops.ops.push(SsaOp::VarStore {
                        slot: *slot,
                        value: *value,
                    });
                    written_vars.insert(*slot);
                    if let Some(ty) = cfg.val_types.get(value) {
                        var_types.entry(*slot).or_insert_with(|| ty.clone());
                    }
                }
                _ => {}
            }
        }
    }

    SsaInfo {
        written_vars,
        read_vars,
        var_types,
        entry_param_defs,
        block_ops,
    }
}

// -- Step 2: SSABuilder execution ------------------------------------

/// Allocate a typed ValueId. This is the ONLY way to create new ValueIds
/// in the SSA pass. val_factory.next() must never be called directly.
fn alloc_val(
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
    ty: Ty,
) -> ValueId {
    let val = val_factory.next();
    val_types.insert(val, ty);
    val
}

/// Convenience: allocate a typed ValueId for an SSA variable, looking up the type from ssa_info.
fn alloc_var_val(
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
    var: SsaVar,
    ssa_info: &SsaInfo,
) -> ValueId {
    let SsaVar::Local(slot) = var else {
        panic!("SSA pass tracks only local variables, got {:?}", var);
    };
    // Every SSA variable MUST have a known type. If not, it's a collect_ssa_info bug.
    let ty = ssa_info
        .var_types
        .get(&slot)
        .cloned()
        .unwrap_or_else(|| panic!("SSA variable {:?} has no type in ssa_info", var));
    alloc_val(val_factory, val_types, ty)
}

fn run_ssa_builder(
    blocks: &[crate::cfg::Block],
    all_successors: &[SmallVec<[BlockIdx; 2]>],
    preds: &FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
    ssa_info: &SsaInfo,
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
) -> SsaBuild {
    let mut ssa = SSABuilder::new();

    let block_label = |bi: BlockIdx| -> Label { blocks[bi.0].label };

    // -- Detect loop headers (backedge target: succ index <= current index) --
    let mut loop_headers: BTreeSet<BlockIdx> = BTreeSet::default();
    for (bi, _) in blocks.iter().enumerate() {
        for &succ in &all_successors[bi] {
            if succ.0 <= bi {
                loop_headers.insert(succ);
            }
        }
    }

    // -- Register predecessors --
    for (block_idx, block_preds) in preds {
        let label = block_label(*block_idx);
        for pred in block_preds {
            ssa.add_predecessor(label, block_label(*pred));
        }
    }

    // -- Define initial values in entry block --
    let mut entry_defs = EntryDefs::default();

    // Param entry defs (from ParamLoad in entry region).
    for (&slot, &val) in &ssa_info.entry_param_defs {
        ssa.define(ENTRY_BLOCK, SsaVar::Local(slot), val);
    }
    // ALL variables that appear in any SsaOp (read or write) need entry defines.
    // This is the LLVM alloca pattern: every variable has a definition at entry.
    let all_local_vars: BTreeSet<ValueId> = ssa_info
        .written_vars
        .iter()
        .chain(ssa_info.read_vars.iter())
        .copied()
        .collect();
    for &slot in &all_local_vars {
        if !ssa_info.entry_param_defs.contains_key(&slot) {
            let undef_val = alloc_var_val(val_factory, val_types, SsaVar::Local(slot), ssa_info);
            ssa.define(ENTRY_BLOCK, SsaVar::Local(slot), undef_val);
            entry_defs.undef_locals.push(undef_val);
        }
    }

    // Typed alloc closure for SSA builder - every ValueId gets a type at birth.
    let mut typed_alloc =
        |var: SsaVar| -> ValueId { alloc_var_val(val_factory, val_types, var, ssa_info) };

    // Seal ENTRY_BLOCK (virtual predecessor of block 0) unless block 0 is a loop header.
    if !loop_headers.contains(&BlockIdx(0)) {
        ssa.seal_block(ENTRY_BLOCK, &mut typed_alloc);
    }

    // -- Single-pass: define and use in program order (Braun algorithm) --
    //
    // For each block, process ops in instruction order:
    //   - VarStore -> ssa.define() (updates current_defs)
    //   - VarLoad/ParamLoad -> ssa.use_var() (reads current_defs or predecessor)
    //
    // This ensures a VarLoad BEFORE a VarStore for the same slot gets the
    // predecessor's value (not the Store's value), because define() hasn't
    // been called yet at that point. A VarLoad AFTER a VarStore correctly
    // gets the stored value via current_defs (intra-block forwarding).
    //
    // Non-loop-header blocks are sealed BEFORE processing their ops,
    // since all predecessors (earlier in block order) are already processed.
    // Loop headers are sealed after all blocks (backedge sources come later).
    let mut var_subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();

    for (bi, _) in blocks.iter().enumerate() {
        let block_idx = BlockIdx(bi);
        let label = block_label(block_idx);

        // Seal non-loop-header blocks before processing (all predecessors done).
        if bi > 0 && !loop_headers.contains(&block_idx) {
            ssa.seal_block(label, &mut typed_alloc);
        }

        if let Some(ops) = ssa_info.block_ops.get(&block_idx) {
            for op in &ops.ops {
                match op {
                    // A stored value may itself be a promoted load; define the
                    // variable with the SSA value that load stands for, so no
                    // definition names a load the pass removes.
                    SsaOp::VarStore { slot, value } => {
                        let value = var_subst.get(value).copied().unwrap_or(*value);
                        ssa.define(label, SsaVar::Local(*slot), value);
                    }
                    SsaOp::VarLoad { dst, slot } | SsaOp::ParamLoad { dst, slot } => {
                        let ssa_val = ssa.use_var(label, SsaVar::Local(*slot), &mut typed_alloc);
                        if ssa_val != *dst {
                            var_subst.insert(*dst, ssa_val);
                        }
                    }
                }
            }
        }
    }

    // -- Seal loop headers (deferred: backedge predecessors processed above) --
    for &header in &loop_headers {
        ssa.seal_block(block_label(header), &mut typed_alloc);
    }

    // -- Trigger PHIs at merge points --
    //
    // Explicitly request the merged value for every written variable at each
    // merge block. This ensures PHI nodes exist even when the variable is not
    // loaded at the merge point (needed for loop backedge args).
    let mut merge_blocks: Vec<_> = preds.iter().filter(|(_, p)| p.len() > 1).collect();
    merge_blocks.sort_by_key(|(idx, _)| *idx);
    for (block_idx, _) in merge_blocks {
        let label = block_label(*block_idx);
        for &slot in &ssa_info.written_vars {
            let _ = ssa.use_var(label, SsaVar::Local(slot), &mut typed_alloc);
        }
    }

    let (phi_insertions, trivial_subst) = ssa.finish();

    // Resolve trivial-phi references in var_subst.
    // A VarLoad may have been mapped to a pending phi ValueId that was later
    // eliminated as trivial. Walk the trivial_subst chain to reach the real value.
    if !trivial_subst.is_empty() {
        for val in var_subst.values_mut() {
            let mut resolved = *val;
            while let Some(&next) = trivial_subst.get(&resolved) {
                resolved = next;
            }
            *val = resolved;
        }
    }

    SsaBuild {
        phi_insertions,
        var_subst,
        entry_defs,
    }
}

/// What SSA construction produced for the body.
struct SsaBuild {
    phi_insertions: Vec<super::ssa::PhiInsertion>,
    /// VarLoad/ParamLoad result -> the SSA value that replaces it.
    var_subst: FxHashMap<ValueId, ValueId>,
    entry_defs: EntryDefs,
}

/// Initial SSA values the entry block must define before its first instruction.
#[derive(Default)]
struct EntryDefs {
    /// A local variable has no value before its first store.
    undef_locals: Vec<ValueId>,
}

/// Prepend the entry definitions to block 0.
fn materialize_entry_defs(cfg: &mut CfgBody, entry_defs: EntryDefs) {
    let insts: Vec<Inst> = entry_defs
        .undef_locals
        .into_iter()
        .map(|dst| Inst {
            span: acvus_ast::Span::ZERO,
            kind: InstKind::Undef { dst },
        })
        .collect();
    if !insts.is_empty() {
        cfg.blocks[0].insts.splice(0..0, insts);
    }
}

// -- Step 3: Patch instructions --------------------------------------

fn patch_instructions(cfg: &mut CfgBody, phi_insertions: &[super::ssa::PhiInsertion]) {
    // PHI lookup tables.
    let mut block_phis: BTreeMap<Label, Vec<&super::ssa::PhiInsertion>> = BTreeMap::default();
    for phi in phi_insertions {
        block_phis.entry(phi.block).or_default().push(phi);
    }
    // Sort PHIs by SsaVar for deterministic block param ordering.
    for phis in block_phis.values_mut() {
        phis.sort_by_key(|p| p.var);
    }

    // Build jump args in the same SsaVar-sorted order as block_phis.
    let mut jump_extra_args: FxHashMap<(Label, Label), Vec<ValueId>> = FxHashMap::default();
    for (&label, phis) in &block_phis {
        for phi in phis {
            for &(pred, val) in &phi.incoming {
                jump_extra_args.entry((pred, label)).or_default().push(val);
            }
        }
    }

    // Add PHI results as block params.
    for (&label, phis) in &block_phis {
        if let Some(&block_idx) = cfg.label_to_block.get(&label) {
            let block = &mut cfg.blocks[block_idx.0];
            for phi in phis {
                block.params.push(phi.result);
            }
        }
    }

    // Add jump args to terminators of predecessor blocks.
    for block in cfg.blocks.iter_mut() {
        let pred_label = block.label;

        match &mut block.terminator {
            Terminator::Jump { label, args } => {
                if let Some(extra) = jump_extra_args.get(&(pred_label, *label)) {
                    args.extend_from_slice(extra);
                }
            }
            Terminator::JumpIf {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                if let Some(extra) = jump_extra_args.get(&(pred_label, *then_label)) {
                    then_args.extend_from_slice(extra);
                }
                if let Some(extra) = jump_extra_args.get(&(pred_label, *else_label)) {
                    else_args.extend_from_slice(extra);
                }
            }
            _ => {}
        }
    }
}

// -- Step 4: Apply var substitutions + remove VarLoad/VarStore/ParamLoad --

fn apply_var_subst(cfg: &mut CfgBody, var_subst: &FxHashMap<ValueId, ValueId>, ssa_info: &SsaInfo) {
    use crate::ir::RefTarget;

    let substituted: FxHashSet<ValueId> = var_subst.keys().copied().collect();
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, var_subst);
        }
        apply_subst_terminator(&mut block.terminator, var_subst);

        // A promoted read stands for its SSA value; a promoted write is the
        // SSA definition itself.
        block.insts.retain(|inst| match &inst.kind {
            InstKind::Take { dst, .. } if substituted.contains(dst) => false,
            InstKind::Assign {
                target: RefTarget::Var(slot),
                path,
                ..
            } if path.is_empty() && ssa_info.written_vars.contains(slot) => false,
            _ => true,
        });
    }
}

// -- Tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{self, CfgBody};
    use crate::test::{compile_script, compile_template};
    use crate::ty::Ty;
    use acvus_utils::Interner;

    fn count_phi_blocks(cfg_body: &CfgBody) -> usize {
        cfg_body
            .blocks
            .iter()
            .filter(|b| !b.params.is_empty())
            .count()
    }

    fn count_commits(cfg_body: &CfgBody) -> usize {
        cfg_body
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(&i.kind, InstKind::Commit { .. }))
            .count()
    }

    /// Every value an instruction or terminator uses is defined by an
    /// instruction, a block param, or an entry definition.
    fn every_use_is_defined(cfg_body: &CfgBody) -> bool {
        let mut defs: FxHashSet<ValueId> = cfg_body.entry_defs().collect();
        for b in &cfg_body.blocks {
            defs.extend(b.params.iter().copied());
            for i in &b.insts {
                defs.extend(crate::analysis::inst_info::defs(&i.kind));
            }
        }
        cfg_body.blocks.iter().all(|b| {
            b.insts
                .iter()
                .flat_map(|i| crate::analysis::inst_info::uses(&i.kind))
                .all(|u| defs.contains(&u))
                && match &b.terminator {
                    Terminator::Return { value, order } => {
                        defs.contains(value) && order.is_none_or(|o| defs.contains(&o))
                    }
                    _ => true,
                }
        })
    }

    #[test]
    fn a_stored_copy_of_a_promoted_load_defines_the_ssa_value() {
        let i = Interner::new();
        let module = compile_script(
            &i,
            "let x = 1; let y = x; @out = y; @out",
            &[("out", Ty::I64)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        run(&mut cfg_body);
        assert!(
            every_use_is_defined(&cfg_body),
            "a copy through two promoted variables reaches the constant, not a removed load"
        );
    }

    // -- Completeness: PHI inserted when needed --

    #[test]
    fn match_one_arm_write_phi() {
        let i = Interner::new();
        let module = compile_template(
            &i,
            r#"{{ true = @n == 1 }}{{ @x = 42 }}{{ _ }}noop{{/}}"#,
            &[("x", Ty::I64), ("n", Ty::I64)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        run(&mut cfg_body);
        assert!(
            count_phi_blocks(&cfg_body) >= 1,
            "merge should have PHI for @x"
        );
    }

    #[test]
    fn match_both_arms_write_phi() {
        let i = Interner::new();
        let module = compile_template(
            &i,
            r#"{{ true = @n == 1 }}{{ @x = 1 }}{{ _ }}{{ @x = 2 }}{{/}}"#,
            &[("x", Ty::I64), ("n", Ty::I64)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        run(&mut cfg_body);
        assert!(count_phi_blocks(&cfg_body) >= 1);
    }

    // -- Soundness: page ops are neither removed nor duplicated --

    #[test]
    fn match_no_write_keeps_commits() {
        let i = Interner::new();
        let module = compile_template(
            &i,
            r#"{{ true = @n == 1 }}yes{{ _ }}no{{/}}"#,
            &[("n", Ty::I64)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        let commits_before = count_commits(&cfg_body);
        run(&mut cfg_body);
        assert_eq!(count_commits(&cfg_body), commits_before);
    }

    #[test]
    fn straight_line_write_keeps_commits() {
        let i = Interner::new();
        let module = compile_script(&i, "@x = 42; @x", &[("x", Ty::I64)]).unwrap();
        let mut cfg_body = cfg::promote(module.main);
        let commits_before = count_commits(&cfg_body);
        run(&mut cfg_body);
        assert_eq!(count_commits(&cfg_body), commits_before);
    }
}
