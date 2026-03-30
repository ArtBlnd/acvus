//! Register coloring — compact ValueId allocation via SSA-aware greedy coloring.
//!
//! After SSA, every value is defined exactly once. The interference graph of
//! an SSA program is chordal, so greedy coloring in definition order is optimal.
//!
//! # Algorithm
//!
//! 1. **Liveness**: backward dataflow → live_in/live_out per block.
//! 2. **Last-use**: for each value, where in its block is the final use?
//!    Values in live_out die in a later block and are excluded.
//! 3. **Greedy coloring**: walk blocks in order, maintaining a set of
//!    currently occupied colors. At each definition, assign the smallest
//!    color not occupied by any live value of the same type.
//!    At each last-use, free the color for reuse.
//! 4. **Rewrite**: map old ValueIds to compacted slots in-place.
//!
//! # Kill order
//!
//! Within one instruction, the sequence is:
//!   **color defs → kill dying uses → kill dead defs**
//!
//! Defs are colored while uses are still live, so a def never steals
//! a color from its own operand. Uses die after the instruction completes.
//! Dead defs (defined but never used) are freed immediately.

use crate::analysis::inst_info;
use crate::analysis::liveness::{self, LivenessResult};
use crate::cfg::{CfgBody, Terminator};
use crate::ir::{Callee, InstKind, ValueId};
use crate::ty::Ty;
use acvus_utils::LocalFactory;
use rustc_hash::{FxHashMap, FxHashSet};

// ── Public API ─────────────────────────────────────────────────────

/// Color with type-compatible slot reuse (default — typed interpreter).
pub fn color_body(cfg: &mut CfgBody) {
    color_body_inner(cfg, false);
}

/// Color with untyped scalar slot reuse (kovac — all scalars share slots).
pub fn color_body_untyped(cfg: &mut CfgBody) {
    color_body_inner(cfg, true);
}

fn color_body_inner(cfg: &mut CfgBody, untyped_scalars: bool) {
    if cfg.blocks.is_empty() {
        return;
    }

    let liveness = liveness::analyze(cfg);
    let last_use = LastUseMap::build(cfg, &liveness);

    let coloring = compute_coloring(cfg, &liveness, &last_use, untyped_scalars);

    if !coloring.is_improvement() {
        return;
    }

    apply_coloring(cfg, &coloring);
}

/// Returns true for scalar types that can share register slots in untyped mode.
fn is_scalar_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Bool | Ty::Float | Ty::Unit)
}

// ── Coloring ───────────────────────────────────────────────────────

/// Color assignment state: which slot (color) each ValueId is mapped to.
struct Coloring {
    /// ValueId → slot number.
    color_of: FxHashMap<ValueId, u32>,
    /// slot number → type constraint. Same-type values share slots.
    slot_types: Vec<Option<Ty>>,
}

impl Coloring {
    fn new() -> Self {
        Coloring {
            color_of: FxHashMap::default(),
            slot_types: Vec::new(),
        }
    }

    /// Assign the smallest available color with matching type.
    /// If `untyped_scalars` is true, all scalar types (Int, Bool, Float, Unit)
    /// are treated as compatible for slot reuse.
    fn assign(
        &mut self,
        live: &FxHashSet<u32>,
        val: ValueId,
        ty: Option<&Ty>,
        untyped_scalars: bool,
    ) -> u32 {
        let color = (0u32..)
            .find(|&c| {
                if live.contains(&c) {
                    return false;
                }
                match self.slot_types.get(c as usize) {
                    Some(Some(slot_ty)) => {
                        if untyped_scalars && is_scalar_ty(slot_ty) && ty.map_or(false, is_scalar_ty) {
                            true
                        } else {
                            ty == Some(slot_ty)
                        }
                    }
                    Some(None) => ty.is_none(),
                    None => true, // New slot — any type.
                }
            })
            .unwrap();

        self.color_of.insert(val, color);
        if color as usize >= self.slot_types.len() {
            self.slot_types.resize(color as usize + 1, None);
        }
        if self.slot_types[color as usize].is_none() {
            self.slot_types[color as usize] = ty.cloned();
        }
        color
    }

    fn color_of(&self, val: ValueId) -> Option<u32> {
        self.color_of.get(&val).copied()
    }

    fn is_colored(&self, val: &ValueId) -> bool {
        self.color_of.contains_key(val)
    }

    fn num_slots(&self) -> usize {
        self.slot_types.len()
    }

    fn is_improvement(&self) -> bool {
        self.num_slots() < self.color_of.len()
    }
}

/// Run greedy coloring over all blocks.
fn compute_coloring(
    cfg: &CfgBody,
    liveness: &LivenessResult,
    last_use: &LastUseMap,
    untyped_scalars: bool,
) -> Coloring {
    let mut coloring = Coloring::new();

    // Params and captures are live simultaneously at entry — color them first.
    let mut entry_live = FxHashSet::default();
    for &(_, v) in cfg.params.iter().chain(cfg.captures.iter()) {
        let c = coloring.assign(&entry_live, v, cfg.val_types.get(&v), untyped_scalars);
        entry_live.insert(c);
    }

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let mut live = LiveColors::from_live_in(&liveness.live_in[bi], &coloring, untyped_scalars);

        // Block params: defined at block entry.
        for &param in &block.params {
            live.define(&mut coloring, param, cfg.val_types.get(&param));
        }

        // Body instructions.
        for (i, inst) in block.insts.iter().enumerate() {
            let defs = inst_info::defs(&inst.kind);
            let uses = inst_info::uses(&inst.kind);

            // 1. Color defs (while uses still occupy their colors).
            for d in &defs {
                live.define(&mut coloring, *d, cfg.val_types.get(d));
            }

            // 2. Kill uses whose last use is this instruction.
            for u in &uses {
                if last_use.dies_at(bi, *u, UsePoint::Inst(i)) {
                    live.kill(&coloring, *u);
                }
            }

            // 3. Kill dead defs (defined but never used, not live-out).
            for d in &defs {
                if !liveness.live_out[bi].contains(d) && !last_use.has_use_in(bi, d) {
                    live.kill(&coloring, *d);
                }
            }
        }

        // Terminator.
        for d in terminator_defs(&block.terminator) {
            live.define(&mut coloring, d, cfg.val_types.get(&d));
        }
        for u in terminator_uses(&block.terminator) {
            if last_use.dies_at(bi, u, UsePoint::Terminator) {
                live.kill(&coloring, u);
            }
        }
    }

    coloring
}

// ── LiveColors ─────────────────────────────────────────────────────

/// Set of colors (slots) currently occupied by live values.
struct LiveColors {
    occupied: FxHashSet<u32>,
    untyped_scalars: bool,
}

impl LiveColors {
    /// Initialize from a block's live_in set.
    fn from_live_in(
        live_in: &FxHashSet<ValueId>,
        coloring: &Coloring,
        untyped_scalars: bool,
    ) -> Self {
        let occupied = live_in
            .iter()
            .filter_map(|v| coloring.color_of(*v))
            .collect();
        LiveColors {
            occupied,
            untyped_scalars,
        }
    }

    /// Assign a color to a new definition and mark it live.
    fn define(&mut self, coloring: &mut Coloring, val: ValueId, ty: Option<&Ty>) {
        let c = coloring.assign(&self.occupied, val, ty, self.untyped_scalars);
        self.occupied.insert(c);
    }

    /// Free a value's color (it died at this point).
    fn kill(&mut self, coloring: &Coloring, val: ValueId) {
        if let Some(c) = coloring.color_of(val) {
            self.occupied.remove(&c);
        }
    }
}

// ── Last-use analysis ──────────────────────────────────────────────

/// Where a value is last used within a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UsePoint {
    Inst(usize),
    Terminator,
}

/// Per-block last-use map.
///
/// Records the final use point for each value whose last use is in this block
/// (not live-out). Values that survive to a later block are excluded.
struct LastUseMap {
    blocks: Vec<FxHashMap<ValueId, UsePoint>>,
}

impl LastUseMap {
    fn build(cfg: &CfgBody, liveness: &LivenessResult) -> Self {
        let blocks = cfg
            .blocks
            .iter()
            .enumerate()
            .map(|(bi, block)| {
                let mut last = FxHashMap::default();

                for (i, inst) in block.insts.iter().enumerate() {
                    for u in inst_info::uses(&inst.kind) {
                        last.insert(u, UsePoint::Inst(i));
                    }
                }
                for u in terminator_uses(&block.terminator) {
                    last.insert(u, UsePoint::Terminator);
                }

                // Only keep values that die in this block.
                last.retain(|v, _| !liveness.live_out[bi].contains(v));
                last
            })
            .collect();

        LastUseMap { blocks }
    }

    /// Does `val` die at `point` in block `bi`?
    fn dies_at(&self, bi: usize, val: ValueId, point: UsePoint) -> bool {
        self.blocks.get(bi).and_then(|m| m.get(&val)).copied() == Some(point)
    }

    /// Does `val` have any use in block `bi`?
    fn has_use_in(&self, bi: usize, val: &ValueId) -> bool {
        self.blocks.get(bi).is_some_and(|m| m.contains_key(val))
    }
}

// ── Terminator value access ────────────────────────────────────────

fn terminator_uses(term: &Terminator) -> smallvec::SmallVec<[ValueId; 4]> {
    match term {
        Terminator::Jump { args, .. } => args.iter().copied().collect(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = smallvec::SmallVec::new();
            v.push(*cond);
            v.extend(then_args.iter().copied());
            v.extend(else_args.iter().copied());
            v
        }
        Terminator::ListStep {
            list,
            index_src,
            done_args,
            ..
        } => {
            let mut v = smallvec::SmallVec::new();
            v.push(*list);
            v.push(*index_src);
            v.extend(done_args.iter().copied());
            v
        }
        Terminator::Return(val) => smallvec::smallvec![*val],
        Terminator::Fallthrough => smallvec::SmallVec::new(),
    }
}

fn terminator_defs(term: &Terminator) -> smallvec::SmallVec<[ValueId; 2]> {
    match term {
        Terminator::ListStep { dst, index_dst, .. } => smallvec::smallvec![*dst, *index_dst],
        _ => smallvec::SmallVec::new(),
    }
}

// ── Rewrite ────────────────────────────────────────────────────────

fn apply_coloring(cfg: &mut CfgBody, coloring: &Coloring) {
    let mut new_factory = LocalFactory::<ValueId>::new();
    let mut slot_to_new: Vec<ValueId> = Vec::with_capacity(coloring.num_slots());
    for _ in 0..coloring.num_slots() {
        slot_to_new.push(new_factory.next());
    }

    let remap = |v: ValueId| -> ValueId {
        coloring
            .color_of(v)
            .map(|slot| slot_to_new[slot as usize])
            .unwrap_or(v)
    };

    // Rewrite blocks.
    for block in &mut cfg.blocks {
        for v in &mut block.params {
            *v = remap(*v);
        }
        for inst in &mut block.insts {
            rewrite_inst(&mut inst.kind, &remap);
        }
        rewrite_terminator(&mut block.terminator, &remap);
    }

    // Rewrite param/capture registers.
    for (_, v) in &mut cfg.params {
        *v = remap(*v);
    }
    for (_, v) in &mut cfg.captures {
        *v = remap(*v);
    }

    // Migrate types — colored values only.
    // After DCE, all values referenced by instructions are colored.
    // Uncolored values are dead — their types are discarded.
    let old_types = std::mem::take(&mut cfg.val_types);
    for (vid, ty) in old_types {
        if coloring.is_colored(&vid) {
            cfg.val_types.insert(remap(vid), ty);
        }
    }

    // Reconstruct debug info from the final CFG — no remap needed.
    // This is authoritative: each ValueId's origin is determined by the
    // instruction that defines it in the final IR.
    cfg.debug = reconstruct_debug(cfg);

    cfg.val_factory = new_factory;
}

/// Reconstruct DebugInfo by scanning the final CFG instructions.
/// Each ValueId gets its origin from the instruction that defines it.
fn reconstruct_debug(cfg: &CfgBody) -> crate::ir::DebugInfo {
    use crate::ir::{DebugInfo, RefTarget, ValOrigin};

    let mut debug = DebugInfo::new();

    // Build slot → name lookup from params and captures.
    let mut slot_name: FxHashMap<ValueId, (acvus_utils::Astr, bool)> = FxHashMap::default();
    for (name, reg) in &cfg.params {
        slot_name.insert(*reg, (*name, true)); // true = param
    }
    for (name, reg) in &cfg.captures {
        slot_name.insert(*reg, (*name, false)); // false = capture (local var)
    }
    // Also seed debug info for slot ValueIds themselves (for label lookups).
    for (&slot, &(name, is_param)) in &slot_name {
        if is_param {
            debug.set(slot, ValOrigin::ExternParam(name));
        } else {
            debug.set(slot, ValOrigin::Named(name));
        }
    }

    for block in &cfg.blocks {
        // Block params: no specific origin (SSA phi results).
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Ref { dst, target, path } => {
                    let origin = if !path.is_empty() {
                        ValOrigin::RefField(target.clone(), path.clone())
                    } else {
                        match target {
                            RefTarget::Context(qref) => ValOrigin::Context(qref.name),
                            RefTarget::Var(slot) => {
                                if let Some(&(name, _)) = slot_name.get(slot) {
                                    ValOrigin::Named(name)
                                } else if let Some(origin) = debug.get(*slot) {
                                    origin.clone()
                                } else {
                                    ValOrigin::Expr
                                }
                            }
                            RefTarget::Param(slot) => {
                                if let Some(&(name, _)) = slot_name.get(slot) {
                                    ValOrigin::ExternParam(name)
                                } else if let Some(origin) = debug.get(*slot) {
                                    origin.clone()
                                } else {
                                    ValOrigin::Expr
                                }
                            }
                        }
                    };
                    debug.set(*dst, origin);
                }
                InstKind::Load { dst, src, .. } => {
                    // Inherit origin from the Ref it loads from.
                    if let Some(origin) = debug.get(*src) {
                        debug.set(*dst, origin.clone());
                    }
                }
                InstKind::FieldGet { dst, object, field, .. } => {
                    debug.set(*dst, ValOrigin::Field(*object, *field));
                }
                InstKind::FunctionCall { dst, callee, .. } => {
                    let name = match callee {
                        Callee::Direct(qref) => qref.name,
                        Callee::Indirect(vid) => {
                            // For indirect calls, try to get the name from the closure's origin.
                            if let Some(ValOrigin::Named(n)) = debug.get(*vid) {
                                *n
                            } else {
                                continue;
                            }
                        }
                    };
                    debug.set(*dst, ValOrigin::Call(name));
                }
                // Everything else that defines a dst: generic Expr origin.
                other => {
                    // Extract dst from remaining instruction kinds.
                    if let Some(dst) = inst_def_dst(other) {
                        if debug.get(dst).is_none() {
                            debug.set(dst, ValOrigin::Expr);
                        }
                    }
                }
            }
        }
    }

    debug
}

/// Extract the primary dst ValueId from an instruction, if it has one.
fn inst_def_dst(kind: &InstKind) -> Option<ValueId> {
    let defs = crate::analysis::inst_info::defs(kind);
    defs.first().copied()
}

fn rewrite_inst(kind: &mut InstKind, remap: &impl Fn(ValueId) -> ValueId) {
    let r = |v: &mut ValueId| *v = remap(*v);
    match kind {
        InstKind::Const { dst, .. } => r(dst),
        InstKind::Ref { dst, target, .. } => {
            r(dst);
            match target {
                crate::ir::RefTarget::Var(slot) | crate::ir::RefTarget::Param(slot) => r(slot),
                crate::ir::RefTarget::Context(_) => {}
            }
        }
        InstKind::Load { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::Store { dst, value, .. } => {
            r(dst);
            r(value);
        }
        InstKind::BinOp {
            dst, left, right, ..
        } => {
            r(dst);
            r(left);
            r(right);
        }
        InstKind::UnaryOp { dst, operand, .. } => {
            r(dst);
            r(operand);
        }
        InstKind::FieldGet { dst, object, .. } => {
            r(dst);
            r(object);
        }
        InstKind::FieldSet {
            dst, object, value, ..
        } => {
            r(dst);
            r(object);
            r(value);
        }
        InstKind::LoadFunction { dst, .. } => r(dst),
        InstKind::FunctionCall {
            dst,
            callee,
            args,
            context_uses,
            context_defs,
        } => {
            r(dst);
            if let Callee::Indirect(v) = callee {
                r(v);
            }
            args.iter_mut().for_each(&r);
            context_uses.iter_mut().for_each(|(_, v)| r(v));
            context_defs.iter_mut().for_each(|(_, v)| r(v));
        }
        InstKind::Spawn {
            dst,
            callee,
            args,
            context_uses,
        } => {
            r(dst);
            if let Callee::Indirect(v) = callee {
                r(v);
            }
            args.iter_mut().for_each(&r);
            context_uses.iter_mut().for_each(|(_, v)| r(v));
        }
        InstKind::Eval {
            dst,
            src,
            context_defs,
        } => {
            r(dst);
            r(src);
            context_defs.iter_mut().for_each(|(_, v)| r(v));
        }
        InstKind::MakeDeque { dst, elements } => {
            r(dst);
            elements.iter_mut().for_each(&r);
        }
        InstKind::MakeObject { dst, fields } => {
            r(dst);
            fields.iter_mut().for_each(|(_, v)| r(v));
        }
        InstKind::MakeRange {
            dst, start, end, ..
        } => {
            r(dst);
            r(start);
            r(end);
        }
        InstKind::MakeTuple { dst, elements } => {
            r(dst);
            elements.iter_mut().for_each(&r);
        }
        InstKind::TupleIndex { dst, tuple, .. } => {
            r(dst);
            r(tuple);
        }
        InstKind::TestLiteral { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::TestListLen { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::TestObjectKey { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::TestRange { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::ListIndex { dst, list, .. } => {
            r(dst);
            r(list);
        }
        InstKind::ListGet { dst, list, index } => {
            r(dst);
            r(list);
            r(index);
        }
        InstKind::ListSlice { dst, list, .. } => {
            r(dst);
            r(list);
        }
        InstKind::ObjectGet { dst, object, .. } => {
            r(dst);
            r(object);
        }
        InstKind::MakeClosure { dst, captures, .. } => {
            r(dst);
            captures.iter_mut().for_each(&r);
        }
        InstKind::MakeVariant { dst, payload, .. } => {
            r(dst);
            if let Some(v) = payload {
                r(v);
            }
        }
        InstKind::TestVariant { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::UnwrapVariant { dst, src } => {
            r(dst);
            r(src);
        }
        InstKind::Cast { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::Poison { dst, .. } => r(dst),
        InstKind::Undef { dst } => r(dst),
        InstKind::Nop => {}
        InstKind::BlockLabel { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::ListStep { .. }
        | InstKind::Return(..) => {
            unreachable!("CF instructions must not appear in CfgBody block.insts")
        }
    }
}

fn rewrite_terminator(term: &mut Terminator, remap: &impl Fn(ValueId) -> ValueId) {
    let r = |v: &mut ValueId| *v = remap(*v);
    match term {
        Terminator::Jump { args, .. } => args.iter_mut().for_each(&r),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            r(cond);
            then_args.iter_mut().for_each(&r);
            else_args.iter_mut().for_each(&r);
        }
        Terminator::ListStep {
            dst,
            list,
            index_src,
            index_dst,
            done_args,
            ..
        } => {
            r(dst);
            r(list);
            r(index_src);
            r(index_dst);
            done_args.iter_mut().for_each(&r);
        }
        Terminator::Return(v) => r(v),
        Terminator::Fallthrough => {}
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::ir::*;
    use crate::ty::Ty;
    use acvus_utils::{Astr, Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn dummy_name() -> Astr {
        thread_local! {
            static INTERNER: Interner = Interner::new();
        }
        INTERNER.with(|i| i.intern("_"))
    }

    fn make_body(insts: Vec<InstKind>) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..20 {
            factory.next();
        }
        MirBody {
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
        }
    }

    fn make_cfg(insts: Vec<InstKind>) -> CfgBody {
        cfg::promote(make_body(insts))
    }

    fn make_cfg_with(insts: Vec<InstKind>, f: impl FnOnce(&mut MirBody)) -> CfgBody {
        let mut body = make_body(insts);
        f(&mut body);
        cfg::promote(body)
    }

    fn collect_defs(body: &MirBody) -> FxHashSet<ValueId> {
        let mut set = FxHashSet::default();
        for inst in &body.insts {
            for d in inst_info::defs(&inst.kind) {
                set.insert(d);
            }
        }
        set
    }

    // ── Soundness: interfering values must NOT share slots ──────────

    #[test]
    fn overlapping_values_get_distinct_slots() {
        let mut cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::BinOp {
                dst: v(2),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(1),
            },
            InstKind::Return(v(2)),
        ]);
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let binop = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::BinOp { .. }))
            .unwrap();
        if let InstKind::BinOp { left, right, .. } = &binop.kind {
            assert_ne!(left, right);
        }
    }

    #[test]
    fn cross_block_value_not_clobbered() {
        let mut cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::BinOp {
                dst: v(2),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(1),
            },
            InstKind::Return(v(2)),
        ]);
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let binop = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::BinOp { .. }))
            .unwrap();
        if let InstKind::BinOp { left, right, .. } = &binop.kind {
            assert_ne!(left, right);
        }
    }

    #[test]
    fn loop_param_not_clobbered_by_body() {
        let mut cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![v(0)],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(1)],
                merge_of: None,
            },
            InstKind::BinOp {
                dst: v(2),
                op: acvus_ast::BinOp::Add,
                left: v(1),
                right: v(1),
            },
            InstKind::Const {
                dst: v(3),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(3),
                then_label: Label(0),
                then_args: vec![v(2)],
                else_label: Label(1),
                else_args: vec![v(2)],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![v(4)],
                merge_of: None,
            },
            InstKind::Return(v(4)),
        ]);
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let binop = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::BinOp { .. }))
            .unwrap();
        if let InstKind::BinOp {
            left, right, dst, ..
        } = &binop.kind
        {
            assert_eq!(left, right, "loop param used as both operands");
            assert_ne!(left, dst, "result must differ from operand");
        }
    }

    #[test]
    fn different_types_never_share() {
        let mut cfg = make_cfg_with(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Const {
                    dst: v(2),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::Return(v(2)),
            ],
            |body| {
                body.val_types.insert(v(0), Ty::Int);
                body.val_types.insert(v(1), Ty::Int);
                body.val_types.insert(v(2), Ty::Bool);
            },
        );
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let consts: Vec<_> = body
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::Const { dst, .. } => Some(*dst),
                _ => None,
            })
            .collect();
        assert_ne!(consts[0], consts[1]);
    }

    #[test]
    fn param_regs_remain_valid() {
        let mut cfg = make_cfg_with(
            vec![
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return(v(1)),
            ],
            |body| {
                body.params = vec![(dummy_name(), v(0))];
            },
        );
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let binop = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::BinOp { .. }))
            .unwrap();
        if let InstKind::BinOp { left, .. } = &binop.kind {
            assert_eq!(*left, body.param_regs()[0]);
        }
    }

    #[test]
    fn capture_regs_remain_valid() {
        let mut cfg = make_cfg_with(
            vec![
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return(v(1)),
            ],
            |body| {
                body.captures = vec![(dummy_name(), v(0))];
            },
        );
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let binop = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::BinOp { .. }))
            .unwrap();
        if let InstKind::BinOp { left, .. } = &binop.kind {
            assert_eq!(*left, body.capture_regs()[0]);
        }
    }

    #[test]
    fn unused_param_still_has_slot() {
        let mut cfg = make_cfg_with(
            vec![
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(42),
                },
                InstKind::Return(v(1)),
            ],
            |body| {
                body.params = vec![(dummy_name(), v(0))];
            },
        );
        color_body(&mut cfg);
        assert!(cfg.params[0].1.to_raw() < 20);
    }

    #[test]
    fn branch_both_arms_correct() {
        let mut cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(1),
                then_label: Label(0),
                then_args: vec![v(0)],
                else_label: Label(1),
                else_args: vec![v(0)],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(2)],
                merge_of: None,
            },
            InstKind::Return(v(2)),
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![v(3)],
                merge_of: None,
            },
            InstKind::Return(v(3)),
        ]);
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let ji = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::JumpIf { .. }))
            .unwrap();
        if let InstKind::JumpIf {
            then_args,
            else_args,
            ..
        } = &ji.kind
        {
            assert_eq!(then_args[0], else_args[0]);
        }
    }

    // ── Completeness: non-interfering values SHOULD share ──────────

    #[test]
    fn non_overlapping_same_type_share_slot() {
        let original_body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::BinOp {
                dst: v(1),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(0),
            },
            InstKind::Const {
                dst: v(2),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::BinOp {
                dst: v(3),
                op: acvus_ast::BinOp::Add,
                left: v(2),
                right: v(2),
            },
            InstKind::BinOp {
                dst: v(4),
                op: acvus_ast::BinOp::Add,
                left: v(1),
                right: v(3),
            },
            InstKind::Return(v(4)),
        ]);
        let original_defs = collect_defs(&original_body);
        let mut cfg = cfg::promote(original_body);
        color_body(&mut cfg);
        let colored_defs = collect_defs(&cfg::demote(cfg));
        assert!(colored_defs.len() < original_defs.len());
    }

    #[test]
    fn dead_value_slot_reclaimed() {
        let mut cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::Return(v(1)),
        ]);
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let consts: Vec<_> = body
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::Const { dst, .. } => Some(*dst),
                _ => None,
            })
            .collect();
        assert_eq!(consts[0], consts[1], "dead value slot should be reused");
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn empty_body_no_panic() {
        let mut cfg = CfgBody {
            blocks: vec![],
            label_to_block: FxHashMap::default(),
            val_types: FxHashMap::default(),
            params: vec![],
            captures: vec![],
            debug: DebugInfo::new(),
            val_factory: LocalFactory::<ValueId>::new(),
        };
        color_body(&mut cfg);
    }

    #[test]
    fn single_return_no_panic() {
        let mut cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Return(v(0)),
        ]);
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        if let (InstKind::Const { dst, .. }, InstKind::Return(val)) =
            (&body.insts[0].kind, &body.insts[1].kind)
        {
            assert_eq!(val, dst);
        }
    }

    #[test]
    fn val_types_remapped_correctly() {
        let mut cfg = make_cfg_with(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return(v(1)),
            ],
            |body| {
                body.val_types.insert(v(0), Ty::Int);
                body.val_types.insert(v(1), Ty::Int);
            },
        );
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let defs = collect_defs(&body);
        for vid in body.val_types.keys() {
            assert!(
                defs.contains(vid)
                    || body.param_regs().contains(vid)
                    || body.capture_regs().contains(vid)
            );
        }
    }

    #[test]
    fn dead_val_types_do_not_overwrite_live_types() {
        let mut cfg = make_cfg_with(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::String("x".into()),
                },
                InstKind::Const {
                    dst: v(2),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![v(2)],
                    else_label: Label(1),
                    else_args: vec![v(2)],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Return(v(3)),
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(4)],
                    merge_of: None,
                },
                InstKind::Return(v(4)),
            ],
            |body| {
                body.val_types.insert(v(0), Ty::Bool);
                body.val_types.insert(v(1), Ty::String);
                body.val_types.insert(v(2), Ty::Int);
                body.val_types.insert(v(3), Ty::Int);
                body.val_types.insert(v(4), Ty::Int);
            },
        );
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let ji = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::JumpIf { .. }))
            .unwrap();
        if let InstKind::JumpIf { cond, .. } = &ji.kind {
            assert_eq!(body.val_types.get(cond), Some(&Ty::Bool));
        }
    }

    #[test]
    fn empty_block_param_does_not_alias_early_value() {
        let mut cfg = make_cfg_with(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::String("hello".into()),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(1),
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
                    args: vec![v(0)],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(0)],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![v(2)],
                    merge_of: None,
                },
                InstKind::BinOp {
                    dst: v(3),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(2),
                },
                InstKind::Return(v(3)),
            ],
            |body| {
                body.val_types.insert(v(0), Ty::String);
                body.val_types.insert(v(1), Ty::Bool);
                body.val_types.insert(v(2), Ty::String);
                body.val_types.insert(v(3), Ty::String);
            },
        );
        color_body(&mut cfg);
        let body = cfg::demote(cfg);
        let binop = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::BinOp { .. }))
            .unwrap();
        if let InstKind::BinOp { left, right, .. } = &binop.kind {
            assert_ne!(left, right);
        }
    }

    #[test]
    fn val_factory_reflects_compacted_count() {
        let mut cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::BinOp {
                dst: v(1),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(0),
            },
            InstKind::Const {
                dst: v(2),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::BinOp {
                dst: v(3),
                op: acvus_ast::BinOp::Add,
                left: v(2),
                right: v(2),
            },
            InstKind::BinOp {
                dst: v(4),
                op: acvus_ast::BinOp::Add,
                left: v(1),
                right: v(3),
            },
            InstKind::Return(v(4)),
        ]);
        color_body(&mut cfg);
        let next = cfg.val_factory.next();
        assert!(next.to_raw() < 5);
    }
}
