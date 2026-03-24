//! Register coloring — compact ValueId allocation by reusing dead slots.
//!
//! Computes liveness intervals for each ValueId, then performs a linear scan
//! to assign the minimum number of physical slots. Rewrites all ValueIds
//! in the MIR body in-place.

use crate::ir::{Callee, InstKind, MirBody, MirModule, ValueId};
use acvus_utils::LocalFactory;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Run register coloring on all bodies in a module.
pub fn color(mut module: MirModule) -> MirModule {
    color_body(&mut module.main);
    for closure in module.closures.values_mut() {
        color_body(Arc::make_mut(closure));
    }
    module
}

fn color_body(body: &mut MirBody) {
    if body.insts.is_empty() {
        return;
    }

    // Step 1: Compute liveness intervals (def_pos, last_use_pos) for each ValueId.
    let mut def_pos: FxHashMap<ValueId, usize> = FxHashMap::default();
    let mut last_use: FxHashMap<ValueId, usize> = FxHashMap::default();

    // Params and captures are defined at position 0.
    for &v in &body.param_regs {
        def_pos.entry(v).or_insert(0);
    }
    for &v in &body.capture_regs {
        def_pos.entry(v).or_insert(0);
    }

    for (i, inst) in body.insts.iter().enumerate() {
        for_each_def(&inst.kind, |v| {
            def_pos.entry(v).or_insert(i);
        });
        for_each_use(&inst.kind, |v| {
            last_use.insert(v, i);
        });
    }

    // Step 2: Build intervals sorted by def position.
    let mut intervals: Vec<(ValueId, usize, usize)> = Vec::new();
    for (&vid, &def) in &def_pos {
        let end = last_use.get(&vid).copied().unwrap_or(def);
        intervals.push((vid, def, end));
    }
    intervals.sort_by_key(|&(_, def, _)| def);

    // Step 3: Linear scan — greedily assign slots.
    // Each slot tracks when it becomes free and what type it holds.
    // Only values of the same type can share a slot (val_types is keyed by ValueId).
    let mut slot_free_at: Vec<usize> = Vec::new();
    let mut slot_type: Vec<Option<crate::ty::Ty>> = Vec::new();
    let mut vid_to_slot: FxHashMap<ValueId, u32> = FxHashMap::default();

    for &(vid, def, end) in &intervals {
        let vid_ty = body.val_types.get(&vid);

        // Find a free slot with matching type.
        let mut assigned = None;
        for (slot, free_at) in slot_free_at.iter_mut().enumerate() {
            if *free_at <= def {
                // Type must match for slot reuse.
                let type_matches = match (&slot_type[slot], vid_ty) {
                    (Some(a), Some(b)) => a == b,
                    (None, None) => true,
                    _ => false,
                };
                if type_matches {
                    *free_at = end + 1;
                    assigned = Some(slot as u32);
                    break;
                }
            }
        }
        let slot = match assigned {
            Some(s) => s,
            None => {
                let s = slot_free_at.len() as u32;
                slot_free_at.push(end + 1);
                slot_type.push(vid_ty.cloned());
                s
            }
        };
        vid_to_slot.insert(vid, slot);
    }

    let num_slots = slot_free_at.len();
    let old_count = def_pos.len();
    if num_slots >= old_count {
        return; // No improvement.
    }

    // Step 4: Build new ValueIds (slot-based).
    let mut new_factory = LocalFactory::<ValueId>::new();
    let mut slot_to_new: Vec<ValueId> = Vec::with_capacity(num_slots);
    for _ in 0..num_slots {
        slot_to_new.push(new_factory.next());
    }

    let remap = |v: ValueId| -> ValueId {
        vid_to_slot
            .get(&v)
            .map(|&slot| slot_to_new[slot as usize])
            .unwrap_or(v)
    };

    // Step 5: Rewrite all instructions.
    for inst in &mut body.insts {
        rewrite_inst(&mut inst.kind, &remap);
    }

    // Rewrite param_regs and capture_regs.
    for v in &mut body.param_regs {
        *v = remap(*v);
    }
    for v in &mut body.capture_regs {
        *v = remap(*v);
    }

    // Rewrite val_types.
    let old_types = std::mem::take(&mut body.val_types);
    for (vid, ty) in old_types {
        body.val_types.insert(remap(vid), ty);
    }

    // Update val_factory.
    body.val_factory = new_factory;
}

// ── Helpers: extract defs and uses from an instruction ──────────────

fn for_each_def(kind: &InstKind, mut f: impl FnMut(ValueId)) {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::ContextProject { dst, .. }
        | InstKind::ContextLoad { dst, .. }
        | InstKind::VarLoad { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::LoadFunction { dst, .. }
        | InstKind::FunctionCall { dst, .. }
        | InstKind::Spawn { dst, .. }
        | InstKind::MakeDeque { dst, .. }
        | InstKind::MakeObject { dst, .. }
        | InstKind::MakeRange { dst, .. }
        | InstKind::MakeTuple { dst, .. }
        | InstKind::TupleIndex { dst, .. }
        | InstKind::TestLiteral { dst, .. }
        | InstKind::TestListLen { dst, .. }
        | InstKind::TestObjectKey { dst, .. }
        | InstKind::TestRange { dst, .. }
        | InstKind::ListIndex { dst, .. }
        | InstKind::ListGet { dst, .. }
        | InstKind::ListSlice { dst, .. }
        | InstKind::ObjectGet { dst, .. }
        | InstKind::MakeClosure { dst, .. }
        | InstKind::MakeVariant { dst, .. }
        | InstKind::TestVariant { dst, .. }
        | InstKind::UnwrapVariant { dst, .. }
        | InstKind::Cast { dst, .. }
        | InstKind::Poison { dst, .. } => f(*dst),

        InstKind::IterStep { dst, iter_dst, .. } => {
            f(*dst);
            f(*iter_dst);
        }
        InstKind::BlockLabel { params, .. } => {
            for &p in params {
                f(p);
            }
        }
        InstKind::Eval {
            dst, context_defs, ..
        } => {
            f(*dst);
            for &(_, v) in context_defs {
                f(v);
            }
        }

        InstKind::ContextStore { .. }
        | InstKind::VarStore { .. }
        | InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Return(_)
        | InstKind::Nop => {}
    }
}

fn for_each_use(kind: &InstKind, mut f: impl FnMut(ValueId)) {
    match kind {
        InstKind::Const { .. }
        | InstKind::ContextProject { .. }
        | InstKind::VarLoad { .. }
        | InstKind::LoadFunction { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Nop
        | InstKind::Poison { .. } => {}

        InstKind::ContextLoad { src, .. } => f(*src),
        InstKind::ContextStore { dst, value } => {
            f(*dst);
            f(*value);
        }
        InstKind::VarStore { src, .. } => f(*src),
        InstKind::BinOp { left, right, .. } => {
            f(*left);
            f(*right);
        }
        InstKind::UnaryOp { operand, .. } => f(*operand),
        InstKind::FieldGet { object, .. } => f(*object),
        InstKind::FunctionCall { callee, args, .. } => {
            if let Callee::Indirect(v) = callee {
                f(*v);
            }
            args.iter().for_each(|v| f(*v));
        }
        InstKind::Spawn {
            callee,
            args,
            context_uses,
            ..
        } => {
            if let Callee::Indirect(v) = callee {
                f(*v);
            }
            args.iter().for_each(|v| f(*v));
            context_uses.iter().for_each(|(_, v)| f(*v));
        }
        InstKind::Eval { src, .. } => f(*src),
        InstKind::MakeDeque { elements, .. } => elements.iter().for_each(|v| f(*v)),
        InstKind::MakeObject { fields, .. } => fields.iter().for_each(|(_, v)| f(*v)),
        InstKind::MakeRange { start, end, .. } => {
            f(*start);
            f(*end);
        }
        InstKind::MakeTuple { elements, .. } => elements.iter().for_each(|v| f(*v)),
        InstKind::TupleIndex { tuple, .. } => f(*tuple),
        InstKind::TestLiteral { src, .. }
        | InstKind::TestListLen { src, .. }
        | InstKind::TestObjectKey { src, .. }
        | InstKind::TestRange { src, .. } => f(*src),
        InstKind::ListIndex { list, .. } => f(*list),
        InstKind::ListGet { list, index, .. } => {
            f(*list);
            f(*index);
        }
        InstKind::ListSlice { list, .. } => f(*list),
        InstKind::ObjectGet { object, .. } => f(*object),
        InstKind::MakeClosure { captures, .. } => captures.iter().for_each(|v| f(*v)),
        InstKind::IterStep {
            iter_src,
            done_args,
            ..
        } => {
            f(*iter_src);
            done_args.iter().for_each(|v| f(*v));
        }
        InstKind::MakeVariant { payload, .. } => {
            if let Some(v) = payload {
                f(*v);
            }
        }
        InstKind::TestVariant { src, .. } | InstKind::UnwrapVariant { src, .. } => f(*src),
        InstKind::Cast { src, .. } => f(*src),
        InstKind::Jump { args, .. } => args.iter().for_each(|v| f(*v)),
        InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            f(*cond);
            then_args.iter().for_each(|v| f(*v));
            else_args.iter().for_each(|v| f(*v));
        }
        InstKind::Return(v) => f(*v),
    }
}

fn rewrite_inst(kind: &mut InstKind, remap: &impl Fn(ValueId) -> ValueId) {
    let r = |v: &mut ValueId| *v = remap(*v);
    match kind {
        InstKind::Const { dst, .. } => r(dst),
        InstKind::ContextProject { dst, .. } => r(dst),
        InstKind::ContextLoad { dst, src } => {
            r(dst);
            r(src);
        }
        InstKind::ContextStore { dst, value } => {
            r(dst);
            r(value);
        }
        InstKind::VarLoad { dst, .. } => r(dst),
        InstKind::VarStore { src, .. } => r(src),
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
        InstKind::LoadFunction { dst, .. } => r(dst),
        InstKind::FunctionCall { dst, callee, args } => {
            r(dst);
            if let Callee::Indirect(v) = callee {
                r(v);
            }
            args.iter_mut().for_each(&r);
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
        InstKind::IterStep {
            dst,
            iter_src,
            iter_dst,
            done_args,
            ..
        } => {
            r(dst);
            r(iter_src);
            r(iter_dst);
            done_args.iter_mut().for_each(&r);
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
        InstKind::BlockLabel { params, .. } => {
            params.iter_mut().for_each(&r);
        }
        InstKind::Jump { args, .. } => args.iter_mut().for_each(&r),
        InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            r(cond);
            then_args.iter_mut().for_each(&r);
            else_args.iter_mut().for_each(&r);
        }
        InstKind::Return(v) => r(v),
        InstKind::Cast { dst, src, .. } => {
            r(dst);
            r(src);
        }
        InstKind::Poison { dst, .. } => r(dst),
        InstKind::Nop => {}
    }
}
