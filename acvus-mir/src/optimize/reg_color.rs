//! Register coloring — compact ValueId allocation by reusing dead slots.
//!
//! Uses CFG-aware liveness analysis to compute accurate liveness intervals,
//! then performs a linear scan to assign the minimum number of physical slots.
//! Rewrites all ValueIds in the MIR body in-place.

use crate::analysis::cfg::Cfg;
use crate::analysis::liveness;
use crate::ir::{Callee, InstKind, MirBody, MirModule, ValueId};
use acvus_utils::LocalFactory;
use rustc_hash::FxHashMap;

/// Run register coloring on all bodies in a module.
pub fn color(mut module: MirModule) -> MirModule {
    color_body(&mut module.main);
    for closure in module.closures.values_mut() {
        color_body(closure);
    }
    module
}

fn color_body(body: &mut MirBody) {
    if body.insts.is_empty() {
        return;
    }

    // Step 1: CFG-aware liveness analysis → accurate intervals.
    let liveness = liveness::analyze(body);
    let cfg = Cfg::build(&body.insts);
    let mut intervals = liveness.intervals(&cfg, &body.insts);

    // Ensure param_regs and capture_regs have intervals even if unused.
    // The caller writes into these slots, so they must exist in the remap.
    let has_interval: rustc_hash::FxHashSet<ValueId> =
        intervals.iter().map(|&(v, _, _)| v).collect();
    for &v in body.param_regs.iter().chain(body.capture_regs.iter()) {
        if !has_interval.contains(&v) {
            intervals.push((v, 0, 0));
        }
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
    let old_count = intervals.len();
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
        InstKind::ParamLoad { dst, .. } => r(dst),
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
            for (_, v) in context_uses {
                r(v);
            }
            for (_, v) in context_defs {
                r(v);
            }
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
        InstKind::ListStep {
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
        InstKind::Undef { dst } => r(dst),
        InstKind::Nop => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;
    use crate::ty::Ty;
    use acvus_utils::{LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
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
            param_regs: Vec::new(),
            capture_regs: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        }
    }

    /// Collect all ValueIds that appear as defs in the body after coloring.
    fn collect_defs(body: &MirBody) -> rustc_hash::FxHashSet<ValueId> {
        use crate::analysis::inst_info;
        let mut set = rustc_hash::FxHashSet::default();
        for inst in &body.insts {
            for d in inst_info::defs(&inst.kind) {
                set.insert(d);
            }
        }
        set
    }

    // ── Soundness: overlapping intervals must NOT share slots ────────

    #[test]
    fn overlapping_values_get_distinct_slots() {
        // r0 = const 1; r1 = const 2; r2 = r0 + r1; return r2
        // r0 and r1 are both live at instruction 2, must not share.
        let mut body = make_body(vec![
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

        color_body(&mut body);

        // After coloring, the BinOp's left and right must be different.
        let binop = body.insts.iter().find(|i| matches!(i.kind, InstKind::BinOp { .. })).unwrap();
        if let InstKind::BinOp { left, right, .. } = &binop.kind {
            assert_ne!(left, right, "overlapping values must have distinct slots");
        }
    }

    #[test]
    fn cross_block_value_not_clobbered() {
        // Block 0: r0 = const 1; jump L0()
        // Block 1 (L0): r1 = const 2; r2 = r0 + r1; return r2
        // r0 is live across blocks — must not share slot with r1.
        let mut body = make_body(vec![
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

        color_body(&mut body);

        // r0 and r1 are both live at the BinOp — must be distinct.
        let binop = body.insts.iter().find(|i| matches!(i.kind, InstKind::BinOp { .. })).unwrap();
        if let InstKind::BinOp { left, right, .. } = &binop.kind {
            assert_ne!(left, right, "cross-block live value must not be clobbered");
        }
    }

    #[test]
    fn loop_param_not_clobbered_by_body() {
        // Block 0: r0 = const 0; jump L0(r0)
        // Block 1 (L0, params=[r1]):
        //   r2 = r1 + r1
        //   r3 = const true
        //   jump_if r3 then L0(r2) else L1(r2)
        // Block 2 (L1, params=[r4]): return r4
        let mut body = make_body(vec![
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

        color_body(&mut body);

        // The BinOp in the loop uses r1 (loop param) — it must be valid.
        let binop = body.insts.iter().find(|i| matches!(i.kind, InstKind::BinOp { .. })).unwrap();
        if let InstKind::BinOp { left, right, dst, .. } = &binop.kind {
            assert_eq!(left, right, "loop param used as both operands");
            assert_ne!(left, dst, "result must differ from operand");
        }
    }

    #[test]
    fn different_types_never_share() {
        // r0 = const 1 (Int); r1 = r0 + r0; r2 = const true (Bool); return r2
        // r0 is dead after r1. r2 is Bool. Even though intervals don't overlap,
        // Int and Bool must not share.
        let mut body = make_body(vec![
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
        ]);
        body.val_types.insert(v(0), Ty::Int);
        body.val_types.insert(v(1), Ty::Int);
        body.val_types.insert(v(2), Ty::Bool);

        color_body(&mut body);

        // r2 (Bool) must not share with r0 (Int) even though r0 is dead.
        let consts: Vec<_> = body
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::Const { dst, .. } => Some(*dst),
                _ => None,
            })
            .collect();
        assert_eq!(consts.len(), 2);
        assert_ne!(consts[0], consts[1], "different types must not share slots");
    }

    #[test]
    fn param_regs_remain_valid() {
        // param r0; r1 = r0 + r0; return r1
        let mut body = make_body(vec![
            InstKind::BinOp {
                dst: v(1),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(0),
            },
            InstKind::Return(v(1)),
        ]);
        body.param_regs = vec![v(0)];

        color_body(&mut body);

        // param_regs[0] must reference a ValueId that exists in the body.
        let defs = collect_defs(&body);
        // param_regs aren't "defs" in instructions — they're implicit.
        // But the BinOp must use the same ValueId as param_regs[0].
        let binop = body.insts.iter().find(|i| matches!(i.kind, InstKind::BinOp { .. })).unwrap();
        if let InstKind::BinOp { left, .. } = &binop.kind {
            assert_eq!(
                *left, body.param_regs[0],
                "BinOp must reference the remapped param"
            );
        }
    }

    #[test]
    fn capture_regs_remain_valid() {
        // capture r0; r1 = r0 + r0; return r1
        let mut body = make_body(vec![
            InstKind::BinOp {
                dst: v(1),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(0),
            },
            InstKind::Return(v(1)),
        ]);
        body.capture_regs = vec![v(0)];

        color_body(&mut body);

        let binop = body.insts.iter().find(|i| matches!(i.kind, InstKind::BinOp { .. })).unwrap();
        if let InstKind::BinOp { left, .. } = &binop.kind {
            assert_eq!(
                *left, body.capture_regs[0],
                "BinOp must reference the remapped capture"
            );
        }
    }

    #[test]
    fn unused_param_still_has_slot() {
        // param r0; r1 = const 42; return r1
        // r0 is unused but must still get a valid slot.
        let mut body = make_body(vec![
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(42),
            },
            InstKind::Return(v(1)),
        ]);
        body.param_regs = vec![v(0)];

        color_body(&mut body);

        // param_regs[0] must be a valid ValueId (from the factory range).
        assert!(
            body.param_regs[0].to_raw() < 20,
            "unused param must still have a valid slot"
        );
    }

    #[test]
    fn branch_both_arms_correct() {
        // r0 = const 1; r1 = const true
        // jump_if r1 then L0(r0) else L1(r0)
        // L0(r2): return r2
        // L1(r3): return r3
        let mut body = make_body(vec![
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

        color_body(&mut body);

        // Both branch arms should receive the value from r0 via jump args.
        // The JumpIf's then_args and else_args should reference the same slot as r0.
        let jump_if = body.insts.iter().find(|i| matches!(i.kind, InstKind::JumpIf { .. })).unwrap();
        if let InstKind::JumpIf { then_args, else_args, .. } = &jump_if.kind {
            assert_eq!(then_args.len(), 1);
            assert_eq!(else_args.len(), 1);
            // Both branches receive the same value.
            assert_eq!(then_args[0], else_args[0]);
        }
    }

    // ── Completeness: non-overlapping intervals SHOULD share ────────

    #[test]
    fn non_overlapping_same_type_share_slot() {
        // r0 = const 1; r1 = r0 + r0; r2 = const 2; r3 = r2 + r2; r4 = r1 + r3; return r4
        // r0 dead after inst 1, r2 starts at inst 2 — same type → share.
        let mut body = make_body(vec![
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

        let original_defs = collect_defs(&body);
        color_body(&mut body);
        let colored_defs = collect_defs(&body);

        // Must have fewer unique ValueIds after coloring.
        assert!(
            colored_defs.len() < original_defs.len(),
            "non-overlapping values should be compacted: {} -> {}",
            original_defs.len(),
            colored_defs.len()
        );
    }

    #[test]
    fn dead_value_slot_reclaimed() {
        // r0 = const 1; r1 = const 2; return r1
        // r0 is completely dead — its slot should be reusable.
        let mut body = make_body(vec![
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

        color_body(&mut body);

        // With dead value reclaimed, we should have fewer slots.
        // r0 is dead, r1 is live → could share the same slot.
        let consts: Vec<_> = body
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::Const { dst, .. } => Some(*dst),
                _ => None,
            })
            .collect();
        // r0 (dead) and r1 can share since r0 interval = [0,0], r1 starts at 1.
        assert_eq!(consts[0], consts[1], "dead value slot should be reused");
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn empty_body_no_panic() {
        let mut body = make_body(vec![]);
        color_body(&mut body);
        assert!(body.insts.is_empty());
    }

    #[test]
    fn single_return_no_panic() {
        let mut body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Return(v(0)),
        ]);
        color_body(&mut body);

        // Should still work — single value, no sharing possible.
        let ret = body.insts.last().unwrap();
        if let InstKind::Return(val) = &ret.kind {
            let c = &body.insts[0];
            if let InstKind::Const { dst, .. } = &c.kind {
                assert_eq!(val, dst);
            }
        }
    }

    #[test]
    fn val_types_remapped_correctly() {
        // r0 = const 1 (Int); r1 = r0 + r0 (Int); return r1
        let mut body = make_body(vec![
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
        ]);
        body.val_types.insert(v(0), Ty::Int);
        body.val_types.insert(v(1), Ty::Int);

        color_body(&mut body);

        // All val_types keys must reference ValueIds that exist in the colored body.
        let defs = collect_defs(&body);
        for (vid, _) in &body.val_types {
            // val_types may include params/captures too, but for this test
            // all typed values should be reachable as defs.
            assert!(
                defs.contains(vid) || body.param_regs.contains(vid) || body.capture_regs.contains(vid),
                "val_types key {:?} not found in colored body",
                vid
            );
        }
    }

    #[test]
    fn val_factory_reflects_compacted_count() {
        // r0 = const 1; r1 = r0+r0; r2 = const 2; r3 = r2+r2; r4 = r1+r3; return r4
        // 5 original values → should compact.
        let mut body = make_body(vec![
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

        color_body(&mut body);

        // val_factory.next() should return the next ID after the compacted count.
        let next = body.val_factory.next();
        assert!(
            next.to_raw() < 5,
            "factory should reflect compacted count, got {:?}",
            next
        );
    }
}
