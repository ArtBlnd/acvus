//! Spawn-split pass: convert IO FunctionCalls into Spawn + Eval pairs.
//!
//! This is a pure IR transformation with no reordering.
//! After this pass, IO calls are expressed as:
//!   handle = Spawn { callee, args }
//!   result = Eval { src: handle }
//!
//! The work starts at Spawn, so the Spawn takes the call's entry `Order`
//! and the Eval yields the `Order` that follows. Reordering is left to a
//! separate pass that can move independent instructions between them.

use crate::cfg::CfgBody;
use crate::ir::*;
use crate::ty::Ty;

/// Split IO FunctionCalls into Spawn + Eval pairs, in-place.
pub fn run(cfg: &mut CfgBody) {
    for block in &mut cfg.blocks {
        let mut new_insts = Vec::with_capacity(block.insts.len() + 4);

        for inst in block.insts.drain(..) {
            match inst.kind {
                InstKind::FunctionCall {
                    dst,
                    callee: ref callee @ (Callee::Direct(_) | Callee::Extern { .. }),
                    ref callee_ty,
                    ref args,
                    order,
                } if is_io_call(callee_ty) => {
                    // Allocate a Handle ValueId.
                    let handle = cfg.val_factory.next();

                    // Register Handle type: Handle<ReturnTy>.
                    if let Ty::Fn { ret, .. } = callee_ty {
                        cfg.val_types.insert(handle, Ty::Handle(ret.clone()));
                    }

                    new_insts.push(Inst {
                        span: inst.span,
                        kind: InstKind::Spawn {
                            dst: handle,
                            callee: callee.clone(),
                            callee_ty: callee_ty.clone(),
                            args: args.clone(),
                            order: order.map(|edge| edge.before),
                        },
                    });
                    new_insts.push(Inst {
                        span: inst.span,
                        kind: InstKind::Eval {
                            dst,
                            src: handle,
                            order: order.map(|edge| edge.after),
                        },
                    });
                }
                // Everything else: pass through.
                _ => {
                    new_insts.push(inst);
                }
            }
        }

        block.insts = new_insts;
    }
}

fn is_io_call(callee_ty: &Ty) -> bool {
    matches!(callee_ty.effect(), Some(e) if !e.is_pure())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::graph::QualifiedRef;
    use crate::ty::Param;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for _ in 0..val_count {
            let vid = factory.next();
            val_types.insert(vid, Ty::I64);
        }
        promote(MirBody {
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
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
            order_param: None,
            task: crate::ty::Task::Sync,
        })
    }

    /// Collect all instructions from all blocks (flattened).
    fn all_insts(cfg: &CfgBody) -> Vec<&Inst> {
        cfg.blocks.iter().flat_map(|b| b.insts.iter()).collect()
    }

    #[test]
    fn split_io_call() {
        let i = Interner::new();
        let fetch_id = QualifiedRef::root(i.intern("fetch"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            fetch_id,
            Ty::Fn {
                params: vec![Param::new(i.intern("id"), Ty::I64)],
                ret: Box::new(Ty::String),
                captures: vec![],

                effect: crate::ty::Effect::OPAQUE.into(),
            },
        );

        let fetch_ty = fn_metadata[&fetch_id].clone();
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(fetch_id),
                    callee_ty: fetch_ty,
                    args: vec![v(0)],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            2,
        );

        run(&mut cfg);

        // Should be: Const, Spawn, Eval in block insts (Return is terminator).
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 3);
        assert!(matches!(insts[1].kind, InstKind::Spawn { .. }));
        assert!(matches!(insts[2].kind, InstKind::Eval { .. }));

        // Verify Spawn dst -> Eval src chain.
        if let (InstKind::Spawn { dst: handle, .. }, InstKind::Eval { src, dst, .. }) =
            (&insts[1].kind, &insts[2].kind)
        {
            assert_eq!(handle, src, "Eval.src must reference Spawn.dst");
            assert_eq!(*dst, v(1), "Eval.dst must be original FunctionCall.dst");
        } else {
            panic!("expected Spawn + Eval");
        }

        // Handle should have Handle type.
        if let InstKind::Spawn { dst: handle, .. } = &insts[1].kind {
            let handle_ty = cfg.val_types.get(handle).unwrap();
            assert!(
                matches!(handle_ty, Ty::Handle(..)),
                "Spawn dst must have Handle type"
            );
        }
    }

    #[test]
    fn pure_call_unchanged() {
        let i = Interner::new();
        let add_id = QualifiedRef::root(i.intern("add"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            add_id,
            Ty::Fn {
                params: vec![
                    Param::new(i.intern("a"), Ty::I64),
                    Param::new(i.intern("b"), Ty::I64),
                ],
                ret: Box::new(Ty::I64),
                captures: vec![],

                effect: crate::ty::Effect::OPAQUE.into(),
            },
        );

        let mut cfg = make_cfg(
            vec![InstKind::FunctionCall {
                dst: v(0),
                callee: Callee::Direct(add_id),
                callee_ty: Ty::error(),
                args: vec![v(1), v(2)],
                order: None,
            }],
            3,
        );

        run(&mut cfg);

        // Pure call should NOT be split.
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 1);
        assert!(matches!(insts[0].kind, InstKind::FunctionCall { .. }));
    }

    #[test]
    fn indirect_call_unchanged() {
        // Indirect calls are never split (no QualifiedRef to look up).
        let mut cfg = make_cfg(
            vec![InstKind::FunctionCall {
                dst: v(0),
                callee: Callee::Indirect(v(1)),
                callee_ty: Ty::error(),
                args: vec![],
                order: None,
            }],
            2,
        );

        run(&mut cfg);

        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 1);
        assert!(matches!(insts[0].kind, InstKind::FunctionCall { .. }));
    }

    #[test]
    fn multiple_io_calls_all_split() {
        let i = Interner::new();
        let fetch_a = QualifiedRef::root(i.intern("fetch_a"));
        let fetch_b = QualifiedRef::root(i.intern("fetch_b"));

        let mut fn_metadata = FxHashMap::default();
        for &fid in &[fetch_a, fetch_b] {
            fn_metadata.insert(
                fid,
                Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::String),
                    captures: vec![],

                    effect: crate::ty::Effect::OPAQUE.into(),
                },
            );
        }

        let io_fn_ty = fn_metadata[&fetch_a].clone();
        let mut cfg = make_cfg(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(fetch_a),
                    callee_ty: io_fn_ty.clone(),
                    args: vec![],
                    order: None,
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(fetch_b),
                    callee_ty: io_fn_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(1),
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            3,
        );

        run(&mut cfg);

        // 2 calls -> 2 Spawn + 2 Eval + BinOp = 5 (Return is terminator)
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 5);

        let spawn_count = insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::Spawn { .. }))
            .count();
        let eval_count = insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::Eval { .. }))
            .count();
        assert_eq!(spawn_count, 2);
        assert_eq!(eval_count, 2);
    }
}
