//! Spawn-split pass: a `FunctionCall` whose effect `runs_apart` becomes a
//! `Spawn` and an `Eval`, whether or not an argument holds a loan
//! (RFC-0046 rule 3, RFC-0079 rule 9). The loan is held in the `Handle`'s
//! in-flight position until the `Eval`, and a run that ends before the
//! `Eval` keeps its storage until the task has finished (the interpreter's
//! `flight`).
//!
//! This pass does not reorder. Moving independent instructions between a
//! `Spawn` and its `Eval` is `optimize::reorder`'s job.

use crate::cfg::CfgBody;
use crate::ir::*;
use crate::ty::{Task, Ty};

/// An `Eval` awaits, so a body this pass splits anything in runs at
/// `Task::Async` however synchronous its callees were declared: the pass
/// is a third source of suspension beside the two RFC-0046's table names,
/// and it raises the body's task itself (found by the interpreter's
/// `may_suspend` assertion, RFC-0046).
pub fn run(cfg: &mut CfgBody) {
    let mut split_one = false;
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
                } if runs_apart(callee_ty) => {
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
                    split_one = true;
                }
                // Everything else: pass through.
                _ => {
                    new_insts.push(inst);
                }
            }
        }

        block.insts = new_insts;
    }
    if split_one {
        cfg.task = cfg.task.join(Task::Async);
    }
}

fn runs_apart(callee_ty: &Ty) -> bool {
    matches!(callee_ty.effect(), Some(e) if e.runs_apart())
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
            demoted_diamonds: Default::default(),
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

    /// RFC-0079 rule 9: a call handed a reference is spawned like any
    /// other; the `Handle` holds the loan until its `Eval`.
    #[test]
    fn a_call_handed_a_reference_is_spawned() {
        let i = Interner::new();
        let print_id = QualifiedRef::root(i.intern("print"));
        let taken = Ty::Ref(
            crate::ty::Mutability::Shared,
            Box::new(crate::ty::TypeArg::uniform(Ty::String)),
        );
        let print_ty = Ty::Fn {
            params: vec![Param::new(i.intern("s"), taken.clone())],
            ret: Box::new(Ty::Unit),
            captures: vec![],
            effect: crate::ty::Effect::OPAQUE.into(),
            flows: crate::ty::Flows::none().into(),
        };
        let mut cfg = make_cfg(
            vec![
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(print_id),
                    callee_ty: print_ty,
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
        cfg.val_types.insert(v(0), taken);

        run(&mut cfg);

        let insts = all_insts(&cfg);
        assert!(
            matches!(&insts[0].kind, InstKind::Spawn { args, .. } if *args == vec![v(0)]),
            "a call whose argument holds a loan is spawned with that argument"
        );
        assert!(matches!(insts[1].kind, InstKind::Eval { dst, .. } if dst == v(1)));
        assert_eq!(cfg.task, Task::Async);
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
                flows: crate::ty::Flows::Every.into(),
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
                flows: crate::ty::Flows::Every.into(),
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

    fn burner(i: &Interner, effect: crate::ty::Effect) -> Ty {
        Ty::Fn {
            params: vec![Param::new(i.intern("seed"), Ty::I64)],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: effect.into(),
            flows: crate::ty::Flows::Every.into(),
        }
    }

    fn split_once(i: &Interner, effect: crate::ty::Effect) -> CfgBody {
        let hash = QualifiedRef::root(i.intern("hash"));
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(hash),
                    callee_ty: burner(i, effect),
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
        cfg
    }

    #[test]
    fn a_pure_heavy_call_splits_into_a_pair_with_no_order() {
        let i = Interner::new();
        let cfg = split_once(&i, crate::ty::Effect::PURE.at_task(Task::Heavy));
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 3);
        assert!(
            matches!(insts[1].kind, InstKind::Spawn { order: None, .. }),
            "a pure call carries no entry Order, so its Spawn takes none"
        );
        assert!(
            matches!(insts[2].kind, InstKind::Eval { order: None, .. }),
            "and its Eval yields none"
        );
    }

    #[test]
    fn a_pure_sync_call_is_not_split() {
        let i = Interner::new();
        let cfg = split_once(&i, crate::ty::Effect::PURE);
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 2);
        assert!(matches!(insts[1].kind, InstKind::FunctionCall { .. }));
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
                    flows: crate::ty::Flows::Every.into(),
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
                    op: crate::ir::BinOp::Add(crate::ir::Overflow::Trap),
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
