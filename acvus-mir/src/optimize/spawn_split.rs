//! Spawn-split pass: a `FunctionCall` whose effect may be split becomes a
//! `Spawn` and a `TakeEffect` (effects §6), whether or not an argument
//! holds a loan. The loan is held in the `Handle`'s in-flight position
//! until the `TakeEffect`, and a run that ends before the `TakeEffect`
//! keeps its storage until the task has finished (the interpreter's
//! `flight`).
//!
//! This pass does not reorder. Moving independent instructions between a
//! `Spawn` and its `TakeEffect` is `optimize::reorder`'s job.

use crate::cfg::CfgBody;
use crate::ir::*;
use crate::ty::{Concrete, SplitCost, Ty, TyKind, Types};

pub fn run(cfg: &mut CfgBody, types: &mut Types<Concrete>) {
    for block in &mut cfg.blocks {
        let mut new_insts = Vec::with_capacity(block.insts.len() + 4);

        for inst in block.insts.drain(..) {
            match inst.kind {
                InstKind::FunctionCall {
                    dst,
                    callee: ref callee @ (Callee::Direct(_) | Callee::Extern { .. }),
                    callee_ty,
                    ref args,
                    order,
                } if splits(types, callee_ty) => {
                    let TyKind::Fn { ret, .. } = types.kind(callee_ty) else {
                        panic!("a call's callee type is a function type")
                    };
                    let handle_ty = types.intern(TyKind::Handle(*ret));
                    let handle = cfg.val_factory.next();
                    cfg.val_types.insert(handle, handle_ty);

                    new_insts.push(Inst {
                        span: inst.span,
                        kind: InstKind::Spawn {
                            dst: handle,
                            callee: callee.clone(),
                            callee_ty,
                            args: args.clone(),
                            order: order.map(|edge| edge.before),
                        },
                    });
                    new_insts.push(Inst {
                        span: inst.span,
                        kind: InstKind::TakeEffect {
                            dst,
                            src: handle,
                            order: order.map(|edge| edge.after),
                        },
                    });
                }
                _ => {
                    new_insts.push(inst);
                }
            }
        }

        block.insts = new_insts;
    }
}

/// Whether a call of `callee_ty` may be split (effects §6).
pub fn splits(types: &Types<Concrete>, callee_ty: Ty) -> bool {
    let Some(effect) = fn_effect(types, callee_ty) else {
        panic!("a call's callee type is a function type")
    };
    matches!(effect.split(), SplitCost::WithCost(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use acvus_utils::QualifiedRef;
    use crate::ty::{
        Calls, Contexts, Cost, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, ParamTerm, Reissue,
        SplitCost, TypeArg,
    };
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(types: &mut Types<Concrete>, insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for _ in 0..val_count {
            let vid = factory.next();
            val_types.insert(vid, i64_ty);
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
        })
    }

    /// A type value these tests do not read further: a placeholder
    /// `callee_ty` for a call whose test is not about its type.
    fn dummy_ty(types: &mut Types<Concrete>) -> Ty {
        types.intern(TyKind::Int(IntTy::I64))
    }

    /// A call whose effect may be split (effects §6), the cost being of no
    /// interest to these tests.
    fn splittable() -> Effect {
        Effect::new(
            Reissue::Opaque,
            false,
            Contexts::new(),
            Contexts::new(),
            SplitCost::WithCost(Cost::stated(1)),
        )
    }

    /// Collect all instructions from all blocks (flattened).
    fn all_insts(cfg: &CfgBody) -> Vec<&Inst> {
        cfg.blocks.iter().flat_map(|b| b.insts.iter()).collect()
    }

    /// RFC-0079 rule 9: a call handed a reference is spawned like any
    /// other; the `Handle` holds the loan until its `TakeEffect`.
    #[test]
    fn a_call_handed_a_reference_is_spawned() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let print_id = QualifiedRef::root(i.intern("print"));
        let string_ty = types.intern(TyKind::String);
        let unit_ty = types.intern(TyKind::Unit);
        let taken = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(string_ty)));
        let print_ty = types.intern(TyKind::Fn {
            params: Box::new([ParamTerm {
                name: i.intern("s"),
                ty: taken,
            }]),
            ret: unit_ty,
            effect: EffectTerm::Known(splittable()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });
        let mut cfg = make_cfg(
            &mut types,
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

        run(&mut cfg, &mut types);

        let insts = all_insts(&cfg);
        assert!(
            matches!(&insts[0].kind, InstKind::Spawn { args, .. } if *args == vec![v(0)]),
            "a call whose argument holds a loan is spawned with that argument"
        );
        assert!(matches!(insts[1].kind, InstKind::TakeEffect { dst, .. } if dst == v(1)));
        // `cfg.task` no longer exists: this pass now decides splitting from
        // the callee's `SplitCost` alone (effects §6), so a body carries no
        // task of its own to assert on.
    }

    #[test]
    fn split_io_call() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let fetch_id = QualifiedRef::root(i.intern("fetch"));

        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let string_ty = types.intern(TyKind::String);
        let fetch_ty = types.intern(TyKind::Fn {
            params: Box::new([ParamTerm {
                name: i.intern("id"),
                ty: i64_ty,
            }]),
            ret: string_ty,
            effect: EffectTerm::Known(splittable()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });
        let mut cfg = make_cfg(
            &mut types,
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

        run(&mut cfg, &mut types);

        // Should be: Const, Spawn, TakeEffect in block insts (Return is terminator).
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 3);
        assert!(matches!(insts[1].kind, InstKind::Spawn { .. }));
        assert!(matches!(insts[2].kind, InstKind::TakeEffect { .. }));

        // Verify Spawn dst -> TakeEffect src chain.
        if let (InstKind::Spawn { dst: handle, .. }, InstKind::TakeEffect { src, dst, .. }) =
            (&insts[1].kind, &insts[2].kind)
        {
            assert_eq!(handle, src, "TakeEffect.src must reference Spawn.dst");
            assert_eq!(*dst, v(1), "TakeEffect.dst must be original FunctionCall.dst");
        } else {
            panic!("expected Spawn + TakeEffect");
        }

        // Handle should have Handle type.
        if let InstKind::Spawn { dst: handle, .. } = &insts[1].kind {
            let handle_ty = cfg.val_types.get(handle).unwrap();
            assert!(
                matches!(types.kind(*handle_ty), TyKind::Handle(..)),
                "Spawn dst must have Handle type"
            );
        }
    }

    #[test]
    fn pure_call_unchanged() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let add_id = QualifiedRef::root(i.intern("add"));

        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let add_ty = types.intern(TyKind::Fn {
            params: Box::new([
                ParamTerm {
                    name: i.intern("a"),
                    ty: i64_ty,
                },
                ParamTerm {
                    name: i.intern("b"),
                    ty: i64_ty,
                },
            ]),
            ret: i64_ty,
            effect: EffectTerm::Known(Effect::opaque()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });

        let mut cfg = make_cfg(
            &mut types,
            vec![InstKind::FunctionCall {
                dst: v(0),
                callee: Callee::Direct(add_id),
                callee_ty: add_ty,
                args: vec![v(1), v(2)],
                order: None,
            }],
            3,
        );

        run(&mut cfg, &mut types);

        // Pure call should NOT be split.
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 1);
        assert!(matches!(insts[0].kind, InstKind::FunctionCall { .. }));
    }

    fn burner(types: &mut Types<Concrete>, i: &Interner, effect: Effect) -> Ty {
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        types.intern(TyKind::Fn {
            params: Box::new([ParamTerm {
                name: i.intern("seed"),
                ty: i64_ty,
            }]),
            ret: i64_ty,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
        })
    }

    fn split_once(i: &Interner, effect: Effect) -> CfgBody {
        let mut types = Types::<Concrete>::new();
        let hash = QualifiedRef::root(i.intern("hash"));
        let burner_ty = burner(&mut types, i, effect);
        let mut cfg = make_cfg(
            &mut types,
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(hash),
                    callee_ty: burner_ty,
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
        run(&mut cfg, &mut types);
        cfg
    }

    #[test]
    fn a_pure_heavy_call_splits_into_a_pair_with_no_order() {
        let i = Interner::new();
        let heavy = Effect::new(
            Reissue::Pure,
            true,
            Contexts::new(),
            Contexts::new(),
            SplitCost::WithCost(Cost::stated(1)),
        );
        let cfg = split_once(&i, heavy);
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 3);
        assert!(
            matches!(insts[1].kind, InstKind::Spawn { order: None, .. }),
            "a pure call carries no entry Order, so its Spawn takes none"
        );
        assert!(
            matches!(insts[2].kind, InstKind::TakeEffect { order: None, .. }),
            "and its TakeEffect yields none"
        );
    }

    #[test]
    fn a_pure_sync_call_is_not_split() {
        let i = Interner::new();
        let cfg = split_once(&i, Effect::pure());
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 2);
        assert!(matches!(insts[1].kind, InstKind::FunctionCall { .. }));
    }

    #[test]
    fn indirect_call_unchanged() {
        // Indirect calls are never split (no QualifiedRef to look up).
        let mut types = Types::<Concrete>::new();
        let dummy = dummy_ty(&mut types);
        let mut cfg = make_cfg(
            &mut types,
            vec![InstKind::FunctionCall {
                dst: v(0),
                callee: Callee::Indirect(v(1)),
                callee_ty: dummy,
                args: vec![],
                order: None,
            }],
            2,
        );

        run(&mut cfg, &mut types);

        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 1);
        assert!(matches!(insts[0].kind, InstKind::FunctionCall { .. }));
    }

    #[test]
    fn multiple_io_calls_all_split() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let fetch_a = QualifiedRef::root(i.intern("fetch_a"));
        let fetch_b = QualifiedRef::root(i.intern("fetch_b"));

        let string_ty = types.intern(TyKind::String);
        let io_fn_ty = types.intern(TyKind::Fn {
            params: Box::new([]),
            ret: string_ty,
            effect: EffectTerm::Known(splittable()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
        });
        let mut cfg = make_cfg(
            &mut types,
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(fetch_a),
                    callee_ty: io_fn_ty,
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

        run(&mut cfg, &mut types);

        // 2 calls -> 2 Spawn + 2 TakeEffect + BinOp = 5 (Return is terminator)
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 5);

        let spawn_count = insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::Spawn { .. }))
            .count();
        let eval_count = insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::TakeEffect { .. }))
            .count();
        assert_eq!(spawn_count, 2);
        assert_eq!(eval_count, 2);
    }
}
