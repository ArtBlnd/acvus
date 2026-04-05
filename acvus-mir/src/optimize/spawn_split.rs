//! Spawn-split pass: convert IO FunctionCalls into Spawn + Eval pairs.
//!
//! This is a pure IR transformation with no reordering.
//! After this pass, IO calls are expressed as:
//!   handle = Spawn { callee, args, context_uses }
//!   result = Eval { src: handle, context_defs }
//!
//! The Spawn is pure (no side effects). Effects happen at Eval.
//! Reordering is left to a separate pass that can move independent
//! instructions between Spawn and Eval.

use rustc_hash::FxHashMap;

use crate::cfg::CfgBody;
use crate::graph::{FnMetadata, QualifiedRef};
use crate::ir::*;
use crate::ty::{Hint, Ty};

/// Split IO FunctionCalls into Spawn + Eval pairs, in-place.
///
/// `fn_metadata`: QualifiedRef → FnMetadata mapping for callee effect/type/hint lookup.
pub fn run(
    cfg: &mut CfgBody,
    fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>,
) {
    for block in &mut cfg.blocks {
        let mut new_insts = Vec::with_capacity(block.insts.len() + 4);

        for inst in block.insts.drain(..) {
            match inst.kind {
                InstKind::FunctionCall {
                    dst,
                    callee: Callee::Direct(ref callee_id),
                    ref args,
                    ref context_uses,
                    ref context_defs,
                } if is_io_call(fn_metadata, callee_id) => {
                    // Allocate a Handle ValueId.
                    let handle = cfg.val_factory.next();

                    // Register Handle type: Handle<ReturnTy, Effect>.
                    if let Some(Ty::Fn { ret, effect, .. }) =
                        fn_metadata.get(callee_id).map(|m| &m.ty)
                    {
                        cfg.val_types
                            .insert(handle, Ty::Handle(ret.clone(), effect.clone()));
                    }

                    new_insts.push(Inst {
                        span: inst.span,
                        kind: InstKind::Spawn {
                            dst: handle,
                            callee: Callee::Direct(*callee_id),
                            args: args.clone(),
                            context_uses: context_uses.clone(),
                        },
                    });
                    new_insts.push(Inst {
                        span: inst.span,
                        kind: InstKind::Eval {
                            dst,
                            src: handle,
                            context_defs: context_defs.clone(),
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

/// Check if a Direct callee has Hint::Io (candidates for spawn-split).
fn is_io_call(fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>, callee: &QualifiedRef) -> bool {
    matches!(
        fn_metadata.get(callee).and_then(|m| m.hint),
        Some(Hint::Io)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ty::{Effect, Param};
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for _ in 0..val_count {
            let vid = factory.next();
            val_types.insert(vid, Ty::Int);
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
        })
    }

    fn io_effect() -> Effect {
        // IO functions have no context/token effects in the effect set.
        // Their "IO-ness" is conveyed by Hint::Io, not the effect set.
        Effect::pure()
    }

    fn pure_effect() -> Effect {
        Effect::pure()
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
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![Param::new(i.intern("id"), Ty::Int)],
                    ret: Box::new(Ty::String),
                    captures: vec![],
                    effect: io_effect(),
                },
                hint: Some(Hint::Io),
            },
        );

        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(fetch_id),
                    args: vec![v(0)],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            2,
        );

        run(&mut cfg, &fn_metadata);

        // Should be: Const, Spawn, Eval in block insts (Return is terminator).
        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 3);
        assert!(matches!(insts[1].kind, InstKind::Spawn { .. }));
        assert!(matches!(insts[2].kind, InstKind::Eval { .. }));

        // Verify Spawn dst → Eval src chain.
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
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![
                        Param::new(i.intern("a"), Ty::Int),
                        Param::new(i.intern("b"), Ty::Int),
                    ],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: pure_effect(),
                },
                hint: None,
            },
        );

        let mut cfg = make_cfg(
            vec![InstKind::FunctionCall {
                dst: v(0),
                callee: Callee::Direct(add_id),
                args: vec![v(1), v(2)],
                context_uses: vec![],
                context_defs: vec![],
            }],
            3,
        );

        run(&mut cfg, &fn_metadata);

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
                args: vec![],
                context_uses: vec![],
                context_defs: vec![],
            }],
            2,
        );

        run(&mut cfg, &FxHashMap::default());

        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 1);
        assert!(matches!(insts[0].kind, InstKind::FunctionCall { .. }));
    }

    #[test]
    fn context_uses_defs_distributed() {
        let i = Interner::new();
        let fetch_id = QualifiedRef::root(i.intern("fetch"));
        let ctx_a = QualifiedRef::root(i.intern("a"));
        let ctx_b = QualifiedRef::root(i.intern("b"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            fetch_id,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: io_effect(),
                },
                hint: Some(Hint::Io),
            },
        );

        let mut cfg = make_cfg(
            vec![InstKind::FunctionCall {
                dst: v(0),
                callee: Callee::Direct(fetch_id),
                args: vec![],
                context_uses: vec![(ctx_a, v(1))],
                context_defs: vec![(ctx_b, v(2))],
            }],
            3,
        );

        run(&mut cfg, &fn_metadata);

        let insts = all_insts(&cfg);
        assert_eq!(insts.len(), 2);

        // Spawn gets context_uses, Eval gets context_defs.
        if let InstKind::Spawn { context_uses, .. } = &insts[0].kind {
            assert_eq!(context_uses.len(), 1);
            assert_eq!(context_uses[0].0, ctx_a);
        } else {
            panic!("expected Spawn");
        }

        if let InstKind::Eval { context_defs, .. } = &insts[1].kind {
            assert_eq!(context_defs.len(), 1);
            assert_eq!(context_defs[0].0, ctx_b);
        } else {
            panic!("expected Eval");
        }
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
                FnMetadata {
                    ty: Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::String),
                        captures: vec![],
                        effect: io_effect(),
                    },
                    hint: Some(Hint::Io),
                },
            );
        }

        let mut cfg = make_cfg(
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(fetch_a),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(fetch_b),
                    args: vec![],
                    context_uses: vec![],
                    context_defs: vec![],
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(1),
                },
                InstKind::Return(v(2)),
            ],
            3,
        );

        run(&mut cfg, &fn_metadata);

        // 2 calls → 2 Spawn + 2 Eval + BinOp = 5 (Return is terminator)
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
