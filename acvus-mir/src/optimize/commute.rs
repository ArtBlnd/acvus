//! Commutative runs (RFC-0013).
//!
//! A run is a maximal sequence of calls whose effect commutes and that are
//! neighbours on the `Order` chain: each takes the `Order` the previous one
//! yielded. Two calls of commutative functions are the same program in
//! either order, so the run is lowered as an `anyorder` block would be:
//! every call in the run takes the run's entry `Order`, and one `Merge` of
//! everything they yielded stands where the last call's `Order` stood.
//!
//! The chain is read after SSA construction, on `FunctionCall` edges and
//! within one block; a phi is not a neighbour, so a run never crosses a
//! branch. Nothing else reads the commutes axis.

use rustc_hash::FxHashMap;

use crate::cfg::CfgBody;
use crate::ir::{Inst, InstKind, OrderEdge, ValueId};
use crate::ty::Ty;

pub fn run(cfg: &mut CfgBody) {
    let mut subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    for bi in 0..cfg.blocks.len() {
        let runs = runs_in_block(&cfg.blocks[bi].insts);
        for run in runs.into_iter().rev() {
            let merge = release(cfg, bi, &run);
            subst.insert(run.last_after, merge);
        }
    }
    if subst.is_empty() {
        return;
    }
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            if let InstKind::Merge { dst, .. } = &inst.kind
                && subst.values().any(|m| m == dst)
            {
                continue;
            }
            super::ssa_pass::apply_subst(&mut inst.kind, &subst);
        }
        super::ssa_pass::apply_subst_terminator(&mut block.terminator, &subst);
    }
}

/// The calls of one run, by instruction index within their block, in
/// chain order.
struct Run {
    calls: Vec<usize>,
    /// The `Order` the first call took; the whole run takes it.
    entry: ValueId,
    /// The `Order` the last call yielded; the merge replaces it.
    last_after: ValueId,
}

/// Find the runs of a block. A call joins the current run when its effect
/// commutes and it takes the `Order` the run's last call yielded.
fn runs_in_block(insts: &[Inst]) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    let mut open: Option<Run> = None;
    for (idx, inst) in insts.iter().enumerate() {
        let InstKind::FunctionCall {
            callee_ty,
            order: Some(edge),
            ..
        } = &inst.kind
        else {
            continue;
        };
        let commutes = callee_ty.effect().is_some_and(|e| e.commutes);
        match open.take() {
            Some(mut run) if commutes && run.last_after == edge.before => {
                run.calls.push(idx);
                run.last_after = edge.after;
                open = Some(run);
            }
            prev => {
                runs.extend(prev.filter(|r| r.calls.len() > 1));
                open = commutes.then(|| Run {
                    calls: vec![idx],
                    entry: edge.before,
                    last_after: edge.after,
                });
            }
        }
    }
    runs.extend(open.filter(|r| r.calls.len() > 1));
    runs
}

/// Rewrite one run: every call takes the entry `Order`, and a `Merge` of
/// what they yielded follows the last call. Returns the merge's value.
fn release(cfg: &mut CfgBody, bi: usize, run: &Run) -> ValueId {
    let mut yielded = Vec::with_capacity(run.calls.len());
    for &idx in &run.calls {
        let InstKind::FunctionCall { order, .. } = &mut cfg.blocks[bi].insts[idx].kind else {
            unreachable!("a run holds only calls");
        };
        let edge = order.expect("a run holds only calls with an order edge");
        *order = Some(OrderEdge {
            before: run.entry,
            after: edge.after,
        });
        yielded.push(edge.after);
    }
    let merge = cfg.val_factory.next();
    cfg.val_types.insert(merge, Ty::Order);
    let last = *run.calls.last().expect("a run has at least two calls");
    let span = cfg.blocks[bi].insts[last].span;
    cfg.blocks[bi].insts.insert(
        last + 1,
        Inst {
            span,
            kind: InstKind::Merge {
                dst: merge,
                orders: yielded,
            },
        },
    );
    merge
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::graph::QualifiedRef;
    use crate::ir::{Callee, DebugInfo, MirBody};
    use crate::ty::Effect;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn fn_ty(effect: Effect) -> Ty {
        Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Int),
            captures: vec![],
            effect: effect.into(),
        }
    }

    /// The value slots a test call uses: its result and its order edge.
    struct Slots {
        dst: usize,
        before: usize,
        after: usize,
    }

    fn call(i: &Interner, name: &str, effect: Effect, slots: Slots) -> InstKind {
        InstKind::FunctionCall {
            dst: v(slots.dst),
            callee: Callee::Direct(QualifiedRef::root(i.intern(name))),
            callee_ty: fn_ty(effect),
            args: vec![],
            order: Some(OrderEdge {
                before: v(slots.before),
                after: v(slots.after),
            }),
        }
    }

    /// A body with `order_param` v0 and `val_count` values; orders are
    /// the values named by the caller.
    fn body(insts: Vec<InstKind>, val_count: usize, orders: &[usize]) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for n in 0..val_count {
            let vid = factory.next();
            let ty = if orders.contains(&n) {
                Ty::Order
            } else {
                Ty::Int
            };
            val_types.insert(vid, ty);
        }
        cfg::promote(MirBody {
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
            order_param: Some(v(0)),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn edges(cfg: &CfgBody) -> Vec<OrderEdge> {
        cfg.blocks[0]
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::FunctionCall { order, .. } => *order,
                _ => None,
            })
            .collect()
    }

    struct MergeShape {
        dst: ValueId,
        orders: Vec<ValueId>,
    }

    fn merges(cfg: &CfgBody) -> Vec<MergeShape> {
        cfg.blocks[0]
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::Merge { dst, orders } => Some(MergeShape {
                    dst: *dst,
                    orders: orders.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_run_shares_its_entry_and_ends_in_a_merge() {
        let i = Interner::new();
        let c = Effect::IDEMPOTENT.commutative();
        // o0 -> draw -> o2 -> draw -> o4 -> return [o4]
        let mut cfg = body(
            vec![
                call(
                    &i,
                    "draw",
                    c,
                    Slots {
                        dst: 1,
                        before: 0,
                        after: 2,
                    },
                ),
                call(
                    &i,
                    "draw",
                    c,
                    Slots {
                        dst: 3,
                        before: 2,
                        after: 4,
                    },
                ),
                InstKind::Return {
                    value: v(1),
                    order: Some(v(4)),
                },
            ],
            5,
            &[0, 2, 4],
        );
        run(&mut cfg);

        assert_eq!(
            edges(&cfg),
            vec![
                OrderEdge {
                    before: v(0),
                    after: v(2)
                },
                OrderEdge {
                    before: v(0),
                    after: v(4)
                },
            ],
            "both calls take the entry order"
        );
        let m = merges(&cfg);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].orders, vec![v(2), v(4)]);
        assert!(
            matches!(cfg.blocks[0].terminator, cfg::Terminator::Return { order: Some(o), .. } if o == m[0].dst),
            "the return waits for the merge"
        );
    }

    #[test]
    fn a_call_that_does_not_commute_breaks_the_run() {
        let i = Interner::new();
        let c = Effect::IDEMPOTENT.commutative();
        // draw, put, draw: no run.
        let mut cfg = body(
            vec![
                call(
                    &i,
                    "draw",
                    c,
                    Slots {
                        dst: 1,
                        before: 0,
                        after: 2,
                    },
                ),
                call(
                    &i,
                    "put",
                    Effect::IDEMPOTENT,
                    Slots {
                        dst: 3,
                        before: 2,
                        after: 4,
                    },
                ),
                call(
                    &i,
                    "draw",
                    c,
                    Slots {
                        dst: 5,
                        before: 4,
                        after: 6,
                    },
                ),
                InstKind::Return {
                    value: v(1),
                    order: Some(v(6)),
                },
            ],
            7,
            &[0, 2, 4, 6],
        );
        let before = edges(&cfg);
        run(&mut cfg);
        assert_eq!(edges(&cfg), before, "the chain is unchanged");
        assert!(merges(&cfg).is_empty());
    }

    #[test]
    fn the_call_after_a_run_waits_for_the_merge() {
        let i = Interner::new();
        let c = Effect::IDEMPOTENT.commutative();
        // draw, draw, put: put takes the merge of both draws.
        let mut cfg = body(
            vec![
                call(
                    &i,
                    "draw",
                    c,
                    Slots {
                        dst: 1,
                        before: 0,
                        after: 2,
                    },
                ),
                call(
                    &i,
                    "draw",
                    c,
                    Slots {
                        dst: 3,
                        before: 2,
                        after: 4,
                    },
                ),
                call(
                    &i,
                    "put",
                    Effect::OPAQUE,
                    Slots {
                        dst: 5,
                        before: 4,
                        after: 6,
                    },
                ),
                InstKind::Return {
                    value: v(1),
                    order: Some(v(6)),
                },
            ],
            7,
            &[0, 2, 4, 6],
        );
        run(&mut cfg);
        let m = merges(&cfg);
        assert_eq!(m.len(), 1);
        let put = edges(&cfg)[2];
        assert_eq!(put.before, m[0].dst, "put follows the merge");
        assert_eq!(put.after, v(6));
    }
}
