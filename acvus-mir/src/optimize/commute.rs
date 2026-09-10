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
//! branch. Before runs are read, a commutative call whose block
//! post-dominates the block of the call it follows is moved there, right
//! after that call: every path through that block reaches the call, so
//! issuing it there speculates nothing. Nothing else reads the commutes
//! axis.

use rustc_hash::FxHashMap;

use crate::analysis::domtree::{DomTree, PostDomTree};
use crate::analysis::inst_info;
use crate::cfg::{BlockIdx, CfgBody};
use crate::ir::{Inst, InstKind, OrderEdge, ValueId};
use crate::ty::Ty;

pub fn run(cfg: &mut CfgBody) {
    while let Some(m) = find_move(cfg) {
        let inst = cfg.blocks[m.from.0].insts.remove(m.at);
        cfg.blocks[m.to.0].insts.insert(m.after + 1, inst);
    }
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

/// A commutative call to move next to the call it follows on the chain.
struct Move {
    from: BlockIdx,
    /// Index of the call within `from`.
    at: usize,
    to: BlockIdx,
    /// Index within `to` of the call whose yielded Order the moved call takes.
    after: usize,
}

/// Where a value is defined: the block, and the instruction index within it
/// for an instruction; `None` for a block param or an entry definition.
#[derive(Clone, Copy)]
struct DefSite {
    block: BlockIdx,
    index: Option<usize>,
}

fn def_sites(cfg: &CfgBody) -> FxHashMap<ValueId, DefSite> {
    let mut sites = FxHashMap::default();
    for v in cfg.entry_defs() {
        sites.insert(
            v,
            DefSite {
                block: BlockIdx(0),
                index: None,
            },
        );
    }
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for &p in &block.params {
            sites.insert(
                p,
                DefSite {
                    block: BlockIdx(bi),
                    index: None,
                },
            );
        }
        for (ii, inst) in block.insts.iter().enumerate() {
            for d in inst_info::defs(&inst.kind) {
                sites.insert(
                    d,
                    DefSite {
                        block: BlockIdx(bi),
                        index: Some(ii),
                    },
                );
            }
        }
    }
    sites
}

/// The first commutative call that can be issued next to the commutative
/// call it follows: its block post-dominates that call's block, and every
/// operand is available there.
fn find_move(cfg: &mut CfgBody) -> Option<Move> {
    let dom = DomTree::build(cfg);
    let pdom = PostDomTree::build(cfg);
    let sites = def_sites(cfg);
    let commutes = |inst: &Inst| match &inst.kind {
        InstKind::FunctionCall {
            callee_ty,
            order: Some(edge),
            ..
        } if callee_ty.effect().is_some_and(|e| e.commutes) => Some(*edge),
        _ => None,
    };
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let from = BlockIdx(bi);
        for (ii, inst) in block.insts.iter().enumerate() {
            let Some(edge) = commutes(inst) else {
                continue;
            };
            let Some(prev) = sites.get(&edge.before) else {
                continue;
            };
            let Some(after) = prev.index else {
                continue;
            };
            if prev.block == from || commutes(&cfg.blocks[prev.block.0].insts[after]).is_none() {
                continue;
            }
            if !pdom.post_dominates(from, prev.block) {
                continue;
            }
            let available = inst_info::uses(&inst.kind)
                .into_iter()
                .filter(|u| *u != edge.before)
                .all(|u| match sites.get(&u) {
                    None => false,
                    Some(site) if site.block == prev.block => site.index.is_none_or(|i| i <= after),
                    Some(site) => dom.dominates(site.block, prev.block) && site.block != prev.block,
                });
            if available {
                return Some(Move {
                    from,
                    at: ii,
                    to: prev.block,
                    after,
                });
            }
        }
    }
    None
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
            lent: Vec::new(),
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
                    c.clone(),
                    Slots {
                        dst: 1,
                        before: 0,
                        after: 2,
                    },
                ),
                call(
                    &i,
                    "draw",
                    c.clone(),
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

    /// B0: `first`, branch; L0 and L1 empty; L2: `second`, return. `second`
    /// takes the Order `first` yielded.
    fn diamond(first: InstKind, second: InstKind, early_return: bool) -> CfgBody {
        use crate::ir::Label;
        let l0 = Label(0);
        let l1 = Label(1);
        let l2 = Label(2);
        let mut insts = vec![
            first,
            InstKind::Const {
                dst: v(9),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(9),
                then_label: l0,
                then_args: vec![],
                else_label: l1,
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: l0,
                params: vec![],
                merge_of: None,
            },
        ];
        if early_return {
            insts.push(InstKind::Return {
                value: v(1),
                order: Some(v(2)),
            });
        } else {
            insts.push(InstKind::Jump {
                label: l2,
                args: vec![],
            });
        }
        insts.extend([
            InstKind::BlockLabel {
                label: l1,
                params: vec![],
                merge_of: None,
            },
            InstKind::Jump {
                label: l2,
                args: vec![],
            },
            InstKind::BlockLabel {
                label: l2,
                params: vec![],
                merge_of: None,
            },
            second,
            InstKind::Return {
                value: v(1),
                order: Some(v(4)),
            },
        ]);
        body(insts, 10, &[0, 2, 4, 6])
    }

    fn block_of_call(cfg: &CfgBody, dst: ValueId) -> Option<usize> {
        cfg.blocks.iter().position(|b| {
            b.insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::FunctionCall { dst: d, .. } if *d == dst))
        })
    }

    #[test]
    fn a_commutative_call_after_a_branch_is_issued_before_it() {
        let i = Interner::new();
        let c = Effect::IDEMPOTENT.commutative();
        let mut cfg = diamond(
            call(
                &i,
                "draw",
                c.clone(),
                Slots {
                    dst: 1,
                    before: 0,
                    after: 2,
                },
            ),
            call(
                &i,
                "draw",
                c.clone(),
                Slots {
                    dst: 3,
                    before: 2,
                    after: 4,
                },
            ),
            false,
        );
        run(&mut cfg);
        assert_eq!(
            block_of_call(&cfg, v(3)),
            Some(0),
            "the second draw moves into the entry block"
        );
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
            "both draws take the entry order"
        );
        assert_eq!(merges(&cfg).len(), 1);
    }

    #[test]
    fn a_call_is_not_issued_on_a_path_that_may_skip_it() {
        let i = Interner::new();
        let c = Effect::IDEMPOTENT.commutative();
        let mut cfg = diamond(
            call(
                &i,
                "draw",
                c.clone(),
                Slots {
                    dst: 1,
                    before: 0,
                    after: 2,
                },
            ),
            call(
                &i,
                "draw",
                c.clone(),
                Slots {
                    dst: 3,
                    before: 2,
                    after: 4,
                },
            ),
            true,
        );
        run(&mut cfg);
        assert_eq!(
            block_of_call(&cfg, v(3)),
            Some(3),
            "one arm returns early: the draw stays"
        );
        assert!(merges(&cfg).is_empty());
    }

    #[test]
    fn a_call_following_one_that_does_not_commute_stays() {
        let i = Interner::new();
        let c = Effect::IDEMPOTENT.commutative();
        let mut cfg = diamond(
            call(
                &i,
                "put",
                Effect::IDEMPOTENT,
                Slots {
                    dst: 1,
                    before: 0,
                    after: 2,
                },
            ),
            call(
                &i,
                "draw",
                c.clone(),
                Slots {
                    dst: 3,
                    before: 2,
                    after: 4,
                },
            ),
            false,
        );
        run(&mut cfg);
        assert_eq!(block_of_call(&cfg, v(3)), Some(3));
        assert!(merges(&cfg).is_empty());
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
                    c.clone(),
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
                    c.clone(),
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
                    c.clone(),
                    Slots {
                        dst: 1,
                        before: 0,
                        after: 2,
                    },
                ),
                call(
                    &i,
                    "draw",
                    c.clone(),
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
