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
//!
//! A call is ordered against the run's own context accesses by its effect's
//! read and write sets (RFC-0025 rule 10): it joins a run only if no `Commit`
//! between the run's first call and itself names a context it touches, and
//! no `Fetch` names one it writes; a call that touches any context is never moved
//! across blocks.

use rustc_hash::FxHashMap;

use crate::analysis::domtree::{DomTree, PostDomTree};
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::cfg::{BlockIdx, CfgBody};
use crate::ir::{Inst, InstKind, OrderEdge, ValueId};
use crate::optimize::context_ops::{context_read, context_written};
use crate::ir::fn_effect;
use crate::ty::{Concrete, Effect, Types};
use crate::decl::Declarations;

pub fn run(cfg: &mut CfgBody, types: &Types<Concrete>, decls: &Declarations) {
    while let Some(m) = find_move(cfg, types, decls) {
        let inst = cfg.blocks[m.from.0].insts.remove(m.at);
        cfg.blocks[m.to.0].insts.insert(m.after + 1, inst);
    }
    let mut subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    for bi in 0..cfg.blocks.len() {
        let runs = runs_in_block(types, &cfg.blocks[bi].insts);
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
fn find_move(cfg: &mut CfgBody, types: &Types<Concrete>, decls: &Declarations) -> Option<Move> {
    let dom = DomTree::build(cfg);
    let pdom = PostDomTree::build(cfg);
    let sites = def_sites(cfg);
    let commutes = |inst: &Inst| match &inst.kind {
        InstKind::FunctionCall {
            callee_ty,
            order: Some(edge),
            ..
        } if fn_effect(types, *callee_ty).is_some_and(|e| e.commutes()) => Some(*edge),
        _ => None,
    };
    let loans = Loans::build(cfg, types, decls);
    let touches_nothing = |inst: &Inst| match &inst.kind {
        InstKind::FunctionCall { callee_ty, .. } => {
            fn_effect(types, *callee_ty).is_some_and(|e| e.reads().is_empty() && e.writes().is_empty())
                && loans.storage_effect(&inst.kind).is_empty()
        }
        _ => false,
    };
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let from = BlockIdx(bi);
        for (ii, inst) in block.insts.iter().enumerate() {
            let Some(edge) = commutes(inst) else {
                continue;
            };
            if !touches_nothing(inst) {
                continue;
            }
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
fn runs_in_block(types: &Types<Concrete>, insts: &[Inst]) -> Vec<Run> {
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
        let Some(effect) = fn_effect(types, *callee_ty) else {
            continue;
        };
        let commutes = effect.commutes();
        match open.take() {
            Some(mut run)
                if commutes
                    && run.last_after == edge.before
                    && !accesses_between(&insts[run.calls[0] + 1..idx], effect) =>
            {
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

/// Whether issuing a call of `effect` before `insts` would reorder it
/// against a page op there: a `Commit` of a context it touches, or a
/// `Fetch` of a context it writes.
fn accesses_between(insts: &[Inst], effect: &Effect) -> bool {
    insts.iter().any(|inst| {
        context_written(&inst.kind)
            .is_some_and(|ctx| effect.reads().contains(&ctx) || effect.writes().contains(&ctx))
            || context_read(&inst.kind).is_some_and(|ctx| effect.writes().contains(&ctx))
    })
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
    let order_ty = cfg.val_types[&run.entry];
    cfg.val_types.insert(merge, order_ty);

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
    use acvus_utils::QualifiedRef;
    use crate::decl::{Core, LoopMarkers, RawDeclarations};
    use crate::ir::{Callee, DebugInfo, MirBody};
    use crate::ty::{Calls, Contexts, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Reissue, SplitCost, Ty, TyKind};
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn empty_declarations(interner: &Interner) -> Declarations {
        Declarations::verify(RawDeclarations {
            lang_impls: Vec::new(),
            types: crate::ty::Types::new(),
            functions: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            inputs: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .unwrap()
    }

    /// `run` at an empty `Declarations`, over `types`.
    fn run_test(cfg: &mut CfgBody, types: &Types<Concrete>) {
        let decls = empty_declarations(&Interner::new());
        run(cfg, types, &decls);
    }

    /// `Idempotent`, not commutative: two of these in a row do not form a
    /// run.
    fn idempotent() -> Effect {
        Effect::new(Reissue::Idempotent, false, Contexts::new(), Contexts::new(), SplitCost::Unavailable)
    }

    /// `Idempotent` and commutative, touching no context.
    fn commutative_idempotent() -> Effect {
        Effect::new(Reissue::Idempotent, true, Contexts::new(), Contexts::new(), SplitCost::Unavailable)
    }

    fn fn_ty(effect: Effect, types: &mut Types<Concrete>) -> Ty {
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        types.intern(TyKind::Fn {
            params: Box::new([]),
            ret: i64_ty,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    }

    /// The value slots a test call uses: its result and its order edge.
    struct Slots {
        dst: usize,
        before: usize,
        after: usize,
    }

    /// `types` trails every helper here, so a `call` nested in the `Vec` an
    /// enclosing `body`/`diamond` builds never borrows it at the same time
    /// as that outer call's own trailing borrow.
    fn call(i: &Interner, name: &str, effect: Effect, slots: Slots, types: &mut Types<Concrete>) -> InstKind {
        InstKind::FunctionCall {
            dst: v(slots.dst),
            callee: Callee::Direct(QualifiedRef::root(i.intern(name))),
            callee_ty: fn_ty(effect, types),
            args: vec![],
            order: Some(OrderEdge {
                before: v(slots.before),
                after: v(slots.after),
            }),
        }
    }

    /// A body with `order_param` v0 and `val_count` values; orders are
    /// the values named by the caller.
    fn body(insts: Vec<InstKind>, val_count: usize, orders: &[usize], types: &mut Types<Concrete>) -> CfgBody {
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let order_ty = types.intern(TyKind::Order);
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for n in 0..val_count {
            let vid = factory.next();
            let ty = if orders.contains(&n) { order_ty } else { i64_ty };
            val_types.insert(vid, ty);
        }
        cfg::promote(MirBody {
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
    fn a_commit_to_a_read_context_breaks_a_run() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let n = QualifiedRef::root(i.intern("n"));
        // Two commutative calls that read @n, with a commit of @n between
        // them: o0 -> f -> o2 ; commit @n ; o2 -> f -> o4. They do not merge.
        let c = Effect::new(
            Reissue::Idempotent,
            true,
            [n].into_iter().collect(),
            Contexts::new(),
            SplitCost::Unavailable,
        );
        let mut cfg = body(
            vec![
                call(
                    &i,
                    "f",
                    c.clone(),
                    Slots {
                        dst: 1,
                        before: 0,
                        after: 2,
                    },
                    &mut types,
                ),
                InstKind::Commit {
                    context: n,
                    value: v(5),
                    wrote: true,
                },
                call(
                    &i,
                    "f",
                    c.clone(),
                    Slots {
                        dst: 3,
                        before: 2,
                        after: 4,
                    },
                    &mut types,
                ),
                InstKind::Return {
                    value: v(1),
                    order: Some(v(4)),
                },
            ],
            6,
            &[0, 2, 4],
            &mut types,
        );
        run_test(&mut cfg, &types);

        assert_eq!(
            edges(&cfg),
            vec![
                OrderEdge {
                    before: v(0),
                    after: v(2)
                },
                OrderEdge {
                    before: v(2),
                    after: v(4)
                },
            ],
            "the commit of @n keeps the second call on the chain after the first"
        );
        assert!(merges(&cfg).is_empty(), "no run formed, so no merge");
    }

    #[test]
    fn a_run_shares_its_entry_and_ends_in_a_merge() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let c = commutative_idempotent();
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
                    &mut types,
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
                    &mut types,
                ),
                InstKind::Return {
                    value: v(1),
                    order: Some(v(4)),
                },
            ],
            5,
            &[0, 2, 4],
            &mut types,
        );
        run_test(&mut cfg, &types);

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
    fn diamond(first: InstKind, second: InstKind, early_return: bool, types: &mut Types<Concrete>) -> CfgBody {
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
            },
            InstKind::Jump {
                label: l2,
                args: vec![],
            },
            InstKind::BlockLabel {
                label: l2,
                params: vec![],
            },
            second,
            InstKind::Return {
                value: v(1),
                order: Some(v(4)),
            },
        ]);
        body(insts, 10, &[0, 2, 4, 6], types)
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
        let mut types = Types::<Concrete>::new();
        let c = commutative_idempotent();
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
                &mut types,
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
                &mut types,
            ),
            false,
            &mut types,
        );
        run_test(&mut cfg, &types);
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
        let mut types = Types::<Concrete>::new();
        let c = commutative_idempotent();
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
                &mut types,
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
                &mut types,
            ),
            true,
            &mut types,
        );
        run_test(&mut cfg, &types);
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
        let mut types = Types::<Concrete>::new();
        let c = commutative_idempotent();
        let mut cfg = diamond(
            call(
                &i,
                "put",
                idempotent(),
                Slots {
                    dst: 1,
                    before: 0,
                    after: 2,
                },
                &mut types,
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
                &mut types,
            ),
            false,
            &mut types,
        );
        run_test(&mut cfg, &types);
        assert_eq!(block_of_call(&cfg, v(3)), Some(3));
        assert!(merges(&cfg).is_empty());
    }

    #[test]
    fn a_call_that_does_not_commute_breaks_the_run() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let c = commutative_idempotent();
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
                    &mut types,
                ),
                call(
                    &i,
                    "put",
                    idempotent(),
                    Slots {
                        dst: 3,
                        before: 2,
                        after: 4,
                    },
                    &mut types,
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
                    &mut types,
                ),
                InstKind::Return {
                    value: v(1),
                    order: Some(v(6)),
                },
            ],
            7,
            &[0, 2, 4, 6],
            &mut types,
        );
        let before = edges(&cfg);
        run_test(&mut cfg, &types);
        assert_eq!(edges(&cfg), before, "the chain is unchanged");
        assert!(merges(&cfg).is_empty());
    }

    #[test]
    fn the_call_after_a_run_waits_for_the_merge() {
        let i = Interner::new();
        let mut types = Types::<Concrete>::new();
        let c = commutative_idempotent();
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
                    &mut types,
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
                    &mut types,
                ),
                call(
                    &i,
                    "put",
                    Effect::opaque(),
                    Slots {
                        dst: 5,
                        before: 4,
                        after: 6,
                    },
                    &mut types,
                ),
                InstKind::Return {
                    value: v(1),
                    order: Some(v(6)),
                },
            ],
            7,
            &[0, 2, 4, 6],
            &mut types,
        );
        run_test(&mut cfg, &types);
        let m = merges(&cfg);
        assert_eq!(m.len(), 1);
        let put = edges(&cfg)[2];
        assert_eq!(put.before, m[0].dst, "put follows the merge");
        assert_eq!(put.after, v(6));
    }
}
