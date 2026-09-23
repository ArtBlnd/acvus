//! Reorder pass: dependency-preserving instruction scheduling.
//!
//! After spawn_split, IO calls are Spawn (async start) + Eval (blocking wait).
//! This pass reorders instructions within each basic block to maximize the
//! distance between Spawn and Eval, hiding IO latency.
//!
//! # Scheduling strategy
//!
//! - **Spawn**: as early as possible (fire async work before anything blocks).
//! - **Eval**: just before its result's first use (not unconditionally last).
//!   This prevents one Eval from blocking another Eval's consumer.
//! - **Normal**: original order (stability for non-IO instructions).
//!
//! # Dependency constraints (soundness)
//!
//! - SSA use-def: B uses value from A -> A before B.
//! - Page order (RFC-0025): for each context `c`, every `Fetch c`, `Commit c`,
//!   and every call whose summary touches `c` keep their original order among
//!   themselves.
//!
//! These constraints are edges in a dependency graph. The scheduler picks from
//! the ready set (zero in-degree) ordered by priority.

use std::collections::BTreeSet;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::inst_info;
use crate::analysis::loans::{Loans, Summaries};
use crate::cfg::CfgBody;
use crate::graph::QualifiedRef;
use crate::ir::*;
use crate::optimize::context_ops::{context_read, context_written};
use crate::ty::Ty;

/// Reorder instructions within each basic block for optimal Spawn/Eval scheduling.
pub fn run(cfg: &mut CfgBody) {
    let loans = Loans::build(cfg, Summaries::NONE);
    for block in &mut cfg.blocks {
        reorder_block(&mut block.insts, &cfg.val_types, &loans);
    }
}

/// Priority for scheduling. Lower value = scheduled earlier.
///
/// Spawn goes first (fire-and-forget, maximizes async overlap).
/// Normal instructions keep their original order.
/// Eval is placed just before the first use of its result -
/// not at the end of the block, so independent Evals don't
/// block each other's consumers.
///
/// The `sub` field breaks ties at the same position:
/// `0` (Eval) sorts before `1` (Normal), so an Eval lands
/// right before the instruction that consumes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Priority {
    /// Spawn - schedule as early as possible.
    Spawn,
    /// Scheduled at a position. (desired_position, 0=eval-before / 1=normal).
    Scheduled(usize, u8),
}

/// Reorder instructions within a single basic block, in-place.
fn reorder_block(insts: &mut Vec<Inst>, val_types: &FxHashMap<ValueId, Ty>, loans: &Loans) {
    let n = insts.len();
    if n <= 1 {
        return;
    }

    let deps = build_dependency_graph(insts, val_types, loans);
    let priorities = compute_priorities(insts, &deps);

    *insts = priority_topo_sort(insts, &deps, &priorities);
}

// -- Page order -------------------------------------------------------

/// The contexts an instruction may touch through the page.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Touch {
    Every,
    Of(BTreeSet<QualifiedRef>),
}

impl Touch {
    fn join(self, other: Touch) -> Touch {
        match (self, other) {
            (Touch::Of(mut a), Touch::Of(b)) => {
                a.extend(b);
                Touch::Of(a)
            }
            _ => Touch::Every,
        }
    }
}

fn effect_touch(ty: &Ty) -> Touch {
    match ty.effect() {
        Some(effect) => Touch::Of(effect.reads.union(&effect.writes).copied().collect()),
        None => Touch::Every,
    }
}

/// The summary of a call: the callee's effect joined with the effect of
/// every function-typed argument.
fn call_touch(callee_ty: &Ty, args: &[ValueId], val_types: &FxHashMap<ValueId, Ty>) -> Touch {
    args.iter()
        .filter_map(|arg| val_types.get(arg))
        .filter(|ty| matches!(ty, Ty::Fn { .. }))
        .fold(effect_touch(callee_ty), |touch, ty| {
            touch.join(effect_touch(ty))
        })
}

/// What each instruction of the block touches; `None` for one that never
/// reaches the page.
fn touches(insts: &[Inst], val_types: &FxHashMap<ValueId, Ty>) -> Vec<Option<Touch>> {
    let mut spawn_of: FxHashMap<ValueId, usize> = FxHashMap::default();
    let mut out: Vec<Option<Touch>> = Vec::with_capacity(insts.len());
    for (i, inst) in insts.iter().enumerate() {
        let touch = match &inst.kind {
            InstKind::FunctionCall {
                callee_ty, args, ..
            } => Some(call_touch(callee_ty, args, val_types)),
            InstKind::Spawn {
                dst,
                callee_ty,
                args,
                ..
            } => {
                spawn_of.insert(*dst, i);
                Some(call_touch(callee_ty, args, val_types))
            }
            InstKind::Eval { src, .. } => Some(
                spawn_of
                    .get(src)
                    .and_then(|&si| out[si].clone())
                    .unwrap_or(Touch::Every),
            ),
            kind => context_read(kind)
                .or_else(|| context_written(kind))
                .map(|c| Touch::Of(BTreeSet::from([c]))),
        };
        out.push(touch);
    }
    out
}

// -- Dependency graph -----------------------------------------------

/// Build dependency edges: `deps[i]` = instructions that must execute before `i`.
fn build_dependency_graph(
    insts: &[Inst],
    val_types: &FxHashMap<ValueId, Ty>,
    loans: &Loans,
) -> Vec<SmallVec<[usize; 4]>> {
    let n = insts.len();
    let mut deps: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];

    // Use-def: a use follows the nearest preceding def of the value.
    let mut def_map: FxHashMap<ValueId, usize> = FxHashMap::default();
    for (i, inst) in insts.iter().enumerate() {
        for u in inst_info::uses(&inst.kind) {
            if let Some(&def_idx) = def_map.get(&u) {
                deps[i].push(def_idx);
            }
        }
        for d in inst_info::defs(&inst.kind) {
            def_map.insert(d, i);
        }
    }

    // Storage order (RFC-0018): a touch of a slot follows its last write,
    // and a write follows every touch since the previous write.
    let mut last_write: FxHashMap<ValueId, usize> = FxHashMap::default();
    let mut reads_since: FxHashMap<ValueId, Vec<usize>> = FxHashMap::default();
    for (i, inst) in insts.iter().enumerate() {
        let effect = loans.storage_effect(&inst.kind);
        for s in effect.reads.iter().chain(&effect.writes) {
            deps[i].extend(last_write.get(s).copied().filter(|&w| w != i));
        }
        for s in &effect.writes {
            deps[i].extend(
                reads_since
                    .remove(s)
                    .into_iter()
                    .flatten()
                    .filter(|&r| r != i),
            );
            last_write.insert(*s, i);
        }
        for s in &effect.reads {
            reads_since.entry(*s).or_default().push(i);
        }
    }

    // Page order: the last instruction that touched each context, and the
    // last that touched every context, precede the next one that touches it.
    let mut last_of: FxHashMap<QualifiedRef, usize> = FxHashMap::default();
    let mut last_every: Option<usize> = None;
    for (i, touch) in touches(insts, val_types).into_iter().enumerate() {
        match touch {
            None => {}
            Some(Touch::Every) => {
                deps[i].extend(last_of.values().copied());
                deps[i].extend(last_every);
                last_of.clear();
                last_every = Some(i);
            }
            Some(Touch::Of(contexts)) => {
                deps[i].extend(last_every);
                for c in contexts {
                    deps[i].extend(last_of.insert(c, i));
                }
            }
        }
    }

    deps
}

// -- Priority assignment --------------------------------------------

/// Assign scheduling priority to each instruction.
///
/// An Eval, and every instruction its result reaches through the dependency
/// graph, runs as late as what depends on it allows: it cannot run before the
/// Eval anyway, so running it later delays nothing, and the Spawns and the
/// instructions they wait for are free to run first.
fn compute_priorities(insts: &[Inst], deps: &[SmallVec<[usize; 4]>]) -> Vec<Priority> {
    let n = insts.len();
    let mut after_an_eval = vec![false; n];
    for i in 0..n {
        after_an_eval[i] = matches!(insts[i].kind, InstKind::Eval { .. })
            || deps[i].iter().any(|&d| after_an_eval[d]);
    }
    let mut successors: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];
    for (i, d) in deps.iter().enumerate() {
        for &dep in d {
            successors[dep].push(i);
        }
    }
    let mut latest = vec![usize::MAX; n];
    for i in (0..n).rev() {
        latest[i] = successors[i]
            .iter()
            .map(|&s| if after_an_eval[s] { latest[s] } else { s })
            .min()
            .unwrap_or(usize::MAX);
    }

    insts
        .iter()
        .enumerate()
        .map(|(i, inst)| match &inst.kind {
            InstKind::Spawn { .. } => Priority::Spawn,
            _ if after_an_eval[i] => Priority::Scheduled(latest[i], 0),
            _ => Priority::Scheduled(i, 1),
        })
        .collect()
}

// -- Topological sort -----------------------------------------------

/// Priority-driven topological sort. Picks the highest-priority ready
/// instruction (lowest Priority value) at each step.
fn priority_topo_sort(
    insts: &[Inst],
    deps: &[SmallVec<[usize; 4]>],
    priorities: &[Priority],
) -> Vec<Inst> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = insts.len();
    let mut in_degree = vec![0u32; n];
    let mut rdeps: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];
    for (i, d) in deps.iter().enumerate() {
        in_degree[i] = d.len() as u32;
        for &dep in d {
            rdeps[dep].push(i);
        }
    }

    let mut ready: BinaryHeap<Reverse<(Priority, usize)>> = BinaryHeap::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            ready.push(Reverse((priorities[i], i)));
        }
    }

    let mut result = Vec::with_capacity(n);
    while let Some(Reverse((_, idx))) = ready.pop() {
        result.push(insts[idx].clone());
        for &succ in &rdeps[idx] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                ready.push(Reverse((priorities[succ], succ)));
            }
        }
    }

    assert_eq!(
        result.len(),
        n,
        "reorder: cycle in dependency graph ({} emitted, {n} total)",
        result.len()
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::ty::Ty;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

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
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
            order_param: None,
            task: crate::ty::Task::Sync,
        })
    }

    /// `commit @x = v0; v1 = fetch @x; commit @x = v1`: the three keep their
    /// order whatever priorities say.
    #[test]
    fn page_ops_keep_their_order() {
        let i = Interner::new();
        let x = QualifiedRef::root(i.intern("x"));
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Commit {
                    context: x,
                    value: v(0),
                },
                InstKind::Fetch {
                    dst: v(1),
                    context: x,
                },
                InstKind::Commit {
                    context: x,
                    value: v(1),
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            2,
        );
        run(&mut cfg);
        assert_eq!(page_ops(&cfg), ["commit", "fetch", "commit"]);
    }

    fn page_ops(cfg: &CfgBody) -> Vec<&'static str> {
        cfg.blocks[0]
            .insts
            .iter()
            .filter_map(|inst| match &inst.kind {
                InstKind::Commit { .. } => Some("commit"),
                InstKind::Fetch { .. } => Some("fetch"),
                _ => None,
            })
            .collect()
    }

    fn fn_ty_touching(i: &Interner, name: &str, context: QualifiedRef) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        let effect = crate::ty::Effect::with_contexts(
            crate::ty::Reissue::Idempotent,
            false,
            Default::default(),
            [context].into_iter().collect(),
        );
        (
            qref,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: effect.into(),
            },
        )
    }

    /// `commit @x; call f (touches @x); fetch @x` keeps its order; a spawn
    /// that touches nothing still moves to the front.
    #[test]
    fn a_call_touching_a_context_stays_between_its_page_ops() {
        let i = Interner::new();
        let x = QualifiedRef::root(i.intern("x"));
        let (f, f_ty) = fn_ty_touching(&i, "f", x);
        let (g, g_ty) = io_fn_ty(&i, "g");
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Commit {
                    context: x,
                    value: v(0),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(f),
                    callee_ty: f_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Fetch {
                    dst: v(2),
                    context: x,
                },
                InstKind::Spawn {
                    dst: v(3),
                    callee: Callee::Direct(g),
                    callee_ty: g_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(4),
                    src: v(3),
                    order: None,
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            5,
        );
        cfg.val_types.insert(v(3), Ty::Handle(Box::new(Ty::String)));
        run(&mut cfg);

        let commit = find_idx(&cfg, |k| matches!(k, InstKind::Commit { .. })).unwrap();
        let call = find_idx(&cfg, |k| matches!(k, InstKind::FunctionCall { .. })).unwrap();
        let fetch = find_idx(&cfg, |k| matches!(k, InstKind::Fetch { .. })).unwrap();
        let spawn = find_idx(&cfg, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        assert!(
            commit < call && call < fetch,
            "commit {commit}, call {call}, fetch {fetch}"
        );
        assert!(
            spawn < commit,
            "the unrelated spawn moves first (spawn {spawn}, commit {commit})"
        );
    }

    /// `spawn f; eval; b = it; y = 1; spawn g(y); eval; read b`: the store of
    /// the first result cannot run before its Eval, so it waits with the Eval
    /// for `b`'s read, and the second Spawn is issued before the first Eval
    /// blocks.
    #[test]
    fn a_store_of_an_eval_result_waits_with_it_for_its_reader() {
        let i = Interner::new();
        let f = QualifiedRef::root(i.intern("f"));
        let f_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::String),
            captures: vec![],
            effect: crate::ty::Effect::PURE.into(),
        };
        let b = v(9);
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(1),
                    callee: Callee::Direct(f),
                    callee_ty: f_ty.clone(),
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(2),
                    src: v(1),
                    order: None,
                },
                InstKind::Assign {
                    target: RefTarget::Var(b),
                    path: vec![],
                    value: v(2),
                },
                InstKind::Const {
                    dst: v(3),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Spawn {
                    dst: v(4),
                    callee: Callee::Direct(f),
                    callee_ty: f_ty,
                    args: vec![v(3)],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(5),
                    src: v(4),
                    order: None,
                },
                InstKind::Take {
                    dst: v(6),
                    target: RefTarget::Var(b),
                    path: vec![],
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );
        for handle in [v(1), v(4)] {
            cfg.val_types.insert(handle, Ty::Handle(Box::new(Ty::String)));
        }
        for text in [v(2), v(5), v(6), b] {
            cfg.val_types.insert(text, Ty::String);
        }
        run(&mut cfg);

        let spawns: Vec<usize> = cfg.blocks[0]
            .insts
            .iter()
            .enumerate()
            .filter(|(_, inst)| matches!(inst.kind, InstKind::Spawn { .. }))
            .map(|(at, _)| at)
            .collect();
        let first_eval = find_idx(&cfg, |k| matches!(k, InstKind::Eval { .. })).unwrap();
        assert!(
            spawns.iter().all(|&spawn| spawn < first_eval),
            "both spawns precede the first eval (spawns {spawns:?}, eval {first_eval})"
        );
    }

    /// `commit @x; spawn f (touches @x); eval h; fetch @x` keeps the fetch
    /// after the eval: the spawn's summary is the eval's.
    #[test]
    fn a_fetch_waits_for_the_eval_of_a_spawn_touching_it() {
        let i = Interner::new();
        let x = QualifiedRef::root(i.intern("x"));
        let (f, f_ty) = fn_ty_touching(&i, "f", x);
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Commit {
                    context: x,
                    value: v(0),
                },
                InstKind::Spawn {
                    dst: v(1),
                    callee: Callee::Direct(f),
                    callee_ty: f_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(2),
                    src: v(1),
                    order: None,
                },
                InstKind::Fetch {
                    dst: v(3),
                    context: x,
                },
                InstKind::Return {
                    value: v(3),
                    order: None,
                },
            ],
            4,
        );
        cfg.val_types.insert(v(1), Ty::Handle(Box::new(Ty::String)));
        run(&mut cfg);

        let commit = find_idx(&cfg, |k| matches!(k, InstKind::Commit { .. })).unwrap();
        let spawn = find_idx(&cfg, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let eval = find_idx(&cfg, |k| matches!(k, InstKind::Eval { .. })).unwrap();
        let fetch = find_idx(&cfg, |k| matches!(k, InstKind::Fetch { .. })).unwrap();
        assert!(
            commit < spawn && spawn < eval && eval < fetch,
            "commit {commit}, spawn {spawn}, eval {eval}, fetch {fetch}"
        );
    }

    fn io_fn_ty(i: &Interner, name: &str) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        (
            qref,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        )
    }

    /// Collect all instructions from all blocks (flattened).
    fn all_insts(cfg: &CfgBody) -> Vec<&Inst> {
        cfg.blocks.iter().flat_map(|b| b.insts.iter()).collect()
    }

    /// Find the index of the first instruction matching a predicate (across all blocks).
    fn find_idx(cfg: &CfgBody, pred: impl Fn(&InstKind) -> bool) -> Option<usize> {
        all_insts(cfg).iter().position(|i| pred(&i.kind))
    }

    // -- Basic: Spawn moves before Eval ------------------------------

    #[test]
    fn two_independent_spawns_before_evals() {
        // Before: spawn_a, eval_a, spawn_b, eval_b, add, return
        // After:  spawn_a, spawn_b, eval_a, eval_b, add, return
        //   (spawns first, evals later)
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_ty(&i, "fetch_a");
        let (fb, fb_ty) = io_fn_ty(&i, "fetch_b");

        // h0 = spawn fetch_a(); r0 = eval h0;
        // h1 = spawn fetch_b(); r1 = eval h1;
        // r2 = r0 + r1; return r2
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    callee_ty: fa_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
                },
                InstKind::Spawn {
                    dst: v(2),
                    callee: Callee::Direct(fb),
                    callee_ty: fb_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(3),
                    src: v(2),
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(3),
                },
                InstKind::Return {
                    value: v(4),
                    order: None,
                },
            ],
            5,
        );
        // Set Handle types for eval.
        cfg.val_types.insert(v(0), Ty::Handle(Box::new(Ty::String)));
        cfg.val_types.insert(v(2), Ty::Handle(Box::new(Ty::String)));

        run(&mut cfg);

        // Both spawns should come before both evals.
        let spawn_a = find_idx(
            &cfg,
            |k| matches!(k, InstKind::Spawn { dst, .. } if *dst == v(0)),
        )
        .unwrap();
        let spawn_b = find_idx(
            &cfg,
            |k| matches!(k, InstKind::Spawn { dst, .. } if *dst == v(2)),
        )
        .unwrap();
        let eval_a = find_idx(
            &cfg,
            |k| matches!(k, InstKind::Eval { src, .. } if *src == v(0)),
        )
        .unwrap();
        let eval_b = find_idx(
            &cfg,
            |k| matches!(k, InstKind::Eval { src, .. } if *src == v(2)),
        )
        .unwrap();

        assert!(spawn_a < eval_a, "spawn_a must come before eval_a");
        assert!(spawn_b < eval_b, "spawn_b must come before eval_b");
        assert!(
            spawn_a < eval_b,
            "spawn_a should come before eval_b (parallelism)"
        );
        assert!(
            spawn_b < eval_a,
            "spawn_b should come before eval_a (parallelism)"
        );
    }

    // -- Dependency: Eval must wait for its Spawn --------------------

    #[test]
    fn eval_after_its_spawn() {
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_ty(&i, "fetch");

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    callee_ty: fa_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            2,
        );
        cfg.val_types.insert(v(0), Ty::Handle(Box::new(Ty::String)));

        run(&mut cfg);

        let spawn_idx = find_idx(&cfg, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let eval_idx = find_idx(&cfg, |k| matches!(k, InstKind::Eval { .. })).unwrap();
        assert!(spawn_idx < eval_idx);
    }

    // -- Independent work fills Spawn-Eval gap -----------------------

    #[test]
    fn independent_work_between_spawn_and_eval() {
        // spawn, const, const, eval, add, return
        // const instructions should land between spawn and eval.
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_ty(&i, "fetch");

        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(5),
                    value: acvus_ast::Literal::Int(10),
                },
                InstKind::Const {
                    dst: v(6),
                    value: acvus_ast::Literal::Int(20),
                },
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    callee_ty: fa_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(7),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(6),
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            8,
        );
        cfg.val_types.insert(v(0), Ty::Handle(Box::new(Ty::String)));

        run(&mut cfg);

        let spawn_idx = find_idx(&cfg, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let eval_idx = find_idx(&cfg, |k| matches!(k, InstKind::Eval { .. })).unwrap();

        // Spawn should be early, eval should be late.
        // BinOp(v5+v6) is independent of spawn/eval, can go between.
        assert!(spawn_idx < eval_idx);
    }

    // -- No-op: no spawns, no change ---------------------------------

    #[test]
    fn no_spawns_preserves_order() {
        let mut cfg = make_cfg(
            vec![
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
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            3,
        );

        let original: Vec<_> = all_insts(&cfg)
            .iter()
            .map(|i| std::mem::discriminant(&i.kind))
            .collect();
        run(&mut cfg);
        let after: Vec<_> = all_insts(&cfg)
            .iter()
            .map(|i| std::mem::discriminant(&i.kind))
            .collect();

        assert_eq!(original, after, "no spawns -> order unchanged");
    }

    // -- Use-def chain prevents wrong reorder ------------------------

    #[test]
    fn use_def_prevents_reorder() {
        // r0 = const 1; r1 = r0 + r0; return r1
        // r0 must come before r1 (use-def).
        let mut cfg = make_cfg(
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
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            2,
        );

        run(&mut cfg);

        let const_idx = find_idx(&cfg, |k| matches!(k, InstKind::Const { .. })).unwrap();
        let binop_idx = find_idx(&cfg, |k| matches!(k, InstKind::BinOp { .. })).unwrap();
        assert!(const_idx < binop_idx);
    }
}
