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
//! - SSA use-def: B uses value from A → A before B.
//! - Token ordering: instructions sharing a token preserve original order.
//! - ContextStore ordering: stores to the same context preserve original order.
//!
//! These constraints are edges in a dependency graph. The scheduler picks from
//! the ready set (zero in-degree) ordered by priority.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::inst_info;
use crate::cfg::CfgBody;
use crate::graph::{FnMetadata, QualifiedRef};
use crate::ir::*;
use crate::ty::{Effect, EffectTarget, Ty};

/// Reorder instructions within each basic block for optimal Spawn/Eval scheduling.
///
/// `fn_metadata`: QualifiedRef → FnMetadata for looking up callee effects (Token extraction).
pub fn run(cfg: &mut CfgBody, fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>) {
    for block in &mut cfg.blocks {
        reorder_block(&mut block.insts, &cfg.val_types, fn_metadata);
    }
}

/// Priority for scheduling. Lower value = scheduled earlier.
///
/// Spawn goes first (fire-and-forget, maximizes async overlap).
/// Normal instructions keep their original order.
/// Eval is placed just before the first use of its result —
/// not at the end of the block, so independent Evals don't
/// block each other's consumers.
///
/// The `sub` field breaks ties at the same position:
/// `0` (Eval) sorts before `1` (Normal), so an Eval lands
/// right before the instruction that consumes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Priority {
    /// Spawn — schedule as early as possible.
    Spawn,
    /// Scheduled at a position. (desired_position, 0=eval-before / 1=normal).
    Scheduled(usize, u8),
}

/// Extract token QualifiedRefs from a FunctionCall/Spawn/Eval's effect.
fn token_deps(
    kind: &InstKind,
    val_types: &FxHashMap<ValueId, Ty>,
    fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>,
) -> SmallVec<[QualifiedRef; 2]> {
    let effect_set = match kind {
        InstKind::FunctionCall {
            callee: Callee::Direct(qref),
            ..
        }
        | InstKind::Spawn {
            callee: Callee::Direct(qref),
            ..
        } => fn_metadata.get(qref).and_then(|m| match &m.ty {
            Ty::Fn {
                effect: Effect::Resolved(eff),
                ..
            } => Some(eff),
            _ => None,
        }),
        InstKind::Eval { src, .. } => val_types.get(src).and_then(|ty| match ty {
            Ty::Handle(_, Effect::Resolved(eff)) => Some(eff),
            _ => None,
        }),
        _ => None,
    };

    let Some(eff) = effect_set else {
        return SmallVec::new();
    };

    eff.reads
        .iter()
        .chain(eff.writes.iter())
        .filter_map(|target| match target {
            EffectTarget::Token(tid) => Some(*tid),
            _ => None,
        })
        .collect()
}

/// Reorder instructions within a single basic block, in-place.
fn reorder_block(
    insts: &mut Vec<Inst>,
    val_types: &FxHashMap<ValueId, Ty>,
    fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>,
) {
    let n = insts.len();
    if n <= 1 {
        return;
    }

    let deps = build_dependency_graph(insts, val_types, fn_metadata);
    let priorities = compute_priorities(insts);

    *insts = priority_topo_sort(insts, &deps, &priorities);
}

// ── Dependency graph ───────────────────────────────────────────────

/// Build dependency edges: `deps[i]` = instructions that must execute before `i`.
fn build_dependency_graph(
    insts: &[Inst],
    val_types: &FxHashMap<ValueId, Ty>,
    fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>,
) -> Vec<SmallVec<[usize; 4]>> {
    let n = insts.len();
    let mut deps: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];

    // def_map: ValueId → defining instruction index in this block.
    let mut def_map: FxHashMap<ValueId, usize> = FxHashMap::default();
    for (i, inst) in insts.iter().enumerate() {
        for d in inst_info::defs(&inst.kind) {
            def_map.insert(d, i);
        }
    }

    // SSA use-def: if B uses a value defined by A, then A → B.
    for (i, inst) in insts.iter().enumerate() {
        for u in inst_info::uses(&inst.kind) {
            if let Some(&def_idx) = def_map.get(&u)
                && def_idx != i
            {
                deps[i].push(def_idx);
            }
        }
    }

    // Token ordering: instructions sharing a token preserve original order.
    let mut last_token_user: FxHashMap<QualifiedRef, usize> = FxHashMap::default();
    for (i, inst) in insts.iter().enumerate() {
        for tid in token_deps(&inst.kind, val_types, fn_metadata) {
            if let Some(&prev) = last_token_user.get(&tid) {
                deps[i].push(prev);
            }
            last_token_user.insert(tid, i);
        }
    }

    // Store ordering: stores to the same context preserve original order.
    let mut last_ctx_store: FxHashMap<QualifiedRef, usize> = FxHashMap::default();
    for (i, inst) in insts.iter().enumerate() {
        if let InstKind::Store { dst, .. } = &inst.kind
            && let Some(&def_idx) = def_map.get(dst)
            && let InstKind::Ref {
                target: crate::ir::RefTarget::Context(ctx),
                ..
            } = &insts[def_idx].kind
        {
            if let Some(&prev) = last_ctx_store.get(ctx)
                && prev != i
            {
                deps[i].push(prev);
            }
            last_ctx_store.insert(*ctx, i);
        }
    }

    deps
}

// ── Priority assignment ────────────────────────────────────────────

/// Assign scheduling priority to each instruction.
fn compute_priorities(insts: &[Inst]) -> Vec<Priority> {
    // For each value, the earliest instruction that uses it.
    let mut first_use_of: FxHashMap<ValueId, usize> = FxHashMap::default();
    for (i, inst) in insts.iter().enumerate() {
        for u in inst_info::uses(&inst.kind) {
            first_use_of.entry(u).or_insert(i);
        }
    }

    insts
        .iter()
        .enumerate()
        .map(|(i, inst)| match &inst.kind {
            InstKind::Spawn { .. } => Priority::Spawn,
            InstKind::Eval { dst, .. } => {
                let pos = first_use_of.get(dst).copied().unwrap_or(usize::MAX);
                Priority::Scheduled(pos, 0) // just before consumer
            }
            _ => Priority::Scheduled(i, 1), // original position
        })
        .collect()
}

// ── Topological sort ───────────────────────────────────────────────

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
    use crate::cfg::{self, CfgBody};
    use crate::ty::{Effect, EffectSet, EffectTarget};
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    /// Helper: non-pure effect with a dummy write target.
    fn effectful(i: &Interner, tag: &str) -> Effect {
        Effect::Resolved(EffectSet {
            reads: std::collections::BTreeSet::new(),
            writes: std::collections::BTreeSet::from([EffectTarget::Token(
                QualifiedRef::root(i.intern(tag)),
            )]),
        })
    }

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
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn io_fn_type(i: &Interner, name: &str) -> (QualifiedRef, FnMetadata) {
        let qref = QualifiedRef::root(i.intern(name));
        (
            qref,
            FnMetadata {
                // IO functions have no context/token effects — their "IO-ness"
                // is conveyed by Hint::Io. The effect set has no tokens so that
                // token liveness does not block spawn hoisting.
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::String),
                    captures: vec![],
                    effect: Effect::pure(),
                },
                hint: Some(crate::ty::Hint::Io),
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

    // ── Basic: Spawn moves before Eval ──────────────────────────────

    #[test]
    fn two_independent_spawns_before_evals() {
        // Before: spawn_a, eval_a, spawn_b, eval_b, add, return
        // After:  spawn_a, spawn_b, eval_a, eval_b, add, return
        //   (spawns first, evals later)
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_type(&i, "fetch_a");
        let (fb, fb_ty) = io_fn_type(&i, "fetch_b");

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(fa, fa_ty);
        fn_metadata.insert(fb, fb_ty);

        // h0 = spawn fetch_a(); r0 = eval h0;
        // h1 = spawn fetch_b(); r1 = eval h1;
        // r2 = r0 + r1; return r2
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::Spawn {
                    dst: v(2),
                    callee: Callee::Direct(fb),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(3),
                    src: v(2),
                    context_defs: vec![],
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(3),
                },
                InstKind::Return(v(4)),
            ],
            5,
        );
        // Set Handle types for eval.
        cfg.val_types.insert(
            v(0),
            Ty::Handle(
                Box::new(Ty::String),
                effectful(&i, "__handle"),
            ),
        );
        cfg.val_types.insert(
            v(2),
            Ty::Handle(
                Box::new(Ty::String),
                effectful(&i, "__handle"),
            ),
        );

        run(&mut cfg, &fn_metadata);

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

    // ── Dependency: Eval must wait for its Spawn ────────────────────

    #[test]
    fn eval_after_its_spawn() {
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_type(&i, "fetch");
        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(fa, fa_ty);

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            2,
        );
        cfg.val_types.insert(
            v(0),
            Ty::Handle(
                Box::new(Ty::String),
                effectful(&i, "__handle"),
            ),
        );

        run(&mut cfg, &fn_metadata);

        let spawn_idx = find_idx(&cfg, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let eval_idx = find_idx(&cfg, |k| matches!(k, InstKind::Eval { .. })).unwrap();
        assert!(spawn_idx < eval_idx);
    }

    // ── Independent work fills Spawn-Eval gap ───────────────────────

    #[test]
    fn independent_work_between_spawn_and_eval() {
        // spawn, const, const, eval, add, return
        // const instructions should land between spawn and eval.
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_type(&i, "fetch");
        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(fa, fa_ty);

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
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::BinOp {
                    dst: v(7),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(6),
                },
                InstKind::Return(v(1)),
            ],
            8,
        );
        cfg.val_types.insert(
            v(0),
            Ty::Handle(
                Box::new(Ty::String),
                effectful(&i, "__handle"),
            ),
        );

        run(&mut cfg, &fn_metadata);

        let spawn_idx = find_idx(&cfg, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let eval_idx = find_idx(&cfg, |k| matches!(k, InstKind::Eval { .. })).unwrap();

        // Spawn should be early, eval should be late.
        // BinOp(v5+v6) is independent of spawn/eval, can go between.
        assert!(spawn_idx < eval_idx);
    }

    // ── Token dependency preserves order ────────────────────────────

    #[test]
    fn token_dependency_preserves_order() {
        let i = Interner::new();
        let fa = QualifiedRef::root(i.intern("write_a"));
        let fb = QualifiedRef::root(i.intern("write_b"));
        let token = QualifiedRef::root(i.intern("db"));

        let mut fn_metadata = FxHashMap::default();
        let token_effect = Effect::Resolved(EffectSet {
            writes: std::collections::BTreeSet::from([EffectTarget::Token(token)]),
            ..Default::default()
        });
        for &qref in &[fa, fb] {
            fn_metadata.insert(
                qref,
                FnMetadata {
                    ty: Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::Unit),
                        captures: vec![],
                        effect: token_effect.clone(),
                    },
                    hint: None,
                },
            );
        }

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::Spawn {
                    dst: v(2),
                    callee: Callee::Direct(fb),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(3),
                    src: v(2),
                    context_defs: vec![],
                },
                InstKind::Return(v(3)),
            ],
            4,
        );
        cfg.val_types
            .insert(v(0), Ty::Handle(Box::new(Ty::Unit), token_effect.clone()));
        cfg.val_types
            .insert(v(2), Ty::Handle(Box::new(Ty::Unit), token_effect.clone()));

        run(&mut cfg, &fn_metadata);

        // With shared token: spawn_a must come before spawn_b (original order preserved).
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

        // Token forces sequential: a before b entirely.
        assert!(spawn_a < spawn_b, "token: spawn_a before spawn_b");
        assert!(eval_a < eval_b, "token: eval_a before eval_b");
    }

    // ── No-op: no spawns, no change ─────────────────────────────────

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
                InstKind::Return(v(2)),
            ],
            3,
        );

        let original: Vec<_> = all_insts(&cfg)
            .iter()
            .map(|i| std::mem::discriminant(&i.kind))
            .collect();
        run(&mut cfg, &FxHashMap::default());
        let after: Vec<_> = all_insts(&cfg)
            .iter()
            .map(|i| std::mem::discriminant(&i.kind))
            .collect();

        assert_eq!(original, after, "no spawns → order unchanged");
    }

    // ── Use-def chain prevents wrong reorder ────────────────────────

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
                InstKind::Return(v(1)),
            ],
            2,
        );

        run(&mut cfg, &FxHashMap::default());

        let const_idx = find_idx(&cfg, |k| matches!(k, InstKind::Const { .. })).unwrap();
        let binop_idx = find_idx(&cfg, |k| matches!(k, InstKind::BinOp { .. })).unwrap();
        assert!(const_idx < binop_idx);
    }
}
