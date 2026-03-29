//! Context key reachability analysis.
//!
//! Determines which `@context` keys a MIR module actually needs at runtime,
//! partitioned by confidence:
//!
//! - **eager**: on unconditionally reachable paths — safe to pre-fetch.
//! - **lazy**: behind unknown branch conditions — resolve on-demand.
//! - **pruned**: in dead branches (known-false conditions) — type-inject only.
//!
//! # Algorithm (two-pass forward analysis)
//!
//! 1. **Value domain** (`forward_analysis` + `ValueDomainTransfer`):
//!    propagate abstract values through the CFG, evaluating branch conditions
//!    where context values are known. This determines which branches are
//!    definitely taken, definitely dead, or unknown.
//!
//! 2. **Reachability** (forward BFS over CFG successors):
//!    propagate `Reach` levels using the value domain's branch verdicts.
//!    Unconditional edges carry the parent's reach; unknown branches
//!    downgrade to `Conditional`; known-dead branches are not followed.
//!
//! Context keys are then collected per-block based on their block's reach level.

use std::collections::VecDeque;

use crate::cfg::{BlockIdx, CfgBody, Terminator, promote};
use crate::analysis::dataflow::{DataflowResult, DataflowState, forward_analysis};
use crate::analysis::domain::AbstractValue;
use crate::analysis::value_transfer::ValueDomainTransfer;
use crate::graph::QualifiedRef;
use crate::ir::{InstKind, MirModule, ValueId};
use acvus_ast::Literal;
use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

// ── Public types ───────────────────────────────────────────────────

/// Context keys partitioned by reachability confidence.
#[derive(Debug, Clone, Default)]
pub struct ContextKeyPartition {
    /// Keys on unconditionally reachable paths — safe to pre-fetch.
    pub eager: FxHashSet<QualifiedRef>,
    /// Keys behind unknown branch conditions — resolve on-demand.
    pub lazy: FxHashSet<QualifiedRef>,
    /// Known keys on reachable paths (already resolved, tracked for UI discovery).
    pub reachable_known: FxHashSet<QualifiedRef>,
    /// Keys in dead branches — type-inject but don't fetch.
    pub pruned: FxHashSet<QualifiedRef>,
}

/// A known context value for branch pruning.
#[derive(Debug, Clone)]
pub enum KnownValue {
    Literal(Literal),
    Variant {
        tag: Astr,
        payload: Option<Box<KnownValue>>,
    },
}

// ── Public API ─────────────────────────────────────────────────────

/// All context keys needed at runtime (eager ∪ lazy).
pub fn reachable_context_keys(
    module: &MirModule,
    known: &FxHashMap<QualifiedRef, KnownValue>,
) -> FxHashSet<QualifiedRef> {
    let p = partition_context_keys(module, known);
    let mut all = p.eager;
    all.extend(p.lazy);
    all
}

/// Partition context keys into eager / lazy / pruned.
pub fn partition_context_keys(
    module: &MirModule,
    known: &FxHashMap<QualifiedRef, KnownValue>,
) -> ContextKeyPartition {
    let mut partition = ContextKeyPartition::default();

    // Main body: full reachability analysis.
    analyze_body(&module.main, known, &mut partition);

    // Closures: conservatively treat all context loads as lazy
    // (closures may be called from any reachable point).
    for closure in module.closures.values() {
        for inst in &closure.insts {
            if let InstKind::ContextProject { ctx, .. } = &inst.kind {
                if known.contains_key(ctx) {
                    partition.reachable_known.insert(*ctx);
                } else {
                    partition.lazy.insert(*ctx);
                }
            }
        }
    }

    // Eager wins over lazy (if a key is on an unconditional path, don't defer it).
    partition.lazy.retain(|k| !partition.eager.contains(k));
    partition
}

// ── Reachability level ─────────────────────────────────────────────

/// How confidently a block is reachable from the entry.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Reach {
    /// Not reachable (dead branch).
    Unreachable,
    /// Reachable only through unknown branch conditions.
    Conditional,
    /// Reachable through unconditional jumps or known branch conditions.
    Definite,
}

// ── Core analysis ──────────────────────────────────────────────────

fn analyze_body(
    body: &crate::ir::MirBody,
    known: &FxHashMap<QualifiedRef, KnownValue>,
    partition: &mut ContextKeyPartition,
) {
    let cfg = promote(body.clone());
    if cfg.blocks.is_empty() {
        return;
    }

    // Pass 1: Value domain — evaluate branch conditions.
    let value_result = {
        let transfer = ValueDomainTransfer {
            val_types: &cfg.val_types,
            known_context: known,
        };
        forward_analysis(&cfg, &transfer, DataflowState::new())
    };

    // Pass 2: Reachability — propagate Reach levels using branch verdicts.
    let reach = compute_reach(&cfg, &value_result);

    // Collect: classify context keys by their block's reach level.
    collect_context_keys(&cfg, &reach, known, partition);
}

/// Forward BFS: propagate `Reach` levels through CFG edges.
///
/// Uses `value_result.block_exit` to evaluate JumpIf conditions:
/// - Definite true/false → follow only the taken branch (same reach).
/// - Unknown → follow both branches (downgrade to Conditional).
fn compute_reach(
    cfg: &CfgBody,
    value_result: &DataflowResult<ValueId, AbstractValue>,
) -> Vec<Reach> {
    let n = cfg.blocks.len();
    let mut reach = vec![Reach::Unreachable; n];
    let mut queue = VecDeque::new();

    reach[0] = Reach::Definite;
    queue.push_back(0usize);

    while let Some(idx) = queue.pop_front() {
        let block = &cfg.blocks[idx];
        let block_reach = resolve_merge_upgrade(idx, block, cfg, &mut reach);

        match &block.terminator {
            Terminator::JumpIf { cond, then_label, else_label, .. } => {
                let verdict = value_result.block_exit[idx].get(*cond).as_definite_bool();
                match verdict {
                    Some(true) => propagate_to(*then_label, block_reach, cfg, &mut reach, &mut queue),
                    Some(false) => propagate_to(*else_label, block_reach, cfg, &mut reach, &mut queue),
                    None => {
                        propagate_to(*then_label, Reach::Conditional, cfg, &mut reach, &mut queue);
                        propagate_to(*else_label, Reach::Conditional, cfg, &mut reach, &mut queue);
                    }
                }
            }
            // All other terminators: propagate to every successor at the same reach level.
            _ => {
                for succ in cfg.successors(BlockIdx(idx)) {
                    if block_reach > reach[succ.0] {
                        reach[succ.0] = block_reach;
                        queue.push_back(succ.0);
                    }
                }
            }
        }
    }

    reach
}

/// Merge-point upgrade: a block marked `merge_of` inherits the reach level
/// of the block that started the match, since the match structure guarantees
/// the merge is reached whenever the first arm's test block is reached.
fn resolve_merge_upgrade(
    idx: usize,
    block: &crate::cfg::Block,
    cfg: &CfgBody,
    reach: &mut [Reach],
) -> Reach {
    let mut block_reach = reach[idx];
    if let Some(source_label) = block.merge_of
        && let Some(&source_idx) = cfg.label_to_block.get(&source_label)
        && reach[source_idx.0] > block_reach
    {
        block_reach = reach[source_idx.0];
        reach[idx] = block_reach;
    }
    block_reach
}

/// Propagate reach level to a labeled successor block.
fn propagate_to(
    label: crate::ir::Label,
    new_reach: Reach,
    cfg: &CfgBody,
    reach: &mut [Reach],
    queue: &mut VecDeque<usize>,
) {
    if let Some(&idx) = cfg.label_to_block.get(&label)
        && new_reach > reach[idx.0]
    {
        reach[idx.0] = new_reach;
        queue.push_back(idx.0);
    }
}

// ── Context key collection ─────────────────────────────────────────

/// Walk all blocks, classify each ContextProject by its block's reach level.
fn collect_context_keys(
    cfg: &CfgBody,
    reach: &[Reach],
    known: &FxHashMap<QualifiedRef, KnownValue>,
    partition: &mut ContextKeyPartition,
) {
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let block_reach = reach[bi];

        for inst in &block.insts {
            let InstKind::ContextProject { ctx, .. } = &inst.kind else {
                continue;
            };

            match block_reach {
                Reach::Unreachable => {
                    partition.pruned.insert(*ctx);
                }
                Reach::Definite | Reach::Conditional => {
                    if known.contains_key(ctx) {
                        partition.reachable_known.insert(*ctx);
                    } else if block_reach == Reach::Definite {
                        partition.eager.insert(*ctx);
                    } else {
                        partition.lazy.insert(*ctx);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::QualifiedRef;
    use crate::ir::{DebugInfo, Inst, Label, MirBody};
    use crate::ty::Ty;
    use acvus_ast::{Literal, RangeKind, Span};
    use acvus_utils::{Interner, LocalFactory};

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                param_regs: Vec::new(),
                capture_regs: Vec::new(),
                debug: DebugInfo::new(),
                val_factory: LocalFactory::new(),
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    }

    fn inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::new(0, 0),
            kind,
        }
    }

    /// Shared interner for tests.
    fn shared_interner() -> &'static Interner {
        use std::sync::LazyLock;
        static INTERNER: LazyLock<Interner> = LazyLock::new(Interner::new);
        &INTERNER
    }

    /// Create a unique QualifiedRef for tests. Uses a shared interner with
    /// monotonically increasing names to ensure uniqueness.
    fn alloc_qref() -> QualifiedRef {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        QualifiedRef::root(shared_interner().intern(&format!("ctx_{n}")))
    }

    /// No branches -- all context loads are needed.
    #[test]
    fn no_branches_all_needed() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::ContextProject { dst: v1, ctx: id1 , volatile: false }),
        ]);
        let needed = reachable_context_keys(&module, &FxHashMap::default());
        assert_eq!(needed, FxHashSet::from_iter([id0, id1]));
    }

    /// Known context key is excluded from needed set.
    #[test]
    fn known_key_excluded() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::ContextProject { dst: v1, ctx: id1 , volatile: false }),
        ]);
        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("alice".into())))]);
        let needed = reachable_context_keys(&module, &known);
        assert_eq!(needed, FxHashSet::from_iter([id1]));
    }

    /// Match on known context value -- dead branch pruned.
    #[test]
    fn branch_then_taken() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::TestLiteral {
                dst: v1,
                src: v0,
                value: Literal::String("search".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v1,
                then_label: Label(1),
                then_args: vec![],
                else_label: Label(2),
                else_args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
            inst(InstKind::Return(v3)),
        ]);

        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("search".into())))]);
        let needed = reachable_context_keys(&module, &known);

        assert!(needed.contains(&id1));
        assert!(!needed.contains(&id2));
        assert!(!needed.contains(&id0)); // already known
    }

    /// Match on known context value -- else branch taken.
    #[test]
    fn branch_else_taken() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::TestLiteral {
                dst: v1,
                src: v0,
                value: Literal::String("search".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v1,
                then_label: Label(1),
                then_args: vec![],
                else_label: Label(2),
                else_args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
            inst(InstKind::Return(v3)),
        ]);

        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("other".into())))]);
        let needed = reachable_context_keys(&module, &known);

        assert!(!needed.contains(&id1));
        assert!(needed.contains(&id2));
    }

    /// Unknown condition -- both branches are live (conservative).
    #[test]
    fn unknown_condition_both_live() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::TestLiteral {
                dst: v1,
                src: v0,
                value: Literal::String("search".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v1,
                then_label: Label(1),
                then_args: vec![],
                else_label: Label(2),
                else_args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
            inst(InstKind::Return(v3)),
        ]);

        // mode is NOT known -> can't evaluate condition
        let needed = reachable_context_keys(&module, &FxHashMap::default());

        assert!(needed.contains(&id0));
        assert!(needed.contains(&id1));
        assert!(needed.contains(&id2));
    }

    /// Nested match -- chained dead branch elimination.
    #[test]
    fn nested_match_known_condition() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::TestLiteral {
                dst: v1,
                src: v0,
                value: Literal::String("admin".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v1,
                then_label: Label(3),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(3),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(0),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(0),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            }),
        ]);

        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("admin".into())))]);
        let needed = reachable_context_keys(&module, &known);

        assert!(needed.contains(&id1));
        assert!(!needed.contains(&id2));
    }

    /// Range test with known value.
    #[test]
    fn range_condition_evaluated() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::TestRange {
                dst: v1,
                src: v0,
                start: 1,
                end: 10,
                kind: RangeKind::Exclusive,
            }),
            inst(InstKind::JumpIf {
                cond: v1,
                then_label: Label(1),
                then_args: vec![],
                else_label: Label(2),
                else_args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
            inst(InstKind::Return(v3)),
        ]);

        let known = FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::Int(5)))]);
        let needed = reachable_context_keys(&module, &known);

        assert!(needed.contains(&id1));
        assert!(!needed.contains(&id2));
    }

    /// Multi-arm match -- chained tests, middle arm matched.
    #[test]
    fn multi_arm_match_middle() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::TestLiteral {
                dst: v1,
                src: v0,
                value: Literal::String("admin".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v1,
                then_label: Label(10),
                then_args: vec![],
                else_label: Label(20),
                else_args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(10),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(20),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::TestLiteral {
                dst: v3,
                src: v0,
                value: Literal::String("user".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v3,
                then_label: Label(30),
                then_args: vec![],
                else_label: Label(40),
                else_args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(30),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v4, ctx: id2 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(40),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v5, ctx: id3 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(99),
                params: vec![],
                merge_of: None,
            }),
        ]);

        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("user".into())))]);
        let needed = reachable_context_keys(&module, &known);

        assert!(!needed.contains(&id1));
        assert!(needed.contains(&id2));
        assert!(!needed.contains(&id3));
    }

    fn make_module_with_types(insts: Vec<Inst>, val_types: FxHashMap<ValueId, Ty>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types,
                param_regs: Vec::new(),
                capture_regs: Vec::new(),
                debug: DebugInfo::new(),
                val_factory: LocalFactory::new(),
                label_count: 0,
            },
            closures: FxHashMap::default(),
        }
    }

    /// Multi-arm enum match: TestVariant(A) -> TestVariant(B) -> fallback.
    /// When type has {A, B, C}, variant D test -> pruned (always false).
    #[test]
    fn enum_variant_nonexistent_pruned() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");
        let d = i.intern("D"); // not in enum

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v0,
            Ty::Enum {
                name: i.intern("MyEnum"),
                variants: FxHashMap::from_iter([(a, None), (b, None), (i.intern("C"), None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                // %0 = ContextLoad "val"
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                // %1 = TestVariant(%0, "D")  -- D not in {A,B,C} -> always false
                inst(InstKind::TestVariant {
                    dst: v1,
                    src: v0,
                    tag: d,
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(10),
                    then_args: vec![],
                    else_label: Label(20),
                    else_args: vec![],
                }),
                // Label(10): D arm -> dead
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // Label(20): else -> live
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(99),
                    params: vec![],
                    merge_of: None,
                }),
            ],
            val_types,
        );

        let needed = reachable_context_keys(&module, &FxHashMap::default());

        assert!(needed.contains(&id0));
        assert!(needed.contains(&id2));
        assert!(!needed.contains(&id1));
    }

    /// Single-variant enum: TestVariant for that variant is always true.
    #[test]
    fn single_variant_enum_always_true() {
        let i = Interner::new();
        let only = i.intern("Only");

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v0,
            Ty::Enum {
                name: i.intern("Wrapper"),
                variants: FxHashMap::from_iter([(only, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                inst(InstKind::TestVariant {
                    dst: v1,
                    src: v0,
                    tag: only,
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(1),
                    then_args: vec![],
                    else_label: Label(2),
                    else_args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
                inst(InstKind::Return(v2)),
                inst(InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
                inst(InstKind::Return(v3)),
            ],
            val_types,
        );

        let p = partition_context_keys(&module, &FxHashMap::default());

        // then_data is eager (single variant -> always matches)
        assert!(p.eager.contains(&id1));
        // else_data is unreachable
        assert!(!p.eager.contains(&id2));
        assert!(!p.lazy.contains(&id2));
    }

    /// Multi-arm enum variant match with type pruning.
    /// Source has {A, B} but match tests A -> C -> fallback.
    /// TestVariant(C) is always false -> C arm is dead, fallback is reached.
    #[test]
    fn enum_multi_arm_unknown_all_conditional() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");
        let c = i.intern("C");

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v0,
            Ty::Enum {
                name: i.intern("ABC"),
                variants: FxHashMap::from_iter([(a, None), (b, None), (c, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                // TestVariant A
                inst(InstKind::TestVariant {
                    dst: v1,
                    src: v0,
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(10),
                    then_args: vec![],
                    else_label: Label(20),
                    else_args: vec![],
                }),
                // A arm
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // else -> test B
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::TestVariant {
                    dst: v3,
                    src: v0,
                    tag: b,
                }),
                inst(InstKind::JumpIf {
                    cond: v3,
                    then_label: Label(30),
                    then_args: vec![],
                    else_label: Label(40),
                    else_args: vec![],
                }),
                // B arm
                inst(InstKind::BlockLabel {
                    label: Label(30),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v4, ctx: id2 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // fallback (catch-all for C)
                inst(InstKind::BlockLabel {
                    label: Label(40),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v5, ctx: id3 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(99),
                    params: vec![],
                    merge_of: None,
                }),
            ],
            val_types,
        );

        let p = partition_context_keys(&module, &FxHashMap::default());

        // src is eager (before any branch)
        assert!(p.eager.contains(&id0));
        // All arms are conditional (variant test can't be resolved without known value)
        assert!(p.lazy.contains(&id1));
        assert!(p.lazy.contains(&id2));
        assert!(p.lazy.contains(&id3));
    }

    /// Multi-arm enum variant match with type pruning.
    /// Source has {A, B} but match tests A -> C -> fallback.
    /// TestVariant(C) is always false -> C arm is dead, fallback is reached.
    #[test]
    fn enum_multi_arm_type_prune_middle() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");
        let c = i.intern("C"); // not in enum

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v0,
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([(a, None), (b, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                // Test A
                inst(InstKind::TestVariant {
                    dst: v1,
                    src: v0,
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(10),
                    then_args: vec![],
                    else_label: Label(20),
                    else_args: vec![],
                }),
                // A arm
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // else -> Test C (not in type!)
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::TestVariant {
                    dst: v3,
                    src: v0,
                    tag: c,
                }),
                inst(InstKind::JumpIf {
                    cond: v3,
                    then_label: Label(30),
                    then_args: vec![],
                    else_label: Label(40),
                    else_args: vec![],
                }),
                // C arm -> dead (C not in {A, B})
                inst(InstKind::BlockLabel {
                    label: Label(30),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v4, ctx: id2 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // fallback -> this is where B goes
                inst(InstKind::BlockLabel {
                    label: Label(40),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v5, ctx: id3 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(99),
                    params: vec![],
                    merge_of: None,
                }),
            ],
            val_types,
        );

        let p = partition_context_keys(&module, &FxHashMap::default());

        // src is eager
        assert!(p.eager.contains(&id0));
        // A arm: conditional (we don't know if it's A or B)
        assert!(p.lazy.contains(&id1));
        // C arm: dead (C not in enum type)
        assert!(!p.eager.contains(&id2));
        assert!(!p.lazy.contains(&id2));
        // fallback: reached when A fails -> conditional, AND when C fails -> definite from Label(20)
        // Label(20) itself is conditional (reached from else of A test).
        // TestVariant(C) is Some(false), so only else_label(40) is enqueued with Label(20)'s reach.
        // Label(20) is Conditional -> Label(40) inherits Conditional.
        assert!(p.lazy.contains(&id3));
    }

    /// Partition: eager vs lazy with enum variant type pruning.
    /// Match on enum {A, B}: test A -> test D(dead) -> fallback.
    /// A arm is conditional, D arm is dead, fallback is definite-from-else.
    #[test]
    fn partition_enum_eager_lazy() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id10 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v_pre = vf.next();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v0,
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([(a, None), (b, None)]),
            },
        );

        // Unconditional context load, then branch on A
        let module = make_module_with_types(
            vec![
                // Eager load before any branch
                inst(InstKind::ContextProject {
                    dst: v_pre,
                    ctx: id10, volatile: false }),
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                inst(InstKind::TestVariant {
                    dst: v1,
                    src: v0,
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(10),
                    then_args: vec![],
                    else_label: Label(20),
                    else_args: vec![],
                }),
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
                inst(InstKind::Return(v2)),
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
                inst(InstKind::Return(v3)),
            ],
            val_types,
        );

        let p = partition_context_keys(&module, &FxHashMap::default());

        // pre and src are eager (before branch)
        assert!(p.eager.contains(&id10));
        assert!(p.eager.contains(&id0));
        // Both arms are conditional (can't resolve TestVariant without known value)
        assert!(p.lazy.contains(&id1));
        assert!(p.lazy.contains(&id2));
        assert!(!p.eager.contains(&id1));
        assert!(!p.eager.contains(&id2));
    }

    /// Match merge point upgrades reachability to Definite.
    #[test]
    fn merge_point_upgrades_to_definite() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v0,
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([(a, None), (b, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                // Entry: load scrutinee then jump to first test
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(1),
                    args: vec![],
                }),
                // Label(1): first arm test block
                inst(InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::TestVariant {
                    dst: v1,
                    src: v0,
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(10),
                    then_args: vec![],
                    else_label: Label(20),
                    else_args: vec![],
                }),
                // Label(10): A arm body
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // Label(20): B arm body
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // Label(99): merge point of the match, merge_of = Label(1)
                inst(InstKind::BlockLabel {
                    label: Label(99),
                    params: vec![],
                    merge_of: Some(Label(1)),
                }),
                inst(InstKind::ContextProject { dst: v4, ctx: id3 , volatile: false }),
                inst(InstKind::Return(v4)),
            ],
            val_types,
        );

        let p = partition_context_keys(&module, &FxHashMap::default());

        // "post_match" should be eager (Definite) because the merge point
        // inherits reachability from the first test block (Label(1) = Definite).
        assert!(
            p.eager.contains(&id3),
            "post_match should be eager, got lazy={}, eager={}",
            p.lazy.contains(&id3),
            p.eager.contains(&id3),
        );
        assert!(!p.lazy.contains(&id3));

        // arm_data and other_arm are behind unknown branches -> lazy
        assert!(p.lazy.contains(&id1));
        assert!(p.lazy.contains(&id2));
    }

    /// When the scrutinee block is itself Conditional (behind an unknown branch),
    /// the merge point should inherit Conditional, not Definite.
    #[test]
    fn merge_point_inherits_conditional() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();
        let id4 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v10 = vf.next();
        let v11 = vf.next();
        let v12 = vf.next();
        let v13 = vf.next();
        let v14 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v10,
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([(a, None), (b, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                // Entry: unknown branch -> the match is only conditionally reachable
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                inst(InstKind::TestLiteral {
                    dst: v1,
                    src: v0,
                    value: Literal::String("yes".into()),
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(1), // goes to the match
                    then_args: vec![],
                    else_label: Label(50), // skips the match entirely
                    else_args: vec![],
                }),
                // Label(1): first arm test (Conditional because flag is unknown)
                inst(InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v10, ctx: id1 , volatile: false }),
                inst(InstKind::TestVariant {
                    dst: v11,
                    src: v10,
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: v11,
                    then_label: Label(10),
                    then_args: vec![],
                    else_label: Label(20),
                    else_args: vec![],
                }),
                // Label(10): A arm
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v12, ctx: id2 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // Label(20): B arm
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v13, ctx: id3 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // Label(99): merge point, merge_of = Label(1)
                inst(InstKind::BlockLabel {
                    label: Label(99),
                    params: vec![],
                    merge_of: Some(Label(1)),
                }),
                inst(InstKind::ContextProject { dst: v14, ctx: id4 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(50),
                    args: vec![],
                }),
                // Label(50): after everything
                inst(InstKind::BlockLabel {
                    label: Label(50),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::Return(v0)),
            ],
            val_types,
        );

        let p = partition_context_keys(&module, &FxHashMap::default());

        // Label(1) is Conditional (reached via unknown branch on "flag").
        // The merge point Label(99) inherits Conditional from Label(1).
        // Therefore "post_match" should be lazy (Conditional), not eager.
        assert!(
            p.lazy.contains(&id4),
            "post_match should be lazy (conditional), got eager={}, lazy={}",
            p.eager.contains(&id4),
            p.lazy.contains(&id4),
        );
        assert!(!p.eager.contains(&id4));

        // scrutinee is also lazy (behind unknown branch)
        assert!(p.lazy.contains(&id1));
    }

    /// Known variant value prunes dead match arms.
    #[test]
    fn known_variant_prunes_match_arms() {
        let i = Interner::new();
        let ooc = i.intern("OOC");
        let normal = i.intern("Normal");

        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();

        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();

        let mut val_types = FxHashMap::default();
        val_types.insert(
            v0,
            Ty::Enum {
                name: i.intern("Output"),
                variants: FxHashMap::from_iter([(ooc, None), (normal, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                // %0 = ContextLoad "Output"
                inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
                // Test Normal
                inst(InstKind::TestVariant {
                    dst: v1,
                    src: v0,
                    tag: normal,
                }),
                inst(InstKind::JumpIf {
                    cond: v1,
                    then_label: Label(10),
                    then_args: vec![],
                    else_label: Label(20),
                    else_args: vec![],
                }),
                // Normal arm
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v2, ctx: id1 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // OOC arm (catch-all)
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject { dst: v3, ctx: id2 , volatile: false }),
                inst(InstKind::Jump {
                    label: Label(99),
                    args: vec![],
                }),
                // merge
                inst(InstKind::BlockLabel {
                    label: Label(99),
                    params: vec![],
                    merge_of: Some(Label(10)),
                }),
                inst(InstKind::ContextProject { dst: v4, ctx: id3 , volatile: false }),
            ],
            val_types,
        );

        // Output is known to be OOC -> Normal arm should be pruned
        let known = FxHashMap::from_iter([(
            id0,
            KnownValue::Variant {
                tag: ooc,
                payload: None,
            },
        )]);
        let p = partition_context_keys(&module, &known);

        // Output is known -> goes to reachable_known
        assert!(p.reachable_known.contains(&id0));
        // Normal arm is dead (TestVariant Normal on OOC value -> false)
        assert!(!p.eager.contains(&id1));
        assert!(!p.lazy.contains(&id1));
        // OOC arm is live and definite (TestVariant Normal is false -> else branch is definite)
        assert!(p.eager.contains(&id2));
        // post_match is eager (merge_of restores definite)
        assert!(p.eager.contains(&id3));
    }

    /// Tuple destructuring: context values packed into a tuple, then extracted
    /// via TupleIndex. The dataflow should track through MakeTuple -> TupleIndex
    /// so that TestLiteral on the extracted element can evaluate against the
    /// known context value.
    #[test]
    fn tuple_destructure_multi_arm() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();
        let id4 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();
        let v10 = vf.next();
        let v11 = vf.next();
        let v12 = vf.next();
        let module = make_module(vec![
            // Pack two known context values into a tuple
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::ContextProject { dst: v1, ctx: id1 , volatile: false }),
            inst(InstKind::MakeTuple {
                dst: v2,
                elements: vec![v0, v1],
            }),
            // Extract first element and match on it
            inst(InstKind::TupleIndex {
                dst: v3,
                tuple: v2,
                index: 0,
            }),
            // Test "admin"
            inst(InstKind::TestLiteral {
                dst: v4,
                src: v3,
                value: Literal::String("admin".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v4,
                then_label: Label(10),
                then_args: vec![],
                else_label: Label(20),
                else_args: vec![],
            }),
            // admin arm -> dead (role = "user", not "admin")
            inst(InstKind::BlockLabel {
                label: Label(10),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v10, ctx: id2 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            // else -> test "user"
            inst(InstKind::BlockLabel {
                label: Label(20),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::TestLiteral {
                dst: v5,
                src: v3,
                value: Literal::String("user".into()),
            }),
            inst(InstKind::JumpIf {
                cond: v5,
                then_label: Label(30),
                then_args: vec![],
                else_label: Label(40),
                else_args: vec![],
            }),
            // user arm -> live (role = "user")
            inst(InstKind::BlockLabel {
                label: Label(30),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v11, ctx: id3 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            // default arm -> dead (role matched "user" above)
            inst(InstKind::BlockLabel {
                label: Label(40),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v12, ctx: id4 , volatile: false }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(99),
                params: vec![],
                merge_of: None,
            }),
        ]);

        let known = FxHashMap::from_iter([
            (id0, KnownValue::Literal(Literal::String("user".into()))),
            (id1, KnownValue::Literal(Literal::Int(5))),
        ]);
        let needed = reachable_context_keys(&module, &known);

        // admin_data is dead (role != "admin")
        assert!(!needed.contains(&id2));
        // user_data is live (role = "user")
        assert!(needed.contains(&id3));
        // default_data is dead (role = "user", matched above)
        assert!(!needed.contains(&id4));
    }

    /// Tuple destructuring with second element: TupleIndex(_, 1) extracts the
    /// second context value and uses it for range testing.
    #[test]
    fn tuple_destructure_second_element_range() {
        let id0 = alloc_qref();
        let id1 = alloc_qref();
        let id2 = alloc_qref();
        let id3 = alloc_qref();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();
        let v6 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject { dst: v0, ctx: id0 , volatile: false }),
            inst(InstKind::ContextProject { dst: v1, ctx: id1 , volatile: false }),
            inst(InstKind::MakeTuple {
                dst: v2,
                elements: vec![v0, v1],
            }),
            // Extract second element (score)
            inst(InstKind::TupleIndex {
                dst: v3,
                tuple: v2,
                index: 1,
            }),
            inst(InstKind::TestRange {
                dst: v4,
                src: v3,
                start: 0,
                end: 50,
                kind: RangeKind::Exclusive,
            }),
            inst(InstKind::JumpIf {
                cond: v4,
                then_label: Label(1),
                then_args: vec![],
                else_label: Label(2),
                else_args: vec![],
            }),
            // low arm -> dead (score = 80, not in [0, 50))
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v5, ctx: id2 , volatile: false }),
            inst(InstKind::Return(v5)),
            // high arm -> live
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject { dst: v6, ctx: id3 , volatile: false }),
            inst(InstKind::Return(v6)),
        ]);

        let known = FxHashMap::from_iter([
            (id0, KnownValue::Literal(Literal::String("alice".into()))),
            (id1, KnownValue::Literal(Literal::Int(80))),
        ]);
        let needed = reachable_context_keys(&module, &known);

        assert!(!needed.contains(&id2));
        assert!(needed.contains(&id3));
    }
}
