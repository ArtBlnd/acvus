use std::collections::VecDeque;

use crate::graph::ContextId;
use crate::ir::{InstKind, Label, MirModule, ValueId};
use acvus_ast::Literal;
use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::cfg::{Cfg, Terminator};
use crate::analysis::dataflow::{BooleanDomain, DataflowState, forward_analysis};
use crate::analysis::domain::AbstractValue;
use crate::analysis::val_def::ValDefMap;
use crate::analysis::value_transfer::ValueDomainTransfer;

/// Context keys partitioned by reachability confidence.
#[derive(Debug, Clone, Default)]
pub struct ContextKeyPartition {
    /// Keys on unconditionally reachable paths -- fetch upfront.
    pub eager: FxHashSet<ContextId>,
    /// Keys behind unknown branch conditions -- resolve lazily via coroutine.
    pub lazy: FxHashSet<ContextId>,
    /// Known keys that appear on reachable (non-dead) paths.
    /// These are excluded from eager/lazy (already resolved for orchestration)
    /// but tracked separately for UI discovery.
    pub reachable_known: FxHashSet<ContextId>,
    /// Keys in dead (pruned) branches -- not needed at runtime, but the
    /// typechecker still sees these references and needs their types injected.
    /// Callers should include these in type injection but NOT in unresolved params.
    pub pruned: FxHashSet<ContextId>,
}

/// A known context value for branch pruning.
/// Extends `Literal` to also cover variant (tagged union) values.
#[derive(Debug, Clone)]
pub enum KnownValue {
    Literal(Literal),
    Variant {
        tag: Astr,
        payload: Option<Box<KnownValue>>,
    },
}

/// Determine which context keys are actually needed by a MIR module,
/// given a set of already-known context values.
///
/// Performs forward reachability from the entry block, evaluating branch
/// conditions where possible (when the condition depends on a known context
/// value). Dead branches are pruned, and only `ContextLoad` instructions
/// on live paths are collected.
///
/// Returns context keys that are referenced on live paths and are NOT
/// already in `known`.
pub fn reachable_context_keys(
    module: &MirModule,
    known: &FxHashMap<ContextId, KnownValue>,
    val_def: &ValDefMap,
) -> FxHashSet<ContextId> {
    let p = partition_context_keys(module, known, val_def);
    let mut all = p.eager;
    all.extend(p.lazy);
    all
}

/// Partition context keys into eager (definitely needed) and lazy
/// (conditionally needed behind unknown branches).
///
/// - **eager**: on paths reachable through unconditional jumps or known
///   branch conditions -- safe to pre-fetch.
/// - **lazy**: on paths reachable only through unknown branch conditions
///   -- resolve on-demand via coroutine.
pub fn partition_context_keys(
    module: &MirModule,
    known: &FxHashMap<ContextId, KnownValue>,
    val_def: &ValDefMap,
) -> ContextKeyPartition {
    let mut partition = ContextKeyPartition::default();

    partition_from_body(
        &module.main.insts,
        &module.main.val_types,
        known,
        val_def,
        &mut partition,
    );

    // Closures: conservatively treat all context loads as lazy
    for closure in module.closures.values() {
        for inst in &closure.insts {
            if let InstKind::ContextProject { id, .. } = &inst.kind {
                if known.contains_key(id) {
                    partition.reachable_known.insert(*id);
                } else {
                    partition.lazy.insert(*id);
                }
            }
        }
    }

    // eager wins over lazy
    partition.lazy.retain(|k| !partition.eager.contains(k));
    partition
}

/// Reachability level for a block.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Reach {
    Unreachable,
    /// Reachable only through unknown branch conditions.
    Conditional,
    /// Reachable through unconditional jumps or known branch conditions.
    Definite,
}

fn partition_from_body(
    insts: &[crate::ir::Inst],
    val_types: &FxHashMap<ValueId, crate::ty::Ty>,
    known: &FxHashMap<ContextId, KnownValue>,
    _val_def: &ValDefMap,
    partition: &mut ContextKeyPartition,
) {
    let cfg = Cfg::build(insts);
    if cfg.blocks.is_empty() {
        return;
    }

    // Run dataflow analysis
    let transfer = ValueDomainTransfer {
        val_types,
        known_context: known,
    };
    let dataflow = forward_analysis(&cfg, insts, &transfer, DataflowState::new());

    // Compute reach levels using dataflow results
    let reach = compute_reach(&cfg, &dataflow);

    // Collect ContextLoads by reach level
    for (i, block) in cfg.blocks.iter().enumerate() {
        let block_reach = reach[i];
        for &inst_idx in &block.inst_indices {
            if let InstKind::ContextProject { id, .. } = &insts[inst_idx].kind {
                match block_reach {
                    Reach::Unreachable => {
                        partition.pruned.insert(*id);
                    }
                    _ => {
                        if known.contains_key(id) {
                            partition.reachable_known.insert(*id);
                        } else {
                            match block_reach {
                                Reach::Definite => partition.eager.insert(*id),
                                Reach::Conditional => partition.lazy.insert(*id),
                                Reach::Unreachable => unreachable!(),
                            };
                        }
                    }
                }
            }
        }
    }
}

fn compute_reach(
    cfg: &Cfg,
    dataflow: &crate::analysis::dataflow::DataflowResult<AbstractValue>,
) -> Vec<Reach> {
    let n = cfg.blocks.len();
    let mut reach = vec![Reach::Unreachable; n];
    let mut queue = VecDeque::new();

    reach[0] = Reach::Definite;
    queue.push_back(0);

    while let Some(idx) = queue.pop_front() {
        let block = &cfg.blocks[idx];
        let mut block_reach = reach[idx];

        // Merge point upgrade: the match structure guarantees this block
        // is reached whenever the first arm's test block is reached.
        if let Some(source_label) = block.merge_of
            && let Some(&source_idx) = cfg.label_to_block.get(&source_label)
            && reach[source_idx.0] > block_reach
        {
            block_reach = reach[source_idx.0];
            reach[idx] = block_reach;
        }

        match &block.terminator {
            Terminator::Jump { target, .. } => {
                enqueue_reach(*target, block_reach, cfg, &mut reach, &mut queue);
            }
            Terminator::JumpIf {
                cond,
                then_label,
                else_label,
                ..
            } => {
                let cond_val = dataflow.block_exit[idx].get(*cond);
                match cond_val.as_definite_bool() {
                    Some(true) => {
                        enqueue_reach(*then_label, block_reach, cfg, &mut reach, &mut queue);
                    }
                    Some(false) => {
                        enqueue_reach(*else_label, block_reach, cfg, &mut reach, &mut queue);
                    }
                    None => {
                        enqueue_reach(*then_label, Reach::Conditional, cfg, &mut reach, &mut queue);
                        enqueue_reach(*else_label, Reach::Conditional, cfg, &mut reach, &mut queue);
                    }
                }
            }
            Terminator::IterStep { done, .. } => {
                // Fallthrough.
                let next = idx + 1;
                if next < n && block_reach > reach[next] {
                    reach[next] = block_reach;
                    queue.push_back(next);
                }
                // Done branch.
                enqueue_reach(*done, block_reach, cfg, &mut reach, &mut queue);
            }
            Terminator::Fallthrough => {
                let next = idx + 1;
                if next < n && block_reach > reach[next] {
                    reach[next] = block_reach;
                    queue.push_back(next);
                }
            }
            Terminator::Return => {}
        }
    }

    reach
}

fn enqueue_reach(
    label: Label,
    new_reach: Reach,
    cfg: &Cfg,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ContextId;
    use crate::ir::{DebugInfo, Inst, MirBody};
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

    fn build_val_def(module: &MirModule) -> ValDefMap {
        use crate::analysis::val_def::ValDefMapAnalysis;
        use crate::pass::AnalysisPass;
        ValDefMapAnalysis.run(module, ())
    }

    /// No branches -- all context loads are needed.
    #[test]
    fn no_branches_all_needed() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
            inst(InstKind::ContextProject {
                dst: v1,
                id: id1,
                ty: Ty::error(),
            }),
        ]);
        let val_def = build_val_def(&module);
        let needed = reachable_context_keys(&module, &FxHashMap::default(), &val_def);
        assert_eq!(needed, FxHashSet::from_iter([id0, id1]));
    }

    /// Known context key is excluded from needed set.
    #[test]
    fn known_key_excluded() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
            inst(InstKind::ContextProject {
                dst: v1,
                id: id1,
                ty: Ty::error(),
            }),
        ]);
        let val_def = build_val_def(&module);
        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("alice".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);
        assert_eq!(needed, FxHashSet::from_iter([id1]));
    }

    /// Match on known context value -- dead branch pruned.
    #[test]
    fn branch_then_taken() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v2,
                id: id1,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject {
                dst: v3,
                id: id2,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v3)),
        ]);

        let val_def = build_val_def(&module);
        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("search".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains(&id1));
        assert!(!needed.contains(&id2));
        assert!(!needed.contains(&id0)); // already known
    }

    /// Match on known context value -- else branch taken.
    #[test]
    fn branch_else_taken() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v2,
                id: id1,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject {
                dst: v3,
                id: id2,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v3)),
        ]);

        let val_def = build_val_def(&module);
        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("other".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(!needed.contains(&id1));
        assert!(needed.contains(&id2));
    }

    /// Unknown condition -- both branches are live (conservative).
    #[test]
    fn unknown_condition_both_live() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v2,
                id: id1,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject {
                dst: v3,
                id: id2,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v3)),
        ]);

        let val_def = build_val_def(&module);
        // mode is NOT known -> can't evaluate condition
        let needed = reachable_context_keys(&module, &FxHashMap::default(), &val_def);

        assert!(needed.contains(&id0));
        assert!(needed.contains(&id1));
        assert!(needed.contains(&id2));
    }

    /// Nested match -- chained dead branch elimination.
    #[test]
    fn nested_match_known_condition() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v2,
                id: id1,
                ty: Ty::error(),
            }),
            inst(InstKind::Jump {
                label: Label(0),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject {
                dst: v3,
                id: id2,
                ty: Ty::error(),
            }),
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

        let val_def = build_val_def(&module);
        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("admin".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains(&id1));
        assert!(!needed.contains(&id2));
    }

    /// Range test with known value.
    #[test]
    fn range_condition_evaluated() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v2,
                id: id1,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v2)),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject {
                dst: v3,
                id: id2,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v3)),
        ]);

        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::Int(5)))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains(&id1));
        assert!(!needed.contains(&id2));
    }

    /// Multi-arm match -- chained tests, middle arm matched.
    #[test]
    fn multi_arm_match_middle() {
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v2,
                id: id1,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v4,
                id: id2,
                ty: Ty::error(),
            }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            inst(InstKind::BlockLabel {
                label: Label(40),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject {
                dst: v5,
                id: id3,
                ty: Ty::error(),
            }),
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

        let val_def = build_val_def(&module);
        let known =
            FxHashMap::from_iter([(id0, KnownValue::Literal(Literal::String("user".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

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

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();

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
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v2,
                    id: id1,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v3,
                    id: id2,
                    ty: Ty::error(),
                }),
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

        let val_def = build_val_def(&module);
        let needed = reachable_context_keys(&module, &FxHashMap::default(), &val_def);

        assert!(needed.contains(&id0));
        assert!(needed.contains(&id2));
        assert!(!needed.contains(&id1));
    }

    /// Single-variant enum: TestVariant for that variant is always true.
    #[test]
    fn single_variant_enum_always_true() {
        let i = Interner::new();
        let only = i.intern("Only");

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();

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
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v2,
                    id: id1,
                    ty: Ty::error(),
                }),
                inst(InstKind::Return(v2)),
                inst(InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject {
                    dst: v3,
                    id: id2,
                    ty: Ty::error(),
                }),
                inst(InstKind::Return(v3)),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

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

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();

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
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v2,
                    id: id1,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v4,
                    id: id2,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v5,
                    id: id3,
                    ty: Ty::error(),
                }),
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

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

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

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();

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
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v2,
                    id: id1,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v4,
                    id: id2,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v5,
                    id: id3,
                    ty: Ty::error(),
                }),
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

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

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

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id10 = ContextId::alloc();

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
                    id: id10,
                    ty: Ty::error(),
                }),
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
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
                inst(InstKind::BlockLabel {
                    label: Label(10),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject {
                    dst: v2,
                    id: id1,
                    ty: Ty::error(),
                }),
                inst(InstKind::Return(v2)),
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextProject {
                    dst: v3,
                    id: id2,
                    ty: Ty::error(),
                }),
                inst(InstKind::Return(v3)),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

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

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();

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
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v2,
                    id: id1,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v3,
                    id: id2,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v4,
                    id: id3,
                    ty: Ty::error(),
                }),
                inst(InstKind::Return(v4)),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

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

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();
        let id4 = ContextId::alloc();

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
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v10,
                    id: id1,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v12,
                    id: id2,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v13,
                    id: id3,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v14,
                    id: id4,
                    ty: Ty::error(),
                }),
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

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

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

        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();

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
                inst(InstKind::ContextProject {
                    dst: v0,
                    id: id0,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v2,
                    id: id1,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v3,
                    id: id2,
                    ty: Ty::error(),
                }),
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
                inst(InstKind::ContextProject {
                    dst: v4,
                    id: id3,
                    ty: Ty::error(),
                }),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        // Output is known to be OOC -> Normal arm should be pruned
        let known = FxHashMap::from_iter([(
            id0,
            KnownValue::Variant {
                tag: ooc,
                payload: None,
            },
        )]);
        let p = partition_context_keys(&module, &known, &val_def);

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
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();
        let id4 = ContextId::alloc();
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
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
            inst(InstKind::ContextProject {
                dst: v1,
                id: id1,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v10,
                id: id2,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v11,
                id: id3,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v12,
                id: id4,
                ty: Ty::error(),
            }),
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

        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([
            (id0, KnownValue::Literal(Literal::String("user".into()))),
            (id1, KnownValue::Literal(Literal::Int(5))),
        ]);
        let needed = reachable_context_keys(&module, &known, &val_def);

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
        let id0 = ContextId::alloc();
        let id1 = ContextId::alloc();
        let id2 = ContextId::alloc();
        let id3 = ContextId::alloc();
        let mut vf = LocalFactory::<ValueId>::new();
        let v0 = vf.next();
        let v1 = vf.next();
        let v2 = vf.next();
        let v3 = vf.next();
        let v4 = vf.next();
        let v5 = vf.next();
        let v6 = vf.next();
        let module = make_module(vec![
            inst(InstKind::ContextProject {
                dst: v0,
                id: id0,
                ty: Ty::error(),
            }),
            inst(InstKind::ContextProject {
                dst: v1,
                id: id1,
                ty: Ty::error(),
            }),
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
            inst(InstKind::ContextProject {
                dst: v5,
                id: id2,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v5)),
            // high arm -> live
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextProject {
                dst: v6,
                id: id3,
                ty: Ty::error(),
            }),
            inst(InstKind::Return(v6)),
        ]);

        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([
            (id0, KnownValue::Literal(Literal::String("alice".into()))),
            (id1, KnownValue::Literal(Literal::Int(80))),
        ]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(!needed.contains(&id2));
        assert!(needed.contains(&id3));
    }
}
