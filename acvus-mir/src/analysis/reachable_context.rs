use std::collections::VecDeque;

use acvus_ast::Literal;
use crate::ir::{InstKind, Label, MirModule, ValueId};
use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::cfg::Cfg;
use crate::analysis::dataflow::{BooleanDomain, DataflowState, forward_analysis};
use crate::analysis::domain::AbstractValue;
use crate::analysis::val_def::ValDefMap;
use crate::analysis::value_transfer::ValueDomainTransfer;

/// Context keys partitioned by reachability confidence.
#[derive(Debug, Clone, Default)]
pub struct ContextKeyPartition {
    /// Keys on unconditionally reachable paths -- fetch upfront.
    pub eager: FxHashSet<Astr>,
    /// Keys behind unknown branch conditions -- resolve lazily via coroutine.
    pub lazy: FxHashSet<Astr>,
    /// Known keys that appear on reachable (non-dead) paths.
    /// These are excluded from eager/lazy (already resolved for orchestration)
    /// but tracked separately for UI discovery.
    pub reachable_known: FxHashSet<Astr>,
    /// Keys in dead (pruned) branches -- not needed at runtime, but the
    /// typechecker still sees these references and needs their types injected.
    /// Callers should include these in type injection but NOT in unresolved params.
    pub pruned: FxHashSet<Astr>,
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
    known: &FxHashMap<Astr, KnownValue>,
    val_def: &ValDefMap,
) -> FxHashSet<Astr> {
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
    known: &FxHashMap<Astr, KnownValue>,
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
        for inst in &closure.body.insts {
            if let InstKind::ContextLoad { name, .. } = &inst.kind {
                if known.contains_key(name) {
                    partition.reachable_known.insert(*name);
                } else {
                    partition.lazy.insert(*name);
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
    known: &FxHashMap<Astr, KnownValue>,
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
            if let InstKind::ContextLoad { name, .. } = &insts[inst_idx].kind {
                match block_reach {
                    Reach::Unreachable => {
                        partition.pruned.insert(*name);
                    }
                    _ => {
                        if known.contains_key(name) {
                            partition.reachable_known.insert(*name);
                        } else {
                            match block_reach {
                                Reach::Definite => partition.eager.insert(*name),
                                Reach::Conditional => partition.lazy.insert(*name),
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
        if let Some(source_label) = block.merge_of {
            if let Some(&source_idx) = cfg.label_to_block.get(&source_label) {
                if reach[source_idx.0] > block_reach {
                    block_reach = reach[source_idx.0];
                    reach[idx] = block_reach;
                }
            }
        }

        match &block.terminator {
            crate::analysis::cfg::Terminator::Jump { target, .. } => {
                enqueue_reach(
                    *target,
                    block_reach,
                    cfg,
                    &mut reach,
                    &mut queue,
                );
            }
            crate::analysis::cfg::Terminator::JumpIf {
                cond,
                then_label,
                else_label,
                ..
            } => {
                let cond_val = dataflow.block_exit[idx].get(*cond);
                match cond_val.as_definite_bool() {
                    Some(true) => {
                        enqueue_reach(
                            *then_label,
                            block_reach,
                            cfg,
                            &mut reach,
                            &mut queue,
                        );
                    }
                    Some(false) => {
                        enqueue_reach(
                            *else_label,
                            block_reach,
                            cfg,
                            &mut reach,
                            &mut queue,
                        );
                    }
                    None => {
                        enqueue_reach(
                            *then_label,
                            Reach::Conditional,
                            cfg,
                            &mut reach,
                            &mut queue,
                        );
                        enqueue_reach(
                            *else_label,
                            Reach::Conditional,
                            cfg,
                            &mut reach,
                            &mut queue,
                        );
                    }
                }
            }
            crate::analysis::cfg::Terminator::Fallthrough => {
                let next = idx + 1;
                if next < n && block_reach > reach[next] {
                    reach[next] = block_reach;
                    queue.push_back(next);
                }
            }
            crate::analysis::cfg::Terminator::Return => {}
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
    use acvus_ast::{Literal, RangeKind, Span};
    use crate::ir::{DebugInfo, Inst, MirBody};
    use crate::ty::Ty;
    use acvus_utils::Interner;

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: FxHashMap::default(),
                debug: DebugInfo::new(),
                val_count: 0,
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
        use crate::pass::AnalysisPass;
        use crate::analysis::val_def::ValDefMapAnalysis;
        ValDefMapAnalysis.run(module, ())
    }

    /// No branches -- all context loads are needed.
    #[test]
    fn no_branches_all_needed() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("user"),
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(1),
                name: i.intern("role"),
            }),
        ]);
        let val_def = build_val_def(&module);
        let needed = reachable_context_keys(&module, &FxHashMap::default(), &val_def);
        assert_eq!(needed, FxHashSet::from_iter([i.intern("user"), i.intern("role")]));
    }

    /// Known context key is excluded from needed set.
    #[test]
    fn known_key_excluded() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("user"),
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(1),
                name: i.intern("role"),
            }),
        ]);
        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([(i.intern("user"), KnownValue::Literal(Literal::String("alice".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);
        assert_eq!(needed, FxHashSet::from_iter([i.intern("role")]));
    }

    /// Match on known context value -- dead branch pruned.
    #[test]
    fn branch_then_taken() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("mode"),
            }),
            inst(InstKind::TestLiteral {
                dst: ValueId(1),
                src: ValueId(0),
                value: Literal::String("search".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(1),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: i.intern("query"),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: i.intern("fallback"),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([(i.intern("mode"), KnownValue::Literal(Literal::String("search".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains(&i.intern("query")));
        assert!(!needed.contains(&i.intern("fallback")));
        assert!(!needed.contains(&i.intern("mode"))); // already known
    }

    /// Match on known context value -- else branch taken.
    #[test]
    fn branch_else_taken() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("mode"),
            }),
            inst(InstKind::TestLiteral {
                dst: ValueId(1),
                src: ValueId(0),
                value: Literal::String("search".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(1),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: i.intern("query"),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: i.intern("fallback"),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([(i.intern("mode"), KnownValue::Literal(Literal::String("other".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(!needed.contains(&i.intern("query")));
        assert!(needed.contains(&i.intern("fallback")));
    }

    /// Unknown condition -- both branches are live (conservative).
    #[test]
    fn unknown_condition_both_live() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("mode"),
            }),
            inst(InstKind::TestLiteral {
                dst: ValueId(1),
                src: ValueId(0),
                value: Literal::String("search".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(1),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: i.intern("query"),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: i.intern("fallback"),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        // mode is NOT known -> can't evaluate condition
        let needed = reachable_context_keys(&module, &FxHashMap::default(), &val_def);

        assert!(needed.contains(&i.intern("mode")));
        assert!(needed.contains(&i.intern("query")));
        assert!(needed.contains(&i.intern("fallback")));
    }

    /// Nested match -- chained dead branch elimination.
    #[test]
    fn nested_match_known_condition() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("role"),
            }),
            inst(InstKind::TestLiteral {
                dst: ValueId(1),
                src: ValueId(0),
                value: Literal::String("admin".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(1),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: i.intern("level"),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: i.intern("guest_data"),
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
        let known = FxHashMap::from_iter([(i.intern("role"), KnownValue::Literal(Literal::String("admin".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains(&i.intern("level")));
        assert!(!needed.contains(&i.intern("guest_data")));
    }

    /// Range test with known value.
    #[test]
    fn range_condition_evaluated() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("level"),
            }),
            inst(InstKind::TestRange {
                dst: ValueId(1),
                src: ValueId(0),
                start: 1,
                end: 10,
                kind: RangeKind::Exclusive,
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(1),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: i.intern("low_data"),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: i.intern("high_data"),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([(i.intern("level"), KnownValue::Literal(Literal::Int(5)))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains(&i.intern("low_data")));
        assert!(!needed.contains(&i.intern("high_data")));
    }

    /// Multi-arm match -- chained tests, middle arm matched.
    #[test]
    fn multi_arm_match_middle() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("role"),
            }),
            inst(InstKind::TestLiteral {
                dst: ValueId(1),
                src: ValueId(0),
                value: Literal::String("admin".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(1),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: i.intern("admin_data"),
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
                dst: ValueId(3),
                src: ValueId(0),
                value: Literal::String("user".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(3),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(4),
                name: i.intern("user_data"),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(5),
                name: i.intern("default_data"),
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
        let known = FxHashMap::from_iter([(i.intern("role"), KnownValue::Literal(Literal::String("user".into())))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(!needed.contains(&i.intern("admin_data")));
        assert!(needed.contains(&i.intern("user_data")));
        assert!(!needed.contains(&i.intern("default_data")));
    }

    fn make_module_with_types(
        insts: Vec<Inst>,
        val_types: FxHashMap<ValueId, Ty>,
    ) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types,
                debug: DebugInfo::new(),
                val_count: 0,
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

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(0),
            Ty::Enum {
                name: i.intern("MyEnum"),
                variants: FxHashMap::from_iter([
                    (a, None),
                    (b, None),
                    (i.intern("C"), None),
                ]),
            },
        );

        let module = make_module_with_types(
            vec![
                // %0 = ContextLoad "val"
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("val"),
                    }),
                // %1 = TestVariant(%0, "D")  -- D not in {A,B,C} -> always false
                inst(InstKind::TestVariant {
                    dst: ValueId(1),
                    src: ValueId(0),
                    tag: d,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(2),
                    name: i.intern("dead_data"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(3),
                    name: i.intern("live_data"),
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

        assert!(needed.contains(&i.intern("val")));
        assert!(needed.contains(&i.intern("live_data")));
        assert!(!needed.contains(&i.intern("dead_data")));
    }

    /// Single-variant enum: TestVariant for that variant is always true.
    #[test]
    fn single_variant_enum_always_true() {
        let i = Interner::new();
        let only = i.intern("Only");

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(0),
            Ty::Enum {
                name: i.intern("Wrapper"),
                variants: FxHashMap::from_iter([(only, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("w"),
                    }),
                inst(InstKind::TestVariant {
                    dst: ValueId(1),
                    src: ValueId(0),
                    tag: only,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(2),
                    name: i.intern("then_data"),
                    }),
                inst(InstKind::Return(ValueId(2))),
                inst(InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextLoad {
                    dst: ValueId(3),
                    name: i.intern("else_data"),
                    }),
                inst(InstKind::Return(ValueId(3))),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

        // then_data is eager (single variant -> always matches)
        assert!(p.eager.contains(&i.intern("then_data")));
        // else_data is unreachable
        assert!(!p.eager.contains(&i.intern("else_data")));
        assert!(!p.lazy.contains(&i.intern("else_data")));
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

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(0),
            Ty::Enum {
                name: i.intern("ABC"),
                variants: FxHashMap::from_iter([
                    (a, None),
                    (b, None),
                    (c, None),
                ]),
            },
        );

        let module = make_module_with_types(
            vec![
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("src"),
                    }),
                // TestVariant A
                inst(InstKind::TestVariant {
                    dst: ValueId(1),
                    src: ValueId(0),
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(2),
                    name: i.intern("data_a"),
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
                    dst: ValueId(3),
                    src: ValueId(0),
                    tag: b,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(3),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(4),
                    name: i.intern("data_b"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(5),
                    name: i.intern("data_c"),
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
        assert!(p.eager.contains(&i.intern("src")));
        // All arms are conditional (variant test can't be resolved without known value)
        assert!(p.lazy.contains(&i.intern("data_a")));
        assert!(p.lazy.contains(&i.intern("data_b")));
        assert!(p.lazy.contains(&i.intern("data_c")));
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

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(0),
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([
                    (a, None),
                    (b, None),
                ]),
            },
        );

        let module = make_module_with_types(
            vec![
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("src"),
                    }),
                // Test A
                inst(InstKind::TestVariant {
                    dst: ValueId(1),
                    src: ValueId(0),
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(2),
                    name: i.intern("data_a"),
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
                    dst: ValueId(3),
                    src: ValueId(0),
                    tag: c,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(3),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(4),
                    name: i.intern("data_c"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(5),
                    name: i.intern("data_fallback"),
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
        assert!(p.eager.contains(&i.intern("src")));
        // A arm: conditional (we don't know if it's A or B)
        assert!(p.lazy.contains(&i.intern("data_a")));
        // C arm: dead (C not in enum type)
        assert!(!p.eager.contains(&i.intern("data_c")));
        assert!(!p.lazy.contains(&i.intern("data_c")));
        // fallback: reached when A fails -> conditional, AND when C fails -> definite from Label(20)
        // Label(20) itself is conditional (reached from else of A test).
        // TestVariant(C) is Some(false), so only else_label(40) is enqueued with Label(20)'s reach.
        // Label(20) is Conditional -> Label(40) inherits Conditional.
        assert!(p.lazy.contains(&i.intern("data_fallback")));
    }

    /// Partition: eager vs lazy with enum variant type pruning.
    /// Match on enum {A, B}: test A -> test D(dead) -> fallback.
    /// A arm is conditional, D arm is dead, fallback is definite-from-else.
    #[test]
    fn partition_enum_eager_lazy() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(0),
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([(a, None), (b, None)]),
            },
        );

        // Unconditional context load, then branch on A
        let module = make_module_with_types(
            vec![
                // Eager load before any branch
                inst(InstKind::ContextLoad {
                    dst: ValueId(10),
                    name: i.intern("pre"),
                    }),
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("src"),
                    }),
                inst(InstKind::TestVariant {
                    dst: ValueId(1),
                    src: ValueId(0),
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(2),
                    name: i.intern("a_data"),
                    }),
                inst(InstKind::Return(ValueId(2))),
                inst(InstKind::BlockLabel {
                    label: Label(20),
                    params: vec![],
                    merge_of: None,
                }),
                inst(InstKind::ContextLoad {
                    dst: ValueId(3),
                    name: i.intern("b_data"),
                    }),
                inst(InstKind::Return(ValueId(3))),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

        // pre and src are eager (before branch)
        assert!(p.eager.contains(&i.intern("pre")));
        assert!(p.eager.contains(&i.intern("src")));
        // Both arms are conditional (can't resolve TestVariant without known value)
        assert!(p.lazy.contains(&i.intern("a_data")));
        assert!(p.lazy.contains(&i.intern("b_data")));
        assert!(!p.eager.contains(&i.intern("a_data")));
        assert!(!p.eager.contains(&i.intern("b_data")));
    }

    /// Match merge point upgrades reachability to Definite.
    #[test]
    fn merge_point_upgrades_to_definite() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(0),
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([(a, None), (b, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                // Entry: load scrutinee then jump to first test
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("scrutinee"),
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
                    dst: ValueId(1),
                    src: ValueId(0),
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(2),
                    name: i.intern("arm_data"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(3),
                    name: i.intern("other_arm"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(4),
                    name: i.intern("post_match"),
                    }),
                inst(InstKind::Return(ValueId(4))),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

        // "post_match" should be eager (Definite) because the merge point
        // inherits reachability from the first test block (Label(1) = Definite).
        assert!(
            p.eager.contains(&i.intern("post_match")),
            "post_match should be eager, got lazy={}, eager={}",
            p.lazy.contains(&i.intern("post_match")),
            p.eager.contains(&i.intern("post_match")),
        );
        assert!(!p.lazy.contains(&i.intern("post_match")));

        // arm_data and other_arm are behind unknown branches -> lazy
        assert!(p.lazy.contains(&i.intern("arm_data")));
        assert!(p.lazy.contains(&i.intern("other_arm")));
    }

    /// When the scrutinee block is itself Conditional (behind an unknown branch),
    /// the merge point should inherit Conditional, not Definite.
    #[test]
    fn merge_point_inherits_conditional() {
        let i = Interner::new();
        let a = i.intern("A");
        let b = i.intern("B");

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(10),
            Ty::Enum {
                name: i.intern("AB"),
                variants: FxHashMap::from_iter([(a, None), (b, None)]),
            },
        );

        let module = make_module_with_types(
            vec![
                // Entry: unknown branch -> the match is only conditionally reachable
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("flag"),
                    }),
                inst(InstKind::TestLiteral {
                    dst: ValueId(1),
                    src: ValueId(0),
                    value: Literal::String("yes".into()),
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(10),
                    name: i.intern("scrutinee"),
                    }),
                inst(InstKind::TestVariant {
                    dst: ValueId(11),
                    src: ValueId(10),
                    tag: a,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(11),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(12),
                    name: i.intern("arm_a"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(13),
                    name: i.intern("arm_b"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(14),
                    name: i.intern("post_match"),
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
                inst(InstKind::Return(ValueId(0))),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        let p = partition_context_keys(&module, &FxHashMap::default(), &val_def);

        // Label(1) is Conditional (reached via unknown branch on "flag").
        // The merge point Label(99) inherits Conditional from Label(1).
        // Therefore "post_match" should be lazy (Conditional), not eager.
        assert!(
            p.lazy.contains(&i.intern("post_match")),
            "post_match should be lazy (conditional), got eager={}, lazy={}",
            p.eager.contains(&i.intern("post_match")),
            p.lazy.contains(&i.intern("post_match")),
        );
        assert!(!p.eager.contains(&i.intern("post_match")));

        // scrutinee is also lazy (behind unknown branch)
        assert!(p.lazy.contains(&i.intern("scrutinee")));
    }

    /// Known variant value prunes dead match arms.
    #[test]
    fn known_variant_prunes_match_arms() {
        let i = Interner::new();
        let ooc = i.intern("OOC");
        let normal = i.intern("Normal");

        let mut val_types = FxHashMap::default();
        val_types.insert(
            ValueId(0),
            Ty::Enum {
                name: i.intern("Output"),
                variants: FxHashMap::from_iter([
                    (ooc, None),
                    (normal, None),
                ]),
            },
        );

        let module = make_module_with_types(
            vec![
                // %0 = ContextLoad "Output"
                inst(InstKind::ContextLoad {
                    dst: ValueId(0),
                    name: i.intern("Output"),
                    }),
                // Test Normal
                inst(InstKind::TestVariant {
                    dst: ValueId(1),
                    src: ValueId(0),
                    tag: normal,
                }),
                inst(InstKind::JumpIf {
                    cond: ValueId(1),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(2),
                    name: i.intern("normal_data"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(3),
                    name: i.intern("ooc_data"),
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
                inst(InstKind::ContextLoad {
                    dst: ValueId(4),
                    name: i.intern("post_match"),
                    }),
            ],
            val_types,
        );

        let val_def = build_val_def(&module);
        // Output is known to be OOC -> Normal arm should be pruned
        let known = FxHashMap::from_iter([(
            i.intern("Output"),
            KnownValue::Variant { tag: ooc, payload: None },
        )]);
        let p = partition_context_keys(&module, &known, &val_def);

        // Output is known -> goes to reachable_known
        assert!(p.reachable_known.contains(&i.intern("Output")));
        // Normal arm is dead (TestVariant Normal on OOC value -> false)
        assert!(!p.eager.contains(&i.intern("normal_data")));
        assert!(!p.lazy.contains(&i.intern("normal_data")));
        // OOC arm is live and definite (TestVariant Normal is false -> else branch is definite)
        assert!(p.eager.contains(&i.intern("ooc_data")));
        // post_match is eager (merge_of restores definite)
        assert!(p.eager.contains(&i.intern("post_match")));
    }

    /// Tuple destructuring: context values packed into a tuple, then extracted
    /// via TupleIndex. The dataflow should track through MakeTuple -> TupleIndex
    /// so that TestLiteral on the extracted element can evaluate against the
    /// known context value.
    #[test]
    fn tuple_destructure_multi_arm() {
        let i = Interner::new();
        let module = make_module(vec![
            // Pack two known context values into a tuple
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("role"),
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(1),
                name: i.intern("level"),
            }),
            inst(InstKind::MakeTuple {
                dst: ValueId(2),
                elements: vec![ValueId(0), ValueId(1)],
            }),
            // Extract first element and match on it
            inst(InstKind::TupleIndex {
                dst: ValueId(3),
                tuple: ValueId(2),
                index: 0,
            }),
            // Test "admin"
            inst(InstKind::TestLiteral {
                dst: ValueId(4),
                src: ValueId(3),
                value: Literal::String("admin".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(4),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(10),
                name: i.intern("admin_data"),
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
                dst: ValueId(5),
                src: ValueId(3),
                value: Literal::String("user".into()),
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(5),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(11),
                name: i.intern("user_data"),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(12),
                name: i.intern("default_data"),
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
            (i.intern("role"), KnownValue::Literal(Literal::String("user".into()))),
            (i.intern("level"), KnownValue::Literal(Literal::Int(5))),
        ]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        // admin_data is dead (role != "admin")
        assert!(!needed.contains(&i.intern("admin_data")));
        // user_data is live (role = "user")
        assert!(needed.contains(&i.intern("user_data")));
        // default_data is dead (role = "user", matched above)
        assert!(!needed.contains(&i.intern("default_data")));
    }

    /// Tuple destructuring with second element: TupleIndex(_, 1) extracts the
    /// second context value and uses it for range testing.
    #[test]
    fn tuple_destructure_second_element_range() {
        let i = Interner::new();
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: i.intern("name"),
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(1),
                name: i.intern("score"),
            }),
            inst(InstKind::MakeTuple {
                dst: ValueId(2),
                elements: vec![ValueId(0), ValueId(1)],
            }),
            // Extract second element (score)
            inst(InstKind::TupleIndex {
                dst: ValueId(3),
                tuple: ValueId(2),
                index: 1,
            }),
            inst(InstKind::TestRange {
                dst: ValueId(4),
                src: ValueId(3),
                start: 0,
                end: 50,
                kind: RangeKind::Exclusive,
            }),
            inst(InstKind::JumpIf {
                cond: ValueId(4),
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
            inst(InstKind::ContextLoad {
                dst: ValueId(5),
                name: i.intern("low_data"),
            }),
            inst(InstKind::Return(ValueId(5))),
            // high arm -> live
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(6),
                name: i.intern("high_data"),
            }),
            inst(InstKind::Return(ValueId(6))),
        ]);

        let val_def = build_val_def(&module);
        let known = FxHashMap::from_iter([
            (i.intern("name"), KnownValue::Literal(Literal::String("alice".into()))),
            (i.intern("score"), KnownValue::Literal(Literal::Int(80))),
        ]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(!needed.contains(&i.intern("low_data")));
        assert!(needed.contains(&i.intern("high_data")));
    }
}
