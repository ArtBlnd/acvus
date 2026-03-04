use std::collections::{HashMap, HashSet, VecDeque};

use acvus_ast::{BinOp, Literal, RangeKind};
use acvus_mir::ir::{InstKind, Label, MirModule, ValueId};

use crate::analysis::val_def::ValDefMap;

/// Context keys partitioned by reachability confidence.
#[derive(Debug, Clone, Default)]
pub struct ContextKeyPartition {
    /// Keys on unconditionally reachable paths — fetch upfront.
    pub eager: HashSet<String>,
    /// Keys behind unknown branch conditions — resolve lazily via coroutine.
    pub lazy: HashSet<String>,
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
    known: &HashMap<String, Literal>,
    val_def: &ValDefMap,
) -> HashSet<String> {
    let p = partition_context_keys(module, known, val_def);
    let mut all = p.eager;
    all.extend(p.lazy);
    all
}

/// Partition context keys into eager (definitely needed) and lazy
/// (conditionally needed behind unknown branches).
///
/// - **eager**: on paths reachable through unconditional jumps or known
///   branch conditions — safe to pre-fetch.
/// - **lazy**: on paths reachable only through unknown branch conditions
///   — resolve on-demand via coroutine.
pub fn partition_context_keys(
    module: &MirModule,
    known: &HashMap<String, Literal>,
    val_def: &ValDefMap,
) -> ContextKeyPartition {
    let mut partition = ContextKeyPartition::default();

    partition_from_body(&module.main.insts, known, val_def, &mut partition);

    // Closures: conservatively treat all context loads as lazy
    for closure in module.closures.values() {
        for inst in &closure.body.insts {
            if let InstKind::ContextLoad { name, .. } = &inst.kind
                && !known.contains_key(name)
            {
                partition.lazy.insert(name.clone());
            }
        }
    }

    // eager wins over lazy
    partition.lazy.retain(|k| !partition.eager.contains(k));
    partition
}

/// Block in the linear instruction stream.
struct Block {
    label: Option<Label>,
    /// Instruction indices (non-terminator)
    insts: Vec<usize>,
    terminator: Term,
}

enum Term {
    Jump(Label),
    JumpIf {
        cond: ValueId,
        then_label: Label,
        else_label: Label,
    },
    Return,
    /// No explicit terminator — falls through to next block.
    Fallthrough,
}

fn build_blocks(insts: &[acvus_mir::ir::Inst]) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut label: Option<Label> = None;
    let mut inst_indices = Vec::new();

    for (i, inst) in insts.iter().enumerate() {
        match &inst.kind {
            InstKind::BlockLabel { label: l, .. } => {
                if !inst_indices.is_empty() || label.is_some() {
                    blocks.push(Block {
                        label: label.take(),
                        insts: std::mem::take(&mut inst_indices),
                        terminator: Term::Fallthrough,
                    });
                }
                label = Some(*l);
            }
            InstKind::Jump { label: target, .. } => {
                blocks.push(Block {
                    label: label.take(),
                    insts: std::mem::take(&mut inst_indices),
                    terminator: Term::Jump(*target),
                });
            }
            InstKind::JumpIf {
                cond,
                then_label,
                else_label,
                ..
            } => {
                blocks.push(Block {
                    label: label.take(),
                    insts: std::mem::take(&mut inst_indices),
                    terminator: Term::JumpIf {
                        cond: *cond,
                        then_label: *then_label,
                        else_label: *else_label,
                    },
                });
            }
            InstKind::Return(_) => {
                blocks.push(Block {
                    label: label.take(),
                    insts: std::mem::take(&mut inst_indices),
                    terminator: Term::Return,
                });
            }
            _ => {
                inst_indices.push(i);
            }
        }
    }

    if !inst_indices.is_empty() || label.is_some() {
        blocks.push(Block {
            label,
            insts: inst_indices,
            terminator: Term::Fallthrough,
        });
    }

    blocks
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
    insts: &[acvus_mir::ir::Inst],
    known: &HashMap<String, Literal>,
    val_def: &ValDefMap,
    partition: &mut ContextKeyPartition,
) {
    let blocks = build_blocks(insts);
    if blocks.is_empty() {
        return;
    }

    // label → block index
    let label_to_block: HashMap<Label, usize> = blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.label.map(|l| (l, i)))
        .collect();

    // Forward reachability from entry block (index 0)
    let mut reach = vec![Reach::Unreachable; blocks.len()];
    let mut queue = VecDeque::new();
    reach[0] = Reach::Definite;
    queue.push_back(0);

    while let Some(idx) = queue.pop_front() {
        let block = &blocks[idx];
        let block_reach = reach[idx];

        match &block.terminator {
            Term::Jump(target) => {
                enqueue_reach(
                    *target,
                    block_reach,
                    &label_to_block,
                    &mut reach,
                    &mut queue,
                );
            }
            Term::JumpIf {
                cond,
                then_label,
                else_label,
            } => match try_eval_condition(*cond, insts, val_def, known) {
                Some(true) => {
                    enqueue_reach(
                        *then_label,
                        block_reach,
                        &label_to_block,
                        &mut reach,
                        &mut queue,
                    );
                }
                Some(false) => {
                    enqueue_reach(
                        *else_label,
                        block_reach,
                        &label_to_block,
                        &mut reach,
                        &mut queue,
                    );
                }
                None => {
                    enqueue_reach(
                        *then_label,
                        Reach::Conditional,
                        &label_to_block,
                        &mut reach,
                        &mut queue,
                    );
                    enqueue_reach(
                        *else_label,
                        Reach::Conditional,
                        &label_to_block,
                        &mut reach,
                        &mut queue,
                    );
                }
            },
            Term::Fallthrough => {
                let next = idx + 1;
                if next < blocks.len() && block_reach > reach[next] {
                    reach[next] = block_reach;
                    queue.push_back(next);
                }
            }
            Term::Return => {}
        }
    }

    // Collect ContextLoads by reach level
    for (i, block) in blocks.iter().enumerate() {
        let target = match reach[i] {
            Reach::Definite => &mut partition.eager,
            Reach::Conditional => &mut partition.lazy,
            Reach::Unreachable => continue,
        };
        for &inst_idx in &block.insts {
            if let InstKind::ContextLoad { name, .. } = &insts[inst_idx].kind
                && !known.contains_key(name)
            {
                target.insert(name.clone());
            }
        }
    }
}

fn enqueue_reach(
    label: Label,
    new_reach: Reach,
    label_to_block: &HashMap<Label, usize>,
    reach: &mut [Reach],
    queue: &mut VecDeque<usize>,
) {
    if let Some(&idx) = label_to_block.get(&label)
        && new_reach > reach[idx]
    {
        reach[idx] = new_reach;
        queue.push_back(idx);
    }
}

// ---------------------------------------------------------------------------
// Condition evaluation
// ---------------------------------------------------------------------------

fn try_eval_condition(
    cond: ValueId,
    insts: &[acvus_mir::ir::Inst],
    val_def: &ValDefMap,
    known: &HashMap<String, Literal>,
) -> Option<bool> {
    let &def_idx = val_def.0.get(&cond)?;

    match &insts[def_idx].kind {
        InstKind::TestLiteral { src, value, .. } => {
            let ctx_name = trace_to_context_load(*src, insts, val_def)?;
            let known_val = known.get(&ctx_name)?;
            Some(known_val == value)
        }
        InstKind::TestRange {
            src,
            start,
            end,
            kind,
            ..
        } => {
            let ctx_name = trace_to_context_load(*src, insts, val_def)?;
            let known_val = known.get(&ctx_name)?;
            let Literal::Int(v) = known_val else {
                return None;
            };
            Some(in_range(*v, *start, *end, *kind))
        }
        InstKind::BinOp {
            op: BinOp::And,
            left,
            right,
            ..
        } => {
            let l = try_eval_condition(*left, insts, val_def, known)?;
            if !l {
                return Some(false);
            }
            try_eval_condition(*right, insts, val_def, known)
        }
        InstKind::BinOp {
            op: BinOp::Or,
            left,
            right,
            ..
        } => {
            let l = try_eval_condition(*left, insts, val_def, known)?;
            if l {
                return Some(true);
            }
            try_eval_condition(*right, insts, val_def, known)
        }
        _ => None,
    }
}

fn trace_to_context_load(
    val: ValueId,
    insts: &[acvus_mir::ir::Inst],
    val_def: &ValDefMap,
) -> Option<String> {
    let &idx = val_def.0.get(&val)?;
    match &insts[idx].kind {
        InstKind::ContextLoad { name, .. } => Some(name.clone()),
        _ => None,
    }
}

fn in_range(v: i64, start: i64, end: i64, kind: RangeKind) -> bool {
    match kind {
        RangeKind::Exclusive => v >= start && v < end,
        RangeKind::InclusiveEnd => v >= start && v <= end,
        RangeKind::ExclusiveStart => v > start && v <= end,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::Span;
    use acvus_mir::ir::{DebugInfo, Inst, MirBody};

    fn make_module(insts: Vec<Inst>) -> MirModule {
        MirModule {
            main: MirBody {
                insts,
                val_types: HashMap::new(),
                debug: DebugInfo::new(),
                val_count: 0,
                label_count: 0,
            },
            closures: HashMap::new(),
            tag_names: Vec::new(),
            extern_names: HashMap::new(),
        }
    }

    fn inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::new(0, 0),
            kind,
        }
    }

    fn build_val_def(module: &MirModule) -> ValDefMap {
        use crate::AnalysisPass;
        use crate::analysis::val_def::ValDefMapAnalysis;
        ValDefMapAnalysis.run(module, ())
    }

    /// No branches — all context loads are needed.
    #[test]
    fn no_branches_all_needed() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "user".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(1),
                name: "role".into(),
                bindings: Vec::new(),
            }),
        ]);
        let val_def = build_val_def(&module);
        let needed = reachable_context_keys(&module, &HashMap::new(), &val_def);
        assert_eq!(needed, HashSet::from(["user".into(), "role".into()]));
    }

    /// Known context key is excluded from needed set.
    #[test]
    fn known_key_excluded() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "user".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(1),
                name: "role".into(),
                bindings: Vec::new(),
            }),
        ]);
        let val_def = build_val_def(&module);
        let known = HashMap::from([("user".into(), Literal::String("alice".into()))]);
        let needed = reachable_context_keys(&module, &known, &val_def);
        assert_eq!(needed, HashSet::from(["role".into()]));
    }

    /// Match on known context value — dead branch pruned.
    ///
    /// ```text
    /// r0 = context_load @mode           // known: "search"
    /// r1 = test r0 == "search"
    /// jump_if r1 then L1 else L2
    /// L1:  r2 = context_load @query     // live (mode == "search")
    /// L2:  r3 = context_load @fallback  // dead
    /// ```
    #[test]
    fn branch_then_taken() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "mode".into(),
                bindings: Vec::new(),
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
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: "query".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: "fallback".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        let known = HashMap::from([("mode".into(), Literal::String("search".into()))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains("query"));
        assert!(!needed.contains("fallback"));
        assert!(!needed.contains("mode")); // already known
    }

    /// Match on known context value — else branch taken.
    #[test]
    fn branch_else_taken() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "mode".into(),
                bindings: Vec::new(),
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
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: "query".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: "fallback".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        let known = HashMap::from([("mode".into(), Literal::String("other".into()))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(!needed.contains("query"));
        assert!(needed.contains("fallback"));
    }

    /// Unknown condition — both branches are live (conservative).
    #[test]
    fn unknown_condition_both_live() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "mode".into(),
                bindings: Vec::new(),
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
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: "query".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: "fallback".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        // mode is NOT known → can't evaluate condition
        let needed = reachable_context_keys(&module, &HashMap::new(), &val_def);

        assert!(needed.contains("mode"));
        assert!(needed.contains("query"));
        assert!(needed.contains("fallback"));
    }

    /// Nested match — chained dead branch elimination.
    ///
    /// ```text
    /// r0 = context_load @role       // known: "admin"
    /// test r0 == "admin" → L3 (then), L1 (else)
    /// L3: r2 = context_load @level  // live
    /// L1: r6 = const "guest"        // dead
    /// ```
    #[test]
    fn nested_match_known_condition() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "role".into(),
                bindings: Vec::new(),
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
            // L3: admin branch
            inst(InstKind::BlockLabel {
                label: Label(3),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: "level".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Jump {
                label: Label(0),
                args: vec![],
            }),
            // L1: else branch
            inst(InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: "guest_data".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Jump {
                label: Label(0),
                args: vec![],
            }),
            // L0: merge
            inst(InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            }),
        ]);

        let val_def = build_val_def(&module);
        let known = HashMap::from([("role".into(), Literal::String("admin".into()))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains("level"));
        assert!(!needed.contains("guest_data"));
    }

    /// Range test with known value.
    #[test]
    fn range_condition_evaluated() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "level".into(),
                bindings: Vec::new(),
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
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: "low_data".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(2))),
            inst(InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(3),
                name: "high_data".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Return(ValueId(3))),
        ]);

        let val_def = build_val_def(&module);
        // level = 5, in range [1, 10) → then branch live
        let known = HashMap::from([("level".into(), Literal::Int(5))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(needed.contains("low_data"));
        assert!(!needed.contains("high_data"));
    }

    /// Multi-arm match — chained tests, middle arm matched.
    ///
    /// ```text
    /// r0 = context_load @role       // known: "user"
    /// test r0 == "admin" → L_admin, L_next
    /// L_next: test r0 == "user" → L_user, L_default
    /// L_admin: context_load @admin_data     // dead
    /// L_user:  context_load @user_data      // live
    /// L_default: context_load @default_data // dead
    /// ```
    #[test]
    fn multi_arm_match_middle() {
        let module = make_module(vec![
            inst(InstKind::ContextLoad {
                dst: ValueId(0),
                name: "role".into(),
                bindings: Vec::new(),
            }),
            // Test "admin"
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
            // L10: admin arm
            inst(InstKind::BlockLabel {
                label: Label(10),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(2),
                name: "admin_data".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            // L20: next test
            inst(InstKind::BlockLabel {
                label: Label(20),
                params: vec![],
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
            // L30: user arm
            inst(InstKind::BlockLabel {
                label: Label(30),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(4),
                name: "user_data".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            // L40: default arm
            inst(InstKind::BlockLabel {
                label: Label(40),
                params: vec![],
            }),
            inst(InstKind::ContextLoad {
                dst: ValueId(5),
                name: "default_data".into(),
                bindings: Vec::new(),
            }),
            inst(InstKind::Jump {
                label: Label(99),
                args: vec![],
            }),
            // L99: merge
            inst(InstKind::BlockLabel {
                label: Label(99),
                params: vec![],
            }),
        ]);

        let val_def = build_val_def(&module);
        let known = HashMap::from([("role".into(), Literal::String("user".into()))]);
        let needed = reachable_context_keys(&module, &known, &val_def);

        assert!(!needed.contains("admin_data"));
        assert!(needed.contains("user_data"));
        assert!(!needed.contains("default_data"));
    }
}
