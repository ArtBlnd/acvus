//! Instruction scheduling: reorder instructions within basic blocks while
//! preserving data dependencies and side-effect ordering.
//!
//! Algorithm:
//!   1. Split instruction stream into basic blocks (at control flow boundaries).
//!   2. For each block, build a dependency graph.
//!   3. Topological sort with random tie-breaking.
//!   4. Reassemble.

use std::collections::HashSet;

use acvus_mir::ir::{Inst, InstKind, ValueId};
use rand::Rng;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;

pub fn reorder(insts: Vec<Inst>, rng: &mut StdRng) -> Vec<Inst> {
    let blocks = split_basic_blocks(insts);
    let mut out = Vec::new();

    for block in blocks {
        if block.len() <= 2 {
            out.extend(block);
            continue;
        }
        out.extend(schedule_block(block, rng));
    }

    out
}

/// Split into basic blocks. A block boundary occurs at control flow instructions.
fn split_basic_blocks(insts: Vec<Inst>) -> Vec<Vec<Inst>> {
    let mut blocks: Vec<Vec<Inst>> = Vec::new();
    let mut current = Vec::new();

    for inst in insts {
        let is_cf = is_control_flow(&inst.kind);
        let is_label = matches!(inst.kind, InstKind::BlockLabel { .. });

        if is_label && !current.is_empty() {
            blocks.push(std::mem::take(&mut current));
        }

        current.push(inst);

        if is_cf && !is_label {
            blocks.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        blocks.push(current);
    }

    blocks
}

fn is_control_flow(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::BlockLabel { .. }
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::Return(_)
    )
}

/// Schedule a single basic block using dependency-aware topological sort.
fn schedule_block(block: Vec<Inst>, rng: &mut StdRng) -> Vec<Inst> {
    let n = block.len();
    if n <= 1 {
        return block;
    }

    // Build dependency edges: deps[i] = set of instruction indices that must come before i.
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    // Map: ValueId → index of instruction that defines it.
    let mut def_map: FxHashMap<ValueId, usize> = FxHashMap::default();
    // Track last side-effect instruction index for ordering.
    let mut last_side_effect: Option<usize> = None;

    for (i, inst) in block.iter().enumerate() {
        // Data dependency: if this instruction reads a value defined in this block.
        for used in used_vals(&inst.kind) {
            if let Some(&def_idx) = def_map.get(&used) {
                deps[i].insert(def_idx);
            }
        }

        // Side-effect ordering: Yield, VarStore, Call must preserve order.
        if has_side_effect(&inst.kind) {
            if let Some(prev) = last_side_effect {
                deps[i].insert(prev);
            }
            last_side_effect = Some(i);
        }

        // Control flow must be last.
        if is_control_flow(&inst.kind) {
            for j in 0..i {
                deps[i].insert(j);
            }
        }

        // BlockLabel must be first.
        if matches!(inst.kind, InstKind::BlockLabel { .. }) {
            for j in (i + 1)..n {
                deps[j].insert(i);
            }
        }

        // Register definitions.
        if let Some(dst) = defined_val(&inst.kind) {
            def_map.insert(dst, i);
        }
    }

    // Topological sort with random tie-breaking.
    let mut in_degree: Vec<usize> = deps.iter().map(|d| d.len()).collect();
    let mut ready: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    // Reverse map: which instructions depend on i.
    let mut rdeps: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, d) in deps.iter().enumerate() {
        for &dep in d {
            rdeps[dep].push(i);
        }
    }

    while !ready.is_empty() {
        let pick = rng.random_range(0..ready.len());
        let idx = ready.swap_remove(pick);
        order.push(idx);

        for &next in &rdeps[idx] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                ready.push(next);
            }
        }
    }

    // Fallback: if some instructions weren't scheduled (shouldn't happen), append them.
    if order.len() < n {
        let scheduled: HashSet<usize> = order.iter().copied().collect();
        for i in 0..n {
            if !scheduled.contains(&i) {
                order.push(i);
            }
        }
    }

    order.into_iter().map(|i| block[i].clone()).collect()
}

/// Extract ValueIds read by an instruction.
fn used_vals(kind: &InstKind) -> Vec<ValueId> {
    match kind {
        InstKind::Yield(v) => vec![*v],
        InstKind::BinOp { left, right, .. } => vec![*left, *right],
        InstKind::UnaryOp { operand, .. } => vec![*operand],
        InstKind::FieldGet { object, .. } => vec![*object],
        InstKind::Call { args, .. } | InstKind::AsyncCall { args, .. } => args.clone(),
        InstKind::Await { src, .. } => vec![*src],
        InstKind::MakeList { elements, .. } => elements.clone(),
        InstKind::MakeObject { fields, .. } => fields.iter().map(|(_, v)| *v).collect(),
        InstKind::MakeRange { start, end, .. } => vec![*start, *end],
        InstKind::MakeTuple { elements, .. } => elements.clone(),
        InstKind::TupleIndex { tuple, .. } => vec![*tuple],
        InstKind::TestLiteral { src, .. } => vec![*src],
        InstKind::TestListLen { src, .. } => vec![*src],
        InstKind::TestObjectKey { src, .. } => vec![*src],
        InstKind::TestRange { src, .. } => vec![*src],
        InstKind::ListIndex { list, .. } => vec![*list],
        InstKind::ListGet { list, index, .. } => vec![*list, *index],
        InstKind::ListSlice { list, .. } => vec![*list],
        InstKind::ObjectGet { object, .. } => vec![*object],
        InstKind::MakeClosure { captures, .. } => captures.clone(),
        InstKind::CallClosure { closure, args, .. } => {
            let mut v = vec![*closure];
            v.extend(args);
            v
        }
        InstKind::IterInit { src, .. } => vec![*src],
        InstKind::IterNext { iter, .. } => vec![*iter],
        InstKind::VarStore { src, .. } => vec![*src],
        InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend(then_args);
            v.extend(else_args);
            v
        }
        InstKind::Jump { args, .. } => args.clone(),
        InstKind::Return(v) => vec![*v],
        InstKind::Const { .. }
        | InstKind::VarLoad { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Nop => vec![],
        InstKind::ContextLoad { bindings, .. } => bindings.iter().map(|(_, v)| *v).collect(),
        InstKind::MakeVariant { payload, .. } => payload.iter().copied().collect(),
        InstKind::TestVariant { src, .. } => vec![*src],
        InstKind::UnwrapVariant { src, .. } => vec![*src],
    }
}

/// Extract the ValueId defined by an instruction, if any.
fn defined_val(kind: &InstKind) -> Option<ValueId> {
    match kind {
        InstKind::Const { dst, .. }
        | InstKind::ContextLoad { dst, .. }
        | InstKind::VarLoad { dst, .. }
        | InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::FieldGet { dst, .. }
        | InstKind::Call { dst, .. }
        | InstKind::AsyncCall { dst, .. }
        | InstKind::Await { dst, .. }
        | InstKind::MakeList { dst, .. }
        | InstKind::MakeObject { dst, .. }
        | InstKind::MakeRange { dst, .. }
        | InstKind::MakeTuple { dst, .. }
        | InstKind::TupleIndex { dst, .. }
        | InstKind::TestLiteral { dst, .. }
        | InstKind::TestListLen { dst, .. }
        | InstKind::TestObjectKey { dst, .. }
        | InstKind::TestRange { dst, .. }
        | InstKind::ListIndex { dst, .. }
        | InstKind::ListGet { dst, .. }
        | InstKind::ListSlice { dst, .. }
        | InstKind::ObjectGet { dst, .. }
        | InstKind::MakeClosure { dst, .. }
        | InstKind::CallClosure { dst, .. }
        | InstKind::IterInit { dst, .. }
        | InstKind::MakeVariant { dst, .. }
        | InstKind::TestVariant { dst, .. }
        | InstKind::UnwrapVariant { dst, .. } => Some(*dst),
        InstKind::IterNext { dst_value, .. } => Some(*dst_value),
        _ => None,
    }
}

fn has_side_effect(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Yield(_)
            | InstKind::VarStore { .. }
            | InstKind::VarLoad { .. }
            | InstKind::Call { .. }
            | InstKind::AsyncCall { .. }
            | InstKind::CallClosure { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::{BinOp, Literal, Span};
    use rand::SeedableRng;

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    #[test]
    fn preserves_data_dependency() {
        let mut rng = StdRng::seed_from_u64(42);
        let insts = vec![
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: ValueId(0),
                    value: Literal::Int(1),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::Const {
                    dst: ValueId(1),
                    value: Literal::Int(2),
                },
            },
            Inst {
                span: span(),
                kind: InstKind::BinOp {
                    dst: ValueId(2),
                    op: BinOp::Add,
                    left: ValueId(0),
                    right: ValueId(1),
                },
            },
        ];

        // Run many times — the Add must always come after both Consts.
        for _ in 0..20 {
            let result = reorder(insts.clone(), &mut rng);
            let pos = |vid: u32| {
                result
                    .iter()
                    .position(|i| defined_val(&i.kind) == Some(ValueId(vid)))
                    .unwrap()
            };
            assert!(pos(2) > pos(0));
            assert!(pos(2) > pos(1));
        }
    }

    #[test]
    fn preserves_side_effect_order() {
        let mut rng = StdRng::seed_from_u64(99);
        let insts = vec![
            Inst {
                span: span(),
                kind: InstKind::Yield(ValueId(0)),
            },
            Inst {
                span: span(),
                kind: InstKind::Yield(ValueId(1)),
            },
        ];

        for _ in 0..20 {
            let result = reorder(insts.clone(), &mut rng);
            // First Yield(0) must come before Yield(1).
            assert!(matches!(result[0].kind, InstKind::Yield(ValueId(0))));
        }
    }
}
