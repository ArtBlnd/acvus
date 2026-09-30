//! List scheduling within each block, over the block's def-use graph, order
//! tokens included: an instruction is ready once every instruction of the
//! block that defines one of its operands is placed. Position by position,
//! from the block's start, the ready instruction of highest priority is
//! offered a move to that position, and the next one is offered where a
//! move is refused. The priority is the longest def-use path from the
//! instruction to the block's end, each instruction counting one, so the
//! longest chain starts first; ties go to the earlier instruction of the
//! block as it was, so the order is otherwise kept.
//!
//! The instruction standing at the position is always ready, since the
//! body defines every operand before its use and every instruction before
//! the position is placed, and it is placed by leaving it there. So each
//! position is filled, and the block is scheduled in one pass.
//!
//! This transform runs last: nothing after it moves instructions
//! (passes-pipeline §3).
//!
//! Form: F2, loops canonical, in and out; it changes no block.
//!
//! A refused candidate is not offered again until what refused it can have
//! changed: one whose move would change where it runs (compilation §1)
//! waits until it stands at the position, where it is placed without a
//! move, since whether two positions of one block run on the same runs
//! does not depend on which position; one whose move would cross a
//! conflicting write waits until that write is placed, since until then
//! the write stands between the position and it. Any other refusal puts
//! the candidate back for the next position, as before.
//!
//! Cost: O(n + operands) per block of n instructions for its graph and
//! priorities, and O(n log n) for the order: each candidate is offered once
//! per position it is placed at or refused for a reason that can have
//! changed, and each offer is a `move_to` at the editor's cost.
//!
//! Meaning: a move is made only where `move_to` allows it: every operand
//! and use, order tokens included, stays dominated, so the order of effects
//! is exactly the tokens' and is kept (effects §3); a move within one block
//! runs on exactly the runs it did (compilation §1); and no move crosses a
//! write to storage the instruction names (ownership §5). No type changes.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use crate::ir::cfg::edit::{CannotMove, Editor, InstRef, Motion, Position};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, InstId, Module};
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Schedule;

impl BodyTransform for Schedule {
    const NAME: &'static str = "schedule";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let blocks: Vec<BlockId> = module.body(body).blocks().map(|(block, _)| block).collect();
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Motion>::new(module, body, &mut facts);
        for block in blocks {
            schedule_block(&mut editor, block);
        }
        editor.finish()
    }
}

/// The def-use graph of one block's instructions, each named by its index
/// in the block before scheduling.
struct BlockGraph {
    insts: Vec<InstId>,
    /// For each instruction, how many distinct instructions of the block
    /// define its operands.
    defined_by: Vec<usize>,
    /// For each instruction, the distinct instructions of the block that
    /// use what it defines.
    used_by: Vec<Vec<usize>>,
    index_of: FxHashMap<InstId, usize>,
}

impl BlockGraph {
    /// Cost: O(instructions + operands).
    fn of(body: &Body, block: BlockId) -> BlockGraph {
        let insts: Vec<InstId> = body.block(block).insts().map(|inst| inst.id()).collect();
        let index_of: FxHashMap<InstId, usize> = insts.iter().enumerate().map(|(index, inst)| (*inst, index)).collect();
        let mut defined_by = vec![0; insts.len()];
        let mut used_by = vec![Vec::new(); insts.len()];
        for (user, inst) in insts.iter().enumerate() {
            let mut definers: Vec<usize> = body
                .inst(*inst)
                .operands()
                .into_iter()
                .filter_map(|operand| match body.definition(operand) {
                    Definition::Inst(definer) => index_of.get(&definer).copied(),
                    Definition::BlockParam(..)
                    | Definition::Param(_)
                    | Definition::Capture(_)
                    | Definition::OrderParam
                    | Definition::Slot(_) => None,
                })
                .collect();
            definers.sort_unstable();
            definers.dedup();
            defined_by[user] = definers.len();
            for definer in definers {
                used_by[definer].push(user);
            }
        }
        BlockGraph { insts, defined_by, used_by, index_of }
    }

    /// The longest def-use path from each instruction to the block's end,
    /// itself included. A user stands after what it uses, so the walk from
    /// the block's end sees every user first. Cost: O(instructions + uses).
    fn heights(&self) -> Vec<usize> {
        let mut heights = vec![0; self.used_by.len()];
        for inst in (0..self.used_by.len()).rev() {
            let longest_after = self.used_by[inst].iter().map(|user| heights[*user]).fold(0, usize::max);
            heights[inst] = 1 + longest_after;
        }
        heights
    }
}

/// A ready instruction's place in the queue: the higher path first, then
/// the earlier instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Ready {
    height: usize,
    earlier: Reverse<usize>,
}

impl Ready {
    fn inst(self) -> usize {
        self.earlier.0
    }
}

/// Why a ready instruction is out of the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Held {
    /// Its move would change where it runs; it is placed where it stands.
    UntilStanding,
    /// Its move would cross the write at this index; it waits for it.
    UntilPlaced(usize),
}

/// Fills the block's positions from its start, each with the ready
/// instruction of highest priority whose move there the editor allows.
fn schedule_block(editor: &mut Editor<'_, Motion>, block: BlockId) {
    let graph = BlockGraph::of(editor.body(), block);
    let heights = graph.heights();
    let count = heights.len();
    let ready_of = |inst: usize| Ready { height: heights[inst], earlier: Reverse(inst) };
    let mut waiting = graph.defined_by.clone();
    let mut ready: BinaryHeap<Ready> = (0..count).filter(|inst| waiting[*inst] == 0).map(ready_of).collect();
    let mut held: FxHashMap<usize, Held> = FxHashMap::default();
    let mut waiters: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut last_placed: Option<InstRef> = None;
    for place in 0..count {
        let standing = match last_placed {
            Some(last) => editor.next_inst(last),
            None => editor.first_inst(block),
        };
        let Some(standing) = standing else {
            panic!("{block:?} holds an instruction at place {place}, one not placed yet")
        };
        let standing_index = graph.index_of[&standing.id()];
        if held.get(&standing_index) == Some(&Held::UntilStanding) {
            held.remove(&standing_index);
            ready.push(ready_of(standing_index));
        }
        let mut refused = Vec::new();
        let placed = loop {
            let Some(candidate) = ready.pop() else {
                panic!("the instruction standing at {block:?}[{place}] is ready, since every earlier one is placed")
            };
            if candidate.inst() == standing_index {
                break candidate.inst();
            }
            let inst = editor.inst_ref(graph.insts[candidate.inst()]);
            match editor.move_to(inst, Position::Before(standing)) {
                Ok(()) => break candidate.inst(),
                Err(CannotMove::ChangesWhereItRuns) => {
                    held.insert(candidate.inst(), Held::UntilStanding);
                }
                Err(CannotMove::CrossesConflictingWrite(write)) => {
                    let write = graph.index_of[&write.id()];
                    held.insert(candidate.inst(), Held::UntilPlaced(write));
                    waiters.entry(write).or_default().push(candidate.inst());
                }
                Err(CannotMove::OperandNotDominating(_) | CannotMove::UseNotDominated(_)) => refused.push(candidate),
            }
        };
        last_placed = Some(editor.inst_ref(graph.insts[placed]));
        ready.extend(refused);
        for waiter in waiters.remove(&placed).into_iter().flatten() {
            held.remove(&waiter);
            ready.push(ready_of(waiter));
        }
        for user in &graph.used_by[placed] {
            waiting[*user] -= 1;
            if waiting[*user] == 0 {
                ready.push(ready_of(*user));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::cfg::{Inst, OrderUse};
    use crate::passes::optimize::tests::{Fixture, assert_whole, int_const, ret, wrapping_add};

    /// ```text
    /// a = 1; x = f(); b = 2; c = b + b; d = c + c; y = f();
    /// e = d + a; g = e + x; h = g + y; return h
    /// ```
    /// The paths to the block's end are b 6, c 5, a 4, d 4, x 3, e 3, y 2,
    /// g 2 and h 1, `x` and `y` being chained by their tokens.
    #[test]
    fn independent_values_go_by_their_path_and_the_tokens_keep_their_order() {
        let mut f = Fixture::new();
        let (int, opaque_call) = (f.int, f.call(f.opaque));
        let mut builder = f.module.build_body(true, int);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let x = builder.push_effect(opaque_call.clone(), int, Span::ZERO);
        let b = builder.push(int_const(2), int, Span::ZERO);
        let c = builder.push(wrapping_add(b, b), int, Span::ZERO);
        let d = builder.push(wrapping_add(c, c), int, Span::ZERO);
        let y = builder.push_effect(opaque_call, int, Span::ZERO);
        let e = builder.push(wrapping_add(d, a), int, Span::ZERO);
        let g = builder.push(wrapping_add(e, x), int, Span::ZERO);
        let h = builder.push(wrapping_add(g, y), int, Span::ZERO);
        builder.terminate(ret(h), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();
        let o0 = f.module.body(id).order_param().expect("the body is ordered");

        let changes = f.run::<Schedule>(id);

        let insts: Vec<_> = f.module.body(id).block(entry).insts().collect();
        let results: Vec<_> = insts.iter().map(|inst| inst.result()).collect();
        assert_eq!(results, [b, c, a, d, x, e, y, g, h].map(Some));
        let edge = |inst: &Inst| match inst.order() {
            Some(OrderUse::Edge(edge)) => edge,
            other => panic!("{:?} is on the chain, not {other:?}", inst.op()),
        };
        let (first, second) = (edge(&insts[4]), edge(&insts[6]));
        assert_eq!(first.before, o0, "the first call still consumes the body's token");
        assert_eq!(second.before, first.after, "the second call still follows the first");
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
        assert_whole(&f.module, id);
    }
}
