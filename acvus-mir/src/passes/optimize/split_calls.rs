//! Call splitting (effects §6), in two editors, one after the other.
//!
//! First `Editor<Effect>` offers every call to `split_call`, which splits
//! it into its start and its taking where the callee's type states that it
//! may be split, and refuses it otherwise.
//!
//! Then `Editor<Motion>` moves each start as early as dominance allows and
//! each taking as late as its first use allows:
//! - a start is offered the earliest point at which every operand, order
//!   token included, is defined: just after the last of them in the deepest
//!   block that defines one, or the start of the entry where the body's
//!   parameters give them all; failing that, the start of each block below
//!   that one on the dominator tree down to its own, in turn;
//! - a taking is offered the latest point that still comes before every use
//!   of its result and of the token it yields: in the nearest common
//!   dominator of those uses, just before the first instruction there that
//!   uses one, or its end where none does; failing that, the end of each
//!   block above that one on the dominator tree up to its own, in turn.
//!
//! Every start is taken exactly once on every path from it
//! (`SplitTakenOnce`); both scopes keep it, and the editor checks it after
//! every edit.
//!
//! Form: F2, loops canonical, in and out; it changes no block.
//!
//! Cost: O(calls) splits, each the core's O(the block's instructions after
//! it); then per start and per taking O(dominator depth + uses) to find its
//! points, and at most dominator depth `move_to` offers at the editor's
//! cost.
//!
//! Meaning: a split call means what the call meant at its taking, and its
//! start stands where its arguments and its token are ready (effects §6). A
//! move is made only where `move_to` allows it: every operand and use,
//! order tokens included, stays dominated, so the order of effects is kept
//! (effects §3); a start or a taking, having an effect, moves only where it
//! runs on exactly the runs it did (compilation §1); and no move crosses a
//! write to storage it names, so a start holds its arguments' loans until
//! its taking (ownership §5). A taking's result has the call's type, and
//! the start's handle is the core's.

use crate::ir::ValueId;
use crate::ir::cfg::edit::{CannotMove, CannotSplit, Editor, Effect, InstRef, Motion, Position};
use crate::ir::cfg::{BlockId, BodyId, Changes, Definition, Inst, Module, Op, OrderUse, Site};
use crate::passes::analysis::{AnalysisManager, Dominators};
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::{BodyTransform, nearest_common_dominator, union_of};

pub struct SplitCalls;

impl BodyTransform for SplitCalls {
    const NAME: &'static str = "split_calls";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let split = split_every_call(module, body, am);

        let dominators = am.get::<Dominators>(module, body);
        let mut facts = EditFacts::new(am, module, body);
        let mut editor = Editor::<Motion>::new(module, body, &mut facts);
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            if matches!(op_of(cursor.editor(), inst), Op::Spawn { .. }) {
                hoist_start(cursor.editor(), &dominators, inst);
            }
        }
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            if matches!(op_of(cursor.editor(), inst), Op::TakeEffect { .. }) {
                sink_taking(cursor.editor(), &dominators, inst);
            }
        }
        union_of([split, editor.finish()])
    }
}

/// Offers every call to `split_call`.
fn split_every_call(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
    let mut facts = EditFacts::new(am, module, body);
    let mut editor = Editor::<Effect>::new(module, body, &mut facts);
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        let editor = cursor.editor();
        if !matches!(op_of(editor, inst), Op::FunctionCall { .. }) {
            continue;
        }
        match editor.split_call(inst) {
            Ok(()) | Err(CannotSplit::NotSplittable) => {}
        }
    }
    editor.finish()
}

fn op_of<'e, S>(editor: &'e Editor<'_, S>, inst: InstRef) -> &'e Op
where
    S: crate::ir::cfg::edit::Scope,
{
    editor.body().block(inst.block()).insts()[inst.index()].op()
}

/// Offers `start` the points from the earliest its operands allow down to
/// the start of its own block, until one is taken; a refused offer leaves
/// the body, and so `start`, as it was. A start in a block the entry does
/// not reach has no dominator above it and stays.
fn hoist_start(editor: &mut Editor<'_, Motion>, dominators: &Dominators, start: InstRef) {
    if !dominators.reachable(start.block()) {
        return;
    }
    let body = editor.body();
    let operands = body.block(start.block()).insts()[start.index()].operands();
    // The blocks from the start's own up to the entry, on the dominator tree.
    let above: Vec<BlockId> = std::iter::successors(Some(start.block()), |block| dominators.idom(*block)).collect();
    let depth_of = |block: BlockId| {
        above.iter().position(|on| *on == block).unwrap_or_else(|| {
            panic!("an operand of {start:?} is defined in {block:?}, which dominates the start's block")
        })
    };
    // The deepest block that defines an operand, as its distance up from the
    // start's block, and the first index there after every operand it defines.
    let mut deepest = (above.len() - 1, 0);
    for operand in operands {
        let defined = match body.definition(operand) {
            Definition::Inst(block, index) => (depth_of(block), index + 1),
            Definition::BlockParam(block, _) => (depth_of(block), 0),
            Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => {
                (above.len() - 1, 0)
            }
        };
        if defined.0 < deepest.0 || (defined.0 == deepest.0 && defined.1 > deepest.1) {
            deepest = defined;
        }
    }
    let (distance, after_defs) = deepest;
    let points: Vec<(BlockId, usize)> =
        std::iter::once((above[distance], after_defs)).chain(above[..distance].iter().rev().map(|block| (*block, 0))).collect();
    for (block, index) in points {
        let at = point(editor, block, index);
        match editor.move_to(start, at) {
            Ok(()) => return,
            Err(
                CannotMove::OperandNotDominating(_)
                | CannotMove::UseNotDominated(_)
                | CannotMove::ChangesWhereItRuns
                | CannotMove::CrossesConflictingWrite(_),
            ) => {}
        }
    }
}

/// Offers `taking` the points from the latest its uses allow up to the end
/// of its own block, until one is taken; a refused offer leaves the body,
/// and so `taking`, as it was.
fn sink_taking(editor: &mut Editor<'_, Motion>, dominators: &Dominators, taking: InstRef) {
    let body = editor.body();
    let defs = defined_by(&body.block(taking.block()).insts()[taking.index()]);
    let use_blocks: Vec<BlockId> = defs.iter().flat_map(|value| body.users(*value)).map(site_block).collect();
    let Some((first, rest)) = use_blocks.split_first() else {
        return;
    };
    let Some(meet) = rest.iter().try_fold(*first, |common, block| nearest_common_dominator(dominators, common, *block)) else {
        return;
    };
    // The blocks from where the uses meet up to the taking's own.
    let mut down: Vec<BlockId> = Vec::new();
    for block in std::iter::successors(Some(meet), |block| dominators.idom(*block)) {
        down.push(block);
        if block == taking.block() {
            break;
        }
    }
    if down.last() != Some(&taking.block()) {
        panic!("the uses of {taking:?} meet in {meet:?}, which its own block dominates")
    }
    let first_use = body.block(meet).insts().iter().position(|user| user.operands().iter().any(|operand| defs.contains(operand)));
    let meet_index = match first_use {
        Some(user) => user,
        None => body.block(meet).insts().len(),
    };
    let points: Vec<(BlockId, usize)> = std::iter::once((meet, meet_index))
        .chain(down[1..].iter().map(|block| (*block, body.block(*block).insts().len())))
        .collect();
    for (block, index) in points {
        let at = point(editor, block, index);
        match editor.move_to(taking, at) {
            Ok(()) => return,
            Err(
                CannotMove::OperandNotDominating(_)
                | CannotMove::UseNotDominated(_)
                | CannotMove::ChangesWhereItRuns
                | CannotMove::CrossesConflictingWrite(_),
            ) => {}
        }
    }
}

/// The position before the instruction at `index` of `block`, or the
/// block's end where `index` is its length.
fn point(editor: &Editor<'_, Motion>, block: BlockId, index: usize) -> Position {
    match editor.inst_at(block, index) {
        Some(inst) => Position::Before(inst),
        None => Position::End(block),
    }
}

/// The values an instruction defines: its result and the token it yields.
fn defined_by(inst: &Inst) -> Vec<ValueId> {
    inst.result().into_iter().chain(inst.order().and_then(OrderUse::yielded)).collect()
}

fn site_block(site: Site) -> BlockId {
    match site {
        Site::Inst(block, _) | Site::Term(block) => block,
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::passes::optimize::tests::{Fixture, assert_whole, direct_call, int_const, ret, wrapping_add};

    /// ```text
    /// a = 1; b = a + a; r = S(); c = 5; d = c + c; s = r + d; return s + b
    /// ```
    /// with `S` splittable.
    #[test]
    fn a_splittable_call_starts_early_and_is_taken_just_before_its_first_use() {
        let mut f = Fixture::new();
        let (int, splittable, name) = (f.int, f.splittable, f.name("f"));
        let mut builder = f.module.build_body(true, int);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let b = builder.push(wrapping_add(a, a), int, Span::ZERO);
        let r = builder.push_effect(direct_call(name, splittable, Vec::new()), int, Span::ZERO);
        let c = builder.push(int_const(5), int, Span::ZERO);
        let d = builder.push(wrapping_add(c, c), int, Span::ZERO);
        let s = builder.push(wrapping_add(r, d), int, Span::ZERO);
        let t = builder.push(wrapping_add(s, b), int, Span::ZERO);
        builder.terminate(ret(t), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();
        let o0 = f.module.body(id).order_param().expect("the body is ordered");

        let changes = f.run::<SplitCalls>(id);

        let insts = f.module.body(id).block(entry).insts();
        let ops: Vec<&Op> = insts.iter().map(Inst::op).collect();
        assert!(matches!(ops[0], Op::Spawn { .. }), "the start rises above independent work: {ops:?}");
        assert_eq!(insts[0].order(), Some(OrderUse::Start(o0)));
        let handle = insts[0].result().expect("a start yields its handle");
        let results: Vec<_> = insts[1..].iter().map(Inst::result).collect();
        assert_eq!(results, [a, b, c, d, r, s, t].map(Some), "the taking stands just before its first use");
        assert!(matches!(ops[5], Op::TakeEffect { src } if *src == handle));
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
        assert_whole(&f.module, id);
    }

    /// A call whose callee's type does not allow a split stays whole.
    #[test]
    fn a_call_that_may_not_split_stays() {
        let mut f = Fixture::new();
        let (int, opaque, name) = (f.int, f.opaque, f.name("f"));
        let mut builder = f.module.build_body(true, int);
        let r = builder.push_effect(direct_call(name, opaque, Vec::new()), int, Span::ZERO);
        builder.terminate(ret(r), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);

        f.run::<SplitCalls>(id);

        assert_eq!(f.snapshot(id), before);
    }
}
