//! Sinking: an instruction whose uses, of its result and of the order token
//! it yields, all stand in blocks its own block strictly dominates moves to
//! the nearest common dominator of those blocks, before the first
//! instruction there that uses it, or at that block's end where none does.
//! A target inside a loop that does not hold the instruction's block is
//! not taken, since the instruction would then run on every iteration of
//! that loop. An instruction nothing uses is not sunk; removing it is
//! `dce`'s.
//!
//! Blocks are walked in reverse of a preorder of the dominator tree, and
//! each block's instructions from its last, so a use is sunk before what it
//! reads is offered, and a chain sinks together. A sunk instruction lands in
//! a block walked already and is not offered again.
//!
//! Form: F1, SSA, in and out.
//!
//! Cost: O(instructions × (uses + dominator depth)) to find the targets,
//! and one `move_to` offer per instruction at the editor's cost.
//!
//! Meaning: a move is made only where `move_to` allows it: every operand
//! and use, order tokens included, stays dominated, so the order of effects
//! is kept (effects §3); an instruction that may trap or has an effect
//! moves only where it runs on exactly the runs it did (compilation §1);
//! and no move crosses a write to storage it names (ownership §5). A pure
//! instruction that no longer runs on a path that never read it changes
//! nothing that path observes. No type changes.

use crate::ir::ValueId;
use crate::ir::cfg::edit::{CannotMove, Editor, InstRef, Motion, Position};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Inst, Module, OrderUse, Site};
use crate::passes::analysis::{AnalysisManager, Dominators, LoopTree};
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::{BodyTransform, dominator_preorder, nearest_common_dominator};

pub struct Sink;

impl BodyTransform for Sink {
    const NAME: &'static str = "sink";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let dominators = am.get::<Dominators>(module, body);
        let loops = am.get::<LoopTree>(module, body);
        let order = dominator_preorder(module.body(body), &dominators);
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Motion>::new(module, body, &mut facts);
        for block in order.into_iter().rev() {
            for inst in editor.insts_of(block).into_iter().rev() {
                sink_one(&mut editor, &dominators, &loops, block, inst);
            }
        }
        editor.finish()
    }
}

/// Offers `inst`, of `block`, a move to where its uses meet, when that is
/// strictly later and in no loop that does not hold `block`. A move out of
/// `block` leaves the instructions before it where they stand, so the
/// block's instructions as the walk of it began are the ones to offer.
fn sink_one(editor: &mut Editor<'_, Motion>, dominators: &Dominators, loops: &LoopTree, block: BlockId, inst: InstRef) {
    let defs = defined_by(editor.inst_of(inst));
    let body = editor.body();
    let uses: Vec<Site> = defs.iter().flat_map(|value| body.users(*value)).collect();
    let use_blocks: Vec<BlockId> = uses.iter().map(|site| site_block(body, *site)).collect();
    let Some((first, rest)) = use_blocks.split_first() else {
        return;
    };
    let target = rest.iter().try_fold(*first, |common, block| nearest_common_dominator(dominators, common, *block));
    let Some(target) = target else {
        return;
    };
    if target == block || !dominators.dominates(block, target) {
        return;
    }
    if loops.innermost(target).is_some_and(|of| !loops.contains(of, block)) {
        return;
    }
    let first_use = uses
        .iter()
        .filter_map(|site| match site {
            Site::Inst(user) if body.block_of(*user) == target => Some(*user),
            Site::Inst(_) | Site::Term(_) => None,
        })
        .reduce(|first, other| if body.comes_before(other, first) { other } else { first });
    let at = match first_use {
        Some(user) => Position::Before(editor.inst_ref(user)),
        None => Position::End(target),
    };
    match editor.move_to(inst, at) {
        Ok(())
        | Err(
            CannotMove::OperandNotDominating(_)
            | CannotMove::UseNotDominated(_)
            | CannotMove::ChangesWhereItRuns
            | CannotMove::CrossesConflictingWrite(_),
        ) => {}
    }
}

/// The values an instruction defines: its result and the token it yields.
fn defined_by(inst: &Inst) -> Vec<ValueId> {
    inst.result().into_iter().chain(inst.order().and_then(OrderUse::yielded)).collect()
}

fn site_block(body: &Body, site: Site) -> BlockId {
    match site {
        Site::Inst(inst) => body.block_of(inst),
        Site::Term(block) => block,
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::cfg::Terminator;
    use crate::passes::optimize::tests::{Fixture, assert_whole, int_const, ret, wrapping_add};

    /// ```text
    /// entry: a = 1; b = a + a; if c { then } else { else } join
    /// then:  u = b + a; jump join(u)
    /// else:  jump join(a)
    /// join(r): return r
    /// ```
    #[test]
    fn a_value_used_only_in_one_arm_sinks_into_it() {
        let mut f = Fixture::new();
        let (int, bool, nc) = (f.int, f.bool, f.name("c"));
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let b = builder.push(wrapping_add(a, a), int, Span::ZERO);
        let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        let u = builder.push(wrapping_add(b, a), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: vec![u] }, acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        builder.terminate(Terminator::Jump { to: join, args: vec![a] }, acvus_ast::Span::ZERO);
        builder.switch_to(join);
        let r = builder.block_params(join)[0];
        builder.terminate(ret(r), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();

        let changes = f.run::<Sink>(id);

        let body = f.module.body(id);
        let results = |block: BlockId| body.block(block).insts().map(|inst| inst.result()).collect::<Vec<_>>();
        assert_eq!(results(entry), [Some(a)], "only the value both arms read stays");
        assert_eq!(results(then_block), [Some(b), Some(u)], "the value sinks to just before its use");
        assert!(results(else_block).is_empty());
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
        assert_whole(&f.module, id);
    }
}
