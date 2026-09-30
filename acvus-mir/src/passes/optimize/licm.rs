//! Loop-invariant code motion, in two editors, one after the other.
//!
//! First `Editor<Cfg>` makes preheaders: every edge into a natural loop's
//! header from outside the loop goes through one new block, by
//! `split_edges`, where the loop has no preheader. Every loop then has one,
//! so F2 holds. No loop's header is the body's entry, since the Builder
//! never makes the entry a loop header (D54), so every header has a block
//! outside its loop that enters it.
//!
//! Then `Editor<Motion>` hoists, inner loops first. An instruction is
//! invariant in a loop when every operand, order token included, is defined
//! outside the loop; an instruction hoisted before it stands in the
//! preheader, outside the loop, so invariance runs down chains. Each
//! invariant instruction, walking the loop's blocks in a preorder of the
//! dominator tree, is offered a move to the end of the preheader, and a
//! refused move leaves it where it is. A hoisted instruction stands in the
//! preheader of an inner loop, which is a block of the loop around it, so
//! that loop offers it again.
//!
//! Form: F1, SSA, in; F1 kept, and F2, loops canonical, out.
//!
//! Cost: O(loops) edge splits, each the core's O(the header's parameters);
//! then O(loop depth × instructions) move offers, each at `move_to`'s cost.
//!
//! Meaning: a preheader is an empty block on the edges into the loop,
//! passing each edge's arguments on, so every run takes the same way through the
//! same instructions. A move is made only where `move_to` allows it: every
//! operand and use, order tokens included, stays dominated, so the order of
//! effects is kept (effects §3); an instruction whose running on other
//! runs is observable, one that may trap, has an effect, writes storage or
//! moves a value out of a place, moves only where it runs on exactly the
//! runs it did (compilation §1), which the end of a preheader is not when
//! the loop may run no iteration; and no move crosses a write to storage the instruction names
//! (ownership §5). A hoisted value's operands are the same on every
//! iteration, so it is the value each iteration computed. No type changes.

use crate::ir::cfg::edit::{CannotMove, Cfg, Editor, Motion, NoFacts, Position};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, Module};
use crate::ir::ValueId;
use crate::passes::analysis::{AnalysisManager, Dominators, LoopId, LoopTree};
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::{BodyTransform, dominator_preorder, union_of};

pub struct Licm;

impl BodyTransform for Licm {
    const NAME: &'static str = "licm";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let preheaders = make_preheaders(module, body, am);
        am.invalidate(body, &preheaders);

        let loops = am.get::<LoopTree>(module, body);
        let dominators = am.get::<Dominators>(module, body);
        let order = dominator_preorder(module.body(body), &dominators);
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Motion>::new(module, body, &mut facts);
        for of in inner_first(&loops) {
            let Some(preheader) = loops.preheader(of) else {
                continue;
            };
            for block in order.iter().copied().filter(|block| loops.contains(of, *block)) {
                hoist_from(&mut editor, &loops, of, block, preheader);
            }
        }
        union_of([preheaders, editor.finish()])
    }
}

/// Makes a preheader for every loop without one, from all the blocks
/// outside it that enter its header. A split adds a block after every block
/// the loop tree knows, and the new block enters only its own loop's header,
/// so the tree still answers for every other loop's entering blocks.
fn make_preheaders(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
    let loops = am.get::<LoopTree>(module, body);
    let mut no_facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(module, body, &mut no_facts);
    for of in loops.loops() {
        if loops.preheader(of).is_some() {
            continue;
        }
        let header = loops.header(of);
        let outside = entering_blocks(editor.body(), &loops, of, header);
        assert!(
            !outside.is_empty(),
            "the loop at {header:?} is entered from outside it, since the Builder never makes the entry a loop header (D54)"
        );
        editor.split_edges(&outside, header);
    }
    editor.finish()
}

/// The blocks outside `of` with an edge into its header, each once.
fn entering_blocks(body: &Body, loops: &LoopTree, of: LoopId, header: BlockId) -> Vec<BlockId> {
    let preds = body.predecessors();
    preds[&header].iter().copied().filter(|pred| !loops.contains(of, *pred)).collect()
}

/// Every loop, the deepest first.
fn inner_first(loops: &LoopTree) -> Vec<LoopId> {
    let depth = |of: LoopId| std::iter::successors(Some(of), |at| loops.parent(*at)).count();
    let mut ordered: Vec<LoopId> = loops.loops().collect();
    ordered.sort_by_key(|of| std::cmp::Reverse(depth(*of)));
    ordered
}

/// Offers every invariant instruction of `block` a move to the end of
/// `preheader`, in the block's order. A move only takes an instruction out
/// of `block`, so the instructions it held when the walk began are the ones
/// to offer.
fn hoist_from(editor: &mut Editor<'_, Motion>, loops: &LoopTree, of: LoopId, block: BlockId, preheader: BlockId) {
    for inst in editor.insts_of(block) {
        let record = editor.inst_of(inst);
        let invariant = record.operands().into_iter().all(|operand| defined_outside(editor.body(), loops, of, operand));
        if invariant {
            match editor.move_to(inst, Position::End(preheader)) {
                Ok(())
                | Err(
                    CannotMove::OperandNotDominating(_)
                    | CannotMove::UseNotDominated(_)
                    | CannotMove::ChangesWhereItRuns
                    | CannotMove::CrossesConflictingWrite(_),
                ) => {}
            }
        }
    }
}

/// Whether `value` is defined before the loop `of` is entered: in a block
/// outside it, or before the body's entry runs.
fn defined_outside(body: &Body, loops: &LoopTree, of: LoopId, value: ValueId) -> bool {
    match body.definition(value) {
        Definition::Inst(inst) => !loops.contains(of, body.block_of(inst)),
        Definition::BlockParam(block, _) => !loops.contains(of, block),
        Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::RefTarget;
    use crate::ir::cfg::{Builder, Op, Terminator};
    use crate::passes::optimize::tests::{Fixture, add, assert_whole, int_const, jump, ret, trapping_div};
    use crate::ty::Mutability;

    /// The values defined before the loop that the instruction under test
    /// may read.
    struct Outside {
        a: ValueId,
        b: ValueId,
        /// `&mut s`, for a slot `s`.
        x: ValueId,
        /// A slot of a `String`.
        text: ValueId,
    }

    /// A body built by `loop_around`.
    struct Looped {
        id: BodyId,
        loop_body: BlockId,
        under_test: ValueId,
    }

    /// ```text
    /// entry:  a = 1; b = 2; x = &mut s; if c goto header else done
    ///         (with `text`, a `String` slot, declared)
    /// header: while c { body } exit
    /// body:   <the instruction under test>; jump header
    /// exit:   return a
    /// done:   return a
    /// ```
    /// `entry` enters the loop and also leaves elsewhere, so the loop has no
    /// preheader until licm makes one.
    fn loop_around<F>(f: &mut Fixture, inside: F) -> Looped
    where
        F: FnOnce(&mut Builder<'_>, &Outside) -> ValueId,
    {
        let (int, bool, string, mut_ref_int, nc) = (f.int, f.bool, f.string, f.mut_ref_int, f.name("c"));
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let s = builder.slot(int, Span::ZERO);
        let text = builder.slot(string, Span::ZERO);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let b = builder.push(int_const(2), int, Span::ZERO);
        let x = builder.push(
            Op::Ref { target: RefTarget::Var(s), path: Vec::new(), mutability: Mutability::Mut },
            mut_ref_int,
            Span::ZERO,
        );
        let [header, loop_body, exit, done] = [(); 4].map(|()| builder.new_block(&[]));
        builder.terminate(Terminator::JumpIf {
            cond: c,
            then_to: header,
            then_args: Vec::new(),
            else_to: done,
            else_args: Vec::new(),
        }, acvus_ast::Span::ZERO);
        builder.switch_to(header);
        builder.terminate(Terminator::While { cond: c, body: loop_body, body_args: Vec::new(), exit, exit_args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(loop_body);
        let under_test = inside(&mut builder, &Outside { a, b, x, text });
        builder.terminate(jump(header), acvus_ast::Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
        builder.switch_to(done);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        Looped { id, loop_body, under_test }
    }

    /// The block that now enters the loop's header from outside it.
    fn preheader_of(f: &Fixture, id: BodyId, loop_body: BlockId) -> BlockId {
        let body = f.module.body(id);
        let header = body
            .blocks()
            .find_map(|(block, record)| match record.terminator() {
                Terminator::While { body, .. } if *body == loop_body => Some(block),
                _ => None,
            })
            .expect("the loop keeps its header");
        let preds = body.predecessors();
        let outside: Vec<BlockId> = preds[&header].iter().copied().filter(|pred| *pred != loop_body).collect();
        let [preheader] = outside[..] else { panic!("one block enters the header from outside: {outside:?}") };
        preheader
    }

    #[test]
    fn a_pure_invariant_moves_to_a_preheader_licm_makes() {
        let mut f = Fixture::new();
        let int = f.int;
        let Looped { id, loop_body, under_test: sum } =
            loop_around(&mut f, |builder, outside| builder.push(add(outside.a, outside.b), int, Span::ZERO));
        let entry = f.module.body(id).entry();

        let changes = f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert_ne!(preheader, entry, "a new block stands on the edge into the loop");
        assert!(f.ops(id, loop_body).is_empty(), "the loop's body is left empty");
        let hoisted: Vec<_> = f.module.body(id).block(preheader).insts().collect();
        assert_eq!(hoisted.len(), 1);
        assert_eq!(hoisted[0].result(), Some(sum));
        assert!(changes.touched(crate::ir::cfg::Aspect::CfgShape));
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
        assert_whole(&f.module, id);
    }

    /// The loop may run no iteration, so the preheader runs on runs the
    /// loop's body does not, and a trap there would be a trap the program
    /// never raised.
    #[test]
    fn a_trapping_invariant_stays_in_the_loop() {
        let mut f = Fixture::new();
        let int = f.int;
        let Looped { id, loop_body, under_test: sum } =
            loop_around(&mut f, |builder, outside| builder.push(trapping_div(outside.a, outside.b), int, Span::ZERO));

        f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert!(f.ops(id, preheader).is_empty(), "nothing is hoisted");
        let stayed: Vec<_> = f.module.body(id).block(loop_body).insts().collect();
        assert_eq!(stayed.iter().map(|inst| inst.result()).collect::<Vec<_>>(), [Some(sum)]);
        assert_whole(&f.module, id);
    }

    /// `v = *x` in a loop that writes nothing: the load reads the same
    /// storage on every iteration, and nothing the way crosses writes it.
    #[test]
    fn a_load_in_a_loop_that_writes_nothing_moves() {
        let mut f = Fixture::new();
        let int = f.int;
        let Looped { id, loop_body, under_test: loaded } = loop_around(&mut f, |builder, outside| {
            builder.push(
                Op::Take { target: RefTarget::Through(outside.x), path: Vec::new(), taken_out: false },
                int,
                Span::ZERO,
            )
        });

        f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        let hoisted: Vec<_> = f.module.body(id).block(preheader).insts().collect();
        assert_eq!(hoisted.iter().map(|inst| inst.result()).collect::<Vec<_>>(), [Some(loaded)]);
        assert!(f.ops(id, loop_body).is_empty());
        assert_whole(&f.module, id);
    }

    /// `*x = a` in a loop that may run no iteration: after a run with none,
    /// `s` holds what it held before the loop, so the store stays in the
    /// loop.
    #[test]
    fn a_store_in_a_loop_that_may_not_run_stays() {
        let mut f = Fixture::new();
        let Looped { id, loop_body, .. } = loop_around(&mut f, |builder, &Outside { a, x, .. }| {
            builder.push_unit(
                Op::Assign { target: RefTarget::Through(x), path: Vec::new(), value: a, restores: false },
                Span::ZERO,
            );
            a
        });

        f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert!(f.ops(id, preheader).is_empty(), "nothing is hoisted");
        assert!(matches!(f.ops(id, loop_body)[..], [Op::Assign { .. }]), "the store stays in the loop");
        assert_whole(&f.module, id);
    }

    /// `v = text` in a loop that may run no iteration: the `String` does not
    /// copy, so the take empties `text`, and after a run with no iteration
    /// `text` still holds its value.
    #[test]
    fn a_take_of_a_value_that_does_not_copy_stays_in_the_loop() {
        let mut f = Fixture::new();
        let string = f.string;
        let Looped { id, loop_body, under_test: taken } = loop_around(&mut f, |builder, outside| {
            builder.push(
                Op::Take { target: RefTarget::Var(outside.text), path: Vec::new(), taken_out: false },
                string,
                Span::ZERO,
            )
        });

        f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert!(f.ops(id, preheader).is_empty(), "nothing is hoisted");
        let stayed: Vec<_> = f.module.body(id).block(loop_body).insts().collect();
        assert_eq!(stayed.iter().map(|inst| inst.result()).collect::<Vec<_>>(), [Some(taken)]);
        assert_whole(&f.module, id);
    }

    /// ```text
    /// entry:  a = 1; b = 2; if c goto left else right
    /// left:   goto header
    /// right:  goto header
    /// header: while c { body } exit
    /// body:   v = a + b; goto header
    /// exit:   return a
    /// ```
    /// Two blocks outside the loop enter its header.
    #[test]
    fn a_loop_two_blocks_enter_gets_one_preheader() {
        let mut f = Fixture::new();
        let (int, bool, nc) = (f.int, f.bool, f.name("c"));
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let b = builder.push(int_const(2), int, Span::ZERO);
        let [left, right, header, loop_body, exit] = [(); 5].map(|()| builder.new_block(&[]));
        builder.terminate(Terminator::JumpIf {
            cond: c,
            then_to: left,
            then_args: Vec::new(),
            else_to: right,
            else_args: Vec::new(),
        }, acvus_ast::Span::ZERO);
        builder.switch_to(left);
        builder.terminate(jump(header), acvus_ast::Span::ZERO);
        builder.switch_to(right);
        builder.terminate(jump(header), acvus_ast::Span::ZERO);
        builder.switch_to(header);
        builder.terminate(Terminator::While { cond: c, body: loop_body, body_args: Vec::new(), exit, exit_args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(loop_body);
        let sum = builder.push(add(a, b), int, Span::ZERO);
        builder.terminate(jump(header), acvus_ast::Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(a), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert!(![left, right].contains(&preheader), "a new block stands on both edges into the loop");
        let body = f.module.body(id);
        for from in [left, right] {
            assert!(matches!(body.block(from).terminator(), Terminator::Jump { to, .. } if *to == preheader));
        }
        let hoisted: Vec<_> = body.block(preheader).insts().collect();
        assert_eq!(hoisted.iter().map(|inst| inst.result()).collect::<Vec<_>>(), [Some(sum)]);
        assert!(f.ops(id, loop_body).is_empty());
        assert!(changes.touched(crate::ir::cfg::Aspect::CfgShape));
        assert_whole(&f.module, id);
    }

    /// `v = *x; *x = v + a` in the loop: the load's operand is defined
    /// outside, but every iteration's store writes what the next one loads.
    #[test]
    fn a_load_across_a_write_in_the_loop_stays() {
        let mut f = Fixture::new();
        let int = f.int;
        let Looped { id, loop_body, under_test: loaded } = loop_around(&mut f, |builder, &Outside { a, x, .. }| {
            let loaded = builder.push(
                Op::Take { target: RefTarget::Through(x), path: Vec::new(), taken_out: false },
                int,
                Span::ZERO,
            );
            let next = builder.push(add(loaded, a), int, Span::ZERO);
            builder.push_unit(
                Op::Assign { target: RefTarget::Through(x), path: Vec::new(), value: next, restores: false },
                Span::ZERO,
            );
            loaded
        });

        f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert!(f.ops(id, preheader).is_empty(), "nothing is hoisted");
        let stayed: Vec<_> = f.module.body(id).block(loop_body).insts().collect();
        assert_eq!(stayed.len(), 3);
        assert_eq!(stayed[0].result(), Some(loaded));
        assert_whole(&f.module, id);
    }
}
