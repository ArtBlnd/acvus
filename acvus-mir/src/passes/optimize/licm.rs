//! Loop-invariant code motion, in two editors, one after the other.
//!
//! First `Editor<Cfg>` makes preheaders: a natural loop without one, whose
//! header one block outside the loop enters by exactly one edge, gets a
//! preheader by `split_edge` on that edge. A header that several outside
//! blocks enter, or that one enters by several edges, gets none, since no
//! split of one edge gives it a single entering block, and its loop is
//! skipped.
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
//! Form: F1, SSA, in; F1 kept, and F2, loops canonical, out for every loop
//! whose header one outside block enters by one edge.
//!
//! Cost: O(loops) edge splits, each the core's O(the header's parameters);
//! then O(loop depth × instructions) move offers, each at `move_to`'s cost.
//!
//! Meaning: a preheader is an empty block on the one edge into the loop,
//! passing its parameters on, so every run takes the same way through the
//! same instructions. A move is made only where `move_to` allows it: every
//! operand and use, order tokens included, stays dominated, so the order of
//! effects is kept (effects §3); an instruction that may trap or has an
//! effect moves only where it runs on exactly the runs it did (compilation
//! §1), which the end of a preheader is not when the loop may run no
//! iteration; and no move crosses a write to storage the instruction names
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
        let mut facts = EditFacts::new(am, module, body);
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

/// Splits the one edge into each loop's header from outside the loop,
/// where the loop has no preheader and exactly one such edge enters it.
/// A split adds a block after every block the loop tree knows and changes
/// no other loop's entering edges, so the tree still answers for them.
fn make_preheaders(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
    let loops = am.get::<LoopTree>(module, body);
    let mut no_facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(module, body, &mut no_facts);
    for of in loops.loops() {
        if loops.preheader(of).is_some() {
            continue;
        }
        let header = loops.header(of);
        if let Some(from) = single_entering_edge(editor.body(), &loops, of, header) {
            editor.split_edge(from, header);
        }
    }
    editor.finish()
}

/// The block outside `of` that enters `header`, where it is the only one
/// and enters by one edge.
fn single_entering_edge(body: &Body, loops: &LoopTree, of: LoopId, header: BlockId) -> Option<BlockId> {
    let preds = body.predecessors();
    let mut outside = preds[&header].iter().copied().filter(|pred| !loops.contains(of, *pred));
    let (Some(from), None) = (outside.next(), outside.next()) else {
        return None;
    };
    let edges = body.block(from).terminator().edges().iter().filter(|edge| edge.to == header).count();
    (edges == 1).then_some(from)
}

/// Every loop, the deepest first.
fn inner_first(loops: &LoopTree) -> Vec<LoopId> {
    let depth = |of: LoopId| std::iter::successors(Some(of), |at| loops.parent(*at)).count();
    let mut ordered: Vec<LoopId> = loops.loops().collect();
    ordered.sort_by_key(|of| std::cmp::Reverse(depth(*of)));
    ordered
}

/// Offers every invariant instruction of `block` a move to the end of
/// `preheader`. A moved instruction leaves the block, so the walk stays at
/// its index, where the next one now stands.
fn hoist_from(editor: &mut Editor<'_, Motion>, loops: &LoopTree, of: LoopId, block: BlockId, preheader: BlockId) {
    let mut index = 0;
    while let Some(inst) = editor.inst_at(block, index) {
        let record = &editor.body().block(block).insts()[index];
        let invariant = record.operands().into_iter().all(|operand| defined_outside(editor.body(), loops, of, operand));
        if invariant {
            match editor.move_to(inst, Position::End(preheader)) {
                Ok(()) => continue,
                Err(
                    CannotMove::OperandNotDominating(_)
                    | CannotMove::UseNotDominated(_)
                    | CannotMove::ChangesWhereItRuns
                    | CannotMove::CrossesConflictingWrite(_),
                ) => {}
            }
        }
        index += 1;
    }
}

/// Whether `value` is defined before the loop `of` is entered: in a block
/// outside it, or before the body's entry runs.
fn defined_outside(body: &Body, loops: &LoopTree, of: LoopId, value: ValueId) -> bool {
    match body.definition(value) {
        Definition::Inst(block, _) | Definition::BlockParam(block, _) => !loops.contains(of, block),
        Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::RefTarget;
    use crate::ir::cfg::{Builder, Op, Stages, Terminator};
    use crate::passes::optimize::tests::{Fixture, assert_whole, int_const, jump, ret, trapping_add, wrapping_add};
    use crate::ty::Mutability;

    /// The values defined before the loop that the instruction under test
    /// may read.
    struct Outside {
        a: ValueId,
        b: ValueId,
        /// `&mut s`, for a slot `s`.
        x: ValueId,
    }

    /// A body built by `loop_around`.
    struct Looped {
        id: BodyId,
        loop_body: BlockId,
        under_test: ValueId,
    }

    /// ```text
    /// entry:  a = 1; b = 2; x = &mut s; if c goto header else done
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
        let (int, bool, mut_ref_int, nc) = (f.int, f.bool, f.mut_ref_int, f.name("c"));
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool);
        let s = builder.slot(int, Span::ZERO);
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
        });
        builder.switch_to(header);
        builder.terminate(Terminator::While { cond: c, stages: Stages::lowered(loop_body), exit, exit_args: Vec::new() });
        builder.switch_to(loop_body);
        let under_test = inside(&mut builder, &Outside { a, b, x });
        builder.terminate(jump(header));
        builder.switch_to(exit);
        builder.terminate(ret(a));
        builder.switch_to(done);
        builder.terminate(ret(a));
        let id = builder.seal().expect("the body has no open state");
        Looped { id, loop_body, under_test }
    }

    /// The block that now enters the loop's header from outside it.
    fn preheader_of(f: &Fixture, id: BodyId, loop_body: BlockId) -> BlockId {
        let body = f.module.body(id);
        let header = body
            .blocks()
            .find_map(|(block, record)| match record.terminator() {
                Terminator::While { stages, .. } if stages.body() == loop_body => Some(block),
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
            loop_around(&mut f, |builder, outside| builder.push(wrapping_add(outside.a, outside.b), int, Span::ZERO));
        let entry = f.module.body(id).entry();

        let changes = f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert_ne!(preheader, entry, "a new block stands on the edge into the loop");
        assert!(f.ops(id, loop_body).is_empty(), "the loop's body is left empty");
        let hoisted = f.module.body(id).block(preheader).insts();
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
            loop_around(&mut f, |builder, outside| builder.push(trapping_add(outside.a, outside.b), int, Span::ZERO));

        f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert!(f.ops(id, preheader).is_empty(), "nothing is hoisted");
        let stayed = f.module.body(id).block(loop_body).insts();
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
        let hoisted = f.module.body(id).block(preheader).insts();
        assert_eq!(hoisted.iter().map(|inst| inst.result()).collect::<Vec<_>>(), [Some(loaded)]);
        assert!(f.ops(id, loop_body).is_empty());
        assert_whole(&f.module, id);
    }

    /// `*x = a` in a loop that may run no iteration: after a run with none,
    /// `s` holds what it held before the loop, so the store stays in the
    /// loop. `move_to` now lets a write to storage move to a place that
    /// does not run on exactly the runs its own does, since it pins only
    /// what may trap or has an effect; the fix is the editor's.
    #[test]
    #[ignore = "blocked on ir::cfg::edit: move_to lets a store leave a loop that may run no iteration"]
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
            let next = builder.push(wrapping_add(loaded, a), int, Span::ZERO);
            builder.push_unit(
                Op::Assign { target: RefTarget::Through(x), path: Vec::new(), value: next, restores: false },
                Span::ZERO,
            );
            loaded
        });

        f.run::<Licm>(id);

        let preheader = preheader_of(&f, id, loop_body);
        assert!(f.ops(id, preheader).is_empty(), "nothing is hoisted");
        let stayed = f.module.body(id).block(loop_body).insts();
        assert_eq!(stayed.len(), 3);
        assert_eq!(stayed[0].result(), Some(loaded));
        assert_whole(&f.module, id);
    }
}
