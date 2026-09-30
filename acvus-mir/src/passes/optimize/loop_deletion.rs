//! A loop that does nothing observable goes (LoopDeletion; compilation §1).
//!
//! A natural loop is deleted where every instruction in it has no effect, may
//! not trap and finishes, every loop inside it finishes (`analysis::Raise`),
//! and every value it gives its exit, or defines and has used after it, is
//! invariant in it or computable at the exit by the scalar evolution. Those
//! values are computed in the preheader or after the loop instead, under
//! undefined behaviour §2's rule for a value the run as written did not
//! compute, and the preheader jumps to the exit.
//!
//! Form: F2, SSA, in and out.
//!
//! Cost: O(loops) deletions, each O(the loop's blocks and exit values) edits.
//!
//! Meaning: the loop computes nothing the run observes but the values replaced
//! by equal ones, and it may not trap and finishes, so no run's effects, trap
//! or ending changes (compilation §1). No type changes; extern-defined types
//! are untouched (compilation §3).


use rustc_hash::FxHashSet;

use crate::ir::ValueId;
use crate::ir::cfg::edit::{Editor, Loop, Position};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, Inst, Module, Op, Site, Terminator};
use crate::passes::analysis::{
    AnalysisManager, Dominators, Endings, Evolution, Evolutions, IntervalValues, LoopId, LoopTree, Raise, ScalarEvolution, TripCount,
};
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::expand::Expander;
use crate::passes::optimize::{BodyTransform, inner_first, union_of};
use crate::ty::{IntTy, Ty, TyKind, copies};

pub struct LoopDeletion;

impl BodyTransform for LoopDeletion {
    const NAME: &'static str = "loop_deletion";

    /// One deletion per round over fresh facts: a deletion prunes blocks,
    /// which renumbers them, so no fact taken before it names a block after
    /// it. Cost: O(deleted loops + 1) rounds, each O(loops · their blocks)
    /// checks over the facts the manager recomputes.
    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let mut rounds = Vec::new();
        loop {
            let round = delete_one(module, body, am);
            let deleted = round.aspects().next().is_some();
            am.invalidate(body, &round);
            rounds.push(round);
            if !deleted {
                return union_of(rounds);
            }
        }
    }
}

/// Deletes the first loop, inner loops first, that the conditions admit.
fn delete_one(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
    let loops = am.get::<LoopTree>(module, body);
    let evolutions = am.get::<ScalarEvolution>(module, body);
    let endings = am.get::<Raise>(module, body);
    let intervals = am.get::<IntervalValues>(module, body);
    let dominators = am.get::<Dominators>(module, body);
    let len_ty = module.types_mut().intern(TyKind::Int(IntTy::U64));
    let expander = Expander::new(&evolutions, &loops, &intervals, &dominators, len_ty);
    let facts = Facts { loops: &loops, evolutions: &evolutions, endings: &endings };

    let mut edit_facts = EditFacts::new(am, body);
    let mut editor = Editor::<Loop>::new(module, body, &mut edit_facts);
    for of in inner_first(&loops) {
        let Some(deletion) = facts.deletion(&editor, of) else {
            continue;
        };
        if deletion.apply(&mut editor, &expander) {
            break;
        }
    }
    editor.finish()
}

/// The facts a deletion is judged by, taken before any edit of the round.
struct Facts<'f> {
    loops: &'f LoopTree,
    evolutions: &'f Evolutions,
    endings: &'f Endings,
}

/// What stands for a value the loop gives its exit or has used after it.
#[derive(Debug, Clone)]
enum Leaving {
    /// A value defined outside the loop, as it is.
    Outside(ValueId),
    /// The value's evolution at the exit, made in its type.
    Made { evolution: Evolution, ty: Ty },
}

/// An exit parameter, and what fills it on every edge from the loop.
struct ExitParam {
    param: ValueId,
    leaving: Leaving,
}

/// A value the loop defines and has used after it, and what stands for it.
struct UsedAfter {
    value: ValueId,
    sites: Vec<Site>,
    leaving: Leaving,
}

/// A loop the conditions admit, and how it goes.
struct Deletion {
    header: BlockId,
    preheader: BlockId,
    exit: BlockId,
    exit_params: Vec<ExitParam>,
    used_after: Vec<UsedAfter>,
}

impl Facts<'_> {
    /// The deletion of `of`, a loop its head's terminator heads, where every
    /// instruction in it has no effect, may not trap and finishes, it and
    /// every loop in it finish, every way out of it is an edge to its head's
    /// exit, which nothing outside it enters, and every value it gives the
    /// exit or has used after it is defined outside it or made from its
    /// evolution: invariant, or `S + n·D` from an exact count.
    fn deletion(&self, editor: &Editor<'_, Loop>, of: LoopId) -> Option<Deletion> {
        let body = editor.body();
        let header = self.loops.header(of);
        let exit = body.block(header).terminator().as_loop()?.exit;
        let preheader = self.loops.preheader(of)?;
        let blocks = self.loops.blocks(of);
        let mut nested = self.loops.loops().filter(|inner| self.loops.contains(of, self.loops.header(*inner)));
        if !nested.all(|inner| self.endings.loop_finishes(inner)) {
            return None;
        }
        for block in blocks {
            let record = body.block(*block);
            let observable = |inst: &Inst| {
                editor.has_effect(inst) || moves_or_writes(editor, inst) || self.endings.may_trap(inst.id()) || !self.endings.finishes(inst.id())
            };
            if record.insts().any(observable) {
                return None;
            }
            let edges = record.terminator().edges();
            if edges.is_empty() || edges.iter().any(|edge| !self.loops.contains(of, edge.to) && edge.to != exit) {
                return None;
            }
        }
        let preds = body.predecessors();
        if !preds.get(&exit).is_some_and(|preds| preds.iter().all(|pred| self.loops.contains(of, *pred))) {
            return None;
        }
        let exact = match self.evolutions.trip_count(of) {
            Some(TripCount::Exact(count)) => Some(count),
            Some(TripCount::AtMost(_)) | None => None,
        };

        let mut exit_params = Vec::new();
        for (index, param) in body.block(exit).params().iter().enumerate() {
            let arguments: FxHashSet<ValueId> = blocks
                .iter()
                .flat_map(|block| body.block(*block).terminator().edges())
                .filter(|edge| edge.to == exit)
                .map(|edge| edge.args[index])
                .collect();
            let [argument] = arguments.into_iter().collect::<Vec<_>>()[..] else {
                return None;
            };
            exit_params.push(ExitParam { param: *param, leaving: self.leaving(body, of, argument, exact)? });
        }
        let mut used_after = Vec::new();
        for value in defined_in(body, blocks) {
            let sites: Vec<Site> = body.users(value).filter(|site| !self.loops.contains(of, site_block(body, *site))).collect();
            if sites.is_empty() {
                continue;
            }
            used_after.push(UsedAfter { value, sites, leaving: self.leaving(body, of, value, exact)? });
        }
        Some(Deletion { header, preheader, exit, exit_params, used_after })
    }

    /// What stands for `value` once the loop is gone.
    fn leaving(&self, body: &Body, of: LoopId, value: ValueId, exact: Option<&Evolution>) -> Option<Leaving> {
        if !defined_in_loop(body, self.loops, of, value) {
            return Some(Leaving::Outside(value));
        }
        let ty = body.ty_of(value);
        match self.evolutions.of(value) {
            invariant @ Evolution::Invariant(_) => Some(Leaving::Made { evolution: invariant.clone(), ty }),
            Evolution::Recurrence(recurrence) if recurrence.over == of => {
                let risen = self.evolutions.product(exact?, &Evolution::Invariant(recurrence.step.clone()));
                let evolution = self.evolutions.sum(&recurrence.start, &risen);
                (evolution != Evolution::Unknown).then_some(Leaving::Made { evolution, ty })
            }
            Evolution::Recurrence(_) | Evolution::Unknown => None,
        }
    }
}

impl Deletion {
    /// Makes every leaving value at the end of the preheader, then: each
    /// exit parameter takes its value (`remove_exit`); each use after the
    /// loop reads its value; each header parameter takes its entry argument
    /// (`remove_join`), whose remaining uses are all in the loop; the edges
    /// into the header go to the exit (`retarget`), and the loop, no longer
    /// reached, goes (`prune`). Where a value cannot be made, only values
    /// nothing reads are made, and the loop stays.
    fn apply(self, editor: &mut Editor<'_, Loop>, expander: &Expander<'_>) -> bool {
        let at = Position::End(self.preheader);
        let span = editor.body().block(self.preheader).terminator_span();
        let make = |editor: &mut Editor<'_, Loop>, leaving: &Leaving| match leaving {
            Leaving::Outside(value) => Some(*value),
            Leaving::Made { evolution, ty } => expander.expand(editor, evolution, *ty, at, span),
        };
        let mut exit_values = Vec::new();
        for exit_param in &self.exit_params {
            let Some(made) = make(editor, &exit_param.leaving) else {
                return false;
            };
            exit_values.push(made);
        }
        let mut after_values = Vec::new();
        for used in &self.used_after {
            let Some(made) = make(editor, &used.leaving) else {
                return false;
            };
            after_values.push(made);
        }

        for (exit_param, made) in self.exit_params.iter().zip(exit_values) {
            let param = exit_param.param;
            editor.remove_exit(self.header, param, made).unwrap_or_else(|refusal| {
                panic!("an exit parameter of a loop in F2 entered from the loop alone takes a value made in the preheader, but {param:?} is refused: {refusal:?}")
            });
        }
        for (used, made) in self.used_after.iter().zip(after_values) {
            for site in &used.sites {
                editor.replace_use(*site, used.value, made).unwrap_or_else(|refusal| {
                    panic!("a use after the loop is dominated by its preheader, but {:?} at {site:?} is refused: {refusal:?}", used.value)
                });
            }
        }
        let Terminator::Jump { to, args } = editor.body().block(self.preheader).terminator() else {
            panic!("a preheader jumps to its header (F2)")
        };
        assert_eq!(*to, self.header, "a preheader jumps to its header (F2)");
        let params = editor.body().block(self.header).params().to_vec();
        for (param, init) in params.into_iter().zip(args.clone()) {
            editor.remove_join(self.header, param, init).unwrap_or_else(|refusal| {
                panic!("a header parameter whose uses are all in its loop takes its entry argument, but {param:?} is refused: {refusal:?}")
            });
        }
        editor.retarget(self.header, self.exit).unwrap_or_else(|refusal| {
            panic!("a header and an exit with no parameter agree, but {:?} to {:?} is refused: {refusal:?}", self.header, self.exit)
        });
        editor.prune();
        true
    }
}

/// Whether running `inst` fewer times than its place runs is observable
/// apart from an effect and a trap: it writes storage, or takes a value
/// that does not copy out of its place.
fn moves_or_writes(editor: &Editor<'_, Loop>, inst: &Inst) -> bool {
    let moves_out = matches!(inst.op(), Op::Take { .. })
        && inst.result().is_some_and(|result| !copies(editor.types(), editor.body().ty_of(result)));
    inst.op().writes_storage() || moves_out
}

/// The values the blocks define: their parameters and results.
fn defined_in(body: &Body, blocks: &[BlockId]) -> Vec<ValueId> {
    let mut values = Vec::new();
    for block in blocks {
        let record = body.block(*block);
        values.extend(record.params().iter().copied());
        values.extend(record.insts().filter_map(|inst| inst.result()));
    }
    values
}

fn defined_in_loop(body: &Body, loops: &LoopTree, of: LoopId, value: ValueId) -> bool {
    match body.definition(value) {
        Definition::Inst(inst) => loops.contains(of, body.block_of(inst)),
        Definition::BlockParam(block, _) => loops.contains(of, block),
        Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => false,
    }
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
    use crate::ir::cfg::Builder;
    use crate::ir::{BinOp, ForSource};
    use crate::passes::optimize::tests::{Fixture, add, assert_whole, int_const, ret, trapping_div};
    use crate::ty::CastTy;

    fn binary(op: BinOp, left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op, left, right }
    }

    fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
        Terminator::Jump { to, args }
    }

    /// What the loop's body does beside stepping `i`.
    #[derive(Clone, Copy)]
    enum Work {
        Nothing,
        /// An opaque call, an effect.
        Effect,
        /// `j / x`, `x` possibly zero.
        DivisionByAnyValue,
        /// `j / two`, never by zero.
        DivisionByTwo,
        /// Leaves to the exit when `c` holds.
        LeavesEarly,
    }

    /// ```text
    /// entry(x, c):  zero = 0; j = max(x, zero); zu = 0; n = j as u64; two = 2; jump header(zero)
    /// header(i):    for k in zu..n { body } else exit
    /// body(k):      <work>; one = 1; i2 = i + one; jump header(i2)
    ///               (or jump_if c { exit } else { header(i2) })
    /// exit:         return <i or j>
    /// ```
    struct Counted {
        id: BodyId,
        n: ValueId,
        j: ValueId,
    }

    fn counted(f: &mut Fixture, work: Work, returns_i: bool) -> Counted {
        let (int, bool, nx, nc) = (f.int, f.bool, f.name("x"), f.name("c"));
        let u64 = f.module.types_mut().intern(TyKind::Int(IntTy::U64));
        let opaque_call = f.call(f.opaque);
        let mut builder: Builder<'_> = f.module.build_body(matches!(work, Work::Effect), int);
        let x = builder.param(nx, int, Span::ZERO);
        let c = builder.param(nc, bool, Span::ZERO);
        let zero = builder.push(int_const(0), int, Span::ZERO);
        let j = builder.push(binary(BinOp::Max, x, zero), int, Span::ZERO);
        let zu = builder.push(int_const(0), u64, Span::ZERO);
        let n = builder.push(Op::Cast { src: j, to: CastTy::Int(IntTy::U64) }, u64, Span::ZERO);
        let two = builder.push(int_const(2), int, Span::ZERO);
        let header = builder.new_block(&[int]);
        let i = builder.block_params(header)[0];
        let loop_body = builder.new_block(&[u64]);
        let exit = builder.new_block(&[]);
        builder.terminate(jump(header, vec![zero]), Span::ZERO);
        builder.switch_to(header);
        let head = Terminator::For { source: ForSource::Range { at: zu, hi: n }, body: loop_body, body_args: Vec::new(), exit, exit_args: Vec::new() };
        builder.terminate(head, Span::ZERO);
        builder.switch_to(loop_body);
        match work {
            Work::Nothing | Work::LeavesEarly => {}
            Work::Effect => {
                builder.push_effect(opaque_call, int, Span::ZERO);
            }
            Work::DivisionByAnyValue => {
                builder.push(trapping_div(j, x), int, Span::ZERO);
            }
            Work::DivisionByTwo => {
                builder.push(trapping_div(j, two), int, Span::ZERO);
            }
        }
        let one = builder.push(int_const(1), int, Span::ZERO);
        let i2 = builder.push(add(i, one), int, Span::ZERO);
        let back = match work {
            Work::LeavesEarly => Terminator::JumpIf { cond: c, then_to: exit, then_args: Vec::new(), else_to: header, else_args: vec![i2] },
            _ => jump(header, vec![i2]),
        };
        builder.terminate(back, Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(if returns_i { i } else { j }), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        Counted { id, n, j }
    }

    fn has_a_loop(f: &Fixture, id: BodyId) -> bool {
        f.module.body(id).blocks().any(|(_, block)| block.terminator().as_loop().is_some())
    }

    /// The block the entry jumps to, and what it returns.
    fn returned_after_the_entry(f: &Fixture, id: BodyId) -> ValueId {
        let body = f.module.body(id);
        let Terminator::Jump { to, args } = body.block(body.entry()).terminator() else {
            panic!("the entry jumps on:\n{}", f.snapshot(id))
        };
        assert!(args.is_empty());
        match body.block(*to).terminator() {
            Terminator::Return { value, .. } => *value,
            other => panic!("the entry jumps to the exit, which returns, not {other:?}:\n{}", f.snapshot(id)),
        }
    }

    /// The body steps `i`, which nothing after the loop reads: nothing in it
    /// is observable and a `For` finishes, so the entry jumps to the exit
    /// and the loop's two blocks go.
    #[test]
    fn an_empty_counted_loop_goes() {
        let mut f = Fixture::new();
        let Counted { id, j, .. } = counted(&mut f, Work::Nothing, false);

        f.run::<LoopDeletion>(id);

        assert!(!has_a_loop(&f, id), "{}", f.snapshot(id));
        assert_eq!(returned_after_the_entry(&f, id), j);
        assert_eq!(f.module.body(id).blocks().count(), 2, "{}", f.snapshot(id));
        assert_whole(&f.module, id);
    }

    /// `return i` after the loop: `i` = `{0, +, 1}` leaves as `0 + n·1`
    /// from the exact count, made in the entry as `n as i64`, and the loop
    /// goes.
    #[test]
    fn a_loop_whose_exit_value_is_its_last_recurrence_value_goes_with_the_value_made() {
        let mut f = Fixture::new();
        let Counted { id, n, .. } = counted(&mut f, Work::Nothing, true);

        f.run::<LoopDeletion>(id);

        assert!(!has_a_loop(&f, id), "{}", f.snapshot(id));
        let body = f.module.body(id);
        let leaving = returned_after_the_entry(&f, id);
        let Definition::Inst(made) = body.definition(leaving) else { panic!("the value is made") };
        assert!(matches!(body.inst(made).op(), Op::Cast { src, to: CastTy::Int(IntTy::I64) } if *src == n), "{}", f.snapshot(id));
        assert_eq!(body.block_of(made), body.entry());
        assert_whole(&f.module, id);
    }

    /// The counterfactual of the empty loop with an opaque call in it.
    #[test]
    fn a_loop_with_an_effect_stays() {
        let mut f = Fixture::new();
        let Counted { id, .. } = counted(&mut f, Work::Effect, false);
        let before = f.snapshot(id);

        f.run::<LoopDeletion>(id);

        assert_eq!(f.snapshot(id), before);
    }

    /// `j / x` may divide by zero, so the loop may trap and stays.
    #[test]
    fn a_loop_that_may_trap_stays() {
        let mut f = Fixture::new();
        let Counted { id, .. } = counted(&mut f, Work::DivisionByAnyValue, false);
        let before = f.snapshot(id);

        f.run::<LoopDeletion>(id);

        assert_eq!(f.snapshot(id), before);
    }

    /// The counterfactual: `j / two` never divides by zero, which `Raise`
    /// proves, so the loop goes although a division may trap by its kind.
    #[test]
    fn a_loop_whose_division_is_proven_clear_goes() {
        let mut f = Fixture::new();
        let Counted { id, j, .. } = counted(&mut f, Work::DivisionByTwo, false);

        f.run::<LoopDeletion>(id);

        assert!(!has_a_loop(&f, id), "{}", f.snapshot(id));
        assert_eq!(returned_after_the_entry(&f, id), j);
        assert_whole(&f.module, id);
    }

    /// A way out of the body to the exit when `c` holds makes the count at
    /// most `n`, so the `i` the exit returns is not `S + n·D` on every run,
    /// and the loop stays.
    #[test]
    fn an_exit_value_with_only_an_at_most_count_keeps_the_loop() {
        let mut f = Fixture::new();
        let Counted { id, .. } = counted(&mut f, Work::LeavesEarly, true);
        let before = f.snapshot(id);

        f.run::<LoopDeletion>(id);

        assert_eq!(f.snapshot(id), before);
    }

    /// The counterfactual: the same loop returning `j`, which it does not
    /// define, goes, its early way out going to the exit too.
    #[test]
    fn an_at_most_loop_giving_no_value_goes() {
        let mut f = Fixture::new();
        let Counted { id, j, .. } = counted(&mut f, Work::LeavesEarly, false);

        f.run::<LoopDeletion>(id);

        assert!(!has_a_loop(&f, id), "{}", f.snapshot(id));
        assert_eq!(returned_after_the_entry(&f, id), j);
        assert_whole(&f.module, id);
    }

    /// ```text
    /// entry(x, c):  zero = 0; j = max(x, zero); jump header(zero)
    /// header(i):    t = <c, or i < j>; while t { body } else exit
    /// body:         one = 1; i2 = i + one; jump header(i2)
    /// exit:         return j
    /// ```
    fn while_loop(f: &mut Fixture, counted: bool) -> (BodyId, ValueId) {
        let (int, bool, nx, nc) = (f.int, f.bool, f.name("x"), f.name("c"));
        let mut builder: Builder<'_> = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let c = builder.param(nc, bool, Span::ZERO);
        let zero = builder.push(int_const(0), int, Span::ZERO);
        let j = builder.push(binary(BinOp::Max, x, zero), int, Span::ZERO);
        let header = builder.new_block(&[int]);
        let i = builder.block_params(header)[0];
        let [loop_body, exit] = [(); 2].map(|()| builder.new_block(&[]));
        builder.terminate(jump(header, vec![zero]), Span::ZERO);
        builder.switch_to(header);
        let test = match counted {
            true => builder.push(binary(BinOp::Lt, i, j), bool, Span::ZERO),
            false => c,
        };
        builder.terminate(Terminator::While { cond: test, body: loop_body, body_args: Vec::new(), exit, exit_args: Vec::new() }, Span::ZERO);
        builder.switch_to(loop_body);
        let one = builder.push(int_const(1), int, Span::ZERO);
        let i2 = builder.push(add(i, one), int, Span::ZERO);
        builder.terminate(jump(header, vec![i2]), Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(j), Span::ZERO);
        (builder.seal().expect("the body has no open state"), j)
    }

    /// `while c`: no count, so the loop may not finish and stays.
    #[test]
    fn a_while_with_no_count_stays() {
        let mut f = Fixture::new();
        let (id, _) = while_loop(&mut f, false);
        let before = f.snapshot(id);

        f.run::<LoopDeletion>(id);

        assert_eq!(f.snapshot(id), before);
    }

    /// The counterfactual: `while i < j` counts `j`, so it finishes, and it
    /// goes with its test.
    #[test]
    fn a_counted_while_goes() {
        let mut f = Fixture::new();
        let (id, j) = while_loop(&mut f, true);

        f.run::<LoopDeletion>(id);

        assert!(!has_a_loop(&f, id), "{}", f.snapshot(id));
        assert_eq!(returned_after_the_entry(&f, id), j);
        assert_whole(&f.module, id);
    }
}
