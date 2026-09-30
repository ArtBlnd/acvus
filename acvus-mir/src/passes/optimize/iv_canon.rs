//! Induction-variable simplification over a `For` (IndVarSimplify; compilation
//! §4: a loop's join holds as little as it can).
//!
//! In a loop the scalar evolution reads, the canonical induction variable is
//! the `For`'s counter. The pass does two rewrites, each an edit of the loop
//! composed with the body's value edits:
//! 1. A use in the body of a header parameter whose evolution is an affine
//!    recurrence `{S, +, D}` of the loop, other than its own step, reads
//!    `S + k·D` made by `expand` from the counter's position `k`. Where no use
//!    is left but the step, the parameter leaves the join (`remove_join`); a
//!    use the body does not dominate keeps it.
//! 2. A value the loop gives its exit, or a value defined in the loop and used
//!    after it, whose evolution at the exit is computable from an exact trip
//!    count (`S + n·D`), is made by `expand` after the loop instead
//!    (`remove_exit`, or a replacement of the uses outside the loop).
//!
//! A value computed in either rewrite that the run as written did not compute
//! steers the run only where it is proven within its type by the interval
//! domain or computed wider (undefined behaviour §2).
//!
//! Form: F2 (loops canonical), SSA, in and out.
//!
//! Cost: O(loops · the loop's header and exit parameters) rewrites, each
//! O(the edits it composes), over the scalar evolution the manager keeps.
//!
//! Meaning: a recurrence's value at the k-th iteration is `S + k·D` exactly
//! (integer arithmetic means the integers', undefined behaviour §2), and at
//! the exit `S + n·D`; every rewritten use reads the value it read before, and
//! nothing traps, ends or orders differently (compilation §1). No type
//! changes; extern-defined types are untouched (compilation §3).

use acvus_ast::Span;
use rustc_hash::FxHashSet;

use crate::ir::ValueId;
use crate::ir::cfg::edit::{CannotEditLoop, Editor, Loop, Position};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, InstId, Module, Site, Terminator};
use crate::passes::analysis::{AnalysisManager, Dominators, Evolution, Evolutions, IntervalValues, LoopId, LoopTree, ScalarEvolution, TripCount};
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::expand::Expander;
use crate::passes::optimize::{BodyTransform, inner_first};
use crate::ty::{IntTy, Ty, TyKind};

pub struct IvCanon;

impl BodyTransform for IvCanon {
    const NAME: &'static str = "iv_canon";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let loops = am.get::<LoopTree>(module, body);
        let counted: Vec<Counted> =
            inner_first(&loops).into_iter().filter_map(|of| Counted::of(module.body(body), &loops, of)).collect();
        if counted.is_empty() {
            return Changes::default();
        }
        let evolutions = am.get::<ScalarEvolution>(module, body);
        let intervals = am.get::<IntervalValues>(module, body);
        let dominators = am.get::<Dominators>(module, body);
        let len_ty = module.types_mut().intern(TyKind::Int(IntTy::U64));
        let expander = Expander { evolutions: &evolutions, loops: &loops, intervals: &intervals, dominators: &dominators, len_ty };

        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Loop>::new(module, body, &mut facts);
        for counted in &counted {
            let rewrite = Rewrite { expander: &expander, evolutions: &evolutions, loops: &loops, dominators: &dominators, counted };
            if let Some(TripCount::Exact(count)) = evolutions.trip_count(counted.of) {
                rewrite.exit_values(&mut editor, count);
            }
            rewrite.recurrences(&mut editor);
        }
        editor.finish()
    }
}

/// A loop a `For` heads, its body inside it and its exit outside.
struct Counted {
    of: LoopId,
    header: BlockId,
    loop_body: BlockId,
    exit: BlockId,
}

impl Counted {
    fn of(body: &Body, loops: &LoopTree, of: LoopId) -> Option<Counted> {
        let header = loops.header(of);
        let Terminator::For { body: loop_body, exit, .. } = body.block(header).terminator() else {
            return None;
        };
        (loops.contains(of, *loop_body) && !loops.contains(of, *exit)).then_some(Counted {
            of,
            header,
            loop_body: *loop_body,
            exit: *exit,
        })
    }
}

/// The rewrites of one loop, over the facts computed before any edit.
struct Rewrite<'r> {
    expander: &'r Expander<'r>,
    evolutions: &'r Evolutions,
    loops: &'r LoopTree,
    dominators: &'r Dominators,
    counted: &'r Counted,
}

impl Rewrite<'_> {
    /// Rewrite 1: each use in the body of a header parameter that is a
    /// recurrence of the loop, its steps aside, reads `S + k·D` made at the
    /// use; a parameter only its steps read then leaves the join.
    fn recurrences(&self, editor: &mut Editor<'_, Loop>) {
        let params = editor.body().block(self.counted.header).params().to_vec();
        for param in params {
            let evolution = self.evolutions.of(param).clone();
            if !matches!(&evolution, Evolution::Recurrence(recurrence) if recurrence.over == self.counted.of) {
                continue;
            }
            let header_params = editor.body().block(self.counted.header).params();
            let index = header_params.iter().position(|held| *held == param).expect("a parameter before it left the join is the header's");
            let steps = self.steps(editor.body(), index);
            for site in distinct_users(editor.body(), param) {
                if is_step(site, &steps) || !self.in_the_body(editor.body(), site) {
                    continue;
                }
                let place = before_site(editor, site);
                let Some(made) = self.expander.expand(editor, &evolution, editor.body().ty_of(param), place.at, place.span) else {
                    continue;
                };
                if made != param {
                    editor.replace_use(site, param, made).expect("a value made just before its use dominates it");
                }
            }
            if distinct_users(editor.body(), param).iter().all(|site| is_step(*site, &steps)) {
                self.leave_the_join(editor, param, &evolution);
            }
        }
    }

    /// `param`, read by its steps alone, leaves the join, its steps reading
    /// `S + k·D` made at the start of the body, which dominates every step
    /// the body holds; a step in the header keeps it.
    fn leave_the_join(&self, editor: &mut Editor<'_, Loop>, param: ValueId, evolution: &Evolution) {
        let place = block_start(editor, self.counted.loop_body);
        let Some(made) = self.expander.expand(editor, evolution, editor.body().ty_of(param), place.at, place.span) else {
            return;
        };
        if made == param {
            return;
        }
        match editor.remove_join(self.counted.header, param, made) {
            Ok(()) | Err(CannotEditLoop::DoesNotDominate(_)) => {}
            Err(refusal) => panic!("a header parameter of a loop in F2 read by its steps alone leaves the join, but {param:?} is refused: {refusal:?}"),
        }
    }

    /// The instructions that make the header parameter at `index` on the
    /// loop's back edges.
    fn steps(&self, body: &Body, index: usize) -> FxHashSet<InstId> {
        let mut steps = FxHashSet::default();
        for block in self.loops.blocks(self.counted.of) {
            for edge in body.block(*block).terminator().edges().iter().filter(|edge| edge.to == self.counted.header) {
                let argument = index.checked_sub(edge.supplied).map(|at| edge.args[at]);
                if let Some(Definition::Inst(step)) = argument.map(|argument| body.definition(argument)) {
                    steps.insert(step);
                }
            }
        }
        steps
    }

    /// Whether `site` is in the loop and the body block dominates it.
    fn in_the_body(&self, body: &Body, site: Site) -> bool {
        let block = site_block(body, site);
        self.loops.contains(self.counted.of, block) && self.dominators.dominates(self.counted.loop_body, block)
    }

    /// Rewrite 2, over an exact count `n`: each exit parameter filled by a
    /// recurrence of the loop, and each use after the loop of a value the
    /// loop defines as such a recurrence, reads `S + n·D` made at the start
    /// of the exit. The exit is entered from the header alone, where the
    /// value leaving is the one of the last header visit.
    fn exit_values(&self, editor: &mut Editor<'_, Loop>, count: &Evolution) {
        let exit = self.counted.exit;
        if editor.body().predecessors().get(&exit).map(|preds| preds[..] == [self.counted.header]) != Some(true) {
            return;
        }
        let Terminator::For { exit_args, .. } = editor.body().block(self.counted.header).terminator() else {
            panic!("the loop keeps its For head")
        };
        let filled: Vec<(ValueId, ValueId)> =
            editor.body().block(exit).params().iter().copied().zip(exit_args.iter().copied()).collect();
        for (param, argument) in filled {
            let Some(made) = self.at_exit(editor, argument, count, editor.body().ty_of(param)) else {
                continue;
            };
            match editor.remove_exit(self.counted.header, param, made) {
                Ok(()) => {}
                Err(refusal) => panic!("an exit parameter of a loop in F2 takes a value made at the exit's start, but {param:?} is refused: {refusal:?}"),
            }
        }
        for value in self.defined_in_the_loop(editor.body()) {
            let after: Vec<Site> = distinct_users(editor.body(), value)
                .into_iter()
                .filter(|site| self.dominators.dominates(exit, site_block(editor.body(), *site)))
                .collect();
            if after.is_empty() {
                continue;
            }
            let Some(made) = self.at_exit(editor, value, count, editor.body().ty_of(value)) else {
                continue;
            };
            for site in after {
                editor.replace_use(site, value, made).expect("a value made at the exit's start dominates what the exit dominates");
            }
        }
    }

    /// `value`'s evolution at the exit, `S + n·D`, made at the start of
    /// the exit, where it is a recurrence of the loop.
    fn at_exit(&self, editor: &mut Editor<'_, Loop>, value: ValueId, count: &Evolution, ty: Ty) -> Option<ValueId> {
        let Evolution::Recurrence(recurrence) = self.evolutions.of(value) else {
            return None;
        };
        if recurrence.over != self.counted.of {
            return None;
        }
        let risen = self.evolutions.product(count, &Evolution::Invariant(recurrence.step.clone()));
        let leaving = self.evolutions.sum(&recurrence.start, &risen);
        let place = block_start(editor, self.counted.exit);
        self.expander.expand(editor, &leaving, ty, place.at, place.span)
    }

    /// The values the loop's blocks define: their parameters and results.
    fn defined_in_the_loop(&self, body: &Body) -> Vec<ValueId> {
        let mut values = Vec::new();
        for block in self.loops.blocks(self.counted.of) {
            let record = body.block(*block);
            values.extend(record.params().iter().copied());
            values.extend(record.insts().filter_map(|inst| inst.result()));
        }
        values
    }
}

/// The sites that read `value`, each once.
fn distinct_users(body: &Body, value: ValueId) -> Vec<Site> {
    let mut seen = FxHashSet::default();
    body.users(value).filter(|site| seen.insert(*site)).collect()
}

fn is_step(site: Site, steps: &FxHashSet<InstId>) -> bool {
    matches!(site, Site::Inst(inst) if steps.contains(&inst))
}

fn site_block(body: &Body, site: Site) -> BlockId {
    match site {
        Site::Inst(inst) => body.block_of(inst),
        Site::Term(block) => block,
    }
}

/// Where a value is made, and the span of what stands there.
#[derive(Clone, Copy)]
struct MadeAt {
    at: Position,
    span: Span,
}

/// Just before `site`.
fn before_site(editor: &Editor<'_, Loop>, site: Site) -> MadeAt {
    match site {
        Site::Inst(inst) => MadeAt { at: Position::Before(editor.inst_ref(inst)), span: editor.body().inst(inst).span() },
        Site::Term(block) => MadeAt { at: Position::End(block), span: editor.body().block(block).terminator_span() },
    }
}

/// The first position of `block`.
fn block_start(editor: &Editor<'_, Loop>, block: BlockId) -> MadeAt {
    match editor.first_inst(block) {
        Some(first) => MadeAt { at: Position::Before(first), span: editor.inst_of(first).span() },
        None => MadeAt { at: Position::End(block), span: editor.body().block(block).terminator_span() },
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::cfg::{Builder, Op};
    use crate::ir::{BinOp, ForSource};
    use crate::passes::optimize::tests::{Fixture, add, assert_whole, int_const, ret};
    use crate::ty::CastTy;

    fn binary(op: BinOp, left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op, left, right }
    }

    fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
        Terminator::Jump { to, args }
    }

    fn range_loop(at: ValueId, hi: ValueId, body: BlockId, exit: BlockId) -> Terminator {
        Terminator::For { source: ForSource::Range { at, hi }, body, body_args: Vec::new(), exit, exit_args: Vec::new() }
    }

    /// What every loop here counts: `zero = 0; j = max(x, zero); zu = 0;
    /// n = j as u64`, so the loop runs `0..n`, `n` in `[0, max(ty)]`.
    struct Count {
        zero: ValueId,
        j: ValueId,
        zu: ValueId,
        n: ValueId,
    }

    fn count(builder: &mut Builder<'_>, x: ValueId, ty: Ty, u64: Ty) -> Count {
        let zero = builder.push(int_const(0), ty, Span::ZERO);
        let j = builder.push(binary(BinOp::Max, x, zero), ty, Span::ZERO);
        let zu = builder.push(int_const(0), u64, Span::ZERO);
        let n = builder.push(Op::Cast { src: j, to: CastTy::Int(IntTy::U64) }, u64, Span::ZERO);
        Count { zero, j, zu, n }
    }

    fn defining_op(body: &Body, value: ValueId) -> &Op {
        let Definition::Inst(inst) = body.definition(value) else {
            panic!("{value:?} is an instruction's result")
        };
        body.inst(inst).op()
    }

    fn defined_in(body: &Body, value: ValueId) -> BlockId {
        let Definition::Inst(inst) = body.definition(value) else {
            panic!("{value:?} is an instruction's result")
        };
        body.block_of(inst)
    }

    /// Whether `value` is `counter as <width>`.
    fn is_counter_cast(body: &Body, value: ValueId, counter: ValueId, width: IntTy) -> bool {
        matches!(defining_op(body, value), Op::Cast { src, to: CastTy::Int(to) } if *src == counter && *to == width)
    }

    fn returned(body: &Body, block: BlockId) -> ValueId {
        match body.block(block).terminator() {
            Terminator::Return { value, .. } => *value,
            other => panic!("{block:?} returns, not {other:?}"),
        }
    }

    /// The `i64` loop with `i`, `{0, +, 1}`, carried and its header running
    /// `<header>`:
    ///
    /// ```text
    /// entry(x):  <count>; two = 2; jump header(zero)
    /// header(i): <header>; for k in zu..n { body } else exit
    /// body(k):   w = i * two; one = 1; i2 = i + one; jump header(i2)
    /// exit:      return <i or j>
    /// ```
    struct Carrying {
        id: BodyId,
        header: BlockId,
        loop_body: BlockId,
        exit: BlockId,
        i: ValueId,
        w: ValueId,
        count: Count,
    }

    fn carrying(f: &mut Fixture, header_reads_i: bool, exit_returns_i: bool) -> Carrying {
        let (int, nx) = (f.int, f.name("x"));
        let u64 = f.module.types_mut().intern(TyKind::Int(IntTy::U64));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let count = count(&mut builder, x, int, u64);
        let two = builder.push(int_const(2), int, Span::ZERO);
        let header = builder.new_block(&[int]);
        let i = builder.block_params(header)[0];
        let loop_body = builder.new_block(&[u64]);
        let exit = builder.new_block(&[]);
        builder.terminate(jump(header, vec![count.zero]), Span::ZERO);
        builder.switch_to(header);
        if header_reads_i {
            builder.push(binary(BinOp::Mul, i, two), int, Span::ZERO);
        }
        builder.terminate(range_loop(count.zu, count.n, loop_body, exit), Span::ZERO);
        builder.switch_to(loop_body);
        let w = builder.push(binary(BinOp::Mul, i, two), int, Span::ZERO);
        let one = builder.push(int_const(1), int, Span::ZERO);
        let i2 = builder.push(add(i, one), int, Span::ZERO);
        builder.terminate(jump(header, vec![i2]), Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(if exit_returns_i { i } else { count.j }), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        Carrying { id, header, loop_body, exit, i, w, count }
    }

    /// `w = i * two` reads `i` = `{0, +, 1}` = `k`, made as `k as i64`
    /// before it (`k` in `[0, i64::MAX - 1]`); then only the step reads `i`,
    /// so `i` leaves the join.
    #[test]
    fn a_recurrence_use_reads_the_counter_and_the_parameter_leaves_the_join() {
        let mut f = Fixture::new();
        let Carrying { id, header, loop_body, w, .. } = carrying(&mut f, false, false);

        f.run::<IvCanon>(id);

        let body = f.module.body(id);
        let k = body.block(loop_body).params()[0];
        let Op::BinOp { left, .. } = defining_op(body, w) else { panic!("w stays a product") };
        assert!(is_counter_cast(body, *left, k, IntTy::I64), "{}", f.snapshot(id));
        assert!(body.block(header).params().is_empty(), "i leaves the join:\n{}", f.snapshot(id));
        assert_whole(&f.module, id);
    }

    /// The counterfactual with the header reading `i` too: `w` still reads
    /// the counter, but the header's read, which the body does not
    /// dominate, keeps `i` in the join.
    #[test]
    fn a_use_outside_the_body_keeps_the_parameter() {
        let mut f = Fixture::new();
        let Carrying { id, header, loop_body, i, w, .. } = carrying(&mut f, true, false);

        f.run::<IvCanon>(id);

        let body = f.module.body(id);
        let k = body.block(loop_body).params()[0];
        let Op::BinOp { left, .. } = defining_op(body, w) else { panic!("w stays a product") };
        assert!(is_counter_cast(body, *left, k, IntTy::I64), "{}", f.snapshot(id));
        assert_eq!(body.block(header).params(), [i], "{}", f.snapshot(id));
        assert_whole(&f.module, id);
    }

    /// `return i` after the loop: the count is exactly `n`, so `i` leaves as
    /// `0 + n·1`, made in the exit as `n as i64` (`n` in `[0, i64::MAX]`);
    /// only the step reads `i` then, so it leaves the join.
    #[test]
    fn an_exit_value_is_made_from_the_exact_count() {
        let mut f = Fixture::new();
        let Carrying { id, header, exit, count, .. } = carrying(&mut f, false, true);

        f.run::<IvCanon>(id);

        let body = f.module.body(id);
        let leaving = returned(body, exit);
        assert!(is_counter_cast(body, leaving, count.n, IntTy::I64), "{}", f.snapshot(id));
        assert_eq!(defined_in(body, leaving), exit);
        assert!(body.block(header).params().is_empty(), "{}", f.snapshot(id));
        assert_whole(&f.module, id);
    }

    /// The counterfactual with an early way out of the body: the count is
    /// at most `n`, so the value leaving at the exit is not `S + n·D` on
    /// every run, and `return i` keeps reading `i`, which stays carried.
    ///
    /// ```text
    /// entry(x, c):  <count>; jump header(zero)
    /// header(i):    for k in zu..n { body } else exit
    /// body(k):      one = 1; i2 = i + one; jump_if c { leave } else { header(i2) }
    /// leave:        return j
    /// exit:         return i
    /// ```
    #[test]
    fn an_at_most_count_leaves_the_exit_value() {
        let mut f = Fixture::new();
        let (int, bool, nx, nc) = (f.int, f.bool, f.name("x"), f.name("c"));
        let u64 = f.module.types_mut().intern(TyKind::Int(IntTy::U64));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let c = builder.param(nc, bool, Span::ZERO);
        let count = count(&mut builder, x, int, u64);
        let header = builder.new_block(&[int]);
        let i = builder.block_params(header)[0];
        let loop_body = builder.new_block(&[u64]);
        let [leave, exit] = [(); 2].map(|()| builder.new_block(&[]));
        builder.terminate(jump(header, vec![count.zero]), Span::ZERO);
        builder.switch_to(header);
        builder.terminate(range_loop(count.zu, count.n, loop_body, exit), Span::ZERO);
        builder.switch_to(loop_body);
        let one = builder.push(int_const(1), int, Span::ZERO);
        let i2 = builder.push(add(i, one), int, Span::ZERO);
        let leaving =
            Terminator::JumpIf { cond: c, then_to: leave, then_args: Vec::new(), else_to: header, else_args: vec![i2] };
        builder.terminate(leaving, Span::ZERO);
        builder.switch_to(leave);
        builder.terminate(ret(count.j), Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(i), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        f.run::<IvCanon>(id);

        let body = f.module.body(id);
        assert_eq!(returned(body, exit), i, "{}", f.snapshot(id));
        assert_eq!(body.block(header).params(), [i]);
        assert_whole(&f.module, id);
    }

    /// Two `i8` recurrences over `0..n`, `n` at most 127, so `k` in
    /// `[0, 126]`:
    ///
    /// ```text
    /// entry(x):     <count over i8>; one = 1; two = 2; jump header(zero, zero)
    /// header(i, q): for k in zu..n { body } else exit
    /// body(k):      wi = i * two; wq = q * two; i2 = i + one; q2 = q + two; jump header(i2, q2)
    /// exit:         return j
    /// ```
    ///
    /// `i` = `{0, +, 1}` is `k as i8`, within `i8`, so `wi` reads it and `i`
    /// leaves; `q` = `{0, +, 2}` would be `(k as i8)·2`, up to 252, which
    /// `i8` does not hold, so `wq` keeps reading `q`, which stays carried.
    #[test]
    fn a_value_not_proven_within_its_type_is_left_alone() {
        let mut f = Fixture::new();
        let (i8, nx) = (f.i8, f.name("x"));
        let u64 = f.module.types_mut().intern(TyKind::Int(IntTy::U64));
        let mut builder = f.module.build_body(false, i8);
        let x = builder.param(nx, i8, Span::ZERO);
        let count = count(&mut builder, x, i8, u64);
        let one = builder.push(int_const(1), i8, Span::ZERO);
        let two = builder.push(int_const(2), i8, Span::ZERO);
        let header = builder.new_block(&[i8, i8]);
        let [i, q] = builder.block_params(header) else { unreachable!() };
        let (i, q) = (*i, *q);
        let loop_body = builder.new_block(&[u64]);
        let exit = builder.new_block(&[]);
        builder.terminate(jump(header, vec![count.zero, count.zero]), Span::ZERO);
        builder.switch_to(header);
        builder.terminate(range_loop(count.zu, count.n, loop_body, exit), Span::ZERO);
        builder.switch_to(loop_body);
        let wi = builder.push(binary(BinOp::Mul, i, two), i8, Span::ZERO);
        let wq = builder.push(binary(BinOp::Mul, q, two), i8, Span::ZERO);
        let i2 = builder.push(add(i, one), i8, Span::ZERO);
        let q2 = builder.push(add(q, two), i8, Span::ZERO);
        builder.terminate(jump(header, vec![i2, q2]), Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(count.j), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        f.run::<IvCanon>(id);

        let body = f.module.body(id);
        let k = body.block(loop_body).params()[0];
        let Op::BinOp { left, .. } = defining_op(body, wi) else { panic!("wi stays a product") };
        assert!(is_counter_cast(body, *left, k, IntTy::I8), "{}", f.snapshot(id));
        let Op::BinOp { left, .. } = defining_op(body, wq) else { panic!("wq stays a product") };
        assert_eq!(*left, q, "{}", f.snapshot(id));
        assert_eq!(body.block(header).params(), [q], "{}", f.snapshot(id));
        assert_whole(&f.module, id);
    }
}
