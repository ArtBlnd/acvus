//! A `While` whose trip count is known becomes a `For` over a range
//! (compilation §4: a transformation reshapes a loop so that its join holds as
//! little as it can; a value handed from one iteration to the next that the
//! iteration's position decides is computed from the position instead).
//!
//! A `While` is promoted where, for its header's test:
//! 1. the scalar evolution names its trip count `n`, exact or at most (a loop
//!    left early keeps its other exits inside the `For`);
//! 2. the test, whose evaluations the promotion removes, has no effect and
//!    may not trap, and finishes (`analysis::Raise`);
//! 3. `n` is computed in the preheader so that it steers the run soundly: it
//!    is proven within its type by the interval domain, or computed wider
//!    (undefined behaviour §2: the run as written never computed it).
//!
//! The promoted loop is `For` over `0..n` with a `u64` counter, `0` and `n`
//! made in the preheader by `expand` from the scalar evolution (the textbook
//! expander: an enclosing loop's recurrence is that loop's header parameter,
//! an atom its value, so a count that varies with an outer loop is made
//! there); the head's test, which the promotion leaves unread, goes. The pass
//! changes the head alone: every header parameter stays, and rewriting the
//! recurrences from the counter and the exit values from the count is
//! `iv_canon`'s, which runs next (the standard split of loop canonicalization
//! from induction-variable simplification). `n` is made in the test's type
//! where the interval domain proves it within that type, else in `i64` where
//! the type is 32 bits or narrower, then cast to `u64`; otherwise condition 3
//! does not hold. A stream head (`while let Some(x) = next(&mut it)`) is not
//! this pass's: its head comes from declared sources, named by the loop
//! analysis to come.
//!
//! Form: F2, SSA, in and out.
//!
//! Cost: O(loops) promotions, each O(the loop's header parameters and its
//! blocks) edits, over the scalar evolution and the endings the manager keeps.
//!
//! Meaning: the promoted loop runs its body `n` times, as the `While` did,
//! every header parameter carried as before; the removed tests had no effect,
//! could not trap and finished, so no run's effects, trap or ending changes
//! (compilation §1). No type changes; extern-defined types are
//! untouched (compilation §3).

use acvus_ast::Span;
use rustc_hash::FxHashSet;

use crate::ir::cfg::edit::{CannotEditLoop, Editor, Head, Loop, NoFacts, Position, Remove, Unremovable};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, InstId, Module, Op, Terminator};
use crate::ir::{ForSource, ValueId};
use crate::passes::analysis::{
    AnalysisManager, Dominators, Endings, Evolution, Evolutions, IntervalValues, LoopId, LoopTree, Raise, ScalarEvolution, TripCount,
};
use crate::passes::domain::Polynomial;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::expand::Expander;
use crate::passes::optimize::{BodyTransform, inner_first, union_of};
use crate::ty::{CastTy, IntTy, Ty, TyKind};

pub struct WhileToFor;

impl BodyTransform for WhileToFor {
    const NAME: &'static str = "while_to_for";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let loops = am.get::<LoopTree>(module, body);
        let evolutions = am.get::<ScalarEvolution>(module, body);
        let endings = am.get::<Raise>(module, body);
        let counted: Vec<Counted> = inner_first(&loops)
            .into_iter()
            .filter_map(|of| Counted::of(module.body(body), &loops, &evolutions, &endings, of))
            .collect();
        if counted.is_empty() {
            return Changes::default();
        }
        let intervals = am.get::<IntervalValues>(module, body);
        let dominators = am.get::<Dominators>(module, body);
        let widths = Widths {
            count: module.types_mut().intern(TyKind::Int(IntTy::U64)),
            wider: module.types_mut().intern(TyKind::Int(IntTy::I64)),
        };
        let expander = Expander::new(&evolutions, &loops, &intervals, &dominators, widths.count);

        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Loop>::new(module, body, &mut facts);
        let promoted: Vec<&Counted> = counted.iter().filter(|counted| promote(&mut editor, &expander, &widths, counted)).collect();
        let heads = editor.finish();

        let mut no_facts = NoFacts;
        let mut editor = Editor::<Remove>::new(module, body, &mut no_facts);
        for counted in promoted {
            remove_test(&mut editor, counted);
        }
        union_of([heads, editor.finish()])
    }
}

/// The types a count is made in besides the test's own: `count`, `u64`, the
/// counter's, and `wider`, `i64`.
struct Widths {
    count: Ty,
    wider: Ty,
}

/// A `While` meeting conditions 1 and 2, as it stood before any edit.
struct Counted {
    header: BlockId,
    preheader: BlockId,
    count: Evolution,
    /// The type of the values the test compares.
    test_ty: Ty,
    /// The instructions of the loop the test reads, its comparison
    /// included, which all stand in the header.
    test: FxHashSet<InstId>,
    span: Span,
}

impl Counted {
    fn of(body: &Body, loops: &LoopTree, evolutions: &Evolutions, endings: &Endings, of: LoopId) -> Option<Counted> {
        let header = loops.header(of);
        let Terminator::While { cond, .. } = body.block(header).terminator() else {
            return None;
        };
        let (TripCount::Exact(count) | TripCount::AtMost(count)) = evolutions.trip_count(of)?;
        let test = test_of(body, loops, of, *cond);
        if !test.iter().all(|inst| !endings.may_trap(*inst) && endings.finishes(*inst)) {
            return None;
        }
        let Definition::Inst(comparison) = body.definition(*cond) else {
            panic!("a While the scalar evolution counts tests a comparison its header makes, not {cond:?}")
        };
        let Op::BinOp { left, .. } = body.inst(comparison).op() else {
            panic!("a While the scalar evolution counts tests a comparison of integers, not {:?}", body.inst(comparison).op())
        };
        let preheader = loops.preheader(of).expect("F2 in: every loop has a preheader");
        Some(Counted {
            header,
            preheader,
            count: count.clone(),
            test_ty: body.ty_of(*left),
            test,
            span: body.inst(comparison).span(),
        })
    }
}

/// `cond`'s defining instruction and every instruction of `of` it reads,
/// transitively.
fn test_of(body: &Body, loops: &LoopTree, of: LoopId, cond: ValueId) -> FxHashSet<InstId> {
    let mut test = FxHashSet::default();
    let mut pending = vec![cond];
    while let Some(value) = pending.pop() {
        let Definition::Inst(inst) = body.definition(value) else {
            continue;
        };
        if !loops.contains(of, body.block_of(inst)) || !test.insert(inst) {
            continue;
        }
        pending.extend(body.inst(inst).operands());
    }
    test
}

/// Makes `0` and `n` in the preheader and replaces the head, where `n` is
/// made (condition 3): in the test's type, else in `i64` where the test's
/// type is 32 bits or narrower, then cast to `u64`.
fn promote(editor: &mut Editor<'_, Loop>, expander: &Expander<'_>, widths: &Widths, counted: &Counted) -> bool {
    let at = Position::End(counted.preheader);
    let span = counted.span;
    let TyKind::Int(test_width) = *editor.types().kind(counted.test_ty) else {
        panic!("a While the scalar evolution counts compares integers")
    };
    let made = expander.expand(editor, &counted.count, counted.test_ty, at, span).or_else(|| match test_width.bits() <= 32 {
        true => expander.expand(editor, &counted.count, widths.wider, at, span),
        false => None,
    });
    let Some(n) = made else {
        return false;
    };
    let n = match editor.body().ty_of(n) == widths.count {
        true => n,
        false => editor.insert(at, Op::Cast { src: n, to: CastTy::Int(IntTy::U64) }, widths.count, span),
    };
    let zero = expander
        .expand(editor, &Evolution::Invariant(Polynomial::zero()), widths.count, at, span)
        .expect("zero lies within u64");
    let head = Head::For { source: ForSource::Range { at: zero, hi: n } };
    match editor.set_head(counted.header, head, &[]) {
        Ok(_) => true,
        Err(CannotEditLoop::BodyEnteredElsewhere) => false,
        Err(refusal) => panic!(
            "a While in F2 takes a For over 0..n made in its preheader, but {:?} was refused: {refusal:?}",
            counted.header
        ),
    }
}

/// Removes the test the new head leaves unread, users before what they
/// read. A test instruction something else still reads stays.
fn remove_test(editor: &mut Editor<'_, Remove>, counted: &Counted) {
    for inst in editor.insts_of(counted.header).into_iter().rev() {
        if !counted.test.contains(&inst.id()) {
            continue;
        }
        match editor.remove(inst) {
            Ok(()) | Err(Unremovable::StillUsed) => {}
            Err(refusal @ (Unremovable::Traps | Unremovable::Writes | Unremovable::Effect)) => panic!(
                "a test the scalar evolution counts is made of constants, lengths and integer arithmetic, which neither trap, write nor have an effect, but {inst:?} is refused: {refusal:?}"
            ),
        }
    }
}

/// No test here blocks a promotion on condition 2, and that is derived, not
/// missed. The scalar evolution names a `While`'s count only where its
/// condition compares two values whose forms it reads at the header, and a
/// form inside a loop is made only by a constant, a length, `+`, `-`, `*`,
/// negation or a parameter; the header dominates the loop, so every
/// instruction of the loop the test reads stands in the header and is one of
/// these, none of which traps, has an effect or fails to finish. Condition 1
/// thus implies condition 2 for every test the evolution reads, and the
/// `Raise` check stays as the guard should the evolution come to read more.
#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::BinOp;
    use crate::ir::cfg::Builder;
    use crate::passes::optimize::tests::{Fixture, add, assert_whole, int_const, ret};

    fn binary(op: BinOp, left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op, left, right }
    }

    fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
        Terminator::Jump { to, args }
    }

    fn while_loop(cond: ValueId, body: BlockId, exit: BlockId) -> Terminator {
        Terminator::While { cond, body, body_args: Vec::new(), exit, exit_args: Vec::new() }
    }

    /// How the counted loop's bound is made from the parameter `x`.
    #[derive(Clone, Copy)]
    enum Bound {
        /// `j = max(x, 0)`, never negative.
        NeverNegative,
        /// `j = x`, of either sign.
        AnySign,
    }

    /// A counted `While` over `ty`:
    ///
    /// ```text
    /// entry(x):   zero = 0; j = <bound>; jump header(zero)
    /// header(i):  c = i <op> j; while c { body } else exit
    /// body:       one = 1; i2 = i + one; jump header(i2)
    /// exit:       return i
    /// ```
    struct Counting {
        id: BodyId,
        header: BlockId,
        loop_body: BlockId,
        entry: BlockId,
        i: ValueId,
        j: ValueId,
    }

    fn counting(f: &mut Fixture, ty: Ty, op: BinOp, bound: Bound) -> Counting {
        let (bool, nx) = (f.bool, f.name("x"));
        let mut builder = f.module.build_body(false, ty);
        let x = builder.param(nx, ty, Span::ZERO);
        let zero = builder.push(int_const(0), ty, Span::ZERO);
        let j = match bound {
            Bound::NeverNegative => builder.push(binary(BinOp::Max, x, zero), ty, Span::ZERO),
            Bound::AnySign => x,
        };
        let header = builder.new_block(&[ty]);
        let i = builder.block_params(header)[0];
        let [loop_body, exit] = [(); 2].map(|()| builder.new_block(&[]));
        builder.terminate(jump(header, vec![zero]), Span::ZERO);
        builder.switch_to(header);
        let c = builder.push(binary(op, i, j), bool, Span::ZERO);
        builder.terminate(while_loop(c, loop_body, exit), Span::ZERO);
        builder.switch_to(loop_body);
        let one = builder.push(int_const(1), ty, Span::ZERO);
        let i2 = builder.push(add(i, one), ty, Span::ZERO);
        builder.terminate(jump(header, vec![i2]), Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(i), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();
        Counting { id, header, loop_body, entry, i, j }
    }

    fn defining_op(body: &Body, value: ValueId) -> &Op {
        let Definition::Inst(inst) = body.definition(value) else {
            panic!("{value:?} is an instruction's result")
        };
        body.inst(inst).op()
    }

    /// The range a `For` heading `header` runs over.
    fn range_of(body: &Body, header: BlockId) -> (ValueId, ValueId) {
        match body.block(header).terminator() {
            Terminator::For { source: ForSource::Range { at, hi }, .. } => (*at, *hi),
            other => panic!("{header:?} heads a For over a range, not {other:?}"),
        }
    }

    /// A value made in `block`: `0` as a `u64`.
    fn assert_zero_in(f: &Fixture, id: BodyId, block: BlockId, value: ValueId) {
        let body = f.module.body(id);
        assert!(matches!(defining_op(body, value), Op::Const { value: acvus_ast::Literal::Int(0) }));
        assert_eq!(f.module.types().kind(body.ty_of(value)), &TyKind::Int(IntTy::U64));
        let Definition::Inst(inst) = body.definition(value) else { unreachable!() };
        assert_eq!(body.block_of(inst), block);
    }

    /// The source of the `u64` cast `value` is.
    fn cast_source(body: &Body, value: ValueId) -> ValueId {
        match defining_op(body, value) {
            Op::Cast { src, to: CastTy::Int(IntTy::U64) } => *src,
            other => panic!("{value:?} is a cast to u64, not {other:?}"),
        }
    }

    /// `i = 0; while i < j { i = i + 1 }` with `j = max(x, 0)`: the count is
    /// `j`, within `i64`, so the loop runs over `0..(j as u64)`, both made in
    /// the entry, its preheader; the unread test goes and `i` stays carried.
    #[test]
    fn a_counted_while_becomes_a_for_over_its_count() {
        let mut f = Fixture::new();
        let int = f.int;
        let Counting { id, header, loop_body, entry, i, j } = counting(&mut f, int, BinOp::Lt, Bound::NeverNegative);

        let changes = f.run::<WhileToFor>(id);

        let body = f.module.body(id);
        let (at, hi) = range_of(body, header);
        assert_zero_in(&f, id, entry, at);
        assert_eq!(cast_source(body, hi), j, "{}", f.snapshot(id));
        assert_eq!(body.block(header).params(), [i], "every header parameter stays");
        assert!(f.ops(id, header).is_empty(), "the test the For leaves unread goes:\n{}", f.snapshot(id));
        let [counter] = body.block(loop_body).params() else { panic!("the body takes the counter") };
        assert_eq!(f.module.types().kind(body.ty_of(*counter)), &TyKind::Int(IntTy::U64));
        assert!(changes.touched(crate::ir::cfg::Aspect::CfgShape));
        assert_whole(&f.module, id);
    }

    /// The counterfactual of the promoted loop with `j = x`: `x - i` may be
    /// negative from the start, so the evolution names no count and the
    /// loop stays as it was.
    #[test]
    fn a_while_with_no_trip_count_stays() {
        let mut f = Fixture::new();
        let int = f.int;
        let Counting { id, .. } = counting(&mut f, int, BinOp::Lt, Bound::AnySign);
        let before = f.snapshot(id);

        let changes = f.run::<WhileToFor>(id);

        assert_eq!(f.snapshot(id), before);
        assert!(changes.aspects().next().is_none());
    }

    /// `i <= j` over `i64`: the count `j + 1` may be `i64::MAX + 1`, which
    /// `i64` does not hold, and no integer type is wider, so condition 3
    /// fails and the loop stays with nothing made.
    #[test]
    fn a_count_not_proven_within_a_64_bit_type_leaves_the_while() {
        let mut f = Fixture::new();
        let int = f.int;
        let Counting { id, .. } = counting(&mut f, int, BinOp::Lte, Bound::NeverNegative);
        let before = f.snapshot(id);

        f.run::<WhileToFor>(id);

        assert_eq!(f.snapshot(id), before);
    }

    /// The counterfactual over `i32`: `j + 1` may be `2^31`, outside `i32`,
    /// so it is made in `i64` from `j as i64`, then cast to `u64`.
    #[test]
    fn a_count_outside_a_32_bit_type_is_made_in_i64() {
        let mut f = Fixture::new();
        let i32 = f.module.types_mut().intern(TyKind::Int(IntTy::I32));
        let Counting { id, header, entry, j, .. } = counting(&mut f, i32, BinOp::Lte, Bound::NeverNegative);

        f.run::<WhileToFor>(id);

        let body = f.module.body(id);
        let (at, hi) = range_of(body, header);
        assert_zero_in(&f, id, entry, at);
        let sum = cast_source(body, hi);
        assert_eq!(f.module.types().kind(body.ty_of(sum)), &TyKind::Int(IntTy::I64));
        let Op::BinOp { op: BinOp::Add, left, right } = defining_op(body, sum) else {
            panic!("the count is j + 1:\n{}", f.snapshot(id))
        };
        let widened = [*left, *right]
            .into_iter()
            .find(|operand| matches!(defining_op(body, *operand), Op::Cast { src, to: CastTy::Int(IntTy::I64) } if *src == j));
        assert!(widened.is_some(), "one addend is j as i64:\n{}", f.snapshot(id));
        assert!(
            [*left, *right].iter().any(|operand| matches!(defining_op(body, *operand), Op::Const { value: acvus_ast::Literal::Int(1) })),
            "the other is 1:\n{}",
            f.snapshot(id)
        );
        assert_whole(&f.module, id);
    }

    /// A triangular nest over `ty`, its inner test `j <op> i`:
    ///
    /// ```text
    /// entry(m):  zero = 0; mm = max(m, zero); jump outer(zero)
    /// outer(i):  c = i < mm; while c { obody } else oexit
    /// obody:     jump inner(zero)
    /// inner(j):  d = j <op> i; while d { ibody } else iexit
    /// ibody:     one = 1; j2 = j + one; jump inner(j2)
    /// iexit:     one' = 1; i2 = i + one'; jump outer(i2)
    /// oexit:     return i
    /// ```
    struct Triangular {
        id: BodyId,
        entry: BlockId,
        outer: BlockId,
        obody: BlockId,
        inner: BlockId,
        i: ValueId,
        j: ValueId,
        mm: ValueId,
    }

    fn triangular(f: &mut Fixture, ty: Ty, op: BinOp) -> Triangular {
        let (bool, nm) = (f.bool, f.name("m"));
        let mut builder: Builder<'_> = f.module.build_body(false, ty);
        let m = builder.param(nm, ty, Span::ZERO);
        let zero = builder.push(int_const(0), ty, Span::ZERO);
        let mm = builder.push(binary(BinOp::Max, m, zero), ty, Span::ZERO);
        let outer = builder.new_block(&[ty]);
        let i = builder.block_params(outer)[0];
        let [obody, oexit] = [(); 2].map(|()| builder.new_block(&[]));
        let inner = builder.new_block(&[ty]);
        let j = builder.block_params(inner)[0];
        let [ibody, iexit] = [(); 2].map(|()| builder.new_block(&[]));
        builder.terminate(jump(outer, vec![zero]), Span::ZERO);
        builder.switch_to(outer);
        let c = builder.push(binary(BinOp::Lt, i, mm), bool, Span::ZERO);
        builder.terminate(while_loop(c, obody, oexit), Span::ZERO);
        builder.switch_to(obody);
        builder.terminate(jump(inner, vec![zero]), Span::ZERO);
        builder.switch_to(inner);
        let d = builder.push(binary(op, j, i), bool, Span::ZERO);
        builder.terminate(while_loop(d, ibody, iexit), Span::ZERO);
        builder.switch_to(ibody);
        let one = builder.push(int_const(1), ty, Span::ZERO);
        let j2 = builder.push(add(j, one), ty, Span::ZERO);
        builder.terminate(jump(inner, vec![j2]), Span::ZERO);
        builder.switch_to(iexit);
        let one = builder.push(int_const(1), ty, Span::ZERO);
        let i2 = builder.push(add(i, one), ty, Span::ZERO);
        builder.terminate(jump(outer, vec![i2]), Span::ZERO);
        builder.switch_to(oexit);
        builder.terminate(ret(i), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();
        Triangular { id, entry, outer, obody, inner, i, j, mm }
    }

    /// `j < i`: the inner count is `{0, +, 1}` over the outer loop, the
    /// evolution of `i`, so it is made in `obody`, the inner preheader, as
    /// `i as u64`; the outer count is `mm`.
    #[test]
    fn a_count_varying_with_an_outer_loop_is_made_from_its_header_parameter() {
        let mut f = Fixture::new();
        let int = f.int;
        let Triangular { id, entry, outer, obody, inner, i, j, mm } = triangular(&mut f, int, BinOp::Lt);

        f.run::<WhileToFor>(id);

        let body = f.module.body(id);
        let (inner_at, inner_hi) = range_of(body, inner);
        assert_zero_in(&f, id, obody, inner_at);
        assert_eq!(cast_source(body, inner_hi), i, "{}", f.snapshot(id));
        let (outer_at, outer_hi) = range_of(body, outer);
        assert_zero_in(&f, id, entry, outer_at);
        assert_eq!(cast_source(body, outer_hi), mm, "{}", f.snapshot(id));
        assert_eq!(body.block(outer).params(), [i]);
        assert_eq!(body.block(inner).params(), [j]);
        assert!(f.ops(id, outer).is_empty() && f.ops(id, inner).is_empty(), "{}", f.snapshot(id));
        assert_whole(&f.module, id);
    }

    /// `j <= i`: the inner count is `{1, +, 1}` over the outer loop, `i`'s
    /// step with a start one higher, so it is `i + 1`. In `obody`, the outer
    /// loop's body, `i` lies in `[0, mm - 1]` by its own range, `mm` at most
    /// the type's greatest value, so `i + 1` is proven within the type and
    /// made there, then cast to `u64`, in `obody`.
    fn assert_the_count_is_i_plus_one(ty_of: fn(&mut Fixture) -> Ty) {
        let mut f = Fixture::new();
        let ty = ty_of(&mut f);
        let Triangular { id, obody, inner, i, .. } = triangular(&mut f, ty, BinOp::Lte);

        f.run::<WhileToFor>(id);

        let body = f.module.body(id);
        let (inner_at, inner_hi) = range_of(body, inner);
        assert_zero_in(&f, id, obody, inner_at);
        let sum = cast_source(body, inner_hi);
        assert_eq!(body.ty_of(sum), ty, "{}", f.snapshot(id));
        let Op::BinOp { op: BinOp::Add, left, right } = defining_op(body, sum) else {
            panic!("the count is i + 1:\n{}", f.snapshot(id))
        };
        assert_eq!(*left, i, "{}", f.snapshot(id));
        assert!(matches!(defining_op(body, *right), Op::Const { value: acvus_ast::Literal::Int(1) }), "{}", f.snapshot(id));
        let Definition::Inst(made) = body.definition(sum) else { unreachable!() };
        assert_eq!(body.block_of(made), obody);
        assert_whole(&f.module, id);
    }

    #[test]
    fn a_count_with_an_outer_parameters_step_is_made_from_it_by_its_offset() {
        assert_the_count_is_i_plus_one(|f| f.int);
    }

    #[test]
    fn a_count_with_an_outer_parameters_step_is_made_in_a_32_bit_type() {
        assert_the_count_is_i_plus_one(|f| f.module.types_mut().intern(TyKind::Int(IntTy::I32)));
    }
}
