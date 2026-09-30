use acvus_ast::{Literal, Span};

use super::*;
use crate::ir::cfg::edit::{Editor, Position, Value};
use crate::ir::cfg::{BlockId, Body, Definition, Stages, Terminator};
use crate::ir::{BinOp, ValueId};
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::tests::{Fixture, add, assign, bool_const, declarations, int_const, jump, ret, take, trapping_div};

/// ```text
/// fn(n, c):
///   x = 1; acc = 0
///   b = x + 2                       folds to 3 once x is a value
///   dead = n * n                    nothing reads it
///   while c {
///     t1 = n + n; t2 = n + n        t2 is t1 again
///     acc = acc + (t1 + t2)         t1 + t2 is invariant in the loop
///   }
///   return acc + b
/// ```
struct Sample {
    body: BodyId,
    n: ValueId,
}

fn sample(f: &mut Fixture) -> Sample {
    let (int, bool, nn, nc) = (f.int, f.bool, f.name("n"), f.name("c"));
    let mut builder = f.module.build_body(false, int);
    let n = builder.param(nn, int, acvus_ast::Span::ZERO);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let x = builder.slot(int, Span::ZERO);
    let acc = builder.slot(int, Span::ZERO);
    let one = builder.push(int_const(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let zero = builder.push(int_const(0), int, Span::ZERO);
    builder.push_unit(assign(acc, zero), Span::ZERO);
    let a = builder.push(take(x), int, Span::ZERO);
    let two = builder.push(int_const(2), int, Span::ZERO);
    let b = builder.push(add(a, two), int, Span::ZERO);
    builder.push(Op::BinOp { op: BinOp::Mul, left: n, right: n }, int, Span::ZERO);
    let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    builder.terminate(Terminator::While { cond: c, stages: Stages::lowered(loop_body), exit, exit_args: Vec::new() }, acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    let t1 = builder.push(add(n, n), int, Span::ZERO);
    let t2 = builder.push(add(n, n), int, Span::ZERO);
    let u = builder.push(add(t1, t2), int, Span::ZERO);
    let current = builder.push(take(acc), int, Span::ZERO);
    let next = builder.push(add(current, u), int, Span::ZERO);
    builder.push_unit(assign(acc, next), Span::ZERO);
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    let last = builder.push(take(acc), int, Span::ZERO);
    let result = builder.push(add(last, b), int, Span::ZERO);
    builder.terminate(ret(result), acvus_ast::Span::ZERO);
    let body = builder.seal().expect("the body has no open state");
    Sample { body, n }
}

fn optimized(f: &mut Fixture, level: Level) {
    let decls = declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    optimize(&mut f.module, &mut am, level);
}

fn defining_op(body: &Body, value: ValueId) -> &Op {
    let Definition::Inst(inst) = body.definition(value) else {
        panic!("{value:?} is an instruction's result: {:?}", body.definition(value))
    };
    body.inst(inst).op()
}

fn is_const_int(body: &Body, value: ValueId, expected: i128) -> bool {
    matches!(defining_op(body, value), Op::Const { value: Literal::Int(held) } if *held == expected)
}

#[test]
fn no_level_changes_nothing() {
    let mut f = Fixture::new();
    let Sample { body, .. } = sample(&mut f);
    let before = f.snapshot(body);

    optimized(&mut f, Level::None);

    assert_eq!(f.snapshot(body), before);
}

/// Derived from the order: P1 promotes both slots, so `acc` becomes the
/// loop header's parameter; P3* folds `b` to 3, replaces `t2` by `t1` and
/// removes the dead product and every constant left unread; P6 sinks the
/// constant 3 to the one block that reads it, then hoists `t1` and
/// `t1 + t1` into the entry, the loop's preheader. What is left:
///
/// ```text
/// entry:  zero = 0; t1 = n + n; u = t1 + t1; jump header(zero)
/// header(acc): while c
/// body:   next = acc + u; jump header(next)
/// exit:   three = 3; r = acc + three; return r
/// ```
#[test]
fn the_full_level_reaches_the_expected_form() {
    let mut f = Fixture::new();
    let Sample { body: id, n } = sample(&mut f);

    optimized(&mut f, Level::Full);

    let body = f.module.body(id);
    assert!(body.slots().is_empty(), "{}", f.snapshot(id));
    let Some((header, cond_stage, exit)) = body.blocks().find_map(|(block, record)| match record.terminator() {
        Terminator::While { stages, exit, .. } => Some((block, stages.body(), *exit)),
        _ => None,
    }) else {
        panic!("the loop stays:\n{}", f.snapshot(id))
    };
    let [acc] = body.block(header).params() else {
        panic!("the header carries acc:\n{}", f.snapshot(id))
    };
    let ops = |block: BlockId| -> Vec<&Op> { body.block(block).insts().map(|inst| inst.op()).collect() };

    let [Op::BinOp { op: BinOp::Add, left, right: u }] = ops(cond_stage)[..] else {
        panic!("the loop body is one addition:\n{}", f.snapshot(id))
    };
    assert_eq!(left, acc);
    let Op::BinOp { op: BinOp::Add, left: t1, right: t1_again } = defining_op(body, *u) else {
        panic!("u is t1 + t1:\n{}", f.snapshot(id))
    };
    assert_eq!(t1, t1_again, "t2 is t1:\n{}", f.snapshot(id));
    assert!(matches!(defining_op(body, *t1), Op::BinOp { left, right, .. } if *left == n && *right == n));
    for hoisted in [*u, *t1] {
        let Definition::Inst(inst) = body.definition(hoisted) else { panic!("{hoisted:?} is an instruction's") };
        assert_eq!(body.block_of(inst), body.entry(), "{hoisted:?} is hoisted into the preheader:\n{}", f.snapshot(id));
    }

    let [Op::Const { .. }, Op::BinOp { op: BinOp::Add, left, right: three }] = ops(exit)[..] else {
        panic!("the exit is a constant and one addition:\n{}", f.snapshot(id))
    };
    assert_eq!(left, acc);
    assert!(is_const_int(body, *three, 3), "b is folded to 3:\n{}", f.snapshot(id));

    let binops = body.blocks().flat_map(|(_, block)| block.insts()).filter(|inst| matches!(inst.op(), Op::BinOp { .. }));
    assert_eq!(binops.count(), 4, "the product and t2 are gone:\n{}", f.snapshot(id));
    let storage = body.blocks().flat_map(|(_, block)| block.insts()).filter(|inst| matches!(inst.op(), Op::Take { .. } | Op::Assign { .. }));
    assert_eq!(storage.count(), 0, "{}", f.snapshot(id));
    assert_eq!(body.blocks().count(), 4, "{}", f.snapshot(id));
}

/// ```text
/// t = true
/// if t { J(5) } else { J(6) }
/// J(p): return p + 1
/// ```
/// The first round takes the then arm and merges `J` into it, so `p` is 5;
/// only then does the second round fold `p + 1` to 6.
#[test]
fn the_scalar_fixpoint_cascades_and_lowers_its_measure_each_round() {
    let mut f = Fixture::new();
    let int = f.int;
    let mut builder = f.module.build_body(false, int);
    let t = builder.push(bool_const(true), f.bool, Span::ZERO);
    let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
    let join = builder.new_block(&[int]);
    let p = builder.block_params(join)[0];
    builder.terminate(Terminator::Diamond {
        cond: t,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    }, acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    let five = builder.push(int_const(5), int, Span::ZERO);
    builder.terminate(Terminator::Jump { to: join, args: vec![five] }, acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    let six = builder.push(int_const(6), int, Span::ZERO);
    builder.terminate(Terminator::Jump { to: join, args: vec![six] }, acvus_ast::Span::ZERO);
    builder.switch_to(join);
    let one = builder.push(int_const(1), int, Span::ZERO);
    let sum = builder.push(add(p, one), int, Span::ZERO);
    builder.terminate(ret(sum), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let decls = declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);

    let measures = scalar_fixpoint(&mut f.module, id, &mut am, scalar_round);

    assert!(measures.len() >= 3, "at least two rounds change the body: {measures:?}\n{}", f.snapshot(id));
    assert!(measures.windows(2).all(|pair| pair[1] < pair[0]), "{measures:?}");
    let body = f.module.body(id);
    let [(_, only)] = body.blocks().collect::<Vec<_>>()[..] else {
        panic!("one block is left:\n{}", f.snapshot(id))
    };
    let Terminator::Return { value, .. } = only.terminator() else { panic!("{:?} returns", only.terminator()) };
    assert!(is_const_int(body, *value, 6), "{}", f.snapshot(id));
    assert_eq!(only.insts().len(), 1, "{}", f.snapshot(id));
}

/// A test-only transform that changes the body without lowering μ: it adds
/// an addition nothing reads, which raises μ's second part.
struct AddsAnUnreadSum;

impl BodyTransform for AddsAnUnreadSum {
    const NAME: &'static str = "adds_an_unread_sum";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let (_, n) = module.body(body).params()[0];
        let (entry, ty) = (module.body(body).entry(), module.body(body).ty_of(n));
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Value>::new(module, body, &mut facts);
        editor.insert(Position::End(entry), add(n, n), ty, Span::ZERO);
        editor.finish()
    }
}

impl Requires for AddsAnUnreadSum {
    const FORM: Form = Form::Lowered;
}

#[test]
#[should_panic(expected = "did not lower its measure")]
fn a_round_that_changes_without_lowering_the_measure_panics() {
    let mut f = Fixture::new();
    let (int, nn) = (f.int, f.name("n"));
    let mut builder = f.module.build_body(false, int);
    let n = builder.param(nn, int, acvus_ast::Span::ZERO);
    builder.terminate(ret(n), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let decls = declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);

    scalar_fixpoint(&mut f.module, id, &mut am, |module, body, am| step::<AddsAnUnreadSum>(module, body, am));
}

/// `x = 1; return x` with `x` a slot nothing lets escape: not F1, which
/// `sccp` requires.
#[test]
#[should_panic(expected = "sccp requires F1")]
fn a_step_that_requires_ssa_refuses_a_promotable_slot() {
    let mut f = Fixture::new();
    let int = f.int;
    let mut builder = f.module.build_body(false, int);
    let x = builder.slot(int, Span::ZERO);
    let one = builder.push(int_const(1), int, Span::ZERO);
    builder.push_unit(assign(x, one), Span::ZERO);
    let read = builder.push(take(x), int, Span::ZERO);
    builder.terminate(ret(read), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");
    let decls = declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);

    step::<Sccp>(&mut f.module, id, &mut am);
}

/// ```text
/// if d { n / n } else { n / 1 }     both may trap, so neither is hoisted
/// while c { n / n }                 may trap, so it stays in the loop
/// return 0
/// ```
/// The loop is entered from both arms, and `licm` gives it a preheader it
/// can hoist nothing into. `licm` runs last in P6 (D73), so no
/// `simplify_cfg` bypasses that empty preheader before P7 requires F2.
#[test]
fn a_loop_keeps_its_empty_preheader_into_p7() {
    let mut f = Fixture::new();
    let (int, bool, nn, nc, nd) = (f.int, f.bool, f.name("n"), f.name("c"), f.name("d"));
    let mut builder = f.module.build_body(false, int);
    let n = builder.param(nn, int, acvus_ast::Span::ZERO);
    let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
    let d = builder.param(nd, bool, acvus_ast::Span::ZERO);
    let [then_block, else_block, join, header, loop_body, exit] = [(); 6].map(|()| builder.new_block(&[]));
    builder.terminate(Terminator::Diamond {
        cond: d,
        then_to: then_block,
        then_args: Vec::new(),
        else_to: else_block,
        else_args: Vec::new(),
        join,
    }, acvus_ast::Span::ZERO);
    builder.switch_to(then_block);
    builder.push(trapping_div(n, n), int, Span::ZERO);
    builder.terminate(jump(join), acvus_ast::Span::ZERO);
    builder.switch_to(else_block);
    let one = builder.push(int_const(1), int, Span::ZERO);
    builder.push(trapping_div(n, one), int, Span::ZERO);
    builder.terminate(jump(join), acvus_ast::Span::ZERO);
    builder.switch_to(join);
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(header);
    builder.terminate(Terminator::While { cond: c, stages: Stages::lowered(loop_body), exit, exit_args: Vec::new() }, acvus_ast::Span::ZERO);
    builder.switch_to(loop_body);
    builder.push(trapping_div(n, n), int, Span::ZERO);
    builder.terminate(jump(header), acvus_ast::Span::ZERO);
    builder.switch_to(exit);
    let zero = builder.push(int_const(0), int, Span::ZERO);
    builder.terminate(ret(zero), acvus_ast::Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    optimized(&mut f, Level::Full);

    let decls = declarations(&f.interner);
    let mut am = AnalysisManager::new(&decls);
    let loops = am.get::<LoopTree>(&f.module, id);
    let [of] = loops.loops().collect::<Vec<_>>()[..] else { panic!("one loop stays:\n{}", f.snapshot(id)) };
    let Some(preheader) = loops.preheader(of) else { panic!("the loop has a preheader:\n{}", f.snapshot(id)) };
    assert!(f.module.body(id).block(preheader).insts().is_empty(), "nothing is hoisted:\n{}", f.snapshot(id));
}
