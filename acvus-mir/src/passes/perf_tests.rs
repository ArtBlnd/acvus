//! Scaling probe (ignored by default): each transform alone, and the whole
//! pipeline, over synthetic bodies of about 100, 1,000 and 10,000
//! instructions in three shapes. It prints each time and the growth from
//! one size to the next, and asserts nothing about time.
//!
//! Run: `cargo test --release -p acvus-mir --lib -- --ignored perf --nocapture --test-threads=1`.
//!
//! A size is skipped where the time it is projected to take, the size
//! below's time multiplied by the growth seen so far (tenfold at least),
//! is over the budget: the probe exists to find the transforms that grow
//! too fast, and those would otherwise run for hours.

use std::time::{Duration, Instant};

use acvus_ast::Span;

use crate::ir::cfg::{BodyId, Module, Op, Stages, Terminator};
use crate::ir::{BinOp, Checked, Overflow, RefTarget, ValueId};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::tests::{Fixture, assign, declarations, int_const, jump, ret, take, wrapping_add};
use crate::passes::optimize::{
    Bce, BodyTransform, Commute, Dce, Dse, Gvn, Licm, Mem2Reg, Reborrow, Schedule, Sccp, SimplifyCfg, Sink, SplitCalls, Sroa,
};
use crate::passes::pipeline::{Level, optimize};
use crate::ty::Mutability;

const SIZES: [usize; 3] = [100, 1_000, 10_000];

/// A size runs only where its projected time is under this.
const BUDGET: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Copy)]
enum Shape {
    /// One block: redundant, foldable and dead values, stores, checks and
    /// calls, one after the other.
    StraightLine,
    /// Nests of three `While` loops, each body computing values invariant
    /// in it and carrying an accumulator.
    NestedLoops,
    /// `Diamond` after `Diamond`, each arm computing a value its join takes.
    Diamonds,
}

const SHAPES: [Shape; 3] = [Shape::StraightLine, Shape::NestedLoops, Shape::Diamonds];

fn wrapping_mul(left: ValueId, right: ValueId) -> Op {
    Op::BinOp { op: BinOp::Mul(Overflow::Wrap), left, right }
}

/// Builds `shape` with about `size` instructions as the module's only body.
fn build(f: &mut Fixture, shape: Shape, size: usize) -> BodyId {
    let (int, bool, ref_int) = (f.int, f.bool, f.ref_int);
    let (commuting, splittable) = (f.call(f.commuting), f.call(f.splittable));
    let (np, nc) = (f.name("p"), f.name("c"));
    let mut builder = f.module.build_body(true, int);
    let p = builder.param(np, int, Span::ZERO);
    let c = builder.param(nc, bool, Span::ZERO);
    let s = builder.slot(int, Span::ZERO);
    let escaping = builder.slot(int, Span::ZERO);
    let acc_slot = builder.slot(int, Span::ZERO);
    let zero = builder.push(int_const(0), int, Span::ZERO);
    builder.push_unit(assign(acc_slot, zero), Span::ZERO);
    builder.push_unit(assign(escaping, zero), Span::ZERO);
    let lent = Op::Ref { target: RefTarget::Var(escaping), path: Vec::new(), mutability: Mutability::Shared };
    let reference = builder.push(lent, ref_int, Span::ZERO);
    let mut acc = zero;
    let mut made = 5;
    while made < size {
        match shape {
            Shape::StraightLine => {
                let k1 = builder.push(int_const(3), int, Span::ZERO);
                let k2 = builder.push(int_const(4), int, Span::ZERO);
                let folded = builder.push(wrapping_add(k1, k2), int, Span::ZERO);
                let t = builder.push(wrapping_add(p, p), int, Span::ZERO);
                let u = builder.push(wrapping_add(p, p), int, Span::ZERO);
                builder.push(wrapping_mul(t, u), int, Span::ZERO);
                builder.push_unit(assign(s, t), Span::ZERO);
                let read = builder.push(take(s), int, Span::ZERO);
                builder.push_unit(assign(escaping, t), Span::ZERO);
                builder.push_unit(assign(escaping, u), Span::ZERO);
                let kept = builder.push(take(escaping), int, Span::ZERO);
                builder.push_unit(Op::Check { op: Checked::Add, left: k1, right: k2 }, Span::ZERO);
                let called = builder.push_effect(commuting.clone(), int, Span::ZERO);
                builder.push_effect(commuting.clone(), int, Span::ZERO);
                let started = builder.push_effect(splittable.clone(), int, Span::ZERO);
                let sum = builder.push(wrapping_add(read, folded), int, Span::ZERO);
                let sum = builder.push(wrapping_add(sum, kept), int, Span::ZERO);
                let sum = builder.push(wrapping_add(sum, called), int, Span::ZERO);
                let sum = builder.push(wrapping_add(sum, started), int, Span::ZERO);
                acc = builder.push(wrapping_add(acc, sum), int, Span::ZERO);
                made += 20;
            }
            Shape::NestedLoops => {
                let headers = [(); 3].map(|()| builder.new_block(&[]));
                let bodies = [(); 3].map(|()| builder.new_block(&[]));
                let exits = [(); 3].map(|()| builder.new_block(&[]));
                builder.terminate(jump(headers[0]), Span::ZERO);
                for depth in 0..3 {
                    builder.switch_to(headers[depth]);
                    let stages = Stages::lowered(bodies[depth]);
                    builder.terminate(Terminator::While { cond: c, stages, exit: exits[depth], exit_args: Vec::new() }, Span::ZERO);
                    builder.switch_to(bodies[depth]);
                    let invariant = builder.push(wrapping_add(p, p), int, Span::ZERO);
                    let scaled = builder.push(wrapping_mul(invariant, p), int, Span::ZERO);
                    let again = builder.push(wrapping_add(p, p), int, Span::ZERO);
                    let current = builder.push(take(acc_slot), int, Span::ZERO);
                    let next = builder.push(wrapping_add(current, scaled), int, Span::ZERO);
                    let next = builder.push(wrapping_add(next, again), int, Span::ZERO);
                    builder.push_unit(assign(acc_slot, next), Span::ZERO);
                    builder.push_effect(commuting.clone(), int, Span::ZERO);
                    match depth {
                        2 => builder.terminate(jump(headers[2]), Span::ZERO),
                        _ => builder.terminate(jump(headers[depth + 1]), Span::ZERO),
                    }
                }
                builder.switch_to(exits[2]);
                builder.terminate(jump(headers[1]), Span::ZERO);
                builder.switch_to(exits[1]);
                builder.terminate(jump(headers[0]), Span::ZERO);
                builder.switch_to(exits[0]);
                made += 24;
            }
            Shape::Diamonds => {
                let t = builder.push(wrapping_add(p, p), int, Span::ZERO);
                let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
                let join = builder.new_block(&[int]);
                builder.terminate(
                    Terminator::Diamond { cond: c, then_to: then_block, then_args: Vec::new(), else_to: else_block, else_args: Vec::new(), join },
                    Span::ZERO,
                );
                builder.switch_to(then_block);
                let again = builder.push(wrapping_add(p, p), int, Span::ZERO);
                let a = builder.push(wrapping_add(t, again), int, Span::ZERO);
                builder.push_effect(commuting.clone(), int, Span::ZERO);
                builder.terminate(Terminator::Jump { to: join, args: vec![a] }, Span::ZERO);
                builder.switch_to(else_block);
                let b = builder.push(wrapping_mul(t, p), int, Span::ZERO);
                builder.push(wrapping_mul(b, b), int, Span::ZERO);
                builder.terminate(Terminator::Jump { to: join, args: vec![b] }, Span::ZERO);
                builder.switch_to(join);
                let joined = builder.block_params(join)[0];
                builder.push_unit(assign(s, joined), Span::ZERO);
                let read = builder.push(take(s), int, Span::ZERO);
                acc = builder.push(wrapping_add(acc, read), int, Span::ZERO);
                made += 9;
            }
        }
    }
    let through = builder.push(Op::Take { target: RefTarget::Through(reference), path: Vec::new(), taken_out: false }, int, Span::ZERO);
    let carried = builder.push(take(acc_slot), int, Span::ZERO);
    let last = builder.push(wrapping_add(acc, through), int, Span::ZERO);
    let last = builder.push(wrapping_add(last, carried), int, Span::ZERO);
    builder.terminate(ret(last), Span::ZERO);
    builder.seal().expect("the body has no open state")
}

fn instructions(module: &Module, body: BodyId) -> usize {
    module.body(body).blocks().map(|(_, block)| block.insts().len()).sum()
}

/// A body of one shape and size, with its preparation already run.
struct Prepared {
    fixture: Fixture,
    body: BodyId,
}

/// A body of `shape` and `size`, with `prepare` run on it untimed.
fn prepared(shape: Shape, size: usize, prepare: fn(&mut Module, BodyId, &mut AnalysisManager<'_>)) -> Prepared {
    let mut fixture = Fixture::new();
    let body = build(&mut fixture, shape, size);
    let decls = declarations(&fixture.interner);
    let mut am = AnalysisManager::new(&decls);
    prepare(&mut fixture.module, body, &mut am);
    Prepared { fixture, body }
}

fn lowered(_: &mut Module, _: BodyId, _: &mut AnalysisManager<'_>) {}

fn ssa(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    Mem2Reg::run(module, body, am);
}

/// Times `measure` over each shape and size, printing one line per run.
fn scale(name: &str, prepare: fn(&mut Module, BodyId, &mut AnalysisManager<'_>), measure: fn(&mut Module, BodyId, &mut AnalysisManager<'_>)) {
    for shape in SHAPES {
        let mut previous: Option<Duration> = None;
        let mut growth_seen = 10.0_f64;
        for size in SIZES {
            if let Some(before) = previous {
                let projected = before.as_secs_f64() * growth_seen;
                if projected > BUDGET.as_secs_f64() {
                    println!("perf {name:<12} {shape:<13?} n={size:>6} skipped: projected {projected:.0}s, over {BUDGET:?}");
                    break;
                }
            }
            let Prepared { mut fixture, body } = prepared(shape, size, prepare);
            let insts = instructions(&fixture.module, body);
            let decls = declarations(&fixture.interner);
            let mut am = AnalysisManager::new(&decls);
            let started = Instant::now();
            measure(&mut fixture.module, body, &mut am);
            let took = started.elapsed();
            match previous {
                Some(before) => {
                    let growth = took.as_secs_f64() / before.as_secs_f64();
                    growth_seen = growth_seen.max(growth);
                    println!("perf {name:<12} {shape:<13?} n={size:>6} insts={insts:>6} {took:>12.3?}  x{growth:.1}");
                }
                None => println!("perf {name:<12} {shape:<13?} n={size:>6} insts={insts:>6} {took:>12.3?}"),
            }
            previous = Some(took);
        }
    }
}

fn run<T>(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>)
where
    T: BodyTransform,
{
    T::run(module, body, am);
}

fn full(module: &mut Module, _: BodyId, am: &mut AnalysisManager<'_>) {
    optimize(module, am, Level::Full);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_mem2reg() {
    scale("mem2reg", lowered, run::<Mem2Reg>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_simplify_cfg() {
    scale("simplify_cfg", ssa, run::<SimplifyCfg>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_sccp() {
    scale("sccp", ssa, run::<Sccp>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_gvn() {
    scale("gvn", ssa, run::<Gvn>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_reborrow() {
    scale("reborrow", ssa, run::<Reborrow>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_dce() {
    scale("dce", ssa, run::<Dce>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_sroa() {
    scale("sroa", ssa, run::<Sroa>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_dse() {
    scale("dse", ssa, run::<Dse>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_bce() {
    scale("bce", ssa, run::<Bce>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_sink() {
    scale("sink", ssa, run::<Sink>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_licm() {
    scale("licm", ssa, run::<Licm>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_commute() {
    scale("commute", ssa, run::<Commute>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_split_calls() {
    scale("split_calls", ssa, run::<SplitCalls>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_schedule() {
    scale("schedule", ssa, run::<Schedule>);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_pipeline_full() {
    scale("pipeline", lowered, full);
}

fn analysis<A>(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>)
where
    A: crate::passes::analysis::Analysis,
{
    am.get::<A>(module, body);
}

/// Every analysis `EditFacts` serves, computed once.
fn all_facts(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    use crate::passes::analysis::{ContentLiveness, Dominators, Escapes, IntervalValues, Loans, LoopTree, PostDominators};
    am.get::<Dominators>(module, body);
    am.get::<PostDominators>(module, body);
    am.get::<LoopTree>(module, body);
    am.get::<Loans>(module, body);
    am.get::<Escapes>(module, body);
    am.get::<ContentLiveness>(module, body);
    am.get::<IntervalValues>(module, body);
}

#[test]
#[ignore = "a scaling probe, run by hand in release"]
fn perf_analyses() {
    use crate::passes::analysis::{
        ConstantValues, ContentLiveness, Dominators, Escapes, IntervalValues, Liveness, Loans, LoopTree, PostDominators,
    };
    scale("dominators", ssa, analysis::<Dominators>);
    scale("postdoms", ssa, analysis::<PostDominators>);
    scale("loop_tree", ssa, analysis::<LoopTree>);
    scale("escapes", ssa, analysis::<Escapes>);
    scale("liveness", ssa, analysis::<Liveness>);
    scale("loans", ssa, analysis::<Loans>);
    scale("content_live", ssa, analysis::<ContentLiveness>);
    scale("constants", ssa, analysis::<ConstantValues>);
    scale("intervals", ssa, analysis::<IntervalValues>);
    scale("all_facts", ssa, all_facts);
}
