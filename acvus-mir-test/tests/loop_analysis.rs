//! The loop nest, the affine values and the carried state of lowered
//! bodies (`analysis::loops`, `analysis::affine`, `analysis::carried`;
//! RFC-0066).

use acvus_ast::Literal;
use acvus_mir::analysis::affine::{Affine, AffineValues, Derivation, for_body};
use acvus_mir::analysis::carried::{Carried, CarriedState, Dependence, MergeOp, Strength};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loans::{Loans, Summaries};
use acvus_mir::analysis::loops::{
    Invariants, Loop, LoopId, LoopKind, LoopNest, Nesting, Term, Trip,
};
use acvus_mir::cfg::{CfgBody, promote};
use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ir::{ForSource, ValueId};
use acvus_mir::optimize::{dce, ssa_pass};
use acvus_mir::ty::{Effect, ParamTerm, Poly, Ty, TyTerm, lift_to_poly};
use acvus_mir_test::lowered_script_module;
use acvus_utils::Interner;

struct Analyzed {
    cfg: CfgBody,
    nest: LoopNest,
    invariants: Invariants,
    loans: Loans,
}

impl Analyzed {
    fn of(source: &str) -> Self {
        Self::with_externs(&Interner::new(), source, &[])
    }

    fn with_externs(i: &Interner, source: &str, externs: &[Function]) -> Self {
        let module =
            lowered_script_module(i, source, externs).unwrap_or_else(|e| panic!("{source}\n{e}"));
        let mut cfg = promote(module.main);
        ssa_pass::run(&mut cfg);
        dce::run(&mut cfg);
        let invariants = Invariants::of(&cfg);
        let nest = LoopNest::of(&cfg, &DomTree::build(&cfg), &invariants);
        let loans = Loans::build(&cfg, Summaries::NONE);
        Self {
            cfg,
            nest,
            invariants,
            loans,
        }
    }

    fn sole_loop(&self) -> &Loop {
        let loops: Vec<&Loop> = self.nest.iter().map(|(_, l)| l).collect();
        let [only] = loops[..] else {
            panic!("one loop, found {}", loops.len());
        };
        only
    }

    fn affine(&self, loop_: &Loop) -> AffineValues {
        AffineValues::of(&self.cfg, loop_, &self.invariants)
    }

    fn state(&self, loop_: &Loop) -> CarriedState {
        CarriedState::of(&self.cfg, loop_, &self.affine(loop_), &self.loans)
    }

    fn carried(&self, loop_: &Loop) -> Vec<Carried> {
        self.state(loop_).params.iter().map(|p| p.carried).collect()
    }

    fn for_counter(&self, loop_: &Loop) -> ValueId {
        let LoopKind::For { source } = loop_.kind else {
            panic!("a `for`, found {:?}", loop_.kind);
        };
        let body = for_body(&self.cfg, loop_.natural.header);
        self.cfg.blocks[body.0].params[source.counter_param()]
    }
}

fn exact_add() -> Carried {
    Carried::Merge {
        op: MergeOp::Add,
        exact: true,
    }
}

#[test]
fn a_range_sum_is_a_weak_exact_merge_run_max_of_hi_minus_at_times() {
    let a = Analyzed::of("let n = 10; let s = 0; for i in 0..n { s = s + i; } s");
    let loop_ = a.sole_loop();
    let LoopKind::For {
        source: ForSource::Range { at, hi },
    } = loop_.kind
    else {
        panic!("a range `for`, found {:?}", loop_.kind);
    };
    assert_eq!(
        loop_.trip,
        Trip::Known(Term::Value(hi).sub(Term::Value(at)).max(Term::int(0)))
    );
    assert_eq!(loop_.nesting, Nesting::Outermost);

    let counter = a.for_counter(loop_);
    assert_eq!(
        a.affine(loop_).get(counter),
        Some(&Affine {
            base: Term::Value(at),
            step: Term::int(1),
            derivation: Derivation::Counter,
        }),
        "a range's element is `at + k`"
    );
    assert_eq!(a.carried(loop_), [exact_add()]);
    assert_eq!(a.state(loop_).strength(), Strength::Weak);
}

#[test]
fn a_slice_sum_is_a_weak_exact_merge_run_len_times() {
    let a = Analyzed::of("let v = vec([1, 2, 3]); let s = 0; for x in &v { s = s + *x; } s");
    let loop_ = a.sole_loop();
    let LoopKind::For {
        source: ForSource::Slice(slice),
    } = loop_.kind
    else {
        panic!("a slice `for`, found {:?}", loop_.kind);
    };
    assert_eq!(loop_.trip, Trip::Known(Term::Len(slice)));

    let index = a.for_counter(loop_);
    assert_eq!(
        a.affine(loop_).get(index),
        Some(&Affine {
            base: Term::int(0),
            step: Term::int(1),
            derivation: Derivation::Counter,
        }),
        "a slice's index is `0 + k`"
    );
    assert_eq!(a.carried(loop_), [exact_add()]);
    assert_eq!(a.state(loop_).strength(), Strength::Weak);
}

#[test]
fn a_float_sum_is_a_weak_inexact_merge() {
    let a = Analyzed::of("let v = vec([1.0, 2.0]); let s = 0.0; for x in &v { s = s + *x; } s");
    let loop_ = a.sole_loop();
    assert_eq!(
        a.carried(loop_),
        [Carried::Merge {
            op: MergeOp::Add,
            exact: false,
        }]
    );
    assert_eq!(a.state(loop_).strength(), Strength::Weak);
}

#[test]
fn a_counter_derived_in_a_while_is_an_iv_and_the_while_has_no_trip_count() {
    let a =
        Analyzed::of("let n = 10; let i = 0; let j = 0; while i < n { j = j + 3; i = i + 1; } j");
    let loop_ = a.sole_loop();
    assert!(matches!(loop_.kind, LoopKind::While), "{:?}", loop_.kind);
    assert_eq!(loop_.trip, Trip::Unknown);
    assert_eq!(a.carried(loop_), [Carried::Iv, Carried::Iv]);

    let header = &a.cfg.blocks[loop_.natural.header.0];
    let affine = a.affine(loop_);
    let steps: Vec<Term> = header
        .params
        .iter()
        .map(|p| affine.get(*p).expect("an induction variable").step.clone())
        .collect();
    assert!(
        steps.contains(&Term::Const(Literal::Int(3))),
        "`j = j + 3` steps by 3, a word the body writes and the term reads as a constant: {steps:?}"
    );
    assert_eq!(a.state(loop_).strength(), Strength::Weak);
}

#[test]
fn a_recurrence_is_strong() {
    let a = Analyzed::of("let n = 10; let a = 1; for i in 0..n { a = a * a + 1; } a");
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    let [param] = state.params[..] else {
        panic!("one carried value");
    };
    assert_eq!(param.carried, Carried::Recurrence);
    assert_eq!(state.dependences, [Dependence::Recurrence(param.param)]);
    assert_eq!(state.strength(), Strength::Strong);
}

fn emit(i: &Interner) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern("emit")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
        },
    }
}

#[test]
fn an_ordered_effect_in_the_body_is_strong() {
    let i = Interner::new();
    let a = Analyzed::with_externs(&i, "let n = 3; for k in 0..n { emit(k); } 0", &[emit(&i)]);
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    assert!(
        state
            .dependences
            .iter()
            .any(|d| matches!(d, Dependence::OrderedEffect { .. })),
        "{:?}",
        state.dependences
    );
    assert_eq!(state.strength(), Strength::Strong);
}

#[test]
fn a_write_through_the_element_is_weak_and_one_elsewhere_is_strong() {
    let through = Analyzed::of("let v = vec([1, 2, 3]); for x in &mut v { *x = 0; } 0");
    let loop_ = through.sole_loop();
    assert_eq!(through.state(loop_).strength(), Strength::Weak);
    assert!(through.state(loop_).runs_apart());

    let elsewhere = Analyzed::of(
        "let v = vec([1, 2, 3]); let t = 0; let r = &mut t; for x in &mut v { *r = *x; } t",
    );
    let loop_ = elsewhere.sole_loop();
    let state = elsewhere.state(loop_);
    assert!(
        matches!(state.dependences[..], [Dependence::StorageWrite(_)]),
        "the write through `r` is the one dependence; the element's is admitted: {:?}",
        state.dependences
    );
    assert_eq!(state.strength(), Strength::Strong);
}

#[test]
fn a_push_in_a_loop_is_a_storage_write_and_strong() {
    let a = Analyzed::of("let v = vec([1, 2, 3]); let w = vec([0]); for x in &v { w.push(*x); } 0");
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    assert!(
        state.params.is_empty(),
        "nothing is carried through the header"
    );
    assert!(
        matches!(state.dependences[..], [Dependence::StorageWrite(_)]),
        "the merge goes through `w`'s storage: {:?}",
        state.dependences
    );
    assert_eq!(state.strength(), Strength::Strong);
}

struct Nested {
    analyzed: Analyzed,
    inner: LoopId,
    outer: LoopId,
}

impl Nested {
    fn of(source: &str) -> Self {
        let analyzed = Analyzed::of(source);
        let ids: Vec<LoopId> = analyzed.nest.iter().map(|(id, _)| id).collect();
        let [inner, outer] = ids[..] else {
            panic!("two loops, found {}", ids.len());
        };
        Self {
            analyzed,
            inner,
            outer,
        }
    }
}

#[test]
fn an_inner_range_of_invariant_bounds_is_rectangular() {
    let Nested {
        analyzed: a,
        inner,
        outer,
    } = Nested::of(
        "let n = 3; let m = 4; let s = 0; for i in 0..n { for j in 0..m { s = s + j; } } s",
    );
    assert_eq!(
        a.nest.get(inner).nesting,
        Nesting::Inside {
            parent: outer,
            rectangular: true,
        }
    );
    assert_eq!(a.nest.get(outer).children, [inner]);
    assert_eq!(a.nest.get(outer).nesting, Nesting::Outermost);
}

#[test]
fn an_inner_range_bounded_by_the_outer_counter_is_not_rectangular() {
    let Nested {
        analyzed: a,
        inner,
        outer,
    } = Nested::of("let n = 3; let s = 0; for i in 0..n { for j in 0..i { s = s + j; } } s");
    assert_eq!(
        a.nest.get(inner).nesting,
        Nesting::Inside {
            parent: outer,
            rectangular: false,
        }
    );
}

/// RFC-0066 rule 6: a loop left from anywhere but its header is ordered,
/// since the iterations after the one that leaves never run. The same loop
/// without the `break` is weak.
#[test]
fn a_break_from_the_body_orders_the_iterations() {
    let a = Analyzed::of("let v = vec([1, 2, 3]); let s = 0; for x in &v { s = s + *x; } s");
    assert_eq!(a.state(a.sole_loop()).strength(), Strength::Weak);

    let a = Analyzed::of(
        "let v = vec([1, 2, 3]); let s = 0; for x in &v { if *x == 2 { break; }; s = s + *x; } s",
    );
    let state = a.state(a.sole_loop());
    assert!(
        state
            .dependences
            .iter()
            .any(|d| matches!(d, Dependence::EarlyExit { .. })),
        "{:?}",
        state.dependences
    );
    assert_eq!(state.strength(), Strength::Strong);
}
