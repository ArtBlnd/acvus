//! The loop nest, the affine values and the carried state of lowered
//! bodies (`analysis::loops`, `analysis::affine`, `analysis::carried`;
//! RFC-0066).

use acvus_ast::Literal;
use acvus_mir::analysis::affine::{Affine, AffineValues, Derivation, for_body};
use acvus_mir::analysis::carried::{Carried, CarriedState, Dependence, MergeOp, Strength};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loans::Loans;
use acvus_mir::analysis::loops::{
    Invariants, Loop, LoopId, LoopKind, LoopNest, Nesting, Term, Trip,
};
use acvus_mir::cfg::{CfgBody, promote};
use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ir::{ForSource, ValueId};
use acvus_mir::optimize::{dce, ssa_pass};
use acvus_mir::ty::{Effect, ParamTerm, Poly, Ty, TyTerm, lift_to_poly};
use acvus_extern::{Registry, TypesOnly, Var, extern_fn, extern_registry, kind};
use acvus_mir::analysis::carried::{ExternMerge, StorageMerge};
use acvus_mir::laws::LawTable;
use acvus_mir_test::{LoweredScript, lowered_script};
use acvus_utils::Interner;

struct Analyzed {
    cfg: CfgBody,
    nest: LoopNest,
    invariants: Invariants,
    loans: Loans,
    laws: LawTable,
}

impl Analyzed {
    fn of(source: &str) -> Self {
        Self::with_externs(&Interner::new(), source, &[])
    }

    fn with_externs(i: &Interner, source: &str, externs: &[Function]) -> Self {
        Self::lowered(source, lowered_script(i, source, externs, vec![]))
    }

    fn with_registries(i: &Interner, source: &str, own: Vec<Registry<TypesOnly>>) -> Self {
        Self::lowered(source, lowered_script(i, source, &[], own))
    }

    fn lowered(source: &str, lowered: Result<LoweredScript, String>) -> Self {
        let LoweredScript { module, laws } = lowered.unwrap_or_else(|e| panic!("{source}\n{e}"));
        let mut cfg = promote(module.main);
        ssa_pass::run(&mut cfg);
        dce::run(&mut cfg);
        let invariants = Invariants::of(&cfg);
        let nest = LoopNest::of(&cfg, &DomTree::build(&cfg), &invariants);
        let loans = Loans::build(&cfg);
        Self {
            cfg,
            nest,
            invariants,
            loans,
            laws,
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
        CarriedState::of(
            &self.cfg,
            loop_,
            &self.affine(loop_),
            &self.loans,
            &self.laws,
        )
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
            vars: acvus_mir::ty::VarsStated::Here(vec![]),
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: acvus_mir::ty::Flows::Every.into(),
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

// -- Merges an extern declares (RFC-0082 rules 2, 3 and 6) ---------------

/// Type-only fixtures: two copies of one binary function and of one
/// storage write, one declaring laws and one declaring none, so that a
/// loop's strength moves with the declaration alone.
mod laws_fx {
    use super::*;

    #[extern_fn(effect = pure, law(associative, commutative))]
    pub fn joined(a: i64, b: i64) -> i64 {
        let _ = (a, b);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure, law(associative))]
    pub fn joined_in_order(a: i64, b: i64) -> i64 {
        let _ = (a, b);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn lawless(a: i64, b: i64) -> i64 {
        let _ = (a, b);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure, law(fold(combine = append, identity = empty)))]
    pub fn put<T>(c: &mut Vec<T>, x: T)
    where
        T: Var<kind::Type>,
    {
        let _ = (c, x);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn put_lawless<T>(c: &mut Vec<T>, x: T)
    where
        T: Var<kind::Type>,
    {
        let _ = (c, x);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn append<T>(c: &mut Vec<T>, part: Vec<T>)
    where
        T: Var<kind::Type>,
    {
        let _ = (c, part);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    pub fn empty<T>() -> Vec<T>
    where
        T: Var<kind::Type>,
    {
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "lawful",
            fns: [joined, joined_in_order, lawless, put, put_lawless, append, empty],
        }
    }
}

fn with_laws(source: &str) -> Analyzed {
    Analyzed::with_registries(&Interner::new(), source, vec![laws_fx::registry()])
}

fn extern_merge(carried: Carried) -> Option<ExternMerge> {
    match carried {
        Carried::Merge {
            op: MergeOp::Extern(merge),
            exact: true,
        } => Some(merge),
        _ => None,
    }
}

#[test]
fn a_max_over_a_range_is_a_weak_merge_on_the_extern() {
    let a = Analyzed::of("let n = 10; let m = 0; for x in 0..n { m = max(m, x); } m");
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    let [param] = state.params[..] else {
        panic!("one carried value: {:?}", state.params);
    };
    let merge = extern_merge(param.carried)
        .unwrap_or_else(|| panic!("`max` declares itself associative: {:?}", param.carried));
    assert!(merge.commutative);
    assert_eq!(state.strength(), Strength::Weak, "{:?}", state.dependences);
}

#[test]
fn the_same_loop_over_an_extern_that_declares_no_law_is_a_strong_recurrence() {
    let lawful = with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::joined(m, x); } m");
    let loop_ = lawful.sole_loop();
    assert!(extern_merge(lawful.carried(loop_)[0]).is_some());
    assert_eq!(lawful.state(loop_).strength(), Strength::Weak);

    let lawless = with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::lawless(m, x); } m");
    let loop_ = lawless.sole_loop();
    let state = lawless.state(loop_);
    assert_eq!(lawless.carried(loop_), [Carried::Recurrence]);
    assert_eq!(state.dependences, [Dependence::Recurrence(state.params[0].param)]);
    assert_eq!(state.strength(), Strength::Strong);
}

#[test]
fn the_state_as_the_second_operand_merges_only_when_the_extern_commutes() {
    let commutes = with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::joined(x, m); } m");
    let loop_ = commutes.sole_loop();
    assert!(extern_merge(commutes.carried(loop_)[0]).is_some());

    let in_order =
        with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::joined_in_order(x, m); } m");
    let loop_ = in_order.sole_loop();
    assert_eq!(in_order.carried(loop_), [Carried::Recurrence]);

    let first = with_laws(
        "let n = 10; let m = 0; for x in 0..n { m = lawful::joined_in_order(m, x); } m",
    );
    let loop_ = first.sole_loop();
    let merge = extern_merge(first.carried(loop_)[0]).expect("the state is the first operand");
    assert!(!merge.commutative);
}

#[test]
fn a_state_read_beside_the_merge_is_a_recurrence() {
    let a = with_laws(
        "let n = 10; let m = 0; let t = 0; for x in 0..n { m = lawful::joined(m, x); t = t + m; } t",
    );
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    assert!(
        state
            .params
            .iter()
            .all(|p| extern_merge(p.carried).is_none()),
        "the loop reads the partial `m`: {:?}",
        state.params
    );
    assert_eq!(state.strength(), Strength::Strong);
}

#[test]
fn a_push_with_a_fold_law_is_a_weak_storage_merge_and_without_it_strong() {
    let lawful = with_laws("let n = 3; let w = vec([0]); for x in 0..n { lawful::put(&mut w, x); } 0");
    let loop_ = lawful.sole_loop();
    let state = lawful.state(loop_);
    let [StorageMerge { fold, .. }] = state.storage_merges[..] else {
        panic!("`w` is merged through its storage: {:?}", state.storage_merges);
    };
    assert!(!fold.commutative);
    assert!(state.dependences.is_empty(), "{:?}", state.dependences);
    assert_eq!(state.strength(), Strength::Weak);
    assert!(!state.runs_apart(), "a storage merge carries state");

    let lawless =
        with_laws("let n = 3; let w = vec([0]); for x in 0..n { lawful::put_lawless(&mut w, x); } 0");
    let loop_ = lawless.sole_loop();
    let state = lawless.state(loop_);
    assert!(state.storage_merges.is_empty());
    assert!(
        matches!(state.dependences[..], [Dependence::StorageWrite(_)]),
        "{:?}",
        state.dependences
    );
    assert_eq!(state.strength(), Strength::Strong);
}

#[test]
fn a_fold_storage_the_loop_also_reads_is_a_strong_write() {
    let a = with_laws(
        "let n = 3; let w = vec([0]); let t = 0; for x in 0..n { lawful::put(&mut w, x); t = t + w.len(); } t",
    );
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    assert!(state.storage_merges.is_empty(), "{:?}", state.storage_merges);
    assert!(
        state
            .dependences
            .iter()
            .any(|d| matches!(d, Dependence::StorageWrite(_))),
        "{:?}",
        state.dependences
    );
}

/// std `Vec::push` declares `fold(combine = extend, identity = new)`
/// (RFC-0082 rule 3), so a loop whose one write of `v` is a push merges
/// through `v`'s storage and stays weak.
#[test]
fn a_std_push_in_a_loop_is_a_weak_storage_merge() {
    let a = Analyzed::of("let n = 3; let v = new(); for x in 0..n { v.push(x); } v.len()");
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    assert!(state.params.is_empty(), "nothing is carried through the header");
    let [StorageMerge { fold, .. }] = state.storage_merges[..] else {
        panic!("`v` is merged through its storage: {:?}", state.storage_merges);
    };
    assert!(!fold.commutative, "`push` keeps order");
    assert!(state.dependences.is_empty(), "{:?}", state.dependences);
    assert_eq!(state.strength(), Strength::Weak);
    assert!(!state.runs_apart(), "a storage merge carries state");
}

#[test]
fn a_std_push_storage_the_loop_also_reads_is_a_strong_write() {
    let a = Analyzed::of(
        "let n = 3; let v = new(); let t = 0u64; for x in 0..n { v.push(x); t = t + v.len(); } t",
    );
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    assert!(state.storage_merges.is_empty(), "{:?}", state.storage_merges);
    assert!(
        state
            .dependences
            .iter()
            .any(|d| matches!(d, Dependence::StorageWrite(_))),
        "{:?}",
        state.dependences
    );
    assert_eq!(state.strength(), Strength::Strong);
}
