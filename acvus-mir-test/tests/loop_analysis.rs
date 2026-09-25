//! The loop nest, the affine values and the carried state of lowered
//! bodies (`analysis::loops`, `analysis::affine`, `analysis::carried`;
//! RFC-0066), and the law `analysis::loop_deps` reads over each state
//! (RFC-0089 rule 4).

use acvus_ast::Literal;
use acvus_extern::{Registry, TypesOnly, Var, extern_fn, extern_registry, kind};
use acvus_mir::analysis::affine::{Affine, AffineValues, Derivation, for_body};
use acvus_mir::analysis::carried::{Carried, CarriedState};
use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loop_deps::{
    Accumulator, CallIdentity, CallLaw, FoldAccumulator, Law, LawOp, LoopDeps, Storage, Token,
};
use acvus_mir::analysis::loops::{
    Invariants, Loop, LoopId, LoopKind, LoopNest, Nesting, Term, Trip,
};
use acvus_mir::cfg::{CfgBody, promote};
use acvus_mir::graph::{Function, QualifiedRef};
use acvus_mir::ir::{Callee, ForSource, InstKind, ValueId};
use acvus_mir::laws::{ExternInstance, LawTable, ResolvedIdentity};
use acvus_mir::optimize::{dce, ssa_pass};
use acvus_mir_test::{LoweredScript, lowered_script};
use acvus_utils::Interner;

struct Analyzed {
    cfg: CfgBody,
    nest: LoopNest,
    invariants: Invariants,
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
        Self {
            cfg,
            nest,
            invariants,
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
        CarriedState::of(&self.cfg, loop_, &self.affine(loop_))
    }

    fn carried(&self, loop_: &Loop) -> Vec<Carried> {
        self.state(loop_).params.iter().map(|p| p.carried).collect()
    }

    /// The law `loop_deps` judges on the cycle holding each token `holds`
    /// picks out of a `for`, in the order of the cycles.
    fn laws_where(&self, loop_: &Loop, holds: impl Fn(&Token) -> bool) -> Vec<Option<Accumulator>> {
        let deps = LoopDeps::of(&self.cfg, &self.laws, loop_.natural.header)
            .unwrap_or_else(|fault| panic!("{}", fault.shown()));
        let judged = deps.judge(&self.cfg, &self.laws);
        deps.cycles
            .iter()
            .zip(judged)
            .filter(|(cycle, _)| cycle.tokens.iter().any(&holds))
            .map(|(_, judged)| judged.law)
            .collect()
    }

    /// The law of the cycle over each carried state of a `for`.
    fn state_laws(&self, loop_: &Loop) -> Vec<Option<Accumulator>> {
        let states: Vec<ValueId> = self
            .state(loop_)
            .params
            .iter()
            .filter(|p| p.carried == Carried::State)
            .map(|p| p.param)
            .collect();
        self.laws_where(
            loop_,
            |token| matches!(token, Token::Carried(param) if states.contains(param)),
        )
    }

    /// The law of the cycle over each storage slot a `for` writes.
    fn storage_laws(&self, loop_: &Loop) -> Vec<Option<Accumulator>> {
        self.laws_where(loop_, |token| {
            matches!(token, Token::Storage(Storage::Slot(_)))
        })
    }

    fn for_counter(&self, loop_: &Loop) -> ValueId {
        let LoopKind::For { source } = loop_.kind else {
            panic!("a `for`, found {:?}", loop_.kind);
        };
        let body = for_body(&self.cfg, loop_.natural.header);
        self.cfg.blocks[body.0].params[source.counter_param()]
    }
}

fn add(exact: bool) -> Option<Accumulator> {
    Some(Accumulator {
        law: Law::Op(LawOp::Add),
        exact,
        commutative: true,
    })
}

#[test]
fn a_range_sum_is_an_exact_merge_run_max_of_hi_minus_at_times() {
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
    assert_eq!(a.carried(loop_), [Carried::State]);
    assert_eq!(a.state_laws(loop_), [add(true)]);
}

#[test]
fn a_slice_sum_is_an_exact_merge_run_len_times() {
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
    assert_eq!(a.carried(loop_), [Carried::State]);
    assert_eq!(a.state_laws(loop_), [add(true)]);
}

#[test]
fn a_float_sum_is_an_inexact_merge() {
    let a = Analyzed::of("let v = vec([1.0, 2.0]); let s = 0.0; for x in &v { s = s + *x; } s");
    let loop_ = a.sole_loop();
    assert_eq!(a.carried(loop_), [Carried::State]);
    assert_eq!(a.state_laws(loop_), [add(false)]);
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
}

#[test]
fn a_state_read_beyond_its_merge_has_no_law() {
    let a = Analyzed::of("let n = 10; let a = 1; for i in 0..n { a = a * a + 1; } a");
    let loop_ = a.sole_loop();
    assert_eq!(a.carried(loop_), [Carried::State]);
    assert_eq!(a.state_laws(loop_), [None]);
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

/// A `break` makes the sum's stage the one the loop leaves from, so its
/// cycle joins the control token and is one join of two tokens with no law
/// (RFC-0089 rule 5). A write through another `&mut` carries nothing
/// through the header, and its storage's cycle has no law.
#[test]
fn a_break_joins_the_sum_to_the_control_token_and_a_write_elsewhere_has_no_law() {
    let a = Analyzed::of(
        "let v = vec([1, 2, 3]); let s = 0; for x in &v { if *x == 2 { break; }; s = s + *x; } s",
    );
    let loop_ = a.sole_loop();
    assert_eq!(a.carried(loop_), [Carried::State]);
    assert_eq!(a.state_laws(loop_), [None]);

    let a = Analyzed::of(
        "let v = vec([1, 2, 3]); let t = 0; let r = &mut t; for x in &mut v { *r = *x; } t",
    );
    let loop_ = a.sole_loop();
    let state = a.state(loop_);
    assert!(state.params.is_empty(), "{:?}", state.params);
    assert_eq!(a.storage_laws(loop_), [None]);
}

// -- Merges an extern declares (RFC-0082 rules 2, 3 and 6) ---------------

/// Type-only fixtures: two copies of one binary function and of one
/// storage write, one declaring laws and one declaring none, so that what
/// a loop carries moves with the declaration alone.
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

/// The one carried state's law, when it is an exact `Call` law.
fn call_law(a: &Analyzed) -> Option<Accumulator> {
    match &a.state_laws(a.sole_loop())[..] {
        [
            Some(
                call @ Accumulator {
                    law: Law::Call(_),
                    exact: true,
                    ..
                },
            ),
        ] => Some(call.clone()),
        [_] => None,
        other => panic!("one carried state: {other:?}"),
    }
}

/// The `Fold` law of the one storage cycle, when it has one.
fn fold_law(a: &Analyzed) -> Option<FoldAccumulator> {
    match &a.storage_laws(a.sole_loop())[..] {
        [
            Some(Accumulator {
                law: Law::Fold(fold),
                exact: true,
                ..
            }),
        ] => Some(*fold),
        [_] => None,
        other => panic!("one storage cycle: {other:?}"),
    }
}

#[test]
fn a_max_over_a_range_has_the_extern_s_call_law() {
    let a = Analyzed::of("let n = 10; let m = 0; for x in 0..n { m = max(m, x); } m");
    let call = call_law(&a).expect("`max` declares itself associative");
    assert!(call.commutative);
    assert!(
        matches!(
            call.law,
            Law::Call(CallLaw {
                identity: CallIdentity::Declared(_),
                ..
            })
        ),
        "{call:?}"
    );
}

#[test]
fn the_same_loop_over_an_extern_that_declares_no_law_has_none() {
    let lawful = with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::joined(m, x); } m");
    assert!(call_law(&lawful).is_some());

    let lawless =
        with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::lawless(m, x); } m");
    assert_eq!(lawless.carried(lawless.sole_loop()), [Carried::State]);
    assert_eq!(call_law(&lawless), None);
}

#[test]
fn the_state_as_the_second_operand_has_a_law_only_when_the_extern_commutes() {
    let commutes =
        with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::joined(x, m); } m");
    assert!(call_law(&commutes).is_some());

    let in_order =
        with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::joined_in_order(x, m); } m");
    assert_eq!(call_law(&in_order), None);

    let first =
        with_laws("let n = 10; let m = 0; for x in 0..n { m = lawful::joined_in_order(m, x); } m");
    let call = call_law(&first).expect("the state is the first operand");
    assert!(!call.commutative);
}

#[test]
fn a_state_read_beside_its_update_has_no_law() {
    let a = with_laws(
        "let n = 10; let m = 0; let t = 0; for x in 0..n { m = lawful::joined(m, x); t = t + m; } t",
    );
    let loop_ = a.sole_loop();
    let laws = a.state_laws(loop_);
    assert!(
        !laws.iter().any(|law| matches!(
            law,
            Some(Accumulator {
                law: Law::Call(_),
                ..
            })
        )),
        "the loop reads the partial `m`: {laws:?}"
    );
}

#[test]
fn a_push_with_a_fold_law_has_the_fold_law_and_without_it_none() {
    let lawful =
        with_laws("let n = 3; let w = vec([0]); for x in 0..n { lawful::put(&mut w, x); } 0");
    let fold = fold_law(&lawful).expect("`w` is combined through its storage");
    assert!(!fold.fold.commutative);

    let lawless = with_laws(
        "let n = 3; let w = vec([0]); for x in 0..n { lawful::put_lawless(&mut w, x); } 0",
    );
    assert_eq!(fold_law(&lawless), None);
}

#[test]
fn a_fold_storage_the_loop_also_reads_has_no_fold_law() {
    let a = with_laws(
        "let n = 3; let w = vec([0]); let t = 0; for x in 0..n { lawful::put(&mut w, x); t = t + w.len(); } t",
    );
    assert_eq!(fold_law(&a), None);
}

/// std `Vec::push` declares `fold(combine = extend, identity = new)`
/// (RFC-0082 rule 3), so a loop whose one write of `v` is a push combines
/// through `v`'s storage.
#[test]
fn a_std_push_in_a_loop_has_the_fold_law() {
    let a = Analyzed::of("let n = 3; let v = new(); for x in 0..n { v.push(x); } v.len()");
    let loop_ = a.sole_loop();
    assert!(
        a.state(loop_).params.is_empty(),
        "nothing is carried through the header"
    );
    let fold = fold_law(&a).expect("`v` is combined through its storage");
    assert!(!fold.fold.commutative, "`push` keeps order");
}

#[test]
fn a_std_push_storage_the_loop_also_reads_has_no_fold_law() {
    let a = Analyzed::of(
        "let n = 3; let v = new(); let t = 0u64; for x in 0..n { v.push(x); t = t + v.len(); } t",
    );
    assert_eq!(fold_law(&a), None);
}

// -- A law names its extern by instance (RFC-0082 rules 2, 3 and 6) ------

/// `joined` again, in a second namespace, declaring no law.
mod plain_fx {
    use super::*;

    #[extern_fn(effect = pure)]
    pub fn joined(a: i64, b: i64) -> i64 {
        let _ = (a, b);
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "plain",
            fns: [joined],
        }
    }
}

/// A fold whose `combine` is a signature with an instance at `f64`
/// registered before the one at `i64`, so that the first instance of the
/// name is not the one of the declaring instance's types; its `identity`
/// is one generic extern, taken at each declaring instance's type.
mod folded_fx {
    use super::*;
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "folded",
        fn put<T>(c: &mut T, x: T)
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "folded",
        fn append<T>(c: &mut T, part: T)
        where
            T: Var<kind::Type>;
    }

    #[extern_fn(
        instance_of = put,
        effect = pure,
        law(fold(combine = append, identity = empty))
    )]
    fn put_float(c: &mut f64, x: f64) {
        let _ = (c, x);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(
        instance_of = put,
        effect = pure,
        law(fold(combine = append, identity = empty))
    )]
    fn put_int(c: &mut i64, x: i64) {
        let _ = (c, x);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(instance_of = append, effect = pure)]
    fn append_float(c: &mut f64, part: f64) {
        let _ = (c, part);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(instance_of = append, effect = pure)]
    fn append_int(c: &mut i64, part: i64) {
        let _ = (c, part);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure)]
    fn empty<T>() -> T
    where
        T: Var<kind::Type>,
    {
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "folded",
            signatures: [put, append],
            fns: [put_float, put_int, append_float, append_int, empty],
        }
    }
}

/// The extern instance every call of `ns::name` in `a`'s body resolved to.
fn called(a: &Analyzed, i: &Interner, ns: &str, name: &str) -> ExternInstance {
    let wanted = QualifiedRef::qualified(i.intern(ns), i.intern(name));
    let mut found: Vec<ExternInstance> = a
        .cfg
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match &inst.kind {
            InstKind::FunctionCall {
                callee: Callee::Extern { id, instance, .. },
                ..
            } if *id == wanted => Some(ExternInstance {
                id: *id,
                instance: *instance,
            }),
            _ => None,
        })
        .collect();
    found.dedup();
    match found[..] {
        [one] => one,
        _ => panic!("calls of {ns}::{name} at one instance: {found:?}"),
    }
}

#[test]
fn of_two_externs_of_one_name_only_the_lawful_one_s_loop_has_a_law() {
    let i = Interner::new();
    let registries = || vec![laws_fx::registry(), plain_fx::registry()];
    let lawful = Analyzed::with_registries(
        &i,
        "let n = 10; let m = 0; for x in 0..n { m = lawful::joined(m, x); } m",
        registries(),
    );
    let call = call_law(&lawful).expect("`lawful::joined` declares itself associative");
    let Law::Call(CallLaw { callee, .. }) = call.law else {
        panic!("{call:?}")
    };
    assert_eq!(
        callee.id,
        QualifiedRef::qualified(i.intern("lawful"), i.intern("joined"))
    );

    let plain = Analyzed::with_registries(
        &i,
        "let n = 10; let m = 0; for x in 0..n { m = plain::joined(m, x); } m",
        registries(),
    );
    assert_eq!(call_law(&plain), None, "`plain::joined` declares no law");
}

#[test]
fn a_min_over_f64_in_a_loop_has_no_law() {
    let a = Analyzed::of("let v = [2.5, 1.5]; let m = 9.5; for x in &v { m = min(m, *x); } m");
    assert_eq!(a.carried(a.sole_loop()), [Carried::State]);
    assert_eq!(call_law(&a), None, "`min` over `f64` declares no law");
}

/// `num::min` over `i32` names `i32::MAX` as its identity, and over `i64`
/// names `i64::MAX`: each loop's law carries its own width's identity, the
/// instance a call of that constant resolves to.
#[test]
fn a_min_over_i32_and_over_i64_each_has_its_own_width_s_identity() {
    for width in ["i32", "i64"] {
        let i = Interner::new();
        let source = format!(
            "let v = [2{width}, 1{width}]; let m = {width}::MAX(); \
             for x in &v {{ m = min(m, *x); }} m"
        );
        let a = Analyzed::with_externs(&i, &source, &[]);
        let call = call_law(&a).unwrap_or_else(|| panic!("`min` over {width} has a law"));
        let Law::Call(CallLaw {
            callee,
            identity: CallIdentity::Declared(ResolvedIdentity::Extern(identity)),
        }) = call.law
        else {
            panic!("{width}: {call:?}")
        };
        assert_eq!(callee, called(&a, &i, "num", "min"), "{width}");
        assert_eq!(identity, called(&a, &i, width, "MAX"), "{width}");
    }
}

/// `vec::push` at `T = i64` combines through `vec::extend` and starts from
/// `vec::new` at the instances calls of them at `i64` resolve to.
#[test]
fn a_push_at_i64_folds_through_extend_and_new_at_i64() {
    let i = Interner::new();
    let a = Analyzed::with_externs(
        &i,
        "let n = 3; let v = new(); for x in 0..n { v.push(x); } \
         let w = vec([7]); w.extend(v); w.len()",
        &[],
    );
    let loops: Vec<&Loop> = a.nest.iter().map(|(_, l)| l).collect();
    let [loop_] = loops[..] else {
        panic!("one loop, found {}", loops.len());
    };
    let [
        Some(Accumulator {
            law: Law::Fold(fold),
            ..
        }),
    ] = a.storage_laws(loop_)[..]
    else {
        panic!("`v` is combined through its storage")
    };
    assert_eq!(fold.fold.combine, called(&a, &i, "vec", "extend"));
    assert_eq!(fold.fold.identity, called(&a, &i, "vec", "new"));
}

/// The `put` instance at `i64` folds through `append` at `i64`, though
/// the instance at `f64` is the first of that name.
#[test]
fn a_fold_combines_through_the_instance_of_its_own_types() {
    let i = Interner::new();
    let a = Analyzed::with_registries(
        &i,
        "let n = 3; let w = folded::empty(); for x in 0..n { folded::put(&mut w, x); } \
         let u = folded::empty(); folded::append(&mut u, w); u",
        vec![folded_fx::registry()],
    );
    let loops: Vec<&Loop> = a.nest.iter().map(|(_, l)| l).collect();
    let [loop_] = loops[..] else {
        panic!("one loop, found {}", loops.len());
    };
    let [
        Some(Accumulator {
            law: Law::Fold(fold),
            ..
        }),
    ] = a.storage_laws(loop_)[..]
    else {
        panic!("`w` is combined through its storage")
    };
    assert_eq!(fold.callee, called(&a, &i, "folded", "put"));
    assert_eq!(fold.fold.combine, called(&a, &i, "folded", "append"));
    assert_eq!(fold.fold.identity, called(&a, &i, "folded", "empty"));
}

#[test]
fn a_difference_with_the_counter_is_affine_with_its_step_kept_or_negated() {
    let a = Analyzed::of("let c = 9; let s = 0; for i in 0..4 { s = s + (i - 2) + (c - i); } s");
    let loop_ = a.sole_loop();
    let LoopKind::For {
        source: ForSource::Range { at, .. },
    } = loop_.kind
    else {
        panic!("a range `for`, found {:?}", loop_.kind);
    };
    let counter = a.for_counter(loop_);
    let affine = a.affine(loop_);
    let differences: Vec<(ValueId, bool)> = loop_
        .natural
        .blocks()
        .flat_map(|block| &a.cfg.blocks[block.0].insts)
        .filter_map(|inst| match inst.kind {
            InstKind::BinOp {
                dst,
                op: acvus_mir::ir::BinOp::Sub(_),
                left,
                right,
            } if left == counter || right == counter => Some((dst, left == counter)),
            _ => None,
        })
        .collect();
    assert_eq!(differences.len(), 2, "`i - 2` and `c - i`");
    for (dst, counter_on_left) in differences {
        let found = affine.get(dst).expect("a difference with the counter is affine");
        match counter_on_left {
            true => {
                assert!(matches!(found.derivation, Derivation::Lowered { of, .. } if of == counter));
                assert!(matches!(&found.base, Term::Sub(base, _) if **base == Term::Value(at)));
                assert_eq!(found.step, Term::int(1), "`i - 2` keeps the counter's step");
            }
            false => {
                assert!(matches!(found.derivation, Derivation::Reflected { of, .. } if of == counter));
                assert!(matches!(&found.base, Term::Sub(_, base) if **base == Term::Value(at)));
                assert_eq!(
                    found.step,
                    Term::int(0).sub(Term::int(1)),
                    "`c - i` negates the counter's step"
                );
            }
        }
    }
}
