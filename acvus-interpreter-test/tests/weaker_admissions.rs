//! An argument the remaining candidates of a bare name take only by weaker
//! admissions of different kinds (RFC-0043): one takes it as a view of what
//! it lends, another only through a declared conversion. `Bag` converts to
//! and from a `Vec<i64>` by one rule each way, so `t::peek` takes a `&Vec`
//! through the reference, and `u::peek` takes the same argument as the
//! slice it lends; the second argument tells the two apart. `weigh` is the
//! same pair with `t::weigh` taking the `Bag` by value, and `pair` the same
//! pair over two lends. `Meters` converts to and from an `i64` the same
//! way, so `w::measure` takes a word through the reference.

use acvus_extern::{Erased, ExternType, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor};
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use std::sync::Arc;

#[derive(ExternType)]
#[extern_type(name = "Bag")]
#[repr(transparent)]
struct Bag(Vec<i64>);

#[extern_fn(effect = pure)]
#[extern_cast]
fn bag(items: Vec<i64>) -> Bag {
    Bag(items)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn unbag(b: Bag) -> Vec<i64> {
    b.0
}

#[extern_fn(name = "peek", effect = pure)]
fn peek_bag(b: &Bag, n: i64) -> i64 {
    b.0.len() as i64 * 100 + n
}

#[extern_fn(name = "stuff", effect = pure)]
fn stuff_bag(b: &mut Bag, n: i64) -> i64 {
    b.0.push(n);
    b.0.len() as i64
}

#[extern_fn(name = "weigh", effect = pure)]
fn weigh_bag(b: Bag, n: i64) -> i64 {
    b.0.len() as i64 * 100 + n
}

#[extern_fn(name = "pair", effect = pure)]
fn pair_bag(a: &Bag, b: &Bag, n: i64) -> i64 {
    (a.0.len() + b.0.len()) as i64 * 100 + n
}

#[extern_fn(name = "peek", effect = pure)]
fn peek_slice<Rt>(s: &[Erased<Rt, i64>], wide: bool) -> i64
where
    Rt: Runtime,
{
    let sum: i64 = s.iter().map(|x| **x).sum();
    if wide { sum * 10 } else { sum }
}

#[extern_fn(name = "weigh", effect = pure)]
fn weigh_slice<Rt>(s: &[Erased<Rt, i64>], wide: bool) -> i64
where
    Rt: Runtime,
{
    let sum: i64 = s.iter().map(|x| **x).sum();
    if wide { sum * 10 } else { sum }
}

#[extern_fn(name = "pair", effect = pure)]
fn pair_slice<Rt>(a: &[Erased<Rt, i64>], b: &[Erased<Rt, i64>], wide: bool) -> i64
where
    Rt: Runtime,
{
    let n = (a.len() + b.len()) as i64;
    if wide { n * 10 } else { n }
}

#[derive(ExternType)]
#[extern_type(name = "Meters")]
#[repr(transparent)]
struct Meters(i64);

#[extern_fn(effect = pure)]
#[extern_cast]
fn meters(n: i64) -> Meters {
    Meters(n)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn unmeters(m: Meters) -> i64 {
    m.0
}

#[extern_fn(name = "measure", effect = pure)]
fn measure_meters(m: &Meters, n: i64) -> i64 {
    m.0 * 100 + n
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = converting_only();
    regs.push(extern_registry! {
        ns: "u",
        fns: [peek_slice, weigh_slice, pair_slice],
    });
    regs
}

/// `w`: a word converted through the reference.
fn measuring() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "w",
        types: [Meters],
        fns: [meters, unmeters, measure_meters],
    });
    regs
}

/// `t` alone, where every argument `t::peek` or `t::weigh` converts meets
/// its parameter on the known path.
fn converting_only() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        types: [Bag],
        fns: [bag, unbag, peek_bag, stuff_bag, weigh_bag, pair_bag],
    });
    regs
}

/// Every stage at `opt`, and the run: the integer it answers, or the
/// refusal's messages.
async fn outcome<R>(source: &str, opt: Opt, registries: R) -> Result<i64, String>
where
    R: FnOnce() -> Vec<Registry<AcvusRuntime>>,
{
    let i = Interner::new();
    let (context_types, snapshot) = split_context(&i, Context::default());
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    let compiled = check_source(&i, ast, &context_types, registries(), Ty::I64, opt, |_| {})
        .map_err(|refusal| refusal.messages.join("\n"))?;
    let (_shared, mut interp) =
        execute_compiled(&i, compiled, snapshot, Arc::new(SequentialExecutor));
    Ok(interp.execute().await.as_int())
}

async fn runs_to(source: &str, expected: i64) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt, registries).await {
            Ok(got) => assert_eq!(got, expected, "at {opt:?}: {source}"),
            Err(why) => panic!("at {opt:?}, refused: {source}\n{why}"),
        }
    }
}

/// RFC-0043 rule 5: where the rest of the call settles on `u::peek`, the
/// view is the checker's at the held argument.
#[tokio::test]
async fn a_held_argument_reaches_a_viewing_candidate_as_its_view() {
    runs_to("let v = vec([1, 2]); peek(&v, true)", 30).await;
}

/// Both registry sets: `t` alone, where `peek` is known to convert its
/// first argument, and `t` with `u`, where that argument is held until the
/// rest of the call settles on `t::peek`.
const BOTH: [(&str, fn() -> Vec<Registry<AcvusRuntime>>); 2] =
    [("t", converting_only), ("t+u", registries)];

/// RFC-0041: a place converted through a reference is taken out of its
/// slot, the call borrows the cast value in a temporary of the parameter's
/// type, and the value is cast back into the place after the call, so the
/// place holds its own `Vec` again, on the known path and the held one.
#[tokio::test]
async fn a_conversion_through_a_reference_lends_a_temporary_and_restores_the_place() {
    let sources = [
        ("let v = vec([1, 2]); peek(&v, 1)", 201),
        ("let v = vec([1, 2]); peek(&v, 1); len(&v) as i64", 2),
        ("let v = vec([1, 2]); peek(&v, 1) + peek(&v, 2)", 403),
    ];
    for (source, expected) in sources {
        for (set, registries) in BOTH {
            for opt in [Opt::None, Opt::Full] {
                match outcome(source, opt, registries).await {
                    Ok(got) => assert_eq!(got, expected, "{set} at {opt:?}: {source}"),
                    Err(why) => panic!("{set} at {opt:?}, refused: {source}\n{why}"),
                }
            }
        }
    }
}

/// RFC-0041: a `&mut` conversion writes back through the cast: what the
/// callee pushed into the `Bag` is in the `Vec` after the call.
#[tokio::test]
async fn a_mutable_conversion_through_a_reference_writes_back_through_the_cast() {
    let source = "let v = vec([1, 2]); stuff(&mut v, 7); len(&v) as i64 * 10 + v[2]";
    for (set, registries) in BOTH {
        for opt in [Opt::None, Opt::Full] {
            match outcome(source, opt, registries).await {
                Ok(got) => assert_eq!(got, 37, "{set} at {opt:?}: {source}"),
                Err(why) => panic!("{set} at {opt:?}, refused: {source}\n{why}"),
            }
        }
    }
}

/// Runs `source` with each set at both levels and expects `expected`.
async fn runs_to_with_both(source: &str, expected: i64) {
    for (set, registries) in BOTH {
        for opt in [Opt::None, Opt::Full] {
            match outcome(source, opt, registries).await {
                Ok(got) => assert_eq!(got, expected, "{set} at {opt:?}: {source}"),
                Err(why) => panic!("{set} at {opt:?}, refused: {source}\n{why}"),
            }
        }
    }
}

/// Refuses `source` with each set at both levels, in exactly `expected`.
async fn refused_with_both(source: &str, expected: &str) {
    for (set, registries) in BOTH {
        for opt in [Opt::None, Opt::Full] {
            match outcome(source, opt, registries).await {
                Ok(got) => panic!("{set} at {opt:?}, expected a refusal, ran to {got}: {source}"),
                Err(why) => assert_eq!(
                    why,
                    format!("[validate:main] {expected}"),
                    "{set} at {opt:?}: {source}"
                ),
            }
        }
    }
}

/// RFC-0041: a place converted for a call is taken out for it until its
/// restore, so a later argument that reads it at its own type, captures it
/// or writes it is refused by the move check, in the same words whether the
/// conversion was known at the argument or held.
#[tokio::test]
async fn a_place_converted_for_a_call_is_lent_to_the_call_for_its_later_arguments() {
    refused_with_both(
        "let v = vec([1, 2]); peek(&v, len(&v) as i64)",
        "`v` is read here while it is lent to a call",
    )
    .await;
    refused_with_both(
        "let v = vec([1, 2]); peek(&v, { let f = |n| -> len(&v) as i64 + n; f(1) })",
        "`v` is read here while it is lent to a call",
    )
    .await;
    refused_with_both(
        "let v = vec([1, 2]); peek(&v, { v = vec([5]); 1 })",
        "`v` is written here while it is lent to a call",
    )
    .await;
}

/// RFC-0041: shared lends of one place at one parameter type inside one
/// call, a later argument or a nested call's, lend the one temporary the
/// place was cast into, and the place is cast back once, after the call
/// that took it out.
#[tokio::test]
async fn shared_lends_of_a_converted_place_in_one_call_lend_one_temporary() {
    runs_to_with_both("let v = vec([1, 2]); pair(&v, &v, 1)", 401).await;
    runs_to_with_both("let v = vec([1, 2]); peek(&v, peek(&v, 1))", 401).await;
    runs_to_with_both(
        "let v = vec([1, 2]); peek(&v, peek(&v, 1)); len(&v) as i64",
        2,
    )
    .await;
}

/// RFC-0041: a `&mut` conversion lends the temporary to no other lend, so
/// a lend of the place inside the call lends the place, which is taken out.
#[tokio::test]
async fn a_place_converted_for_a_mutable_lend_is_lent_exclusively() {
    refused_with_both(
        "let v = vec([1, 2]); stuff(&mut v, peek(&v, 1))",
        "`v` is read here while it is lent to a call",
    )
    .await;
    refused_with_both(
        "let v = vec([1, 2]); stuff(&mut v, stuff(&mut v, 1))",
        "`v` is read here while it is lent to a call",
    )
    .await;
}

/// RFC-0041: the take-out is the converted place's, so another field of
/// the same object is read beside it.
#[tokio::test]
async fn a_field_beside_a_converted_field_is_read_during_the_call() {
    runs_to_with_both("let o = { v: vec([1, 2]), w: 3, }; peek(&o.v, o.w)", 203).await;
}

/// RFC-0041: a conversion through a reference takes the value out of
/// storage the body owns directly, so a place behind a reference or an
/// element is refused by the checker. With `t` alone an element's type is
/// still open where it is lent, so the lend decision takes the parameter's
/// referent and refuses the element's own type first (RFC-0029 rule 4).
#[tokio::test]
async fn a_place_not_owned_directly_is_not_converted_through_a_reference() {
    let not_owned = "[main] converting &Vec<i64> to &Bag takes the value out of its place \
                     for the call, and this argument reaches it through a reference or an \
                     element, not in storage the body owns";
    let cases = [
        (
            "let o = { v: vec([1, 2]), }; let r = &o; peek(&r.v, 1)",
            [not_owned, not_owned],
        ),
        (
            "let a = [vec([1, 2])]; peek(&a[0], 1)",
            [
                "[main] type mismatch: expected Bag, got Vec<i64>",
                not_owned,
            ],
        ),
    ];
    for (source, expected) in cases {
        for ((set, registries), expected) in BOTH.into_iter().zip(expected) {
            for opt in [Opt::None, Opt::Full] {
                match outcome(source, opt, registries).await {
                    Ok(got) => {
                        panic!("{set} at {opt:?}, expected a refusal, ran to {got}: {source}")
                    }
                    Err(why) => assert_eq!(why, expected, "{set} at {opt:?}: {source}"),
                }
            }
        }
    }
}

/// RFC-0041: a word converted through the reference is taken out for the
/// call as any place is: its take copies, and the take-out is marked in the
/// MIR, so the move check does not ask the place's type.
#[tokio::test]
async fn a_word_converted_through_a_reference_is_lent_to_the_call() {
    let cases = [
        ("let m = 5i64; measure(&m, 1)", Ok(501)),
        ("let m = 5i64; measure(&m, 1); m", Ok(5)),
        (
            "let m = 5i64; measure(&m, m)",
            Err("[validate:main] `m` is read here while it is lent to a call"),
        ),
        (
            "let m = 5i64; measure(&m, { m = 3; 1 })",
            Err("[validate:main] `m` is written here while it is lent to a call"),
        ),
    ];
    for (source, expected) in cases {
        for opt in [Opt::None, Opt::Full] {
            let got = outcome(source, opt, measuring).await;
            assert_eq!(
                got,
                expected.map_err(str::to_string),
                "at {opt:?}: {source}"
            );
        }
    }
}

/// RFC-0041: a borrowed temporary is no place: the cast value is lent and
/// nothing is cast back.
#[tokio::test]
async fn a_temporary_converted_through_a_reference_is_lent_and_not_cast_back() {
    let source = "peek(&vec([1, 2]), 1)";
    for (set, registries) in BOTH {
        for opt in [Opt::None, Opt::Full] {
            match outcome(source, opt, registries).await {
                Ok(got) => assert_eq!(got, 201, "{set} at {opt:?}: {source}"),
                Err(why) => panic!("{set} at {opt:?}, refused: {source}\n{why}"),
            }
        }
    }
}

/// RFC-0043 rule 2: a lambda parameter's head is open where `weigh` meets
/// it, so the argument is held, and the settled `t::weigh` takes the `Vec`
/// the call passes through `bag`, in either call form, as the known form
/// takes it.
#[tokio::test]
async fn a_held_argument_converted_by_value_runs_as_the_known_form_does() {
    runs_to("let v = vec([1, 2]); weigh(v, 1)", 201).await;
    runs_to("let f = |x| -> weigh(x, 1); f(vec([1, 2]))", 201).await;
    runs_to("let f = |x| -> x.weigh(1); f(vec([1, 2]))", 201).await;
}

/// The control: a held argument the settled candidate takes directly.
#[tokio::test]
async fn a_held_argument_the_settled_candidate_takes_directly() {
    runs_to("let f = |x| -> weigh(x, 1); f(bag(vec([1, 2, 3])))", 301).await;
}

/// The control: a held argument no candidate admits once the solve names
/// its head refuses the call, which names that head.
#[tokio::test]
async fn a_held_argument_no_candidate_admits_is_refused_at_its_settled_type() {
    let source = "let f = |x| -> weigh(x, 1); f(\"ab\".to_string())";
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt, registries).await {
            Ok(got) => panic!("at {opt:?}, expected a refusal, ran to {got}: {source}"),
            Err(why) => assert!(
                why.contains("no `weigh` takes a call of type Fn(String, i64) -> i64"),
                "at {opt:?}: {why}"
            ),
        }
    }
}
