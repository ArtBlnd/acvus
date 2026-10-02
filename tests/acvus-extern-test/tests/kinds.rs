//! The safe kinds, driven through extern fns written as the macros will write
//! them: each argument taken by its kind, the body called, and its result made
//! by its kind. Every test ends with no value held, so ownership is checked
//! along with the results.
//!
//! The tests and the entries play the compiler: each `unsafe` asserts the
//! premise for a value made here, at the type it was made at, except in the
//! last two tests, which break a premise on purpose to show that the safe
//! runtime panics.

use acvus_extern::{
    Declared, Declaring, ExternFn, FnKind, OptionKind, Registry, ResultKind, Runtime, ScalarKind, StringKind, TupleKind,
    Ty,
};
use acvus_extern_test::{Safe, Value, live};
use acvus_mir::decl::{DeclaredFacts, Scheme};
use acvus_mir::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, TyKind};
use acvus_utils::{Interner, QualifiedRef};

fn pure_fn(cx: &mut Declaring<'_>, params: &[Ty], ret: Ty) -> Ty {
    cx.intern(TyKind::Fn {
        params: params.into(),
        ret,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(Flows::Every),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    })
}

fn declared(cx: &mut Declaring<'_>, name: &str, params: &[(&str, Ty)], ret: Ty) -> (QualifiedRef, Scheme) {
    let param_types: Vec<Ty> = params.iter().map(|(_, ty)| *ty).collect();
    let scheme = Scheme {
        ty: pure_fn(cx, &param_types, ret),
        param_names: params.iter().map(|(name, _)| cx.name(name)).collect(),
        bounds: Box::new([]),
        impl_of: None,
        requires: Box::new([]),
        facts: DeclaredFacts::default(),
        runs: Box::new([]),
        awaits: false,
        specialized: Box::new([]),
    };
    (cx.root(name), scheme)
}

fn int(cx: &mut Declaring<'_>) -> Ty {
    <i64 as Declared>::declare(cx)
}

fn string(cx: &mut Declaring<'_>) -> Ty {
    <String as Declared>::declare(cx)
}

fn make_int<R>(rt: &R, x: i64) -> R::Value
where
    R: Runtime,
{
    <R::Scalar as ScalarKind<R, i64>>::make(rt, x)
}

/// # Safety
///
/// `value` was made as an `i64`, and the caller owns it.
unsafe fn take_int<R>(rt: &R, value: R::Value) -> i64
where
    R: Runtime,
{
    unsafe { <R::Scalar as ScalarKind<R, i64>>::take(rt, value) }
}

// The bodies, as an author writes them.

fn inc(x: i64) -> i64 {
    x + 1
}

fn shout(s: String) -> String {
    s.to_uppercase()
}

fn swap(pair: (i64, String)) -> (String, i64) {
    (pair.1, pair.0)
}

fn or(option: Option<i64>, fallback: i64) -> i64 {
    option.unwrap_or(fallback)
}

fn parse(text: String) -> Result<i64, String> {
    text.parse().map_err(|error: std::num::ParseIntError| format!("{text:?} is no integer: {error}"))
}

// The entries, as the macros will write them.

struct Inc;

impl<R> ExternFn<R> for Inc
where
    R: Runtime,
{
    type Args = [R::Value; 1];

    fn declare(cx: &mut Declaring<'_>) -> (QualifiedRef, Scheme) {
        let int = int(cx);
        declared(cx, "inc", &[("x", int)], int)
    }

    unsafe fn call(rt: &R, [x]: [R::Value; 1]) -> R::Value {
        make_int(rt, inc(unsafe { take_int(rt, x) }))
    }
}

struct Shout;

impl<R> ExternFn<R> for Shout
where
    R: Runtime,
{
    type Args = [R::Value; 1];

    fn declare(cx: &mut Declaring<'_>) -> (QualifiedRef, Scheme) {
        let string = string(cx);
        declared(cx, "shout", &[("s", string)], string)
    }

    unsafe fn call(rt: &R, [s]: [R::Value; 1]) -> R::Value {
        <R::String as StringKind<R>>::make(rt, shout(unsafe { <R::String as StringKind<R>>::take(rt, s) }))
    }
}

struct Swap;

impl<R> ExternFn<R> for Swap
where
    R: Runtime,
{
    type Args = [R::Value; 1];

    fn declare(cx: &mut Declaring<'_>) -> (QualifiedRef, Scheme) {
        let pair = <(i64, String) as Declared>::declare(cx);
        let swapped = <(String, i64) as Declared>::declare(cx);
        declared(cx, "swap", &[("pair", pair)], swapped)
    }

    unsafe fn call(rt: &R, [pair]: [R::Value; 1]) -> R::Value {
        let [first, second] = unsafe { <R::Tuple as TupleKind<R>>::take::<2>(rt, pair) };
        let pair = unsafe { (take_int(rt, first), <R::String as StringKind<R>>::take(rt, second)) };
        let (first, second) = swap(pair);
        let parts = [<R::String as StringKind<R>>::make(rt, first), make_int(rt, second)];
        <R::Tuple as TupleKind<R>>::make(rt, parts)
    }
}

struct Or;

impl<R> ExternFn<R> for Or
where
    R: Runtime,
{
    type Args = [R::Value; 2];

    fn declare(cx: &mut Declaring<'_>) -> (QualifiedRef, Scheme) {
        let int = int(cx);
        let option = <Option<i64> as Declared>::declare(cx);
        declared(cx, "or", &[("option", option), ("fallback", int)], int)
    }

    unsafe fn call(rt: &R, [option, fallback]: [R::Value; 2]) -> R::Value {
        let option = unsafe { <R::Option as OptionKind<R>>::take(rt, option) }.map(|payload| unsafe { take_int(rt, payload) });
        make_int(rt, or(option, unsafe { take_int(rt, fallback) }))
    }
}

struct Parse;

impl<R> ExternFn<R> for Parse
where
    R: Runtime,
{
    type Args = [R::Value; 1];

    fn declare(cx: &mut Declaring<'_>) -> (QualifiedRef, Scheme) {
        let string = string(cx);
        let result = <Result<i64, String> as Declared>::declare(cx);
        declared(cx, "parse", &[("text", string)], result)
    }

    unsafe fn call(rt: &R, [text]: [R::Value; 1]) -> R::Value {
        let parsed = parse(unsafe { <R::String as StringKind<R>>::take(rt, text) })
            .map(|int| make_int(rt, int))
            .map_err(|text| <R::String as StringKind<R>>::make(rt, text));
        <R::Result as ResultKind<R>>::make(rt, parsed)
    }
}

struct Apply;

impl<R> ExternFn<R> for Apply
where
    R: Runtime,
{
    type Args = [R::Value; 2];

    fn declare(cx: &mut Declaring<'_>) -> (QualifiedRef, Scheme) {
        let int = int(cx);
        let function = pure_fn(cx, &[int], int);
        declared(cx, "apply", &[("f", function), ("x", int)], int)
    }

    unsafe fn call(rt: &R, [function, x]: [R::Value; 2]) -> R::Value {
        let applied = unsafe { <R::Fn as FnKind<R>>::call::<1>(rt, &function, [x]) };
        unsafe { R::release(function) };
        applied
    }
}

struct Adder;

impl<R> ExternFn<R> for Adder
where
    R: Runtime,
{
    type Args = [R::Value; 1];

    fn declare(cx: &mut Declaring<'_>) -> (QualifiedRef, Scheme) {
        let int = int(cx);
        let function = pure_fn(cx, &[int], int);
        declared(cx, "adder", &[("n", int)], function)
    }

    unsafe fn call(rt: &R, [n]: [R::Value; 1]) -> R::Value {
        let n = unsafe { take_int(rt, n) };
        // The body is a safe closure that consumes its argument on a premise
        // its type does not state: the extern note's open question on a
        // function value's body.
        <R::Fn as FnKind<R>>::make::<1>(rt, move |rt, [x]| make_int(rt, unsafe { take_int(rt, x) } + n))
    }
}

fn runtime() -> Safe {
    let mut rt = Safe::new(Interner::new());
    rt.register::<Inc>();
    rt.register::<Shout>();
    rt.register::<Swap>();
    rt.register::<Or>();
    rt.register::<Parse>();
    rt.register::<Apply>();
    rt.register::<Adder>();
    rt
}

fn make_string(rt: &Safe, s: &str) -> Value {
    <<Safe as Runtime>::String as StringKind<Safe>>::make(rt, s.to_owned())
}

/// # Safety
///
/// `value` was made as a `String`, and the caller owns it.
unsafe fn take_string(rt: &Safe, value: Value) -> String {
    unsafe { <<Safe as Runtime>::String as StringKind<Safe>>::take(rt, value) }
}

#[test]
fn a_scalar_crosses_by_value() {
    let rt = runtime();
    let out = unsafe { rt.call("inc", &[make_int(&rt, 41)]) };
    assert_eq!(unsafe { take_int(&rt, out) }, 42);
    assert_eq!(live(), 0);
}

#[test]
fn a_string_crosses_by_value_and_is_lent_in_place() {
    let rt = runtime();
    let mut out = unsafe { rt.call("shout", &[make_string(&rt, "ab")]) };
    unsafe { <<Safe as Runtime>::String as StringKind<Safe>>::view_mut(&rt, &mut out, |s| s.push('!')) };
    let len = unsafe { <<Safe as Runtime>::String as StringKind<Safe>>::view(&rt, &out, |s| s.len()) };
    assert_eq!(len, 3);
    assert_eq!(unsafe { take_string(&rt, out) }, "AB!");
    assert_eq!(live(), 0);
}

#[test]
fn a_tuple_crosses_part_by_part() {
    let rt = runtime();
    let pair = <<Safe as Runtime>::Tuple as TupleKind<Safe>>::make(&rt, [make_int(&rt, 7), make_string(&rt, "x")]);
    let mut out = unsafe { rt.call("swap", &[pair]) };
    unsafe {
        <<Safe as Runtime>::Tuple as TupleKind<Safe>>::view_mut::<2, _>(&rt, &mut out, |[_, second]| {
            <<Safe as Runtime>::Scalar as ScalarKind<Safe, i64>>::view_mut(&rt, second, |n| *n *= 2)
        })
    };
    let [first, second] = unsafe { <<Safe as Runtime>::Tuple as TupleKind<Safe>>::take::<2>(&rt, out) };
    assert_eq!(unsafe { (take_string(&rt, first), take_int(&rt, second)) }, ("x".to_owned(), 14));
    assert_eq!(live(), 0);
}

#[test]
fn an_option_lends_its_payload_and_replaces_its_variant() {
    let rt = runtime();
    let some = <<Safe as Runtime>::Option as OptionKind<Safe>>::make(&rt, Some(make_int(&rt, 3)));
    let out = unsafe { rt.call("or", &[some, make_int(&rt, 9)]) };
    assert_eq!(unsafe { take_int(&rt, out) }, 3);

    let mut option = <<Safe as Runtime>::Option as OptionKind<Safe>>::make(&rt, None);
    let none = unsafe { rt.call("or", &[<<Safe as Runtime>::Option as OptionKind<Safe>>::make(&rt, None), make_int(&rt, 9)]) };
    assert_eq!(unsafe { take_int(&rt, none) }, 9);
    let five = make_int(&rt, 5);
    let old = unsafe { <<Safe as Runtime>::Option as OptionKind<Safe>>::view_mut(&rt, &mut option, |option| option.replace(five)) };
    assert_eq!(old, None);
    let incremented = unsafe {
        <<Safe as Runtime>::Option as OptionKind<Safe>>::view_mut(&rt, &mut option, |option| {
            option.as_mut().map(|payload| <<Safe as Runtime>::Scalar as ScalarKind<Safe, i64>>::view_mut(&rt, payload, |n| *n += 1))
        })
    };
    assert_eq!(incremented, Some(()));
    let out = unsafe { rt.call("or", &[option, make_int(&rt, 0)]) };
    assert_eq!(unsafe { take_int(&rt, out) }, 6);
    assert_eq!(live(), 0);
}

#[test]
fn a_result_crosses_with_either_payload() {
    let rt = runtime();
    let ok = unsafe { rt.call("parse", &[make_string(&rt, "12")]) };
    let ok = unsafe { <<Safe as Runtime>::Result as ResultKind<Safe>>::take(&rt, ok) }.map(|int| unsafe { take_int(&rt, int) });
    assert_eq!(ok, Ok(12));
    let err = unsafe { rt.call("parse", &[make_string(&rt, "twelve")]) };
    let err = unsafe { <<Safe as Runtime>::Result as ResultKind<Safe>>::take(&rt, err) }.map_err(|text| unsafe { take_string(&rt, text) });
    assert_eq!(err, Err("\"twelve\" is no integer: invalid digit found in string".to_owned()));
    assert_eq!(live(), 0);
}

#[test]
fn a_function_value_is_given_and_called() {
    let rt = runtime();
    let add_two = unsafe { rt.call("adder", &[make_int(&rt, 2)]) };
    let out = unsafe { rt.call("apply", &[add_two, make_int(&rt, 40)]) };
    assert_eq!(unsafe { take_int(&rt, out) }, 42);
    assert_eq!(live(), 0);
}

#[test]
fn releasing_a_structure_releases_its_parts() {
    let rt = runtime();
    let inner = <<Safe as Runtime>::Option as OptionKind<Safe>>::make(&rt, Some(make_string(&rt, "s")));
    let outer = <<Safe as Runtime>::Tuple as TupleKind<Safe>>::make(&rt, [make_int(&rt, 1), inner]);
    assert_eq!(live(), 4);
    unsafe { Safe::release(outer) };
    assert_eq!(live(), 0);
}

#[test]
fn every_extern_fn_reaches_the_registry_declared() {
    let rt = runtime();
    let declared: Vec<usize> = rt.declared().iter().map(|(_, scheme)| scheme.param_names.len()).collect();
    assert_eq!(declared, [1, 1, 1, 2, 1, 2, 1]);
}

#[test]
#[should_panic(expected = "already borrowed")]
fn two_exclusive_views_of_one_value_panic() {
    let rt = runtime();
    let mut first = make_string(&rt, "a");
    let mut second = first;
    unsafe {
        <<Safe as Runtime>::String as StringKind<Safe>>::view_mut(&rt, &mut first, |_| {
            <<Safe as Runtime>::String as StringKind<Safe>>::view_mut(&rt, &mut second, |_| ())
        })
    };
}

#[test]
#[should_panic(expected = "a released value is used")]
fn a_released_value_is_not_used_again() {
    let rt = runtime();
    let value = make_int(&rt, 1);
    unsafe { Safe::release(value) };
    unsafe { <<Safe as Runtime>::Scalar as ScalarKind<Safe, i64>>::view(&rt, &value, |_| ()) };
}
