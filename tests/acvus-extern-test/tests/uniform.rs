//! `Uniform`, `Lend`, the type variable marker and `#[derive(Opaque)]`: a
//! value held by the runtime, lent through its kind, and an opaque value seen
//! from generic code and from a concrete view alike. Run under Miri.

use acvus_extern::{Crossing, Lend, Never, Opaque, Runtime, TypeVar, Uniform};
use acvus_extern_test::{Safe, live};
use acvus_utils::Interner;

type U<X> = Uniform<Safe, X>;

fn runtime() -> Safe {
    Safe::new(Interner::new())
}

#[derive(Opaque)]
struct Stack<T> {
    items: Vec<T>,
}

#[test]
fn a_scalar_is_got_and_lent_in_place() {
    let rt = runtime();
    let mut n = U::new(&rt, 5_i64);
    n.as_mut(&rt, |n| *n += 1);
    assert_eq!(n.get(&rt), 6);
    drop(n);
    assert_eq!(live(), 0);
}

#[test]
fn a_string_is_lent_in_place() {
    let rt = runtime();
    let mut s = U::new(&rt, String::from("ab"));
    s.as_mut(&rt, |s| s.push('c'));
    assert_eq!(s.as_ref(&rt, |s| s.clone()), "abc");
    drop(s);
    assert_eq!(live(), 0);
}

#[test]
fn an_option_is_lent_whole_and_its_payload_opens_natively() {
    let rt = runtime();
    let mut option = U::new(&rt, Some(7_i64));
    let seven = option.as_ref(&rt, |option| option.as_ref().map(|payload| payload.get(&rt)));
    assert_eq!(seven, Some(7));
    let replaced = U::new(&rt, 9_i64);
    let old = option.as_mut(&rt, |option| option.replace(replaced));
    assert_eq!(old.map(|old| old.get(&rt)), Some(7));
    assert_eq!(option.as_ref(&rt, |option| option.as_ref().map(|payload| payload.get(&rt))), Some(9));
    option.as_mut(&rt, |option| *option = None);
    assert_eq!(live(), 1);
    drop(option);
    assert_eq!(live(), 0);
}

#[test]
fn a_result_is_lent_whole_with_either_payload() {
    let rt = runtime();
    let result: U<Result<i64, String>> = U::new(&rt, Err("no".to_owned()));
    let err = result.as_ref(&rt, |result| result.as_ref().map(|ok| ok.get(&rt)).map_err(|err| err.as_ref(&rt, String::clone)));
    assert_eq!(err, Err("no".to_owned()));
    drop(result);
    assert_eq!(live(), 0);
}

/// An author's generic body: it sees only `T: TypeVar`, and moves values.
fn count<R, T>(stack: &Stack<T>) -> usize
where
    R: Runtime,
    T: TypeVar<R>,
{
    stack.items.len()
}

fn type_variable<R, T>()
where
    R: Runtime,
    T: TypeVar<R>,
{
}

#[test]
fn every_uniform_is_a_type_variable() {
    type_variable::<Safe, U<Never>>();
    type_variable::<Safe, U<i64>>();
    type_variable::<Safe, U<Stack<U<i64>>>>();
}

#[test]
fn an_opaque_value_is_seen_from_generic_code_and_from_a_concrete_view() {
    let rt = runtime();
    let items = vec![U::new(&rt, 1_i64), U::new(&rt, 2_i64)];
    let stack = U::new(&rt, Stack { items });
    let given = stack.give(&rt);

    // The generic entry opens it at the marker, as a type variable's value.
    let generic: U<Stack<U<Never>>> = Crossing::take(&rt, given);
    let counted = generic.as_ref(&rt, count::<Safe, U<Never>>);
    assert_eq!(counted, 2);

    // A concrete view opens the same value at its concrete element type.
    let concrete: U<Stack<U<i64>>> = Crossing::take(&rt, generic.give(&rt));
    let sum: i64 = concrete.as_ref(&rt, |stack| stack.items.iter().map(|item| item.get(&rt)).sum());
    assert_eq!(sum, 3);
    drop(concrete);
    assert_eq!(live(), 0);
}

#[test]
fn an_opaque_value_crosses_by_value_and_is_lent() {
    let rt = runtime();
    let mut stack = U::new(&rt, Stack { items: vec![U::new(&rt, String::from("a"))] });
    let pushed = U::new(&rt, String::from("b"));
    stack.as_mut(&rt, |stack| stack.items.push(pushed));
    let taken: Stack<U<String>> = Crossing::take(&rt, stack.give(&rt));
    let joined: String = taken.items.iter().map(|item| item.as_ref(&rt, String::clone)).collect();
    assert_eq!(joined, "ab");
    drop(taken);
    assert_eq!(live(), 0);
}

fn lends_as<R, X>()
where
    R: Runtime,
    X: Lend<R>,
{
}

#[test]
fn an_opaque_type_lends_as_itself() {
    lends_as::<Safe, Stack<U<i64>>>();
}
