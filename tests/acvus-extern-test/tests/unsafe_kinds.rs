//! The extern crate's unsafe kinds, `StoredOpaque` and `StoredRef`, over the
//! safe runtime's `Stores`. These are the tests Miri runs: each one reads
//! storage back as a Rust type, the one `unsafe` the extern crate writes.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{OpaqueKind, OptionKind, RefKind, Runtime};
use acvus_extern_test::{Safe, Value, live};
use acvus_utils::Interner;

type Opaque = <Safe as Runtime>::Opaque;
type Ref = <Safe as Runtime>::Ref;

fn runtime() -> Safe {
    Safe::new(Interner::new())
}

/// Counts its own drops, so a test can see a value ended exactly once.
struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn an_opaque_value_is_lent_and_taken_back() {
    let rt = runtime();
    let mut held = <Opaque as OpaqueKind<Safe>>::make(&rt, vec![1_i64, 2]);
    let len = <Opaque as OpaqueKind<Safe>>::view(&rt, &held, |xs: &Vec<i64>| xs.len());
    assert_eq!(len, 2);
    <Opaque as OpaqueKind<Safe>>::view_mut(&rt, &mut held, |xs: &mut Vec<i64>| xs.push(3));
    let xs: Vec<i64> = <Opaque as OpaqueKind<Safe>>::take(&rt, held);
    assert_eq!(xs, [1, 2, 3]);
    assert_eq!(live(), 0);
}

#[test]
fn an_opaque_value_ends_once_with_its_last_owner() {
    let rt = runtime();
    let drops = Arc::new(AtomicUsize::new(0));
    let held = <Opaque as OpaqueKind<Safe>>::make(&rt, Counted(Arc::clone(&drops)));
    let shared = rt.share(&held);
    Safe::release(held);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    Safe::release(shared);
    assert_eq!(drops.load(Ordering::Relaxed), 1);

    let alone = <Opaque as OpaqueKind<Safe>>::make(&rt, Counted(Arc::clone(&drops)));
    Safe::release(alone);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
    assert_eq!(live(), 0);
}

#[test]
#[should_panic(expected = "while another owner shares it")]
fn a_shared_opaque_value_is_not_taken() {
    let rt = runtime();
    let held = <Opaque as OpaqueKind<Safe>>::make(&rt, 1_u8);
    let _shared = rt.share(&held);
    let _: u8 = <Opaque as OpaqueKind<Safe>>::take(&rt, held);
}

/// A value that holds a loan, as an extern type may.
struct Cursor<'a> {
    xs: &'a [i64],
    at: usize,
}

/// The author's body.
fn next<'a>(cursor: &mut Cursor<'a>) -> Option<&'a i64> {
    let x = cursor.xs.get(cursor.at)?;
    cursor.at += 1;
    Some(x)
}

/// The entry, as the macros will write it: every loan of the call is opened
/// at `'call`, a borrow of the call's own arguments, and the result is given
/// back before the call ends.
fn next_entry<'call>(rt: &Safe, [cursor]: &'call mut [Value; 1]) -> Value {
    let found = <Opaque as OpaqueKind<Safe>>::view_mut(rt, cursor, |cursor: &mut Cursor<'call>| {
        next(cursor).map(|x| <Ref as RefKind<Safe>>::make(rt, x))
    });
    <<Safe as Runtime>::Option as OptionKind<Safe>>::make(rt, found)
}

#[test]
fn a_value_holding_a_loan_is_held_and_opened_at_the_calls_lifetime() {
    let rt = runtime();
    let xs = vec![10_i64, 20];
    let mut args = [<Opaque as OpaqueKind<Safe>>::make(&rt, Cursor { xs: &xs, at: 0 })];
    let mut seen = Vec::new();
    loop {
        let found = next_entry(&rt, &mut args);
        let Some(found) = <<Safe as Runtime>::Option as OptionKind<Safe>>::take(&rt, found) else {
            break;
        };
        seen.push(<Ref as RefKind<Safe>>::view(&rt, &found, |x: &i64| *x));
        Safe::release(found);
    }
    assert_eq!(seen, [10, 20]);
    let [cursor] = args;
    Safe::release(cursor);
    assert_eq!(live(), 0);
}

#[test]
fn a_reference_crosses_as_its_words_thin_or_fat() {
    let rt = runtime();
    let n = 5_i64;
    let thin = <Ref as RefKind<Safe>>::make(&rt, &n);
    assert_eq!(<Ref as RefKind<Safe>>::view(&rt, &thin, |n: &i64| *n), 5);

    let text = String::from("hello");
    let fat = <Ref as RefKind<Safe>>::make(&rt, text.as_str());
    assert!(<Ref as RefKind<Safe>>::view(&rt, &fat, |s: &str| s == "hello"));

    let mut xs = vec![1_i64, 2, 3];
    let mut run = <Ref as RefKind<Safe>>::make_mut(&rt, xs.as_mut_slice());
    <Ref as RefKind<Safe>>::view_mut(&rt, &mut run, |xs: &mut [i64]| xs[0] = 9);
    for value in [thin, fat, run] {
        Safe::release(value);
    }
    assert_eq!(xs, [9, 2, 3]);
    assert_eq!(live(), 0);
}

#[test]
fn a_reference_into_an_opaque_value_outlives_the_view_it_was_made_in() {
    let rt = runtime();
    let held = <Opaque as OpaqueKind<Safe>>::make(&rt, vec![7_i64, 8]);
    let second = <Opaque as OpaqueKind<Safe>>::view(&rt, &held, |xs: &Vec<i64>| <Ref as RefKind<Safe>>::make(&rt, &xs[1]));
    assert_eq!(<Ref as RefKind<Safe>>::view(&rt, &second, |x: &i64| *x), 8);
    Safe::release(second);
    let xs: Vec<i64> = <Opaque as OpaqueKind<Safe>>::take(&rt, held);
    assert_eq!(xs, [7, 8]);
    assert_eq!(live(), 0);
}

#[repr(transparent)]
struct Celsius(u64);

#[repr(transparent)]
struct Kelvin(u64);

/// The premise the note's type variables rest on: a value made as `X<T>` is
/// read as `X<U>` where `T` and `U` share a layout. Miri checks this build.
#[test]
fn an_opaque_value_is_read_at_another_instantiation_of_its_layout() {
    let rt = runtime();
    let held = <Opaque as OpaqueKind<Safe>>::make(&rt, vec![Celsius(1), Celsius(2)]);
    let sum = <Opaque as OpaqueKind<Safe>>::view(&rt, &held, |xs: &Vec<Kelvin>| xs.iter().map(|k| k.0).sum::<u64>());
    assert_eq!(sum, 3);
    let xs: Vec<Kelvin> = <Opaque as OpaqueKind<Safe>>::take(&rt, held);
    assert_eq!(xs.len(), 2);
    assert_eq!(live(), 0);
}
