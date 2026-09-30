//! The crossing's tests, each run on `WordRuntime` and on
//! `Adapter<WordRuntime>` with the same expected result (structure B-adapter).
//!
//! A `Frame` stands in for the run: it passes its variables through
//! `Entry::new`, as the runtime's boundary does, and the glue each test writes
//! by hand opens them with `Entry::open`, as the attribute's glue will.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::adapter::Adapter;
use acvus_extern::mode::{ByMut, ByRef, ByStr, ByUniform, ByValue};
use acvus_extern::runtime::{Crosses, Mutability, Runtime};
use acvus_extern_test::{Frame, Pair, Pass, Tally, Word, WordRuntime};

use Mutability::{Exclusive, Shared};

/// Runs `case` on `WordRuntime` and on the adapter over it.
macro_rules! on_both {
    ($($name:ident: $case:ident;)+) => {
        mod on_word_runtime {
            $(
                #[test]
                fn $name() {
                    super::$case(&super::WordRuntime::default());
                }
            )+
        }

        mod on_adapter {
            $(
                #[test]
                fn $name() {
                    super::$case(&super::Adapter::new(super::WordRuntime::default()));
                }
            )+
        }
    };
}

on_both! {
    a_value_given_by_value_is_taken_at_its_type: given_by_value;
    a_value_taken_at_once_is_released_by_the_handler_once: taken_at_once_released_once;
    a_held_value_only_dropped_is_released_once: held_and_dropped;
    a_held_value_taken_is_released_by_its_new_owner_once: held_and_taken;
    a_held_value_is_read_and_written_in_place: held_opened_in_place;
    a_value_never_passed_is_released_by_the_run_once: never_passed;
    a_shared_lend_reads_the_variable_in_place: lent_shared;
    an_exclusive_lend_writes_the_variable_in_place: lent_exclusive;
    a_scalar_lent_exclusively_is_written_in_its_word: scalar_lent_exclusive;
    a_string_is_lent_as_text: lent_as_text;
    a_borrowed_result_names_the_whole_lent_value: borrowed_whole;
    a_borrowed_result_names_a_range_of_lent_text: borrowed_str_range;
    a_borrowed_result_names_a_strings_text: borrowed_text;
    a_borrowed_result_names_a_field: borrowed_field;
    a_borrowed_result_names_an_element_the_lent_value_owns: borrowed_element;
    a_writable_borrowed_result_writes_the_lent_value: borrowed_exclusive;
}

/// A variable holding `value`, made as a program makes one: the result of a
/// call.
fn produce<R, T>(frame: &mut Frame<'_, R>, value: T) -> usize
where
    R: Runtime<Value = Word>,
    T: Crosses,
{
    frame.call(&[], |entry| entry.open::<(), ()>(|cx, (), ()| cx.finish(value)))
}

/// A variable's value, given to a call that takes it by value.
fn read<R, T>(frame: &mut Frame<'_, R>, variable: usize) -> T
where
    R: Runtime<Value = Word>,
    T: Crosses,
{
    let mut read = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByValue<T>,), ()>(|cx, (value,), ()| {
            read = Some(value);
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// What a variable holding a `String`, or a reference to one, reads as.
fn read_lent<R>(frame: &mut Frame<'_, R>, pass: Pass) -> String
where
    R: Runtime<Value = Word>,
{
    let mut read = None;
    frame.call(&[pass], |entry| {
        entry.open::<(ByRef<String>,), ()>(|cx, (lent,), ()| {
            read = Some(lent.clone());
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// What a variable holding a `String` or text reads as, lent as `&str`.
fn read_text<R>(frame: &mut Frame<'_, R>, pass: Pass) -> String
where
    R: Runtime<Value = Word>,
{
    let mut read = None;
    frame.call(&[pass], |entry| {
        entry.open::<(ByStr,), ()>(|cx, (text,), ()| {
            read = Some(String::from(&*text));
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

fn tally() -> (Tally, Arc<AtomicUsize>) {
    let drops = Arc::new(AtomicUsize::new(0));
    (Tally(Arc::clone(&drops)), drops)
}

fn given_by_value<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let text = produce(&mut frame, String::from("moved"));
    let number = produce(&mut frame, -41i64);

    assert_eq!(read::<_, String>(&mut frame, text), "moved");
    assert_eq!(read::<_, i64>(&mut frame, number), -41);
}

fn taken_at_once_released_once<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, value);

    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByValue<Tally>,), ()>(|cx, (taken,), ()| {
            drop(taken);
            cx.finish(())
        })
    });
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn held_and_dropped<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, value);

    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByUniform<Tally>,), ()>(|cx, (held,), ()| {
            drop(held);
            cx.finish(())
        })
    });
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn held_and_taken<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, value);

    let mut kept = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByUniform<Tally>,), ()>(|cx, (held,), ()| {
            kept = Some(held.into_inner(&cx));
            cx.finish(())
        })
    });
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(kept);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn held_opened_in_place<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("held"));

    let mut read = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByUniform<String>,), ()>(|cx, (mut held,), ()| {
            let before = held.open(&cx).len();
            held.open_mut(&cx).push_str(" and written");
            read = Some((before, held.into_inner(&cx)));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((4, String::from("held and written"))));
}

fn never_passed<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    produce(&mut frame, value);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn lent_shared<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("lent"));

    assert_eq!(read_lent(&mut frame, Pass::Lend(variable, Shared)), "lent");
    assert_eq!(read::<_, String>(&mut frame, variable), "lent");
}

fn lent_exclusive<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("lent"));

    frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<String>,), ()>(|cx, (mut lent,), ()| {
            lent.push_str(" and written");
            cx.finish(())
        })
    });

    assert_eq!(read::<_, String>(&mut frame, variable), "lent and written");
}

fn scalar_lent_exclusive<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, 5i16);

    let mut before = None;
    frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<i16>,), ()>(|cx, (mut lent,), ()| {
            before = Some(*lent);
            *lent = -300;
            cx.finish(())
        })
    });

    assert_eq!(before, Some(5));
    assert_eq!(read::<_, i16>(&mut frame, variable), -300);
}

fn lent_as_text<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("text"));

    assert_eq!(read_text(&mut frame, Pass::Lend(variable, Shared)), "text");
}

fn borrowed_whole<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("whole"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<String>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |s: &String| s))
    });

    assert_eq!(read_lent(&mut frame, Pass::Same(borrowed, Shared)), "whole");
    assert_eq!(read_text(&mut frame, Pass::Same(borrowed, Shared)), "whole");
}

fn borrowed_str_range<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("a range"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByStr,), ()>(|cx, (text,), ()| cx.finish_lent(text, |s: &str| &s[2..5]))
    });

    assert_eq!(read_text(&mut frame, Pass::Same(borrowed, Shared)), "ran");
}

fn borrowed_text<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("owned text"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<String>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |s: &String| s.as_str()))
    });

    assert_eq!(read_text(&mut frame, Pass::Same(borrowed, Shared)), "owned text");
}

fn pair() -> Pair {
    Pair { name: String::from("field"), items: vec![3, 5, 8] }
}

fn borrowed_field<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, pair());

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |p: &Pair| &p.name))
    });

    assert_eq!(read_lent(&mut frame, Pass::Same(borrowed, Shared)), "field");
}

fn borrowed_element<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, pair());

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |p: &Pair| &p.items[1]))
    });

    let mut read = None;
    frame.call(&[Pass::Same(borrowed, Shared)], |entry| {
        entry.open::<(ByRef<i64>,), ()>(|cx, (element,), ()| {
            read = Some(*element);
            cx.finish(())
        })
    });
    assert_eq!(read, Some(5));
}

fn borrowed_exclusive<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("borrowed"));

    let borrowed = frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<String>,), ()>(|cx, (lent,), ()| cx.finish_lent_mut(lent, |s: &mut String| s))
    });
    frame.call(&[Pass::Same(borrowed, Exclusive)], |entry| {
        entry.open::<(ByMut<String>,), ()>(|cx, (mut target,), ()| {
            target.push_str(" and written");
            cx.finish(())
        })
    });

    assert_eq!(read::<_, String>(&mut frame, variable), "borrowed and written");
}

// Breaches the adapter's tracking covers. Under `WordRuntime` alone each is
// undefined behaviour, so they run on the adapter only.

#[test]
#[should_panic(expected = "overlaps a live Exclusive lend")]
fn the_adapter_refuses_a_shared_lend_over_a_live_exclusive_one() {
    let runtime = Adapter::new(WordRuntime::default());
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, String::from("twice"));

    frame.call(&[Pass::Lend(variable, Exclusive), Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByMut<String>, ByRef<String>), ()>(|cx, _, ()| cx.finish(()))
    });
}

#[test]
#[should_panic(expected = "a Large value is read as i64")]
fn the_adapter_refuses_a_lend_at_a_type_the_value_is_not() {
    let runtime = Adapter::new(WordRuntime::default());
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, String::from("not a number"));

    frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<i64>,), ()>(|cx, _, ()| cx.finish(()))
    });
}

#[test]
#[should_panic(expected = "a I64 value is read as alloc::string::String")]
fn the_adapter_refuses_a_take_at_a_type_the_value_is_not() {
    let runtime = Adapter::new(WordRuntime::default());
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, 7i64);

    read::<_, String>(&mut frame, variable);
}
