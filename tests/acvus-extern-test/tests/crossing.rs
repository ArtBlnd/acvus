//! The crossing's tests, each run on `WordRuntime` and on
//! `Adapter<WordRuntime>` with the same expected result (structure B-adapter).
//!
//! A `Frame` stands in for the run: it passes its variables through
//! `Entry::new`, as the runtime's boundary does, and the glue each test writes
//! by hand opens them with `Entry::open`, as the attribute's glue will.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use acvus_extern::adapter::Adapter;
use acvus_extern::Uniform;
use acvus_extern::callback::{CallsBack, Lent, LentMut};
use acvus_extern::context::Returning;
use acvus_extern::guard::{Guard, GuardMut};
use acvus_extern::mode::{ByCallback, ByMut, ByRef, BySlice, BySliceMut, ByStr, ByUniform, ByValue};
use acvus_extern::done::Done;
use acvus_extern::register::{ExternFn, Refused, Registers};
use acvus_extern::runtime::{Crosses, Mutability, Runtime};
use acvus_extern::{Entry, SiteShape};
use acvus_utils::{Interner, QualifiedRef};
use acvus_extern_test::{Frame, Number, Pair, Pass, Tally, Word, WordRuntime};

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
    a_held_primitive_is_read_and_written_by_a_copy: held_primitive;
    a_value_never_passed_is_released_by_the_run_once: never_passed;
    a_shared_lend_reads_the_variable_in_place: lent_shared;
    an_exclusive_lend_writes_the_variable_in_place: lent_exclusive;
    a_string_is_lent_as_text: lent_as_text;
    a_borrowed_result_names_the_whole_lent_value: borrowed_whole;
    a_borrowed_result_names_a_range_of_lent_text: borrowed_str_range;
    a_borrowed_result_names_a_strings_text: borrowed_text;
    a_borrowed_result_names_a_field: borrowed_field;
    a_borrowed_result_names_an_element_the_lent_value_owns: borrowed_element;
    a_writable_borrowed_result_writes_the_lent_value: borrowed_exclusive;
    an_array_is_lent_as_a_slice: lent_as_slice;
    a_borrowed_result_names_a_run_of_a_lent_slice: borrowed_sub_slice;
    an_awaited_body_takes_its_argument_and_finishes_after_a_wait: awaited_by_value;
    an_awaited_body_reads_its_lend_before_a_wait: awaited_lent;
    an_awaited_body_holds_a_guard_across_a_wait: awaited_guard_held;
    a_function_value_is_called_through_its_dispatch: function_called;
    a_function_value_releases_what_it_captures_once: function_released;
    a_callback_is_lent_a_part_of_the_handlers_storage: callback_lent;
    a_callback_writes_through_a_part_lent_exclusively: callback_lent_mut;
    a_callback_is_lent_text: callback_lent_text;
    an_awaited_callback_takes_its_arguments_by_move: callback_awaited;
    a_function_body_reads_its_held_argument_through_its_context: function_with_context;
    an_option_a_result_and_a_tuple_cross_by_their_parts: aggregates_round_trip;
    an_aggregate_never_taken_releases_its_parts_once: aggregate_released_once;
    a_type_variables_value_crosses_inside_an_option: uniform_inside_an_option;
}

/// Runs `case` on a `WordRuntime` and on an adapter over one, each owned, so
/// the case may register functions with it.
macro_rules! on_both_owned {
    ($($name:ident: $case:ident;)+) => {
        mod owned_on_word_runtime {
            $(
                #[test]
                fn $name() {
                    super::$case(super::WordRuntime::default());
                }
            )+
        }

        mod owned_on_adapter {
            $(
                #[test]
                fn $name() {
                    super::$case(super::Adapter::new(super::WordRuntime::default()));
                }
            )+
        }
    };
}

on_both_owned! {
    a_registered_function_is_called_through_its_dispatch: registered_called;
    a_path_registered_twice_is_refused: registered_twice;
}

/// `add(a: u64, b: u64) -> u64`, its dispatch written as `#[extern_fn]`
/// writes it: the entry over the argument run, each argument opened by its
/// mode, the body, its result.
struct Add;

// SAFETY: this stands in for `#[extern_fn]`: `call` opens the two arguments
// by the declaration's modes, `u64` by value.
#[allow(unsafe_code)]
unsafe impl acvus_extern::seal::ExternFns for Add {}

impl<R: Runtime> ExternFn<R> for Add {
    fn path(&self, names: &Interner) -> QualifiedRef {
        QualifiedRef::root(names.intern("add"))
    }

    fn call(&self, runtime: &R, args: &[R::Value], site: &SiteShape<R>) -> Done<R> {
        Entry::new(runtime, args, site).open::<(ByValue<u64>, ByValue<u64>), ()>(|cx, (a, b), ()| cx.finish(a + b))
    }
}

fn registered_called<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register(Add).expect("the path is free");
    let path = QualifiedRef::root(runtime.interner().intern("add"));

    let mut frame = Frame::new(&runtime);
    let a = produce(&mut frame, 2u64);
    let b = produce(&mut frame, 3u64);
    let sum = frame.call_registered(path, &[Pass::Give(a), Pass::Give(b)]);

    assert_eq!(read::<_, u64>(&mut frame, sum), 5);
}

fn registered_twice<R: Registers<Value = Word>>(mut runtime: R) {
    runtime.register(Add).expect("the path is free");
    let path = QualifiedRef::root(runtime.interner().intern("add"));

    assert_eq!(runtime.register(Add), Err(Refused::PathTaken(path)));
}

/// A variable holding `value`, made as a program makes one: the result of a
/// call.
fn produce<R, T>(frame: &mut Frame<'_, R>, value: T) -> usize
where
    R: Runtime<Value = Word>,
    T: Crosses<R>,
{
    frame.call(&[], |entry| entry.open::<(), ()>(|cx, (), ()| cx.finish(value)))
}

/// The values of the array a pass lends, read by a call that takes a slice.
fn read_array<R>(frame: &mut Frame<'_, R>, pass: Pass) -> Vec<i64>
where
    R: Runtime<Value = Word>,
{
    let mut read = None;
    frame.call(&[pass], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| {
            read = Some(run.iter().map(|element| element.get(&cx)).collect());
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// A variable's value, given to a call that takes it by value.
fn read<R, T>(frame: &mut Frame<'_, R>, variable: usize) -> T
where
    R: Runtime<Value = Word>,
    T: Crosses<R>,
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

/// The name of the `Pair` a variable holds, or a reference to one names.
fn read_name<R>(frame: &mut Frame<'_, R>, pass: Pass) -> String
where
    R: Runtime<Value = Word>,
{
    let mut read = None;
    frame.call(&[pass], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| {
            read = Some(lent.name.clone());
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
    let variable = produce(&mut frame, pair("held"));

    let mut read = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByUniform<Pair>,), ()>(|cx, (mut held,), ()| {
            let before = held.open(&cx).name.len();
            held.open_mut(&cx).name.push_str(" and written");
            read = Some((before, held.into_inner(&cx).name));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((4, String::from("held and written"))));
}

fn held_primitive<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, 5i64);

    let mut read = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByUniform<i64>,), ()>(|cx, (mut held,), ()| {
            let before = held.get(&cx);
            held.set(&cx, 9);
            read = Some((before, held.into_inner(&cx)));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((5, 9)));
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
    let variable = produce(&mut frame, pair("lent"));

    assert_eq!(read_name(&mut frame, Pass::Lend(variable, Shared)), "lent");
    assert_eq!(read::<_, Pair>(&mut frame, variable).name, "lent");
}

fn lent_exclusive<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, pair("lent"));

    frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<Pair>,), ()>(|cx, (mut lent,), ()| {
            lent.name.push_str(" and written");
            cx.finish(())
        })
    });

    assert_eq!(read::<_, Pair>(&mut frame, variable).name, "lent and written");
}

fn lent_as_text<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, String::from("text"));

    assert_eq!(read_text(&mut frame, Pass::Lend(variable, Shared)), "text");
}

fn borrowed_whole<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, pair("whole"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |p: &Pair| p))
    });

    assert_eq!(read_name(&mut frame, Pass::Same(borrowed, Shared)), "whole");
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
    let variable = produce(&mut frame, pair("owned text"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |p: &Pair| p.name.as_str()))
    });

    assert_eq!(read_text(&mut frame, Pass::Same(borrowed, Shared)), "owned text");
}

fn pair(name: &str) -> Pair {
    Pair { name: String::from(name), lead: Number(2), items: vec![Number(3), Number(5), Number(8)] }
}

/// The `Number` a reference value names.
fn read_number<R>(frame: &mut Frame<'_, R>, pass: Pass) -> Number
where
    R: Runtime<Value = Word>,
{
    let mut read = None;
    frame.call(&[pass], |entry| {
        entry.open::<(ByRef<Number>,), ()>(|cx, (number,), ()| {
            read = Some(*number);
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

fn borrowed_field<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, pair("field"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |p: &Pair| &p.lead))
    });

    assert_eq!(read_number(&mut frame, Pass::Same(borrowed, Shared)), Number(2));
}

fn borrowed_element<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, pair("element"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_lent(lent, |p: &Pair| &p.items[1]))
    });

    assert_eq!(read_number(&mut frame, Pass::Same(borrowed, Shared)), Number(5));
}

fn borrowed_exclusive<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, pair("borrowed"));

    let borrowed = frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<Pair>,), ()>(|cx, (lent,), ()| cx.finish_lent_mut(lent, |p: &mut Pair| p))
    });
    frame.call(&[Pass::Same(borrowed, Exclusive)], |entry| {
        entry.open::<(ByMut<Pair>,), ()>(|cx, (mut target,), ()| {
            target.name.push_str(" and written");
            cx.finish(())
        })
    });

    assert_eq!(read::<_, Pair>(&mut frame, variable).name, "borrowed and written");
}

fn lent_as_slice<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = frame.array(&[3i64, 5, 8]);

    let mut sum = None;
    frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| {
            sum = Some(run.iter().map(|element| element.get(&cx)).sum::<i64>());
            cx.finish(())
        })
    });
    frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(BySliceMut<i64>,), ()>(|cx, (mut run,), ()| {
            for element in run.iter_mut() {
                let doubled = element.get(&cx) * 2;
                element.set(&cx, doubled);
            }
            cx.finish(())
        })
    });

    assert_eq!(sum, Some(16));
    assert_eq!(read_array(&mut frame, Pass::Lend(variable, Shared)), [6, 10, 16]);
}

fn borrowed_sub_slice<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = frame.array(&[3i64, 5, 8]);

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| cx.finish_lent(run, |s: &[Uniform<R, i64>]| &s[1..]))
    });

    assert_eq!(read_array(&mut frame, Pass::Same(borrowed, Shared)), [5, 8]);
}

/// The awaited entry's future is `Send` (the user's decision of 2026-10-01).
fn sent<F: Future + Send>(future: F) -> F {
    future
}

fn executor() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().build().expect("a current-thread executor")
}

fn awaited_by_value<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let text = produce(&mut frame, String::from("awaited"));

    let length = executor().block_on(frame.call_awaited(&[Pass::Give(text)], async |entry| {
        let body = entry.open_awaited::<(ByValue<String>,), (), _>(|cx, (text,), ()| async move {
            cx.sleep(Duration::from_millis(1)).await;
            cx.finish(text.len() as i64)
        });
        sent(body).await
    }));

    assert_eq!(read::<_, i64>(&mut frame, length), 7);
}

fn awaited_lent<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let text = produce(&mut frame, pair("held across"));

    let length = executor().block_on(frame.call_awaited(&[Pass::Lend(text, Shared)], async |entry| {
        let body = entry.open_awaited::<(ByRef<Pair>,), (), _>(|cx, (lent,), ()| {
            let length = lent.name.len() as i64;
            drop(lent);
            async move {
                cx.sleep(Duration::from_millis(1)).await;
                cx.finish(length)
            }
        });
        sent(body).await
    }));

    assert_eq!(read::<_, i64>(&mut frame, length), 11);
    assert_eq!(read::<_, Pair>(&mut frame, text).name, "held across");
}

fn awaited_guard_held<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let text = produce(&mut frame, pair("held across a wait"));

    let length = executor().block_on(frame.call_awaited(&[Pass::Lend(text, Shared)], async |entry| {
        let body = entry.open_awaited::<(ByRef<Pair>,), (), _>(|cx, (lent,), ()| async move {
            cx.sleep(Duration::from_millis(1)).await;
            let length = lent.name.len() as i64;
            cx.finish(length)
        });
        sent(body).await
    }));

    assert_eq!(read::<_, i64>(&mut frame, length), 18);
    assert_eq!(read::<_, Pair>(&mut frame, text).name, "held across a wait");
}

/// A function value's dispatch, written as `#[extern_fn]` writes one: the
/// entry over the argument run, each argument opened by its mode `A`, the body
/// `f`, and its result given.
struct Dispatch<A, F> {
    f: F,
    _modes: PhantomData<fn() -> A>,
}

fn dispatch<A, F>(f: F) -> Dispatch<A, F> {
    Dispatch { f, _modes: PhantomData }
}

// SAFETY: this stands in for `#[extern_fn]`: `call` opens the arguments by
// the modes `A` the body `f` takes them at.
#[allow(unsafe_code)]
unsafe impl<A, F> acvus_extern::seal::ExternFns for Dispatch<A, F> {}

impl<R, A, U, F> ExternFn<R> for Dispatch<A, F>
where
    R: Runtime,
    A: acvus_extern::Args<R> + 'static,
    U: Crosses<R>,
    F: for<'c> Fn(&Returning<'c, R>, A::Opened<'c>) -> U + Send + Sync + 'static,
{
    fn path(&self, names: &Interner) -> QualifiedRef {
        QualifiedRef::root(names.intern("lambda"))
    }

    fn call(&self, runtime: &R, args: &[R::Value], site: &SiteShape<R>) -> Done<R> {
        Entry::new(runtime, args, site).open::<A, ()>(|cx, args, ()| {
            let result = (self.f)(&cx, args);
            cx.finish(result)
        })
    }
}

/// The length of text lent to it.
fn length_of<R: Runtime>(_: &Returning<'_, R>, (text,): (Guard<'_, &str>,)) -> i64 {
    text.len() as i64
}

fn function_called<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let function = frame.function(dispatch::<(ByStr,), _>(length_of::<R>));
    let short = produce(&mut frame, String::from("four"));
    let long = produce(&mut frame, String::from("eleven long"));

    let first = frame.call_function(function, &[Pass::Lend(short, Shared)]);
    let second = frame.call_function(function, &[Pass::Lend(long, Shared)]);

    assert_eq!(read::<_, i64>(&mut frame, first), 4);
    assert_eq!(read::<_, i64>(&mut frame, second), 11);
    assert_eq!(read::<_, String>(&mut frame, short), "four");
}

fn function_released<R: Runtime<Value = Word>>(runtime: &R) {
    let (captured, drops) = tally();
    let mut frame = Frame::new(runtime);
    let function = frame.function(dispatch::<(ByValue<i64>,), _>(move |_: &Returning<'_, R>, (n,): (i64,)| {
        n + Arc::strong_count(&captured.0) as i64
    }));

    let forty = produce(&mut frame, 40i64);
    let called = frame.call_function(function, &[Pass::Give(forty)]);
    assert_eq!(read::<_, i64>(&mut frame, called), 42);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn callback_lent<R: CallsBack<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let function = frame.function(dispatch::<(ByStr,), _>(length_of::<R>));
    let text = produce(&mut frame, String::from("lent to a callback"));

    let called = frame.call(&[Pass::Lend(text, Shared), Pass::Function(function)], |entry| {
        entry.open::<(ByStr, ByCallback<(Lent<str>,), i64, ()>), ()>(|cx, (text, callback), ()| {
            let first = callback.call(&cx, (&*text,));
            let second = callback.call(&cx, (&*text,));
            cx.finish(first + second)
        })
    });

    assert_eq!(read::<_, i64>(&mut frame, called), 36);
    assert_eq!(read::<_, String>(&mut frame, text), "lent to a callback");
}

fn append_written<R: Runtime>(_: &Returning<'_, R>, (mut lent,): (GuardMut<'_, &mut Pair>,)) {
    lent.name.push_str(" and written");
}

fn words_in<R: Runtime>(_: &Returning<'_, R>, (text,): (Guard<'_, &str>,)) -> i64 {
    text.split(' ').count() as i64
}

fn callback_lent_mut<R: CallsBack<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let append = frame.function(dispatch::<(ByMut<Pair>,), _>(append_written::<R>));
    let lent = produce(&mut frame, pair("lent"));

    frame.call(&[Pass::Lend(lent, Exclusive), Pass::Function(append)], |entry| {
        entry.open::<(ByMut<Pair>, ByCallback<(LentMut<Pair>,), (), ()>), ()>(|cx, (mut lent, callback), ()| {
            callback.call(&cx, (&mut *lent,));
            cx.finish(())
        })
    });

    assert_eq!(read::<_, Pair>(&mut frame, lent).name, "lent and written");
}

fn callback_lent_text<R: CallsBack<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let count = frame.function(dispatch::<(ByStr,), _>(words_in::<R>));
    let text = produce(&mut frame, String::from("three short words and more"));

    let called = frame.call(&[Pass::Lend(text, Shared), Pass::Function(count)], |entry| {
        entry.open::<(ByStr, ByCallback<(Lent<str>,), i64, ()>), ()>(|cx, (text, callback), ()| {
            let words = callback.call(&cx, (&text[..17],));
            cx.finish(words)
        })
    });

    assert_eq!(read::<_, i64>(&mut frame, called), 3);
}

/// The length of a `String` taken by value; the `Tally` held beside it is
/// dropped unread.
fn length_and_drop<R: Runtime>(_: &Returning<'_, R>, (text, tally): (String, Uniform<R, Tally>)) -> i64 {
    drop(tally);
    text.len() as i64
}

fn callback_awaited<R: CallsBack<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let consume = frame.function(dispatch::<(ByValue<String>, ByUniform<Tally>), _>(length_and_drop::<R>));

    let called = executor().block_on(frame.call_awaited(&[Pass::Function(consume)], async |entry| {
        let body = entry.open_awaited::<(ByCallback<(String, Tally), i64, ()>,), (), _>(|cx, (callback,), ()| async move {
            cx.sleep(Duration::from_millis(1)).await;
            let length = callback.call_await(&cx, (String::from("moved"), value)).await;
            cx.finish(length)
        });
        sent(body).await
    }));

    assert_eq!(read::<_, i64>(&mut frame, called), 5);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

/// The length of a `String` held by value, taken through the body's context.
fn held_length<R: Runtime>(cx: &Returning<'_, R>, (held,): (Uniform<R, String>,)) -> i64 {
    held.into_inner(cx).len() as i64
}

fn function_with_context<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let function = frame.function(dispatch::<(ByUniform<String>,), _>(held_length::<R>));
    let text = produce(&mut frame, String::from("held"));

    let called = frame.call_function(function, &[Pass::Give(text)]);

    assert_eq!(read::<_, i64>(&mut frame, called), 4);
}

fn aggregates_round_trip<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let option = produce(&mut frame, Some((String::from("inside"), -3i64, Some('x'))));
    let absent = produce(&mut frame, None::<u8>);
    let ok = produce(&mut frame, Ok::<(bool, u32), String>((true, 7)));
    let err = produce(&mut frame, Err::<(bool, u32), String>(String::from("failed")));

    assert_eq!(
        read::<_, Option<(String, i64, Option<char>)>>(&mut frame, option),
        Some((String::from("inside"), -3, Some('x')))
    );
    assert_eq!(read::<_, Option<u8>>(&mut frame, absent), None);
    assert_eq!(read::<_, Result<(bool, u32), String>>(&mut frame, ok), Ok((true, 7)));
    assert_eq!(read::<_, Result<(bool, u32), String>>(&mut frame, err), Err(String::from("failed")));
}

fn aggregate_released_once<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    produce(&mut frame, Ok::<_, i64>(Some((1i64, value))));
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn uniform_inside_an_option<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, Some(value));

    let mut taken = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByValue<Option<Uniform<R, Tally>>>,), ()>(|cx, (held,), ()| {
            taken = held.map(|held| held.into_inner(&cx));
            cx.finish(())
        })
    });
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(taken);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

// Breaches the adapter's tracking covers. Under `WordRuntime` alone each is
// undefined behaviour, so they run on the adapter only.

#[test]
#[ignore = "a reference argument is read through target, which the adapter does not yet validate or track: its detection of argument lends waits for call-scoped tracking"]
#[should_panic(expected = "overlaps a live Exclusive lend")]
fn the_adapter_refuses_a_shared_lend_over_a_live_exclusive_one() {
    let runtime = Adapter::new(WordRuntime::default());
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("twice"));

    frame.call(&[Pass::Lend(variable, Exclusive), Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByMut<Pair>, ByRef<Pair>), ()>(|cx, _, ()| cx.finish(()))
    });
}
