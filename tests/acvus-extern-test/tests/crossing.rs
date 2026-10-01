//! The crossing's tests, each run on `WordRuntime`.
//!
//! A `Frame` stands in for the run: it passes its variables through
//! `Entry::new`, as the runtime's boundary does, and the glue each test writes
//! by hand opens them with `Entry::open`, as the attribute's glue will.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use acvus_extern::Uniform;
use acvus_extern::callback::{Lent, LentMut};
use acvus_extern::context::Returning;
use acvus_extern::guard::{Guard, GuardMut};
use acvus_extern::mode::{ByCallback, ByMut, ByRef, BySlice, BySliceMut, ByStr, ByUniform, ByValue};
use acvus_extern::done::Done;
use acvus_extern::register::{Declared, ExternFn, Refused, Registers};
use acvus_extern::runtime::{Crosses, Mutability, Runtime};
use acvus_extern::{Entry, SiteShape};
use acvus_utils::QualifiedRef;
use acvus_extern_test::{Frame, Number, Pair, Pass, Tally, WordRuntime};

use Mutability::{Exclusive, Shared};



#[acvus_extern::extern_fn(effect = pure)]
fn add(a: u64, b: u64) -> u64 {
    a + b
}

/// The path `add` is registered at.
fn add_path(runtime: &WordRuntime) -> QualifiedRef {
    Declared::<WordRuntime>::path(&add::new(), runtime.interner())
}

#[test]
fn a_path_registered_twice_is_refused() {
    let mut runtime = WordRuntime::default();
    runtime.register(add::new()).expect("the path is free");
    let path = add_path(&runtime);

    assert_eq!(runtime.register(add::new()), Err(Refused::PathTaken(path)));
}


/// An awaited function declared at the path `add` names.
#[acvus_extern::extern_fn(name = "add")]
async fn add_awaited() {}

#[test]
fn a_path_registered_returning_and_awaited_is_refused() {
    let mut runtime = WordRuntime::default();
    runtime.register(add::new()).expect("the path is free");
    let path = add_path(&runtime);

    assert_eq!(runtime.register_awaited(add_awaited::new()), Err(Refused::PathTaken(path)));
}

/// A variable holding `value`, made as a program makes one: the result of a
/// call.
fn produce<T: Crosses<WordRuntime>>(frame: &mut Frame<'_>, value: T) -> usize {
    frame.call(&[], |entry| entry.open::<(), ()>(|cx, (), ()| cx.finish(value)))
}

/// The values of the array a pass lends, read by a call that takes a slice.
fn read_array(frame: &mut Frame<'_>, pass: Pass) -> Vec<i64> {
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
fn read<T: Crosses<WordRuntime>>(frame: &mut Frame<'_>, variable: usize) -> T {
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
fn read_name(frame: &mut Frame<'_>, pass: Pass) -> String {
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
fn read_text(frame: &mut Frame<'_>, pass: Pass) -> String {
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

#[test]
fn a_value_given_by_value_is_taken_at_its_type() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, String::from("moved"));
    let number = produce(&mut frame, -41i64);

    assert_eq!(read::<String>(&mut frame, text), "moved");
    assert_eq!(read::<i64>(&mut frame, number), -41);
}

#[test]
fn a_value_taken_at_once_is_released_by_the_handler_once() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
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

#[test]
fn a_held_value_only_dropped_is_released_once() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
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

#[test]
fn a_held_value_taken_is_released_by_its_new_owner_once() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
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

#[test]
fn a_held_value_is_read_and_written_in_place() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
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

#[test]
fn a_held_primitive_is_read_and_written_by_a_copy() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
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

#[test]
fn a_value_never_passed_is_released_by_the_run_once() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
    produce(&mut frame, value);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_shared_lend_reads_the_variable_in_place() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("lent"));

    assert_eq!(read_name(&mut frame, Pass::Lend(variable, Shared)), "lent");
    assert_eq!(read::<Pair>(&mut frame, variable).name, "lent");
}

#[test]
fn an_exclusive_lend_writes_the_variable_in_place() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("lent"));

    frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<Pair>,), ()>(|cx, (mut lent,), ()| {
            lent.name.push_str(" and written");
            cx.finish(())
        })
    });

    assert_eq!(read::<Pair>(&mut frame, variable).name, "lent and written");
}

#[test]
fn a_string_is_lent_as_text() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, String::from("text"));

    assert_eq!(read_text(&mut frame, Pass::Lend(variable, Shared)), "text");
}

#[test]
fn a_borrowed_result_names_the_whole_lent_value() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("whole"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_borrowed(lent, |p: &Pair, sink| sink.lent(p)))
    });

    assert_eq!(read_name(&mut frame, Pass::Same(borrowed, Shared)), "whole");
}

#[test]
fn a_borrowed_result_names_a_range_of_lent_text() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, String::from("a range"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByStr,), ()>(|cx, (text,), ()| cx.finish_borrowed(text, |s: &str, sink| sink.lent(&s[2..5])))
    });

    assert_eq!(read_text(&mut frame, Pass::Same(borrowed, Shared)), "ran");
}

#[test]
fn a_borrowed_result_names_a_strings_text() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("owned text"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_borrowed(lent, |p: &Pair, sink| sink.lent(p.name.as_str())))
    });

    assert_eq!(read_text(&mut frame, Pass::Same(borrowed, Shared)), "owned text");
}

fn pair(name: &str) -> Pair {
    Pair { name: String::from(name), lead: Number(2), items: vec![Number(3), Number(5), Number(8)] }
}

/// The `Number` a reference value names.
fn read_number(frame: &mut Frame<'_>, pass: Pass) -> Number {
    let mut read = None;
    frame.call(&[pass], |entry| {
        entry.open::<(ByRef<Number>,), ()>(|cx, (number,), ()| {
            read = Some(*number);
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

#[test]
fn a_borrowed_result_names_a_field() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("field"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_borrowed(lent, |p: &Pair, sink| sink.lent(&p.lead)))
    });

    assert_eq!(read_number(&mut frame, Pass::Same(borrowed, Shared)), Number(2));
}

#[test]
fn a_borrowed_result_names_an_element_the_lent_value_owns() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("element"));

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(ByRef<Pair>,), ()>(|cx, (lent,), ()| cx.finish_borrowed(lent, |p: &Pair, sink| sink.lent(&p.items[1])))
    });

    assert_eq!(read_number(&mut frame, Pass::Same(borrowed, Shared)), Number(5));
}

#[test]
fn a_writable_borrowed_result_writes_the_lent_value() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("borrowed"));

    let borrowed = frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<Pair>,), ()>(|cx, (lent,), ()| cx.finish_borrowed_mut(lent, |p: &mut Pair, sink| sink.lent_mut(p)))
    });
    frame.call(&[Pass::Same(borrowed, Exclusive)], |entry| {
        entry.open::<(ByMut<Pair>,), ()>(|cx, (mut target,), ()| {
            target.name.push_str(" and written");
            cx.finish(())
        })
    });

    assert_eq!(read::<Pair>(&mut frame, variable).name, "borrowed and written");
}

#[test]
fn an_array_is_lent_as_a_slice() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
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

#[test]
fn a_borrowed_result_names_a_run_of_a_lent_slice() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = frame.array(&[3i64, 5, 8]);

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| cx.finish_borrowed(run, |s: &[Uniform<WordRuntime, i64>], sink| sink.lent(&s[1..])))
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

#[test]
fn an_awaited_body_takes_its_argument_and_finishes_after_a_wait() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, String::from("awaited"));

    let length = executor().block_on(frame.call_awaited(&[Pass::Give(text)], async |entry| {
        let body = entry.open_awaited::<(ByValue<String>,), (), _>(|cx, (text,), ()| async move {
            cx.sleep(Duration::from_millis(1)).await;
            cx.finish(text.len() as i64)
        });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, length), 7);
}

#[test]
fn an_awaited_body_reads_its_lend_before_a_wait() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
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

    assert_eq!(read::<i64>(&mut frame, length), 11);
    assert_eq!(read::<Pair>(&mut frame, text).name, "held across");
}

#[test]
fn an_awaited_body_holds_a_guard_across_a_wait() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, pair("held across a wait"));

    let length = executor().block_on(frame.call_awaited(&[Pass::Lend(text, Shared)], async |entry| {
        let body = entry.open_awaited::<(ByRef<Pair>,), (), _>(|cx, (lent,), ()| async move {
            cx.sleep(Duration::from_millis(1)).await;
            let length = lent.name.len() as i64;
            cx.finish(length)
        });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, length), 18);
    assert_eq!(read::<Pair>(&mut frame, text).name, "held across a wait");
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

#[test]
fn a_function_value_is_called_through_its_dispatch() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(ByStr,), _>(length_of::<WordRuntime>));
    let short = produce(&mut frame, String::from("four"));
    let long = produce(&mut frame, String::from("eleven long"));

    let first = frame.call_function(function, &[Pass::Lend(short, Shared)]);
    let second = frame.call_function(function, &[Pass::Lend(long, Shared)]);

    assert_eq!(read::<i64>(&mut frame, first), 4);
    assert_eq!(read::<i64>(&mut frame, second), 11);
    assert_eq!(read::<String>(&mut frame, short), "four");
}

#[test]
fn a_function_value_releases_what_it_captures_once() {
    let runtime = WordRuntime::default();
    let (captured, drops) = tally();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(ByValue<i64>,), _>(move |_: &Returning<'_, WordRuntime>, (n,): (i64,)| {
        n + Arc::strong_count(&captured.0) as i64
    }));

    let forty = produce(&mut frame, 40i64);
    let called = frame.call_function(function, &[Pass::Give(forty)]);
    assert_eq!(read::<i64>(&mut frame, called), 42);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_callback_is_lent_a_part_of_the_handlers_storage() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(ByStr,), _>(length_of::<WordRuntime>));
    let text = produce(&mut frame, String::from("lent to a callback"));

    let called = frame.call(&[Pass::Lend(text, Shared), Pass::Function(function)], |entry| {
        entry.open::<(ByStr, ByCallback<(Lent<str>,), i64, ()>), ()>(|cx, (text, callback), ()| {
            let first = callback.call(&cx, (&*text,));
            let second = callback.call(&cx, (&*text,));
            cx.finish(first + second)
        })
    });

    assert_eq!(read::<i64>(&mut frame, called), 36);
    assert_eq!(read::<String>(&mut frame, text), "lent to a callback");
}

/// Appends to the name of a `Pair` lent exclusively, in the view its crossing
/// opens.
fn append_written<'c, R: Runtime>(_: &Returning<'c, R>, (mut lent,): (GuardMut<'c, <Pair as Crosses<R>>::OpenMut<'c>>,)) {
    lent.name.push_str(" and written");
}

fn words_in<R: Runtime>(_: &Returning<'_, R>, (text,): (Guard<'_, &str>,)) -> i64 {
    text.split(' ').count() as i64
}

#[test]
fn a_callback_writes_through_a_part_lent_exclusively() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let append = frame.function(dispatch::<(ByMut<Pair>,), _>(append_written::<WordRuntime>));
    let lent = produce(&mut frame, pair("lent"));

    frame.call(&[Pass::Lend(lent, Exclusive), Pass::Function(append)], |entry| {
        entry.open::<(ByMut<Pair>, ByCallback<(LentMut<Pair>,), (), ()>), ()>(|cx, (mut lent, callback), ()| {
            callback.call(&cx, (&mut *lent,));
            cx.finish(())
        })
    });

    assert_eq!(read::<Pair>(&mut frame, lent).name, "lent and written");
}

#[test]
fn a_callback_is_lent_text() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let count = frame.function(dispatch::<(ByStr,), _>(words_in::<WordRuntime>));
    let text = produce(&mut frame, String::from("three short words and more"));

    let called = frame.call(&[Pass::Lend(text, Shared), Pass::Function(count)], |entry| {
        entry.open::<(ByStr, ByCallback<(Lent<str>,), i64, ()>), ()>(|cx, (text, callback), ()| {
            let words = callback.call(&cx, (&text[..17],));
            cx.finish(words)
        })
    });

    assert_eq!(read::<i64>(&mut frame, called), 3);
}

/// The length of a `String` taken by value; the `Tally` held beside it is
/// dropped unread.
fn length_and_drop<R: Runtime>(_: &Returning<'_, R>, (text, tally): (String, Uniform<R, Tally>)) -> i64 {
    drop(tally);
    text.len() as i64
}

#[test]
fn an_awaited_callback_takes_its_arguments_by_move() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
    let consume = frame.function(dispatch::<(ByValue<String>, ByUniform<Tally>), _>(length_and_drop::<WordRuntime>));

    let called = executor().block_on(frame.call_awaited(&[Pass::Function(consume)], async |entry| {
        let body = entry.open_awaited::<(ByCallback<(String, Tally), i64, ()>,), (), _>(|cx, (callback,), ()| async move {
            cx.sleep(Duration::from_millis(1)).await;
            let length = callback.call_await(&cx, (String::from("moved"), value)).await;
            cx.finish(length)
        });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, called), 5);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

/// The length of a `String` held by value, taken through the body's context.
fn held_length<R: Runtime>(cx: &Returning<'_, R>, (held,): (Uniform<R, String>,)) -> i64 {
    held.into_inner(cx).len() as i64
}

#[test]
fn a_function_body_reads_its_held_argument_through_its_context() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(ByUniform<String>,), _>(held_length::<WordRuntime>));
    let text = produce(&mut frame, String::from("held"));

    let called = frame.call_function(function, &[Pass::Give(text)]);

    assert_eq!(read::<i64>(&mut frame, called), 4);
}

#[test]
fn an_option_a_result_and_a_tuple_cross_by_their_parts() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let option = produce(&mut frame, Some((String::from("inside"), -3i64, Some('x'))));
    let absent = produce(&mut frame, None::<u8>);
    let ok = produce(&mut frame, Ok::<(bool, u32), String>((true, 7)));
    let err = produce(&mut frame, Err::<(bool, u32), String>(String::from("failed")));

    assert_eq!(
        read::<Option<(String, i64, Option<char>)>>(&mut frame, option),
        Some((String::from("inside"), -3, Some('x')))
    );
    assert_eq!(read::<Option<u8>>(&mut frame, absent), None);
    assert_eq!(read::<Result<(bool, u32), String>>(&mut frame, ok), Ok((true, 7)));
    assert_eq!(read::<Result<(bool, u32), String>>(&mut frame, err), Err(String::from("failed")));
}

#[test]
fn an_aggregate_never_taken_releases_its_parts_once() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
    produce(&mut frame, Ok::<_, i64>(Some((1i64, value))));
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_type_variables_value_crosses_inside_an_option() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, Some(value));

    let mut taken = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByValue<Option<Uniform<WordRuntime, Tally>>>,), ()>(|cx, (held,), ()| {
            taken = held.map(|held| held.into_inner(&cx));
            cx.finish(())
        })
    });
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(taken);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_string_lent_exclusively_is_written_in_place() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, String::from("lent"));

    frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(ByMut<String>,), ()>(|cx, (mut lent,), ()| {
            lent.push_str(" and written");
            cx.finish(())
        })
    });

    assert_eq!(read::<String>(&mut frame, variable), "lent and written");
}

#[test]
fn a_primitive_lent_exclusively_is_written_in_place() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
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
    assert_eq!(read::<i16>(&mut frame, variable), -300);
}

/// The name of the pair an option lent holds, where it holds one: the option
/// taken as its projection.
#[acvus_extern::extern_fn(effect = pure)]
fn name_if(o: Option<&Pair>) -> Option<String> {
    o.map(|pair| pair.name.clone())
}

#[test]
fn a_lent_option_is_viewed_as_its_part() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let present = produce(&mut frame, Some(pair("inside")));
    let absent = produce(&mut frame, None::<Pair>);
    let function = frame.function(name_if::new());

    let named = frame.call_function(function, &[Pass::Lend(present, Shared)]);
    let unnamed = frame.call_function(function, &[Pass::Lend(absent, Shared)]);

    assert_eq!(read::<Option<String>>(&mut frame, named), Some(String::from("inside")));
    assert_eq!(read::<Option<String>>(&mut frame, unnamed), None);
}

#[test]
fn a_held_tuple_is_opened_as_its_parts() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, (4i64, String::from("held")));

    let mut read = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByUniform<(i64, String)>,), ()>(|cx, (mut held,), ()| {
            {
                let (number, text) = held.open_mut(&cx);
                *number *= 2;
                text.push_str(" and written");
            }
            read = Some(held.into_inner(&cx));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((8, String::from("held and written"))));
}
