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
use acvus_extern::callback::Callable;
use acvus_extern::decl::EffectNth;
use acvus_extern::executor::Executor;
use acvus_extern::done::Done;
use acvus_extern::register::{Declared, ExternFn, Refused, Registers};
use acvus_extern::runtime::{Form, Key, Place, Ref, RefMut, Runtime};
use acvus_extern::{Argument, Entry, SiteShape};
use acvus_utils::QualifiedRef;
use acvus_extern_test::{Frame, Number, Pair, Pass, Tally, WordRuntime};



#[acvus_extern::extern_fn(effect = pure)]
fn add(a: u64, b: u64) -> u64 {
    a + b
}

/// The path `add` is registered at.
fn add_path(runtime: &WordRuntime) -> QualifiedRef {
    <add as Declared<WordRuntime>>::path(runtime.interner())
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
fn produce<T: for<'l> Place<'l, WordRuntime>>(frame: &mut Frame<'_>, value: T) -> usize {
    frame.call([], |entry| entry.open(|cx, [], []| cx.finish(value)))
}

/// The values of the array a pass lends, read by a call that takes a slice.
fn read_array(frame: &mut Frame<'_>, pass: Pass) -> Vec<i64> {
    let mut read = None;
    frame.call([pass], |entry| {
        entry.open(|cx, [__given0], []| { let run = __given0.open::<&[Uniform<WordRuntime, i64>]>(); 
            read = Some(run.iter().map(|element| element.get(&cx)).collect());
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// A variable's value, given to a call that takes it by value.
fn read<T: for<'l> Place<'l, WordRuntime>>(frame: &mut Frame<'_>, variable: usize) -> T {
    let mut read = None;
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let value = __given0.open::<T>(); 
            read = Some(value);
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// The name of the `Pair` a variable holds, or a reference to one names.
fn read_name(frame: &mut Frame<'_>, pass: Pass) -> String {
    let mut read = None;
    frame.call([pass], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&Pair>(); 
            read = Some(lent.name.clone());
            cx.finish(())
        })
    });
    read.expect("the call ran")
}

/// What a variable holding a `String` or text reads as, lent as `&str`.
fn read_text(frame: &mut Frame<'_>, pass: Pass) -> String {
    let mut read = None;
    frame.call([pass], |entry| {
        entry.open(|cx, [__given0], []| { let text = __given0.open::<&str>(); 
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

    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let taken = __given0.open::<Tally>(); 
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

    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let held = __given0.open::<Uniform<WordRuntime, Tally>>(); 
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
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let held = __given0.open::<Uniform<WordRuntime, Tally>>(); 
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
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let mut held = __given0.open::<Uniform<WordRuntime, Pair>>(); 
            let before = held.open(&cx).name.len();
            held.open_mut(&cx).name.push_str(" and written");
            read = Some((before, held.into_inner(&cx).name));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((4, String::from("held and written"))));
}

#[test]
fn a_rust_value_held_at_its_type_leaves_at_it() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);

    let held = frame.call([], |entry| {
        entry.open(|cx, [], []| {
            let held = Uniform::new(&cx, String::from("held"));
            cx.finish(Some(held))
        })
    });

    assert_eq!(read::<Option<String>>(&mut frame, held), Some(String::from("held")));
}

#[test]
fn a_reference_to_a_variable_opens_as_the_value_it_holds() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, String::from("held"));

    let mut read = None;
    frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let held = __given0.open::<&Uniform<WordRuntime, String>>(); 
            read = Some(held.open(&cx).clone());
            cx.finish(())
        })
    });

    assert_eq!(read.as_deref(), Some("held"));
}

#[test]
fn a_rust_value_held_and_dropped_is_released_once() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);

    frame.call([], |entry| {
        entry.open(|cx, [], []| {
            drop(Uniform::new(&cx, value));
            cx.finish(())
        })
    });

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_held_primitive_is_read_and_written_by_a_copy() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, 5i64);

    let mut read = None;
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let mut held = __given0.open::<Uniform<WordRuntime, i64>>(); 
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

    assert_eq!(read_name(&mut frame, Pass::Lend(variable)), "lent");
    assert_eq!(read::<Pair>(&mut frame, variable).name, "lent");
}

#[test]
fn an_exclusive_lend_writes_the_variable_in_place() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("lent"));

    frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&mut Pair>(); 
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

    assert_eq!(read_text(&mut frame, Pass::Lend(variable)), "text");
}

#[test]
fn a_borrowed_result_names_the_whole_lent_value() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("whole"));

    let borrowed = frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&Pair>(); cx.finish(&*lent) })
    });

    assert_eq!(read_name(&mut frame, Pass::Same(borrowed)), "whole");
}

#[test]
fn a_borrowed_result_names_a_range_of_lent_text() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, String::from("a range"));

    let borrowed = frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let text = __given0.open::<&str>(); cx.finish(&text[2..5]) })
    });

    assert_eq!(read_text(&mut frame, Pass::Same(borrowed)), "ran");
}

#[test]
fn a_borrowed_result_names_a_strings_text() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("owned text"));

    let borrowed = frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&Pair>(); cx.finish(lent.name.as_str()) })
    });

    assert_eq!(read_text(&mut frame, Pass::Same(borrowed)), "owned text");
}

fn pair(name: &str) -> Pair {
    Pair { name: String::from(name), lead: Number(2), items: vec![Number(3), Number(5), Number(8)] }
}

/// The `Number` a reference value names.
fn read_number(frame: &mut Frame<'_>, pass: Pass) -> Number {
    let mut read = None;
    frame.call([pass], |entry| {
        entry.open(|cx, [__given0], []| { let number = __given0.open::<&Number>(); 
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

    let borrowed = frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&Pair>(); cx.finish(&lent.lead) })
    });

    assert_eq!(read_number(&mut frame, Pass::Same(borrowed)), Number(2));
}

#[test]
fn a_borrowed_result_names_an_element_the_lent_value_owns() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("element"));

    let borrowed = frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&Pair>(); cx.finish(&lent.items[1]) })
    });

    assert_eq!(read_number(&mut frame, Pass::Same(borrowed)), Number(5));
}

#[test]
fn a_writable_borrowed_result_writes_the_lent_value() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, pair("borrowed"));

    let borrowed = frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&mut Pair>(); cx.finish(&mut *lent) })
    });
    frame.call([Pass::Same(borrowed)], |entry| {
        entry.open(|cx, [__given0], []| { let target = __given0.open::<&mut Pair>(); 
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
    frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let run = __given0.open::<&[Uniform<WordRuntime, i64>]>(); 
            sum = Some(run.iter().map(|element| element.get(&cx)).sum::<i64>());
            cx.finish(())
        })
    });
    frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let run = __given0.open::<&mut [Uniform<WordRuntime, i64>]>(); 
            for element in run.iter_mut() {
                let doubled = element.get(&cx) * 2;
                element.set(&cx, doubled);
            }
            cx.finish(())
        })
    });

    assert_eq!(sum, Some(16));
    assert_eq!(read_array(&mut frame, Pass::Lend(variable)), [6, 10, 16]);
}

#[test]
fn a_borrowed_result_names_a_run_of_a_lent_slice() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let variable = frame.array(&[3i64, 5, 8]);

    let borrowed = frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let run = __given0.open::<&[Uniform<WordRuntime, i64>]>(); cx.finish(&run[1..]) })
    });

    assert_eq!(read_array(&mut frame, Pass::Same(borrowed)), [5, 8]);
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

    let length = executor().block_on(frame.call_awaited([Pass::Give(text)], async |entry| {
        let body = entry.open_awaited(|cx, [__given0], []| { let text = __given0.open::<String>(); async move {
            cx.sleep(Duration::from_millis(1)).await;
            cx.finish(text.len() as i64)
        } });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, length), 7);
}

#[test]
fn an_awaited_body_reads_its_lend_before_a_wait() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let text = produce(&mut frame, pair("held across"));

    let length = executor().block_on(frame.call_awaited([Pass::Lend(text)], async |entry| {
        let body = entry.open_awaited(|cx, [__given0], []| { let lent = __given0.open::<&Pair>(); {
            let length = lent.name.len() as i64;
            async move {
                cx.sleep(Duration::from_millis(1)).await;
                cx.finish(length)
            }
        } });
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

    let length = executor().block_on(frame.call_awaited([Pass::Lend(text)], async |entry| {
        let body = entry.open_awaited(|cx, [__given0], []| { let lent = __given0.open::<&Pair>(); async move {
            cx.sleep(Duration::from_millis(1)).await;
            let length = lent.name.len() as i64;
            cx.finish(length)
        } });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, length), 18);
    assert_eq!(read::<Pair>(&mut frame, text).name, "held across a wait");
}

/// A function value's dispatch, written as `#[extern_fn]` writes one: the
/// entry over the argument run, each argument opened at the shape keys `A`, the body
/// `f`, and its result given.
struct Dispatch<A, F> {
    f: F,
    _keys: PhantomData<fn() -> A>,
}

fn dispatch<A, F>(f: F) -> Dispatch<A, F> {
    Dispatch { f, _keys: PhantomData }
}

// SAFETY: this stands in for `#[extern_fn]`: `call` opens the arguments by
// the shape keys `A` the body `f` takes them at.
#[allow(unsafe_code)]
unsafe impl<A, F> acvus_extern::seal::ExternFns for Dispatch<A, F> {}

impl<R, A, U, F, const N: usize> ExternFn<R, N, 0> for Dispatch<A, F>
where
    R: Runtime,
    A: Keys<R, N> + 'static,
    U: for<'c> Form<'c, R>,
    F: for<'c> Fn(&Executor<'c, R>, A::Opened<'c>) -> U + Send + Sync + 'static,
{
    fn call(&self, runtime: &R, args: [R::Value; N], site: &SiteShape<R, 0>) -> Done<R> {
        Entry::new(runtime, args, site).open(|cx, args, []| {
            let result = (self.f)(&cx, A::open(args));
            cx.finish(result)
        })
    }
}

/// The shape keys of a dispatch's arguments, a tuple of one or two, each
/// argument opened at its own shape's form.
trait Keys<R: Runtime, const N: usize> {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    fn open<'c>(args: [Argument<'c, R>; N]) -> Self::Opened<'c>
    where
        Self: 'c;
}

impl<R, P> Keys<R, 1> for (P,)
where
    R: Runtime,
    P: for<'c> Key<'c, R>,
{
    type Opened<'c>
        = (<P as Key<'c, R>>::At,)
    where
        Self: 'c,
        R: 'c;

    fn open<'c>([p]: [Argument<'c, R>; 1]) -> Self::Opened<'c>
    where
        Self: 'c,
    {
        (p.open::<<P as Key<'c, R>>::At>(),)
    }
}

impl<R, P, Q> Keys<R, 2> for (P, Q)
where
    R: Runtime,
    P: for<'c> Key<'c, R>,
    Q: for<'c> Key<'c, R>,
{
    type Opened<'c>
        = (<P as Key<'c, R>>::At, <Q as Key<'c, R>>::At)
    where
        Self: 'c,
        R: 'c;

    fn open<'c>([p, q]: [Argument<'c, R>; 2]) -> Self::Opened<'c>
    where
        Self: 'c,
    {
        (p.open::<<P as Key<'c, R>>::At>(), q.open::<<Q as Key<'c, R>>::At>())
    }
}

/// The length of text lent to it.
fn length_of<R: Runtime>(_: &Executor<'_, R>, (text,): (&str,)) -> i64 {
    text.len() as i64
}

#[test]
fn a_function_value_is_called_through_its_dispatch() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(Ref<str>,), _>(length_of::<WordRuntime>));
    let short = produce(&mut frame, String::from("four"));
    let long = produce(&mut frame, String::from("eleven long"));

    let first = frame.call_function(function, [Pass::Lend(short)]);
    let second = frame.call_function(function, [Pass::Lend(long)]);

    assert_eq!(read::<i64>(&mut frame, first), 4);
    assert_eq!(read::<i64>(&mut frame, second), 11);
    assert_eq!(read::<String>(&mut frame, short), "four");
}

#[test]
fn a_function_value_releases_what_it_captures_once() {
    let runtime = WordRuntime::default();
    let (captured, drops) = tally();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(i64,), _>(move |_: &Executor<'_, WordRuntime>, (n,): (i64,)| {
        n + Arc::strong_count(&captured.0) as i64
    }));

    let forty = produce(&mut frame, 40i64);
    let called = frame.call_function(function, [Pass::Give(forty)]);
    assert_eq!(read::<i64>(&mut frame, called), 42);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_callback_is_lent_a_part_of_the_handlers_storage() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(Ref<str>,), _>(length_of::<WordRuntime>));
    let text = produce(&mut frame, String::from("lent to a callback"));

    let called = executor().block_on(frame.call_awaited([Pass::Lend(text), Pass::Lend(function)], async |entry| {
        let body = entry.open_awaited(|cx, [__given0, __given1], []| { let text = __given0.open::<&str>(); let callback = __given1.open::<&Uniform<WordRuntime, Callable<'_, (Ref<str>,), i64, EffectNth<0>>>>(); async move {
                let first = callback.call(&cx, (&*text,));
                let second = callback.call(&cx, (&*text,));
                cx.finish(first + second)
            } });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, called), 36);
    assert_eq!(read::<String>(&mut frame, text), "lent to a callback");
}

/// Appends to the name of a `Pair` lent exclusively, in the view its crossing
/// opens.
fn append_written<'c, R: Runtime>(_: &Executor<'c, R>, (lent,): (&'c mut Pair,)) {
    lent.name.push_str(" and written");
}

fn words_in<R: Runtime>(_: &Executor<'_, R>, (text,): (&str,)) -> i64 {
    text.split(' ').count() as i64
}

#[test]
fn a_callback_writes_through_a_part_lent_exclusively() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let append = frame.function(dispatch::<(RefMut<Pair>,), _>(append_written::<WordRuntime>));
    let lent = produce(&mut frame, pair("lent"));

    executor().block_on(frame.call_awaited([Pass::Lend(lent), Pass::Lend(append)], async |entry| {
        let body = entry.open_awaited(|cx, [__given0, __given1], []| { let lent = __given0.open::<&mut Pair>(); let callback = __given1.open::<&Uniform<WordRuntime, Callable<'_, (RefMut<Pair>,), (), EffectNth<0>>>>(); async move {
                callback.call(&cx, (&mut *lent,));
                cx.finish(())
            } });
        sent(body).await
    }));

    assert_eq!(read::<Pair>(&mut frame, lent).name, "lent and written");
}

#[test]
fn a_callback_is_lent_text() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let count = frame.function(dispatch::<(Ref<str>,), _>(words_in::<WordRuntime>));
    let text = produce(&mut frame, String::from("three short words and more"));

    let called = executor().block_on(frame.call_awaited([Pass::Lend(text), Pass::Lend(count)], async |entry| {
        let body = entry.open_awaited(|cx, [__given0, __given1], []| { let text = __given0.open::<&str>(); let callback = __given1.open::<&Uniform<WordRuntime, Callable<'_, (Ref<str>,), i64, EffectNth<0>>>>(); async move {
                let words = callback.call(&cx, (&text[..17],));
                cx.finish(words)
            } });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, called), 3);
}

/// The length of a `String` taken by value; the `Tally` held beside it is
/// dropped unread.
fn length_and_drop<R: Runtime>(_: &Executor<'_, R>, (text, tally): (String, Uniform<R, Tally>)) -> i64 {
    drop(tally);
    text.len() as i64
}

#[test]
fn an_awaited_callback_takes_its_arguments_by_move() {
    let runtime = WordRuntime::default();
    let (value, drops) = tally();
    let mut frame = Frame::new(&runtime);
    let consume = frame.function(dispatch::<(String, Uniform<WordRuntime, Tally>), _>(length_and_drop::<WordRuntime>));

    let called = executor().block_on(frame.call_awaited([Pass::Give(consume)], async |entry| {
        let body = entry.open_awaited(|cx, [__given0], []| { let callback = __given0.open::<Uniform<WordRuntime, Callable<'_, (String, Tally), i64, EffectNth<0>>>>(); async move {
            cx.sleep(Duration::from_millis(1)).await;
            let length = callback.call_await(&cx, (String::from("moved"), value)).await;
            cx.finish(length)
        } });
        sent(body).await
    }));

    assert_eq!(read::<i64>(&mut frame, called), 5);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

/// The length of a `String` held by value, taken through the body's executor.
fn held_length<R: Runtime>(cx: &Executor<'_, R>, (held,): (Uniform<R, String>,)) -> i64 {
    held.into_inner(cx).len() as i64
}

#[test]
fn a_function_body_reads_its_held_argument_through_its_executor() {
    let runtime = WordRuntime::default();
    let mut frame = Frame::new(&runtime);
    let function = frame.function(dispatch::<(Uniform<WordRuntime, String>,), _>(held_length::<WordRuntime>));
    let text = produce(&mut frame, String::from("held"));

    let called = frame.call_function(function, [Pass::Give(text)]);

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
    frame.call([Pass::Give(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let held = __given0.open::<Option<Uniform<WordRuntime, Tally>>>(); 
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

    frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&mut String>(); 
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
    frame.call([Pass::Lend(variable)], |entry| {
        entry.open(|cx, [__given0], []| { let lent = __given0.open::<&mut i16>(); 
            before = Some(*lent);
            *lent = -300;
            cx.finish(())
        })
    });

    assert_eq!(before, Some(5));
    assert_eq!(read::<i16>(&mut frame, variable), -300);
}
