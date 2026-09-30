//! The crossing's tests, each run on `WordRuntime` and on
//! `Adapter<WordRuntime>` with the same expected result (structure B-adapter).
//!
//! A `Frame` stands in for the run: it passes its variables through
//! `Entry::new`, as the runtime's boundary does, and the glue each test writes
//! by hand opens them with `Entry::open`, as the attribute's glue will.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use acvus_extern::adapter::Adapter;
use acvus_extern::callback::{CallsBack, Lent, LentMut, RustFn, RustFns};
use acvus_extern::guard::{Guard, GuardMut};
use acvus_extern::mode::{ByCallback, ByMut, ByProjection, ByRef, BySlice, BySliceMut, ByStr, ByUniform, ByValue};
use acvus_extern::runtime::{Crosses, Mutability, Runtime};
use acvus_extern_test::shapes::{Labelled, Maybe, MaybeOwned, MaybeRef, Nested, Record};
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
    a_held_scalar_is_read_and_written_in_its_word: held_scalar_in_place;
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
    a_returned_vec_is_lent_as_a_slice: lent_as_slice;
    a_borrowed_result_names_an_element_of_a_lent_slice: borrowed_slice_element;
    a_borrowed_result_names_a_run_of_a_lent_slice: borrowed_sub_slice;
    an_awaited_body_takes_its_argument_and_finishes_after_a_wait: awaited_by_value;
    an_awaited_body_reads_its_lend_before_a_wait: awaited_lent;
    a_projection_result_composes_its_leaves_into_a_tuple: projection_tuple;
    a_projection_result_composes_a_lent_payload_into_a_variant: projection_variant;
    a_projection_result_left_in_pieces_panics_and_releases_them_once: projection_uncomposed;
    a_lent_tuple_is_projected_a_guard_at_each_leaf: projected_tuple;
    a_lent_object_is_projected_in_the_order_of_its_names: projected_object;
    a_shape_nested_in_a_lent_tuple_is_projected_in_place: projected_nested;
    a_lent_variant_is_projected_by_its_tag: projected_variant;
    a_held_tuple_is_taken_apart_into_its_owned_view: viewed_tuple;
    a_held_nested_shape_is_viewed_part_by_part: viewed_nested;
    a_held_variant_is_viewed_by_its_tag_and_its_payload_released_once: viewed_variant;
    an_awaited_body_holds_a_guard_across_a_wait: awaited_guard_held;
    a_returned_rust_fn_is_called_as_a_script_calls_a_function_value: rust_fn_called;
    a_rust_fn_releases_what_it_captures_once: rust_fn_released;
    a_callback_is_lent_a_part_of_the_handlers_storage: callback_lent;
    a_callback_writes_through_a_part_lent_exclusively: callback_lent_mut;
    a_callback_is_lent_text: callback_lent_text;
    an_awaited_callback_takes_its_arguments_by_move: callback_awaited;
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

fn held_scalar_in_place<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, 5i64);

    let mut read = None;
    frame.call(&[Pass::Give(variable)], |entry| {
        entry.open::<(ByUniform<i64>,), ()>(|cx, (mut held,), ()| {
            let before = *held.open(&cx);
            *held.open_mut(&cx) = 9;
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

fn lent_as_slice<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, vec![3i64, 5, 8]);

    let mut sum = None;
    frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| {
            sum = Some(run.iter().sum::<i64>());
            cx.finish(())
        })
    });
    frame.call(&[Pass::Lend(variable, Exclusive)], |entry| {
        entry.open::<(BySliceMut<i64>,), ()>(|cx, (mut run,), ()| {
            run.iter_mut().for_each(|element| *element *= 2);
            cx.finish(())
        })
    });

    assert_eq!(sum, Some(16));
    assert_eq!(read::<_, Vec<i64>>(&mut frame, variable), [6, 10, 16]);
}

fn borrowed_slice_element<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, vec![3i64, 5, 8]);

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| cx.finish_lent(run, |s: &[i64]| &s[1]))
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

fn borrowed_sub_slice<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let variable = produce(&mut frame, vec![3i64, 5, 8]);

    let borrowed = frame.call(&[Pass::Lend(variable, Shared)], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| cx.finish_lent(run, |s: &[i64]| &s[1..]))
    });

    let mut read = None;
    frame.call(&[Pass::Same(borrowed, Shared)], |entry| {
        entry.open::<(BySlice<i64>,), ()>(|cx, (run,), ()| {
            read = Some(run.to_vec());
            cx.finish(())
        })
    });
    assert_eq!(read, Some(vec![5, 8]));
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
    let text = produce(&mut frame, String::from("held across"));

    let length = executor().block_on(frame.call_awaited(&[Pass::Lend(text, Shared)], async |entry| {
        let body = entry.open_awaited::<(ByRef<String>,), (), _>(|cx, (lent,), ()| {
            let length = lent.len() as i64;
            drop(lent);
            async move {
                cx.sleep(Duration::from_millis(1)).await;
                cx.finish(length)
            }
        });
        sent(body).await
    }));

    assert_eq!(read::<_, i64>(&mut frame, length), 11);
    assert_eq!(read::<_, String>(&mut frame, text), "held across");
}

fn projection_tuple<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let text = produce(&mut frame, String::from("leaf"));

    frame.call(&[Pass::Lend(text, Shared)], |entry| {
        entry.open::<(ByRef<String>,), ()>(|cx, (lent,), ()| {
            cx.finish_projection(|result| {
                result.lent(lent, |s: &String| s.as_str());
                result.value(value);
                result.tuple(2);
            })
        })
    });
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn projection_variant<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let text = produce(&mut frame, String::from("payload"));

    frame.call(&[Pass::Lend(text, Shared)], |entry| {
        entry.open::<(ByRef<String>,), ()>(|cx, (lent,), ()| {
            cx.finish_projection(|result| {
                result.lent(lent, |s: &String| s);
                result.variant(1, true);
            })
        })
    });

    assert_eq!(read::<_, String>(&mut frame, text), "payload");
}

fn projection_uncomposed<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);

    let unwound = catch_unwind(AssertUnwindSafe(|| {
        frame.call(&[], |entry| {
            entry.open::<(), ()>(|cx, (), ()| {
                cx.finish_projection(|result| {
                    result.value(value);
                    result.value(1i64);
                })
            })
        })
    }));

    assert!(unwound.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

/// A variable holding the tuple `(label, count)`, as a program makes one.
fn produce_labelled<R>(frame: &mut Frame<'_, R>, label: &str, count: i64) -> usize
where
    R: Runtime<Value = Word>,
{
    let label = String::from(label);
    frame.call(&[], |entry| {
        entry.open::<(), ()>(|cx, (), ()| {
            cx.finish_projection(|result| {
                result.value(label);
                result.value(count);
                result.tuple(2);
            })
        })
    })
}

fn projected_tuple<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let tuple = produce_labelled(&mut frame, "leaves", 2);

    let mut read = None;
    frame.call(&[Pass::Lend(tuple, Shared)], |entry| {
        entry.open::<(ByProjection<Labelled>,), ()>(|cx, (lent,), ()| {
            read = Some((String::from(lent.label.as_str()), *lent.count));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((String::from("leaves"), 2)));
}

fn projected_object<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let object = frame.call(&[], |entry| {
        entry.open::<(), ()>(|cx, (), ()| {
            cx.finish_projection(|result| {
                result.value(7i64);
                result.value(String::from("record"));
                result.object(2);
            })
        })
    });

    let mut read = None;
    frame.call(&[Pass::Lend(object, Shared)], |entry| {
        entry.open::<(ByProjection<Record>,), ()>(|cx, (lent,), ()| {
            read = Some((*lent.count, String::from(lent.name.as_str())));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((7, String::from("record"))));
}

fn projected_nested<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let nested = frame.call(&[], |entry| {
        entry.open::<(), ()>(|cx, (), ()| {
            cx.finish_projection(|result| {
                result.value(String::from("inner"));
                result.value(3i64);
                result.tuple(2);
                result.value(true);
                result.tuple(2);
            })
        })
    });

    let mut read = None;
    frame.call(&[Pass::Lend(nested, Shared)], |entry| {
        entry.open::<(ByProjection<Nested>,), ()>(|cx, (lent,), ()| {
            read = Some((String::from(lent.inner.label.as_str()), *lent.inner.count, *lent.flag));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((String::from("inner"), 3, true)));
}

/// A variable holding `Maybe::Some(value)`, or `Maybe::None`.
fn produce_maybe<R>(frame: &mut Frame<'_, R>, value: Option<Tally>) -> usize
where
    R: Runtime<Value = Word>,
{
    frame.call(&[], |entry| {
        entry.open::<(), ()>(|cx, (), ()| {
            cx.finish_projection(|result| match value {
                Some(value) => {
                    result.value(value);
                    result.variant(1, true);
                }
                None => result.variant(0, false),
            })
        })
    })
}

fn projected_variant<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let some = produce_maybe(&mut frame, Some(value));
    let none = produce_maybe(&mut frame, None);

    let mut read = Vec::new();
    frame.call(&[Pass::Lend(some, Shared), Pass::Lend(none, Shared)], |entry| {
        entry.open::<(ByProjection<Maybe>, ByProjection<Maybe>), ()>(|cx, (some, none), ()| {
            for lent in [some, none] {
                read.push(match lent {
                    MaybeRef::Some(tally) => Some(Arc::strong_count(&tally.0)),
                    MaybeRef::None => None,
                });
            }
            cx.finish(())
        })
    });
    assert_eq!(read, [Some(2), None]);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn viewed_tuple<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let tuple = produce_labelled(&mut frame, "owned", 4);

    let mut read = None;
    frame.call(&[Pass::Give(tuple)], |entry| {
        entry.open::<(ByUniform<Labelled>,), ()>(|cx, (held,), ()| {
            let view = held.into_view(&cx);
            read = Some((view.label.into_inner(&cx), view.count.into_inner(&cx)));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((String::from("owned"), 4)));
}

fn viewed_nested<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let nested = frame.call(&[], |entry| {
        entry.open::<(), ()>(|cx, (), ()| {
            cx.finish_projection(|result| {
                result.value(String::from("deep"));
                result.value(5i64);
                result.tuple(2);
                result.value(false);
                result.tuple(2);
            })
        })
    });

    let mut read = None;
    frame.call(&[Pass::Give(nested)], |entry| {
        entry.open::<(ByUniform<Nested>,), ()>(|cx, (held,), ()| {
            let view = held.into_view(&cx);
            let inner = view.inner.into_view(&cx);
            read = Some((*inner.label.open(&cx) == "deep", inner.count.into_inner(&cx), *view.flag.open(&cx)));
            cx.finish(())
        })
    });

    assert_eq!(read, Some((true, 5, false)));
}

fn viewed_variant<R: Runtime<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let some = produce_maybe(&mut frame, Some(value));
    let none = produce_maybe(&mut frame, None);

    let mut read = Vec::new();
    frame.call(&[Pass::Give(some), Pass::Give(none)], |entry| {
        entry.open::<(ByUniform<Maybe>, ByUniform<Maybe>), ()>(|cx, (some, none), ()| {
            for held in [some, none] {
                read.push(matches!(held.into_view(&cx), MaybeOwned::Some(_)));
            }
            cx.finish(())
        })
    });

    assert_eq!(read, [true, false]);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn awaited_guard_held<R: Runtime<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let text = produce(&mut frame, String::from("held across a wait"));

    let length = executor().block_on(frame.call_awaited(&[Pass::Lend(text, Shared)], async |entry| {
        let body = entry.open_awaited::<(ByRef<String>,), (), _>(|cx, (lent,), ()| async move {
            cx.sleep(Duration::from_millis(1)).await;
            let length = lent.len() as i64;
            cx.finish(length)
        });
        sent(body).await
    }));

    assert_eq!(read::<_, i64>(&mut frame, length), 18);
    assert_eq!(read::<_, String>(&mut frame, text), "held across a wait");
}

/// A variable holding the function `f`, as a program holds a function an
/// extern returned.
fn produce_fn<R>(frame: &mut Frame<'_, R>, f: RustFn<R>) -> usize
where
    R: RustFns<Value = Word>,
{
    frame.call(&[], |entry| entry.open::<(), ()>(|cx, (), ()| cx.finish_rust_fn(f)))
}

/// The length of a `String` lent to it.
fn length_of<R: Runtime>((text,): (Guard<'_, R::Shared<'_, String>>,)) -> i64 {
    text.len() as i64
}

/// `length_of`, as a Rust function.
fn length<R: RustFns>() -> RustFn<R> {
    RustFn::new::<(ByRef<String>,), i64>(length_of::<R>)
}

fn rust_fn_called<R: RustFns<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let function = produce_fn(&mut frame, length());
    let short = produce(&mut frame, String::from("four"));
    let long = produce(&mut frame, String::from("eleven long"));

    let first = frame.call_function(function, &[Pass::Lend(short, Shared)]);
    let second = frame.call_function(function, &[Pass::Lend(long, Shared)]);

    assert_eq!(read::<_, i64>(&mut frame, first), 4);
    assert_eq!(read::<_, i64>(&mut frame, second), 11);
    assert_eq!(read::<_, String>(&mut frame, short), "four");
}

fn rust_fn_released<R: RustFns<Value = Word>>(runtime: &R) {
    let (captured, drops) = tally();
    let mut frame = Frame::new(runtime);
    let function = produce_fn(
        &mut frame,
        RustFn::new::<(ByValue<i64>,), i64>(move |(n,)| n + Arc::strong_count(&captured.0) as i64),
    );

    let forty = produce(&mut frame, 40i64);
    let called = frame.call_function(function, &[Pass::Give(forty)]);
    assert_eq!(read::<_, i64>(&mut frame, called), 42);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(frame);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

fn callback_lent<R: CallsBack + RustFns<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let function = produce_fn(&mut frame, length());
    let text = produce(&mut frame, String::from("lent to a callback"));

    let called = frame.call(&[Pass::Lend(text, Shared), Pass::Function(function)], |entry| {
        entry.open::<(ByRef<String>, ByCallback<(Lent<String>,), i64, ()>), ()>(|cx, (text, callback), ()| {
            let first = callback.call(&cx, (&*text,));
            let second = callback.call(&cx, (&*text,));
            cx.finish(first + second)
        })
    });

    assert_eq!(read::<_, i64>(&mut frame, called), 36);
    assert_eq!(read::<_, String>(&mut frame, text), "lent to a callback");
}

fn append_written<R: Runtime>((mut text,): (GuardMut<'_, R::Exclusive<'_, String>>,)) {
    text.push_str(" and written");
}

fn words_in<R: Runtime>((text,): (Guard<'_, R::SharedStr<'_>>,)) -> i64 {
    text.split(' ').count() as i64
}

fn callback_lent_mut<R: CallsBack + RustFns<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let append = produce_fn(
        &mut frame,
        RustFn::new::<(ByMut<String>,), ()>(append_written::<R>),
    );
    let text = produce(&mut frame, String::from("lent"));

    frame.call(&[Pass::Lend(text, Exclusive), Pass::Function(append)], |entry| {
        entry.open::<(ByMut<String>, ByCallback<(LentMut<String>,), (), ()>), ()>(|cx, (mut text, callback), ()| {
            callback.call(&cx, (&mut *text,));
            cx.finish(())
        })
    });

    assert_eq!(read::<_, String>(&mut frame, text), "lent and written");
}

fn callback_lent_text<R: CallsBack + RustFns<Value = Word>>(runtime: &R) {
    let mut frame = Frame::new(runtime);
    let count = produce_fn(
        &mut frame,
        RustFn::new::<(ByStr,), i64>(words_in::<R>),
    );
    let text = produce(&mut frame, String::from("three short words and more"));

    let called = frame.call(&[Pass::Lend(text, Shared), Pass::Function(count)], |entry| {
        entry.open::<(ByStr, ByCallback<(Lent<str>,), i64, ()>), ()>(|cx, (text, callback), ()| {
            let words = callback.call(&cx, (&text[..17],));
            cx.finish(words)
        })
    });

    assert_eq!(read::<_, i64>(&mut frame, called), 3);
}

fn callback_awaited<R: CallsBack + RustFns<Value = Word>>(runtime: &R) {
    let (value, drops) = tally();
    let mut frame = Frame::new(runtime);
    let consume = produce_fn(
        &mut frame,
        RustFn::new::<(ByValue<String>, ByValue<Tally>), i64>(|(text, tally)| {
            drop(tally);
            text.len() as i64
        }),
    );

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
#[should_panic(expected = "a I64 value is called as a closure")]
fn the_adapter_refuses_a_callback_that_is_no_closure() {
    let runtime = Adapter::new(WordRuntime::default());
    let mut frame = Frame::new(&runtime);
    let variable = produce(&mut frame, 7i64);

    frame.call(&[Pass::Function(variable)], |entry| {
        entry.open::<(ByCallback<(), i64, ()>,), ()>(|cx, (callback,), ()| {
            let called = callback.call(&cx, ());
            cx.finish(called)
        })
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
