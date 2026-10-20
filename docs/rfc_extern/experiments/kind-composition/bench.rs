//! What opening a nested value layer by layer costs, beside native Rust, with
//! layers held in boxes or, where they fit, in their `Value`. Boxes are
//! placed in a frame reused batch after batch, so no number carries an
//! allocation. Each pattern is a ring of `BATCH` values whose scalars sum to
//! the index of the next value to open, one cycle through all of them. Per
//! value:
//! - `project`: every value opened through every layer to its scalars, one after
//!   another, so the openings may overlap;
//! - `project chain`: the same, each opening's result naming the next value, so
//!   each waits for the one before: the latency of opening;
//! - `take chain`: values given beforehand, taken back by value along the
//!   chain;
//! - `round trip`: given and taken back.
//!
//! A projection opens as an extern body would, through `Uniform::project`, `Ref::project`
//! and each layer's `Open`. Prints nanoseconds and allocations per value.
//! Run in release, without the counts the tests use:
//! `cargo run --release --no-default-features --bin bench`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use model::boxes::{Boxes, Chained, Frame, Unchained, Value};
use model::{Cross, Runtime, Uniform};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded to `System` unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's premise.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's premise.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const BATCH: usize = 1024;
const BATCHES: usize = 2000;
/// Odd, so coprime to `BATCH`: stepping by it visits every value once.
const STRIDE: usize = 389;

type InValue = Boxes<Frame, Chained>;
type InBoxes = Boxes<Frame, Unchained>;

fn next(at: usize) -> i64 {
    i64::try_from((at + STRIDE) % BATCH).expect("an index fits an i64")
}

fn index(sum: Option<i64>) -> usize {
    usize::try_from(sum.expect("every value of a ring is present")).expect("an index is not negative")
}

/// Runs `batch` `BATCHES` times; it returns the time of its measured part.
fn report(measure: &str, pattern: &str, how: &str, mut batch: impl FnMut() -> Duration) {
    (0..BATCHES / 10).for_each(|_| {
        batch();
    });
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let elapsed: Duration = (0..BATCHES).map(|_| batch()).sum();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations;
    let values = (BATCH * BATCHES) as f64;
    let per_value = elapsed.as_secs_f64() * 1e9 / values;
    let allocations_per_value = allocations as f64 / values;
    println!("{measure:<13} {pattern:<34} {how:<9} {per_value:>6.2} ns {allocations_per_value:>6.3} allocations");
}

fn timed(work: impl FnOnce()) -> Duration {
    let start = Instant::now();
    work();
    start.elapsed()
}

// The openers: each opens its value through every layer, by each layer's
// `Open`, to the sum of its scalars.

fn open_option<R>(held: &Uniform<R, Option<i64>>) -> Option<i64>
where
    R: Runtime,
{
    held.project(|option| option.as_ref().map(|number| number.project(|number| *number)))
}

fn open_option_2<R>(held: &Uniform<R, Option<Option<i64>>>) -> Option<i64>
where
    R: Runtime,
{
    held.project(|option| option.as_ref().and_then(open_option::<R>))
}

fn open_option_3<R>(held: &Uniform<R, Option<Option<Option<i64>>>>) -> Option<i64>
where
    R: Runtime,
{
    held.project(|option| option.as_ref().and_then(open_option_2::<R>))
}

fn open_pair<R>(held: &Uniform<R, (i64, i64)>) -> i64
where
    R: Runtime,
{
    held.project(|(first, second)| first.project(|first| *first) + second.project(|second| *second))
}

fn open_option_pair<R>(held: &Uniform<R, Option<(i64, i64)>>) -> Option<i64>
where
    R: Runtime,
{
    held.project(|option| option.as_ref().map(open_pair::<R>))
}

fn open_option_2_pair<R>(held: &Uniform<R, Option<Option<(i64, i64)>>>) -> Option<i64>
where
    R: Runtime,
{
    held.project(|option| option.as_ref().and_then(open_option_pair::<R>))
}

fn open_result_option_pair<R>(held: &Uniform<R, Result<Option<(i64, i64)>, i64>>) -> Option<i64>
where
    R: Runtime,
{
    held.project(|result| result.as_ref().ok().and_then(open_option_pair::<R>))
}

fn open_tuple_3<R>(held: &Uniform<R, (i64, (i64, (i64, i64)))>) -> Option<i64>
where
    R: Runtime,
{
    held.project(|(first, rest)| {
        let rest = rest.project(|(second, rest)| {
            second.project(|second| *second) + rest.project(|(third, fourth)| third.project(|third| *third) + fourth.project(|fourth| *fourth))
        });
        Some(first.project(|first| *first) + rest)
    })
}

/// One pattern: how a ring's value is made from the next index, and how it is
/// opened natively and through the kinds.
struct Pattern<T, N> {
    name: &'static str,
    make: fn(i64) -> T,
    native: N,
    in_value: fn(&Uniform<InValue, T>) -> Option<i64>,
    in_boxes: fn(&Uniform<InBoxes, T>) -> Option<i64>,
}

fn ring<T>(make: fn(i64) -> T) -> Vec<T> {
    (0..BATCH).map(|at| make(next(at))).collect()
}

fn projections_native<T, N>(pattern: &Pattern<T, N>)
where
    N: Fn(&T) -> Option<i64>,
{
    let values = ring(pattern.make);
    report("project", pattern.name, "native", || {
        timed(|| {
            values.iter().for_each(|value| {
                black_box((pattern.native)(black_box(value)));
            });
        })
    });
    report("project chain", pattern.name, "native", || {
        timed(|| {
            let mut at = 0;
            (0..BATCH).for_each(|_| at = index((pattern.native)(&values[at])));
            black_box(at);
        })
    });
}

fn projections_through_kinds<R, T>(name: &str, how: &str, make: fn(i64) -> T, open: fn(&Uniform<R, T>) -> Option<i64>)
where
    R: Runtime,
    T: Cross<R> + 'static,
{
    let held: Vec<Uniform<R, T>> = ring(make).into_iter().map(Uniform::new).collect();
    report("project", name, how, || {
        timed(|| {
            held.iter().for_each(|value| {
                black_box(open(black_box(value)));
            });
        })
    });
    report("project chain", name, how, || {
        timed(|| {
            let mut at = 0;
            (0..BATCH).for_each(|_| at = index(open(&held[at])));
            black_box(at);
        })
    });
    drop(held);
    // SAFETY: every value the ring placed in the frame was released.
    unsafe { Frame::reset() };
}

fn takes_native<T, N>(pattern: &Pattern<T, N>)
where
    T: Clone,
    N: Fn(&T) -> Option<i64>,
{
    let values = ring(pattern.make);
    report("take chain", pattern.name, "native", || {
        let held = values.clone();
        timed(|| {
            let mut at = 0;
            (0..BATCH).for_each(|_| {
                let taken = held[at].clone();
                at = index((pattern.native)(&taken));
                black_box(taken);
            });
        })
    });
}

fn takes_through_kinds<R, T, N>(pattern: &Pattern<T, N>, how: &str)
where
    R: Runtime<Value = Value>,
    T: Cross<R> + Clone,
    N: Fn(&T) -> Option<i64>,
{
    let values = ring(pattern.make);
    report("take chain", pattern.name, how, || {
        let given: Vec<Value> = values.iter().cloned().map(<T as Cross<R>>::give).collect();
        let elapsed = timed(|| {
            let mut at = 0;
            (0..BATCH).for_each(|_| {
                // SAFETY: given as `T`; the chain takes each value once.
                let taken = unsafe { <T as Cross<R>>::take(given[at]) };
                at = index((pattern.native)(&taken));
                black_box(taken);
            });
        });
        // SAFETY: the chain took every value placed in the frame back.
        unsafe { Frame::reset() };
        elapsed
    });
}

fn round_trips<R, T>(name: &str, how: &str, make: fn(i64) -> T)
where
    R: Runtime,
    T: Cross<R> + Clone,
{
    let values = ring(make);
    report("round trip", name, how, || {
        let elapsed = timed(|| {
            values.iter().cloned().for_each(|value| {
                let given = <T as Cross<R>>::give(black_box(value));
                // SAFETY: given just now as `T`, and owned here.
                black_box(unsafe { <T as Cross<R>>::take(black_box(given)) });
            });
        });
        // SAFETY: every value given in the batch was taken back.
        unsafe { Frame::reset() };
        elapsed
    });
}

fn measure<T, N>(pattern: Pattern<T, N>)
where
    T: Cross<InValue> + Cross<InBoxes> + Clone + 'static,
    N: Fn(&T) -> Option<i64>,
{
    projections_native(&pattern);
    projections_through_kinds::<InValue, T>(pattern.name, "in value", pattern.make, pattern.in_value);
    projections_through_kinds::<InBoxes, T>(pattern.name, "in boxes", pattern.make, pattern.in_boxes);
    takes_native(&pattern);
    takes_through_kinds::<InValue, T, N>(&pattern, "in value");
    takes_through_kinds::<InBoxes, T, N>(&pattern, "in boxes");
    round_trips::<InValue, T>(pattern.name, "in value", pattern.make);
    round_trips::<InBoxes, T>(pattern.name, "in boxes", pattern.make);
}

fn main() {
    measure(Pattern {
        name: "Option<i64>",
        make: Some,
        native: |value: &Option<i64>| *value,
        in_value: open_option::<InValue>,
        in_boxes: open_option::<InBoxes>,
    });
    measure(Pattern {
        name: "Option<Option<i64>>",
        make: |next| Some(Some(next)),
        native: |value: &Option<Option<i64>>| value.flatten(),
        in_value: open_option_2::<InValue>,
        in_boxes: open_option_2::<InBoxes>,
    });
    measure(Pattern {
        name: "Option<Option<Option<i64>>>",
        make: |next| Some(Some(Some(next))),
        native: |value: &Option<Option<Option<i64>>>| value.flatten().flatten(),
        in_value: open_option_3::<InValue>,
        in_boxes: open_option_3::<InBoxes>,
    });
    measure(Pattern {
        name: "Option<(i64, i64)>",
        make: |next| Some((next, 0)),
        native: |value: &Option<(i64, i64)>| value.map(|(first, second)| first + second),
        in_value: open_option_pair::<InValue>,
        in_boxes: open_option_pair::<InBoxes>,
    });
    measure(Pattern {
        name: "Option<Option<(i64, i64)>>",
        make: |next| Some(Some((next, 0))),
        native: |value: &Option<Option<(i64, i64)>>| value.flatten().map(|(first, second)| first + second),
        in_value: open_option_2_pair::<InValue>,
        in_boxes: open_option_2_pair::<InBoxes>,
    });
    measure(Pattern {
        name: "Result<Option<(i64, i64)>, i64>",
        make: |next| Ok(Some((next, 0))),
        native: |value: &Result<Option<(i64, i64)>, i64>| value.ok().flatten().map(|(first, second)| first + second),
        in_value: open_result_option_pair::<InValue>,
        in_boxes: open_result_option_pair::<InBoxes>,
    });
    measure(Pattern {
        name: "(i64, (i64, (i64, i64)))",
        make: |next| (next, (0, (0, 0))),
        native: |(first, (second, (third, fourth))): &(i64, (i64, (i64, i64)))| Some(first + second + third + fourth),
        in_value: open_tuple_3::<InValue>,
        in_boxes: open_tuple_3::<InBoxes>,
    });
}
