//! A run keeps a frame's cells while a task that frame spawned is in flight
//! (RFC-0046 rule 3, RFC-0079 rule 9).
//!
//! Every end of a frame — return, trap, its run dropped — reads the frame's
//! `Tally`: none aloft, the cells are released at once; any, they go to the
//! run's `Flight`. The tally, not the MIR's shape, is what the return reads:
//! a body that returns past a `Spawn` whose `Eval` it never ran keeps its
//! cells as a trap does.
//!
//! A frame that ends with tasks aloft adds them to the frame it ran within,
//! because a task may hold a loan on any frame below its spawner: that
//! frame's end then hands off too.
//!
//! A run that ends before an `Eval` does not wait for the task: that is a
//! decision. An executor cannot stop a future being polled on another thread
//! or a sync handler running on a pool thread, so waiting would block the
//! thread that drops the run, and on a current-thread runtime it would
//! deadlock against the task it waits for. The frame hands its cells to the
//! run's `Flight` instead, and the last task to finish releases them.
//!
//! A run and every run it spawns share one `Flight`, because a loan passed
//! down to a spawned run's own task names the outer run's cells.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use parking_lot::Mutex;

use crate::regs::Store;
use crate::runtime::AcvusRuntime;
use crate::value::{HandleValue, Value};

#[derive(Default)]
pub struct Flight {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    tasks: usize,
    kept: Vec<Store>,
}

impl Flight {
    pub(crate) fn new() -> Arc<Flight> {
        Arc::new(Flight::default())
    }

    pub(crate) fn start(self: &Arc<Flight>) -> Flying {
        self.state.lock().tasks += 1;
        Flying(Arc::clone(self))
    }

    fn release_after_tasks(&self, store: Store) {
        let mut state = self.state.lock();
        if state.tasks > 0 {
            state.kept.push(store);
            return;
        }
        drop(state);
        drop(store);
    }

    #[cfg(test)]
    fn counts(&self) -> Counts {
        let state = self.state.lock();
        Counts {
            tasks: state.tasks,
            kept: state.kept.len(),
        }
    }
}

#[cfg(test)]
#[derive(PartialEq, Eq, Debug)]
struct Counts {
    tasks: usize,
    kept: usize,
}

pub(crate) struct Flying(Arc<Flight>);

impl Drop for Flying {
    fn drop(&mut self) {
        let released = {
            let mut state = self.0.state.lock();
            state.tasks -= 1;
            match state.tasks {
                0 => std::mem::take(&mut state.kept),
                _ => Vec::new(),
            }
        };
        drop(released);
    }
}

/// Work handed to an executor, holding one count of its run's `Flight`.
pub(crate) struct Aloft<W> {
    work: W,
    flying: Flying,
}

impl<W> Aloft<W> {
    pub(crate) fn new(work: W, flying: Flying) -> Aloft<W> {
        Aloft { work, flying }
    }
}

impl<F> Aloft<F>
where
    F: FnOnce() -> Value,
{
    pub(crate) fn run(self) -> Value {
        let Aloft { work, flying } = self;
        let value = work();
        drop(flying);
        value
    }
}

impl<F> Future for Aloft<F>
where
    F: Future<Output = Value> + Unpin,
{
    type Output = Value;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Value> {
        Pin::new(&mut self.get_mut().work).poll(cx)
    }
}

/// The tasks a frame spawned and has not evaluated, and those the frames
/// that ran within it ended with.
pub(crate) struct Tally {
    aloft: AtomicUsize,
    within: Option<Arc<Tally>>,
}

impl Tally {
    /// The tally of a run's host, which no frame's end reads.
    pub(crate) fn outermost() -> Arc<Tally> {
        Arc::new(Tally {
            aloft: AtomicUsize::new(0),
            within: None,
        })
    }

    fn within(enclosing: &Arc<Tally>) -> Arc<Tally> {
        Arc::new(Tally {
            aloft: AtomicUsize::new(0),
            within: Some(Arc::clone(enclosing)),
        })
    }

    pub(crate) fn spawned(self: &Arc<Tally>) -> Unevaluated {
        self.aloft.fetch_add(1, Ordering::AcqRel);
        Unevaluated(Arc::clone(self))
    }

    fn aloft(&self) -> usize {
        self.aloft.load(Ordering::Acquire)
    }
}

/// One task in its spawning frame's `Tally`. Dropped without `evaluated` —
/// its handle lost to a trap, or its `Eval` never reached — the task stays
/// counted.
pub(crate) struct Unevaluated(Arc<Tally>);

impl Unevaluated {
    pub(crate) fn evaluated(self) {
        self.0.aloft.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What a `Handle` register holds: the executor's handle and the task's
/// count in the frame that spawned it.
pub(crate) struct Launched {
    pub(crate) handle: HandleValue,
    pub(crate) unevaluated: Unevaluated,
}

/// The cells of a frame that can suspend.
pub(crate) struct FrameCells {
    store: Store,
    tally: Arc<Tally>,
    flight: Arc<Flight>,
}

impl FrameCells {
    /// The frame's cells, and the runtime its body runs on: every spawn
    /// through that runtime counts in this frame's tally.
    pub(crate) fn open(store: Store, rt: &AcvusRuntime) -> (FrameCells, AcvusRuntime) {
        let cells = FrameCells::within(store, &rt.tally, &rt.flight);
        let within = AcvusRuntime::new(
            Arc::clone(&rt.shared),
            Arc::clone(&rt.page),
            Arc::clone(&rt.flight),
            Arc::clone(&cells.tally),
        );
        (cells, within)
    }

    fn within(store: Store, enclosing: &Arc<Tally>, flight: &Arc<Flight>) -> FrameCells {
        FrameCells {
            store,
            tally: Tally::within(enclosing),
            flight: Arc::clone(flight),
        }
    }

    pub(crate) fn store(&mut self) -> &mut Store {
        &mut self.store
    }
}

impl Drop for FrameCells {
    fn drop(&mut self) {
        let aloft = self.tally.aloft();
        if aloft == 0 {
            return;
        }
        if let Some(enclosing) = &self.tally.within {
            enclosing.aloft.fetch_add(aloft, Ordering::AcqRel);
        }
        self.flight.release_after_tasks(std::mem::take(&mut self.store));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: Counts = Counts { tasks: 0, kept: 0 };

    /// How many tasks a flight counted when a `Witness` of it was dropped.
    #[derive(Clone, Default)]
    struct Seen(Arc<Mutex<Option<usize>>>);

    impl Seen {
        fn tasks_at_drop(&self) -> Option<usize> {
            *self.0.lock()
        }
    }

    struct Witness {
        flight: Arc<Flight>,
        seen: Seen,
    }

    impl Drop for Witness {
        fn drop(&mut self) {
            *self.seen.0.lock() = Some(self.flight.counts().tasks);
        }
    }

    #[test]
    fn a_dropped_work_item_is_gone_before_its_count() {
        let flight = Flight::new();
        let seen = Seen::default();
        let witness = Witness {
            flight: Arc::clone(&flight),
            seen: seen.clone(),
        };
        let aloft = Aloft::new(
            move || -> Value {
                let _held = &witness;
                Value::unit()
            },
            flight.start(),
        );
        drop(aloft);
        assert_eq!(seen.tasks_at_drop(), Some(1));
        assert_eq!(flight.counts(), NONE);
    }

    #[test]
    fn a_work_item_that_panics_is_gone_before_its_count() {
        let flight = Flight::new();
        let seen = Seen::default();
        let witness = Witness {
            flight: Arc::clone(&flight),
            seen: seen.clone(),
        };
        let aloft = Aloft::new(
            move || -> Value {
                let _held = &witness;
                panic!("the work traps")
            },
            flight.start(),
        );
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| aloft.run()));
        assert!(unwound.is_err());
        assert_eq!(seen.tasks_at_drop(), Some(1));
        assert_eq!(flight.counts(), NONE);
    }

    fn frame(enclosing: &Arc<Tally>, flight: &Arc<Flight>) -> FrameCells {
        FrameCells::within(Store::new(), enclosing, flight)
    }

    #[test]
    fn a_frame_that_ends_with_a_task_aloft_keeps_its_cells_until_the_last_task_ends() {
        let flight = Flight::new();
        let cells = frame(&Tally::outermost(), &flight);
        let first = flight.start();
        let second = flight.start();
        drop(cells.tally.spawned());
        drop(cells);
        assert_eq!(flight.counts(), Counts { tasks: 2, kept: 1 });
        drop(first);
        assert_eq!(flight.counts(), Counts { tasks: 1, kept: 1 });
        drop(second);
        assert_eq!(flight.counts(), NONE);
    }

    #[test]
    fn a_frame_whose_tasks_were_evaluated_releases_its_cells_at_once() {
        let flight = Flight::new();
        let _elsewhere = flight.start();
        let cells = frame(&Tally::outermost(), &flight);
        cells.tally.spawned().evaluated();
        drop(cells);
        assert_eq!(flight.counts(), Counts { tasks: 1, kept: 0 });
    }

    #[test]
    fn a_frame_that_spawned_nothing_releases_its_cells_at_once() {
        let flight = Flight::new();
        let _elsewhere = flight.start();
        drop(frame(&Tally::outermost(), &flight));
        assert_eq!(flight.counts(), Counts { tasks: 1, kept: 0 });
    }

    #[test]
    fn a_frame_that_ends_with_a_task_aloft_leaves_it_to_the_frame_it_ran_within() {
        let flight = Flight::new();
        let _task = flight.start();
        let outer = frame(&Tally::outermost(), &flight);
        let inner = frame(&outer.tally, &flight);
        drop(inner.tally.spawned());
        drop(inner);
        assert_eq!(outer.tally.aloft(), 1);
        drop(outer);
        assert_eq!(flight.counts(), Counts { tasks: 1, kept: 2 });
    }
}
