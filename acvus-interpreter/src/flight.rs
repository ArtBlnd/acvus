//! A run keeps its frames' cells while a task it spawned is in flight
//! (RFC-0046 rule 3, RFC-0079 rule 9).
//!
//! A run that traps or is dropped before an `Eval` does not wait for the
//! task: that is a decision. An executor cannot stop a future being polled
//! on another thread or a sync handler running on a pool thread, so waiting
//! would block the thread that drops the run, and on a current-thread
//! runtime it would deadlock against the task it waits for. The frame hands
//! its cells to the run's `Flight` instead, and the last task to finish
//! releases them.
//!
//! A run and every run it spawns share one `Flight`, because a loan passed
//! down to a spawned run's own task names the outer run's cells.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use parking_lot::Mutex;

use crate::regs::Store;
use crate::value::Value;

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

/// The cells of a frame that can suspend.
pub(crate) struct FrameCells {
    store: Store,
    abandoned_to: Option<Arc<Flight>>,
}

impl FrameCells {
    pub(crate) fn new(store: Store, flight: &Arc<Flight>) -> FrameCells {
        FrameCells {
            store,
            abandoned_to: Some(Arc::clone(flight)),
        }
    }

    pub(crate) fn store(&mut self) -> &mut Store {
        &mut self.store
    }

    /// Releases the cells at once. A body reaches its return only past the
    /// `Eval` of every `Spawn` it issued, since `optimize::spawn_split`
    /// puts the two in one block, and acvus-mir's borrow check ends a loan
    /// a `Handle` holds at its `Eval`: no task holds a loan on these cells.
    pub(crate) fn returned(mut self) {
        self.abandoned_to = None;
    }
}

impl Drop for FrameCells {
    fn drop(&mut self) {
        if let Some(flight) = self.abandoned_to.take() {
            flight.release_after_tasks(std::mem::take(&mut self.store));
        }
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

    #[test]
    fn abandoned_cells_stay_until_the_last_task_ends() {
        let flight = Flight::new();
        let first = flight.start();
        let second = flight.start();
        drop(FrameCells::new(Store::new(), &flight));
        assert_eq!(flight.counts(), Counts { tasks: 2, kept: 1 });
        drop(first);
        assert_eq!(flight.counts(), Counts { tasks: 1, kept: 1 });
        drop(second);
        assert_eq!(flight.counts(), NONE);
    }

    #[test]
    fn returned_cells_are_released_at_once() {
        let flight = Flight::new();
        let _task = flight.start();
        FrameCells::new(Store::new(), &flight).returned();
        assert_eq!(flight.counts(), Counts { tasks: 1, kept: 0 });
    }

    #[test]
    fn cells_abandoned_with_no_task_in_flight_are_released_at_once() {
        let flight = Flight::new();
        drop(FrameCells::new(Store::new(), &flight));
        assert_eq!(flight.counts(), NONE);
    }
}
