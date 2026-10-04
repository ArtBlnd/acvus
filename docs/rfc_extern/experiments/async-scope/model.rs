//! A model of a safe runtime, and of the scope through which an extern lends
//! values to an asynchronous call (functions §3): the scope takes the callable
//! and the arguments by value, lends them to the call, and gives them back
//! with its result. The model has no `unsafe`, so memory safety holds by Rust
//! alone; what it checks is that the runtime keeps what a call borrows until
//! the call ends, whatever happens to the future that started it.
//!
//! One runtime per thread, so no `R` parameter. A `Value` is an index into the
//! runtime's slots. An acvus function value is a `Program`, what the compiler
//! would have compiled: it waits `yields` times, adds one to its exclusive
//! argument, and returns the length of its shared one.

#![forbid(unsafe_code)]

use std::any::Any;
use std::cell::RefCell;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Value(usize);

struct Slot {
    value: Option<Box<dyn Any>>,
    owned: bool,
    in_use: usize,
    exclusive: bool,
}

type Task = Pin<Box<dyn Future<Output = ()>>>;

#[derive(Default)]
struct Store {
    slots: Vec<Slot>,
    tasks: Vec<Task>,
}

thread_local! {
    static STORE: RefCell<Store> = RefCell::new(Store::default());
}

fn with_store<O>(f: impl FnOnce(&mut Store) -> O) -> O {
    STORE.with(|store| f(&mut store.borrow_mut()))
}

fn hold(value: Box<dyn Any>) -> Value {
    with_store(|store| {
        store.slots.push(Slot { value: Some(value), owned: true, in_use: 0, exclusive: false });
        Value(store.slots.len() - 1)
    })
}

/// Ends the owner of `value`; the runtime frees it once no call uses it.
fn end_owner(value: Value) {
    with_store(|store| {
        let slot = &mut store.slots[value.0];
        assert!(slot.owned, "a value released twice");
        slot.owned = false;
        if slot.in_use == 0 {
            slot.value = None;
        }
    })
}

fn begin_use(value: Value, exclusive: bool) {
    with_store(|store| {
        let slot = &mut store.slots[value.0];
        assert!(slot.value.is_some(), "a released value lent to a call");
        assert!(!slot.exclusive, "a value lent while a call holds it exclusively");
        slot.in_use += 1;
        slot.exclusive |= exclusive;
    })
}

fn end_use(value: Value) {
    with_store(|store| {
        let slot = &mut store.slots[value.0];
        slot.in_use -= 1;
        if slot.in_use == 0 {
            slot.exclusive = false;
        }
        if slot.in_use == 0 && !slot.owned {
            slot.value = None;
        }
    })
}

fn read<X: 'static, O>(value: Value, f: impl FnOnce(&X) -> O) -> O {
    with_store(|store| {
        let held = store.slots[value.0].value.as_ref().expect("a value read after its release");
        f(held.downcast_ref::<X>().expect("a value read as another type"))
    })
}

fn write<X: 'static, O>(value: Value, f: impl FnOnce(&mut X) -> O) -> O {
    with_store(|store| {
        let held = store.slots[value.0].value.as_mut().expect("a value written after its release");
        f(held.downcast_mut::<X>().expect("a value written as another type"))
    })
}

/// The number of values the runtime still holds.
pub fn live() -> usize {
    with_store(|store| store.slots.iter().filter(|slot| slot.value.is_some()).count())
}

fn spawn(task: Task) {
    with_store(|store| store.tasks.push(task));
}

/// Polls every task once, keeping those still pending. Whether any is left.
pub fn run_tasks() -> bool {
    let tasks = with_store(|store| std::mem::take(&mut store.tasks));
    let mut context = Context::from_waker(Waker::noop());
    let mut pending: Vec<Task> = tasks.into_iter().filter_map(|mut task| task.as_mut().poll(&mut context).is_pending().then_some(task)).collect();
    with_store(|store| {
        pending.append(&mut store.tasks);
        store.tasks = pending;
        !store.tasks.is_empty()
    })
}

/// Runs `future` to its end, running the runtime's tasks between polls.
pub fn block_on<O>(future: impl Future<Output = O>) -> O {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(out) = future.as_mut().poll(&mut context) {
            return out;
        }
        run_tasks();
    }
}

/// Polls `future` once, running the runtime's tasks after.
pub fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    let mut context = Context::from_waker(Waker::noop());
    let polled = future.poll(&mut context);
    run_tasks();
    polled
}

struct YieldNow(bool);

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        match self.0 {
            true => Poll::Ready(()),
            false => {
                self.0 = true;
                Poll::Pending
            }
        }
    }
}

/// A value the runtime holds, its type kept by `X`.
pub struct Uniform<X> {
    value: Value,
    marker: PhantomData<fn(X) -> X>,
}

impl<X: 'static> Uniform<X> {
    pub fn new(x: X) -> Self {
        Uniform { value: hold(Box::new(x)), marker: PhantomData }
    }

    pub fn get(&self) -> X
    where
        X: Clone,
    {
        let held = with_store(|store| store.slots[self.value.0].exclusive);
        assert!(!held, "a value read by its owner while a call holds it exclusively");
        read(self.value, |x: &X| x.clone())
    }
}

impl<X> Drop for Uniform<X> {
    fn drop(&mut self) {
        end_owner(self.value);
    }
}

/// An acvus function value as the runtime holds it.
pub struct Program {
    pub yields: usize,
}

/// A shared reference a scope lends to a call: invariant in `'s`, so only
/// that scope's loans fit its call.
pub struct Lent<'s, X> {
    value: Value,
    marker: PhantomData<(fn(X) -> X, fn(&'s ()) -> &'s ())>,
}

/// An exclusive reference a scope lends to a call.
pub struct LentMut<'s, X> {
    value: Value,
    marker: PhantomData<(fn(X) -> X, fn(&'s ()) -> &'s ())>,
}

/// The negative control: an exclusive loan covariant in `'s`, as a plain
/// reference is, so a longer loan shortens into a scope's.
pub struct CovLentMut<'s, X> {
    value: Value,
    marker: PhantomData<(fn(X) -> X, &'s ())>,
}

impl<'s, X> CovLentMut<'s, X> {
    /// Lends the caller's own value, outside any scope: the hazard the
    /// control exposes.
    pub fn of(owner: &'s mut Uniform<X>) -> Self {
        CovLentMut { value: owner.value, marker: PhantomData }
    }
}

/// The callable a scope lends to a call.
pub struct Calling<'s> {
    function: Value,
    marker: PhantomData<fn(&'s ()) -> &'s ()>,
}

/// An asynchronous call, started when it is made: a started call is taken
/// on every path from it (rfc_compiler effects §6), so dropping this future
/// leaves the call running in the runtime.
pub struct Call<'s> {
    done: Rc<RefCell<Option<usize>>>,
    marker: PhantomData<fn(&'s ()) -> &'s ()>,
}

impl Future for Call<'_> {
    type Output = usize;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<usize> {
        match self.done.borrow_mut().take() {
            Some(out) => Poll::Ready(out),
            None => Poll::Pending,
        }
    }
}

fn start(function: Value, counter: Value, text: Value) -> Rc<RefCell<Option<usize>>> {
    let done = Rc::new(RefCell::new(None));
    let yields = read(function, |program: &Program| program.yields);
    begin_use(function, false);
    begin_use(counter, true);
    begin_use(text, false);
    let finished = done.clone();
    spawn(Box::pin(async move {
        for _ in 0..yields {
            YieldNow(false).await;
        }
        write(counter, |n: &mut i64| *n += 1);
        let len = read(text, |s: &String| s.len());
        *finished.borrow_mut() = Some(len);
        for used in [function, counter, text] {
            end_use(used);
        }
    }));
    done
}

impl<'s> Calling<'s> {
    pub fn call(self, counter: LentMut<'s, i64>, text: Lent<'s, String>) -> Call<'s> {
        Call { done: start(self.function, counter.value, text.value), marker: PhantomData }
    }

    /// The negative control's call, taking a covariant exclusive loan.
    pub fn call_covariant(self, counter: CovLentMut<'s, i64>, text: Lent<'s, String>) -> Call<'s> {
        Call { done: start(self.function, counter.value, text.value), marker: PhantomData }
    }
}

/// What a scope lends of the arguments it holds.
pub trait Lend {
    type Lent<'s>
    where
        Self: 's;

    fn lend(&self) -> Self::Lent<'_>;
}

/// An argument a scope lends exclusively.
pub struct Exclusive<X>(pub Uniform<X>);

/// An argument a scope lends shared.
pub struct Shared<X>(pub Uniform<X>);

impl<X: 'static> Lend for Exclusive<X> {
    type Lent<'s> = LentMut<'s, X>;

    fn lend(&self) -> LentMut<'_, X> {
        LentMut { value: self.0.value, marker: PhantomData }
    }
}

impl<X: 'static> Lend for Shared<X> {
    type Lent<'s> = Lent<'s, X>;

    fn lend(&self) -> Lent<'_, X> {
        Lent { value: self.0.value, marker: PhantomData }
    }
}

impl<A: Lend, B: Lend> Lend for (A, B) {
    type Lent<'s>
        = (A::Lent<'s>, B::Lent<'s>)
    where
        Self: 's;

    fn lend(&self) -> Self::Lent<'_> {
        (self.0.lend(), self.1.lend())
    }
}

impl Uniform<Program> {
    /// Lends the function and `args` to the calls `start` makes, and gives
    /// both back with its result.
    pub async fn scope<A, O>(self, args: A, start: impl for<'s> AsyncFnOnce(Calling<'s>, A::Lent<'s>) -> O) -> (Uniform<Program>, A, O)
    where
        A: Lend,
    {
        let out = start(Calling { function: self.value, marker: PhantomData }, args.lend()).await;
        (self, args, out)
    }
}
