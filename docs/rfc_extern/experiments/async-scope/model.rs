//! A model of a safe runtime, and of the scope through which an extern lends
//! values to an asynchronous call (functions §3): the scope takes the callable
//! and the arguments by value, lends them to the call, and gives them back
//! with its result. The call runs in place, as the future that awaits it is
//! polled, so a future dropped between its polls abandons the call where it
//! suspended. The model has no `unsafe`, so memory safety holds by Rust alone;
//! what it checks is what a scope gives back, whatever happens to its call.
//!
//! One runtime per thread, so no `R` parameter. A `Value` is an index into the
//! runtime's slots. An acvus function value is a `Program`, what the compiler
//! would have compiled: it waits `yields` times, adds one to its exclusive
//! argument, and returns the length of its shared one. One that takes the
//! argument out across its waits is what the compiler refuses (rfc_compiler
//! ownership §4); it is here to show what that rule keeps out.

#![forbid(unsafe_code)]

use std::any::Any;
use std::cell::RefCell;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Value(usize);

struct Slot {
    value: Option<Box<dyn Any>>,
    owned: bool,
    in_use: usize,
    exclusive: bool,
}

#[derive(Default)]
struct Store {
    slots: Vec<Slot>,
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
/// A place released empty is one an optimized runtime would drop again.
fn end_owner(value: Value) {
    with_store(|store| {
        let slot = &mut store.slots[value.0];
        assert!(slot.owned, "a value released twice");
        assert!(slot.value.is_some(), "a value released with its content taken out");
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

/// Takes the content of `value`'s place out, leaving the place empty.
fn take_out<X: 'static>(value: Value) -> Box<X> {
    with_store(|store| {
        store.slots[value.0].value.take().expect("a value taken out twice").downcast::<X>().expect("a value taken as another type")
    })
}

/// Fills `value`'s place again.
fn put_back<X: 'static>(value: Value, content: X) {
    with_store(|store| store.slots[value.0].value = Some(Box::new(content)))
}

/// The number of values the runtime still holds.
pub fn live() -> usize {
    with_store(|store| store.slots.iter().filter(|slot| slot.value.is_some()).count())
}

/// Runs `future` to its end.
pub fn block_on<O>(future: impl Future<Output = O>) -> O {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(out) = future.as_mut().poll(&mut context) {
            return out;
        }
    }
}

/// Polls `future` once.
pub fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    let mut context = Context::from_waker(Waker::noop());
    future.poll(&mut context)
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
    /// Whether its body takes its exclusive argument out before its waits
    /// and writes it back after them.
    pub takes_across_waits: bool,
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

/// An asynchronous call, running in place as it is polled. Dropped before
/// it ends, it abandons the call where the call suspended.
pub struct Call<'s> {
    body: Pin<Box<dyn Future<Output = usize>>>,
    uses: [Value; 3],
    ended: bool,
    marker: PhantomData<fn(&'s ()) -> &'s ()>,
}

impl Future for Call<'_> {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<usize> {
        let polled = self.body.as_mut().poll(context);
        if polled.is_ready() {
            self.ended = true;
            self.uses.iter().for_each(|used| end_use(*used));
        }
        polled
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.uses.iter().for_each(|used| end_use(*used));
        }
    }
}

fn start<'s>(function: Value, counter: Value, text: Value) -> Call<'s> {
    let (yields, takes_across_waits) = read(function, |program: &Program| (program.yields, program.takes_across_waits));
    begin_use(function, false);
    begin_use(counter, true);
    begin_use(text, false);
    let body = Box::pin(async move {
        let taken = takes_across_waits.then(|| take_out::<i64>(counter));
        for _ in 0..yields {
            YieldNow(false).await;
        }
        match taken {
            Some(n) => put_back(counter, *n + 1),
            None => write(counter, |n: &mut i64| *n += 1),
        }
        read(text, |s: &String| s.len())
    });
    Call { body, uses: [function, counter, text], ended: false, marker: PhantomData }
}

impl<'s> Calling<'s> {
    pub fn call(self, counter: LentMut<'s, i64>, text: Lent<'s, String>) -> Call<'s> {
        start(self.function, counter.value, text.value)
    }

    /// The negative control's call, taking a covariant exclusive loan.
    pub fn call_covariant(self, counter: CovLentMut<'s, i64>, text: Lent<'s, String>) -> Call<'s> {
        start(self.function, counter.value, text.value)
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
