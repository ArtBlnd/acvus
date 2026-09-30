//! `WordRuntime`'s values and its operations.


use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use acvus_extern::callback::{CallArgs, CallsBack, LoanForm, MovedArgs};
use acvus_extern::{Settled, SiteShape};
use acvus_extern::runtime::{Crosses, Mutability, Runtime};
use acvus_extern::done::Done;
use acvus_extern::register::{AwaitedFn, Declared, ExternFn, Refused, Registers};
use acvus_utils::{Interner, QualifiedRef};

use crate::boundary::{self, Held, Kinds, Word};

/// The runtime: its interner, and nothing else; every value's storage is its
/// own allocation.
#[derive(Default)]
pub struct WordRuntime {
    interner: Interner,
    /// The functions registered at this runtime, by path.
    registered: HashMap<QualifiedRef, Registered<WordRuntime>>,
}

/// A function registered at the runtime: returning, or awaited, whose call's
/// future this runtime boxes: its tests await few calls, and a box is the
/// plainest way to hold a future whose type is each function's own.
enum Registered<R: Runtime> {
    Returning(Box<dyn ExternFn<R>>),
    Awaited(Box<dyn Boxed<R>>),
}

/// An awaited function whose call's future is boxed.
trait Boxed<R: Runtime>: Send + Sync {
    fn call<'c>(
        &'c self,
        runtime: &'c R,
        args: &'c [R::Value],
        site: &'c SiteShape<R>,
    ) -> Pin<Box<dyn Future<Output = Done<R>> + Send + 'c>>;
}

impl<R: Runtime, F: AwaitedFn<R>> Boxed<R> for F {
    fn call<'c>(
        &'c self,
        runtime: &'c R,
        args: &'c [R::Value],
        site: &'c SiteShape<R>,
    ) -> Pin<Box<dyn Future<Output = Done<R>> + Send + 'c>> {
        Box::pin(AwaitedFn::call(self, runtime, args, site))
    }
}

impl WordRuntime {
    /// Registers `f` at `path`, refused where a function is registered there.
    fn enter(&mut self, path: QualifiedRef, f: Registered<WordRuntime>) -> Result<(), Refused> {
        if self.registered.contains_key(&path) {
            return Err(Refused::PathTaken(path));
        }
        self.registered.insert(path, f);
        Ok(())
    }
}

impl Registers for WordRuntime {
    fn register<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: ExternFn<Self> + Declared<Self>,
    {
        let path = Declared::path(&f, &self.interner);
        self.enter(path, Registered::Returning(Box::new(f)))
    }

    fn register_awaited<F>(&mut self, f: F) -> Result<(), Refused>
    where
        F: AwaitedFn<Self> + Declared<Self>,
    {
        let path = Declared::path(&f, &self.interner);
        self.enter(path, Registered::Awaited(Box::new(f)))
    }

    fn registered(&self, path: QualifiedRef) -> Option<&dyn ExternFn<Self>> {
        match self.registered.get(&path)? {
            Registered::Returning(f) => Some(&**f),
            Registered::Awaited(_) => None,
        }
    }

    fn call_awaited<'c>(
        &'c self,
        path: QualifiedRef,
        args: &'c [Word],
        site: &'c SiteShape<Self>,
    ) -> Option<impl Future<Output = Done<Self>> + Send + 'c> {
        match self.registered.get(&path)? {
            Registered::Awaited(f) => Some(f.call(self, args, site)),
            Registered::Returning(_) => None,
        }
    }
}

impl Runtime for WordRuntime {
    type Value = Word;

    type Opaque = Kinds;
    type Primitive = Kinds;
    type String = Kinds;
    type Option = Kinds;
    type Result = Kinds;
    type Tuple = Kinds;
    type Ref = Kinds;

    fn interner(&self) -> &Interner {
        &self.interner
    }

    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send {
        Deadline(Instant::now() + d)
    }
}

impl CallsBack for WordRuntime {
    fn call_now<'l, R, A, U>(&self, rt: &'l R, closure: Word, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        R: Runtime<Value = Word>,
        A: CallArgs<R>,
        U: LoanForm<R>,
        A: 'l,
        U: 'l,
    {
        let run = A::lend_run(args, rt, at).map(Held::kept).collect();
        let result = call_closure(rt, closure, run, A::lent().collect(), at);
        U::read(result, rt, at)
    }

    fn call_later<'a, R, A, U>(
        &self,
        rt: &'a R,
        closure: Word,
        args: A::Moved,
        at: Settled<'a>,
    ) -> impl Future<Output = U> + Send
    where
        R: Runtime<Value = Word>,
        A: MovedArgs<R>,
        U: Crosses<R>,
    {
        let run: Vec<Held> = A::move_run(args, rt, at).map(Held::kept).collect();
        let lent: Box<[Option<Mutability>]> = A::lent().collect();
        async move {
            let result = call_closure(rt, closure, run, lent, at);
            U::take(rt, result, at)
        }
    }
}

/// Calls the Rust function `closure` is with the argument run `run`, each
/// passed as `lent` says, and gives its result. A value given is the callee's
/// from here on; a reference value lent is the caller's, released after the
/// call.
fn call_closure<'a, R>(
    rt: &'a R,
    closure: Word,
    run: Vec<Held>,
    lent: Box<[Option<Mutability>]>,
    at: Settled<'a>,
) -> Word
where
    R: Runtime<Value = Word>,
{
    let mut values = Vec::with_capacity(run.len());
    let mut references = Vec::new();
    for (held, passed) in run.into_iter().zip(&lent) {
        match passed {
            Some(_) => {
                values.push(held.same());
                references.push(held);
            }
            None => values.push(held.hand_over()),
        }
    }
    let site = SiteShape::new(lent, Box::new([]));

    let done = boundary::callee::<R>(closure, at).call(rt, &values, &site);

    drop(references);
    done.into_value()
}


/// A wait until an instant. This runtime owns no timer and no executor, so
/// the future asks to be polled again until the instant has passed; the
/// crossing's tests wait for short times only, and a timer would bring a
/// thread or a dependency this runtime otherwise does without.
struct Deadline(Instant);

impl Future for Deadline {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.0 {
            return Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

